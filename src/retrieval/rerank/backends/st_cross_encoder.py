"""sentence-transformers ``CrossEncoder`` backend: ms-marco-MiniLM-L12-v2, mxbai-rerank-large-v2.

These two share an API but not an architecture, and the difference matters:

* ``ms-marco-MiniLM-L12-v2`` is a 33M ``BertForSequenceClassification``. Its
  ``config.json`` pins ``sbert_ce_default_activation_function`` to ``Identity``,
  so ``predict()`` returns the **raw logit** (the card's own example prints
  ``[9.218911, -4.0780287]`` — note the negative).

* ``mxbai-rerank-large-v2`` is a 1.54B ``Qwen2ForCausalLM``. Its ``modules.json``
  ends in a ``LogitScore`` module whose ``1_LogitScore/config.json`` sets
  ``{"true_token_id": 16, "false_token_id": 15}`` (the tokens ``"1"`` and ``"0"``),
  so ``predict()`` returns the **log-odds** ``logit("1") - logit("0")`` read at the
  last position. Its instruction lives in the repo's ``chat_template.jinja`` and is
  applied for us. **Left padding is mandatory** and is enforced by the ST
  Transformer module; we assert it rather than trust it.

Neither model has an instruction slot we control, so ``instruction`` must be None.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import gc

import torch
from .base import (
    Reranker,
    TruncationCounter,
    enable_tf32_if_fp32,
    resolve_device,
    resolve_dtype,
)
from reranker_registry import RerankerSpec

logger = logging.getLogger(__name__)


class STCrossEncoderReranker(Reranker):
    def __init__(self, spec: RerankerSpec, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        if spec.max_length is None:
            raise ValueError(f"[{spec.alias}] st_cross_encoder requires an explicit max_length.")

        self.alias = spec.alias
        self.spec = spec
        self.max_length: int = spec.max_length
        self.device = device or resolve_device()
        self.batch_size = batch_size
        self.truncation = TruncationCounter()
        self.tf32 = enable_tf32_if_fp32(spec.dtype, self.device)

        model_kwargs = {"dtype": resolve_dtype(spec.dtype, self.device)}
        self.model = CrossEncoder(
            spec.hf_id,
            max_length=self.max_length,
            device=self.device,
            model_kwargs=model_kwargs,
        )
        self.model.model.eval()

        tokenizer = self.model.tokenizer
        # mxbai is a causal LM read at the last position: right padding would put a
        # pad token there and silently return noise. ST claims to enforce left
        # padding for text-generation cross-encoders; verify instead of assuming.
        if spec.padding_side == "left" and tokenizer.padding_side != "left":
            logger.warning(
                "[%s] ST left the tokenizer at padding_side=%r; forcing 'left'.",
                spec.alias, tokenizer.padding_side,
            )
            tokenizer.padding_side = "left"

        logger.info(
            "[%s] CrossEncoder ready: max_length=%d, dtype=%s, padding_side=%s",
            spec.alias, self.max_length, spec.dtype, tokenizer.padding_side,
        )

    def _count_truncated(self, docs: Sequence[str], query: str) -> int:
        """Documents whose tokenized length exceeds what max_length leaves them.

        The effective document budget is per-query (it depends on the query's own token
        count), which is why this is computed here rather than cached. For ms-marco it
        lands around 490 of its hard 512 — a property of the model's position embeddings,
        reported rather than silently absorbed.

        Uses ``spec.template_overhead``, not a hardcoded constant: ms-marco wraps the pair
        in 3 BERT special tokens, but mxbai wraps it in an 84-token chat template.
        Assuming 4 for mxbai would make ``available`` too large and silently UNDER-report
        truncation — the opposite of the conservative direction.
        """
        tok = self.model.tokenizer
        query_len = len(tok(query, add_special_tokens=False, verbose=False)["input_ids"])
        available = self.max_length - query_len - self.spec.template_overhead
        lens = tok(list(docs), add_special_tokens=False, verbose=False)["input_ids"]
        return sum(1 for ids in lens if len(ids) > available)

    def score(self, query: str, docs: Sequence[str], instruction: str | None = None) -> list[float]:
        if instruction is not None:
            raise ValueError(f"[{self.alias}] has no controllable instruction slot.")
        if not docs:
            return []

        self.truncation.observe(len(docs), self._count_truncated(docs, query))

        pairs = [(query, doc) for doc in docs]
        # activation_fn=Identity is REQUIRED, not cosmetic.
        #
        # sentence-transformers defaults `CrossEncoder.activation_fn` to Sigmoid() unless
        # the repo pins otherwise. ms-marco pins Identity in its config.json, so leaving
        # this unset happened to be right for it. mxbai does not, so its documented
        # log-odds got squashed through a sigmoid and saturated:
        #
        #   default  : [1.0, 1.0, 0.878906, 0.050293, 0.012451]   <- 421/2758 docs at 1.0
        #   Identity : [10.125, 8.125, 2.0, -2.9375, -4.375]      <- the card's scores
        #
        # With 421 documents tied at exactly 1.0, the stable sort in
        # `rerank_ranked_list` returns the FIRST-STAGE order for the whole top-421 block:
        # mxbai's reranking became a silent no-op precisely where P@10 / nDCG@10 look.
        # Identity is a no-op for ms-marco (already pinned) and the fix for mxbai.
        # CrossEncoder.predict batches internally, so the shared score_with_oom_backoff
        # (which re-chunks a doc list) cannot reach the batch that actually OOMs. Halve
        # `batch_size` instead, and keep the reduced value: an OOM on one query means the
        # next query would OOM too. fp32 mxbai (1.5B) is the case that needs this.
        while True:
            try:
                scores = self.model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    activation_fn=torch.nn.Identity(),
                )
                break
            except torch.cuda.OutOfMemoryError:
                if self.batch_size <= 1:
                    raise
                gc.collect()
                torch.cuda.empty_cache()
                self.batch_size = max(1, self.batch_size // 2)
                logger.warning("[%s] CUDA OOM; retrying at batch_size=%d",
                               self.alias, self.batch_size)
        return [float(s) for s in scores]
