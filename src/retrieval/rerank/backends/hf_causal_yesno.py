"""Qwen3-Reranker-4B / -8B: decoder LLM scored from contrasted yes/no token logits.

The card's ``transformers`` recipe, reproduced exactly:

    prefix = '<|im_start|>system\\nJudge whether the Document meets the requirements
              based on the Query and the Instruct provided. Note that the answer can
              only be "yes" or "no".<|im_end|>\\n<|im_start|>user\\n'
    suffix = '<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'
    text   = '<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}'

    batch_scores = model(**inputs).logits[:, -1, :]
    batch_scores = log_softmax(stack([logits[:, no_id], logits[:, yes_id]], dim=1), dim=1)
    scores       = batch_scores[:, 1].exp()          # -> P(yes) in [0, 1]

Three details the card is explicit about and that fail *silently* if ignored:

* ``AutoTokenizer.from_pretrained(..., padding_side='left')`` — the score is read at
  ``logits[:, -1, :]``. ``tokenizer_config.json`` does **not** set ``padding_side``, so
  transformers' ``"right"`` default applies unless we override it. Right padding reads
  a pad position and returns plausible-looking noise for every pair.
* The yes/no tokens are **lowercase**, resolved via ``convert_tokens_to_ids``.
* Truncation happens on the *formatted single string*, so ``longest_first`` degenerates
  to tail truncation — the document's tail is what gets cut, never the instruction or query.

**Scored with transformers, never vLLM.** vllm#21681 reports Qwen3-Reranker scores
compressing under vLLM until relevant and irrelevant documents stop separating
(vLLM ``0.907, 0.810, 0.807, 0.076`` vs transformers ``0.957, 0.001, 0.000, 0.000``).
No maintainer root-cause exists as of 2026-07-08, and this study's headline metric is a
ranking metric — precisely what that bug destroys.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from .base import (
    Reranker,
    TruncationCounter,
    assert_padding_side,
    length_bucketed_batches,
    resolve_device,
    resolve_dtype,
    score_with_oom_backoff,
)
from reranker_registry import RerankerSpec

logger = logging.getLogger(__name__)

PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

DEFAULT_TOKEN_BUDGET = 65_536
MAX_BATCH_SIZE = 64


def format_instruction(instruction: str, query: str, doc: str) -> str:
    """The card's exact template. Field separator is a single ``\\n``.

    (The 4B card's *vLLM* snippet uses ``\\n\\n`` instead — an internal inconsistency
    in the card. We follow the transformers snippet, which is the path we score on.)
    """
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class HFCausalYesNoReranker(Reranker):
    def __init__(
        self,
        spec: RerankerSpec,
        device: str | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        attn_implementation: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        if spec.max_length is None:
            raise ValueError(f"[{spec.alias}] hf_causal_yesno requires an explicit max_length.")

        self.alias = spec.alias
        self.spec = spec
        self.max_length: int = spec.max_length
        self.device = device or resolve_device()
        self.token_budget = token_budget
        self.truncation = TruncationCounter()
        self.tf32 = False   # bf16 weights; TF32 only affects fp32 matmul

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, padding_side="left")
        assert_padding_side(self.tokenizer, "left", spec.alias)

        model_kwargs = {"dtype": resolve_dtype(spec.dtype, self.device)}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)
        self.model.to(self.device).eval()

        # Lowercase, per the card: tokenizer.convert_tokens_to_ids("yes") / ("no").
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.token_true_id is None or self.token_false_id is None:
            raise ValueError(f"[{spec.alias}] could not resolve the 'yes'/'no' token ids.")

        # Score = logit(yes) - logit(no), computed WITHOUT materialising the logits.
        #
        # The card reads `model(**inputs).logits[:, -1, :]` and returns
        # `log_softmax([no, yes])[:, 1].exp()` = P(yes) = sigmoid(logit_yes - logit_no).
        # Two problems with that here:
        #
        #  1. Saturation. P(yes) rounds to exactly 1.0 (or 0.0) once the logit gap exceeds
        #     ~17, so every confidently-relevant document ties at 1.0. A stable sort then
        #     returns first-stage order for the whole tied block -- the reranker silently
        #     becomes a copy of the retriever. This is exactly what sank mxbai.
        #  2. bf16. The logits are bf16 (~8 mantissa bits), so even the gap is quantised.
        #
        # Both vanish if we take the log-odds directly off the fp32-upcast hidden state:
        #     logit_yes - logit_no = h . w_yes - h . w_no = h . (w_yes - w_no)
        # a single fp32 dot product. It is a strictly monotone transform of the card's
        # P(yes), so the RANKING is identical -- and P@R, nDCG and MRR are rank-only.
        # As a bonus it never allocates the [B, L, 151669] logits tensor (~11 GB at
        # batch 32), which is most of why this model is memory-hungry.
        lm_head_w = self.model.lm_head.weight               # [V, H]
        self._w_delta = (
            lm_head_w[self.token_true_id].float() - lm_head_w[self.token_false_id].float()
        ).detach().contiguous()

        self.prefix_tokens: list[int] = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens: list[int] = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        # Budget available to the formatted "<Instruct>...<Query>...<Document>..." string.
        self._body_budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if self._body_budget <= 0:
            raise ValueError(f"[{spec.alias}] max_length={self.max_length} is smaller than the template.")

        logger.info(
            "[%s] yes/no reranker ready: max_length=%d (body=%d), dtype=%s, device=%s, "
            "yes=%d no=%d, padding_side=%s, attn=%s",
            spec.alias, self.max_length, self._body_budget, spec.dtype, self.device,
            self.token_true_id, self.token_false_id, self.tokenizer.padding_side,
            attn_implementation or "default",
        )

    def _scaffold_len(self, query: str, instruction: str) -> int:
        """Tokens the instruction + query + field labels occupy, document excluded."""
        scaffold = format_instruction(instruction, query, "")
        return len(self.tokenizer(scaffold, add_special_tokens=False, verbose=False)["input_ids"])

    def _count_truncated(self, docs: Sequence[str], query: str, instruction: str) -> int:
        """Documents whose tail the body budget cuts off."""
        available = self._body_budget - self._scaffold_len(query, instruction)
        lens = self.tokenizer(list(docs), add_special_tokens=False, verbose=False)["input_ids"]
        return sum(1 for ids in lens if len(ids) > available)

    @torch.no_grad()
    def _score_batch(self, query: str, batch_docs: Sequence[str], instruction: str) -> list[float]:
        texts = [format_instruction(instruction, query, doc) for doc in batch_docs]
        encoded = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self._body_budget,
        )
        encoded["input_ids"] = [
            self.prefix_tokens + ids + self.suffix_tokens for ids in encoded["input_ids"]
        ]
        # tokenizer.pad regenerates attention_mask and honours padding_side="left".
        inputs = self.tokenizer.pad(
            encoded, padding=True, return_tensors="pt", max_length=self.max_length
        ).to(self.device)

        # Base model only: skips lm_head, so no [B, L, V] logits tensor is allocated.
        # Qwen3Model applies its final RMSNorm internally, so this is the same hidden
        # state the lm_head would consume.
        hidden = self.model.model(**inputs).last_hidden_state[:, -1, :].float()
        # log-odds = h . (w_yes - w_no); see __init__ for why not P(yes).
        scores = hidden @ self._w_delta.to(hidden.device)
        return scores.cpu().tolist()

    def score(self, query: str, docs: Sequence[str], instruction: str | None = None) -> list[float]:
        if instruction is None:
            raise ValueError(f"[{self.alias}] is instruction-aware; an instruction is required.")
        if not docs:
            return []

        self.truncation.observe(len(docs), self._count_truncated(docs, query, instruction))

        # The prefix, suffix and instruction/query scaffold ride along with every document
        # in a batch (~60-100 tokens each), so they count against the token budget.
        # Omitting them makes batches larger than intended and provokes avoidable OOM retries.
        overhead = (
            len(self.prefix_tokens) + len(self.suffix_tokens)
            + self._scaffold_len(query, instruction)
        )
        scores = [0.0] * len(docs)
        for idx, batch_docs in length_bucketed_batches(
            docs, self.token_budget, MAX_BATCH_SIZE, per_doc_overhead=overhead
        ):
            batch_scores = score_with_oom_backoff(
                lambda d: self._score_batch(query, d, instruction),
                batch_docs,
                initial_chunk=len(batch_docs),
                alias=self.alias,
            )
            for i, s in zip(idx, batch_scores):
                scores[i] = s
        return scores
