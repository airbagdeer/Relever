"""bge-reranker-v2-gemma: decoder LLM scored from a single, uncontrasted 'Yes' logit.

Reproduces the card's ``get_inputs`` recipe exactly. Three things distinguish it from
the Qwen3 backend, and conflating them would silently change what the scores mean:

* **No softmax, no contrast.** The score is the raw logit of the ``'Yes'`` token
  (capital Y, ``tokenizer('Yes', add_special_tokens=False)['input_ids'][0]``) at the
  last position. Qwen3 contrasts ``yes`` against ``no``; this model does not. The two
  score scales are therefore not comparable — only ranks are.
* **The instruction is a suffix.** Layout is
  ``<bos> A: {query} \\n B: {passage} \\n {prompt}`` — the prompt trails the passage
  rather than prefixing it. Appended *after* pair truncation, so it is never cut.
* **The budget is split.** The query is capped at ``3/4 * max_length`` and
  ``prepare_for_model(truncation='only_second')`` cuts the *passage*, never the query.
  This is the only reranker in the study that protects the query structurally.

``padding_side`` is ``"left"`` by ``GemmaTokenizer``'s class attribute rather than by
``tokenizer_config.json``, so it is correct by default — which is exactly why we assert
it instead of trusting it. ``trust_remote_code`` is not needed (the repo ships no .py).

Implemented against raw ``transformers`` rather than ``FlagEmbedding.FlagLLMReranker``
so the project does not take a dependency that pins an older ``transformers``.
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

SEP = "\n"
DEFAULT_TOKEN_BUDGET = 65_536
MAX_BATCH_SIZE = 64


class HFCausalYesLogitReranker(Reranker):
    def __init__(
        self,
        spec: RerankerSpec,
        device: str | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        attn_implementation: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        if spec.max_length is None:
            raise ValueError(f"[{spec.alias}] hf_causal_yes_logit requires an explicit max_length.")

        self.alias = spec.alias
        self.spec = spec
        self.max_length: int = spec.max_length
        self.device = device or resolve_device()
        self.token_budget = token_budget
        self.truncation = TruncationCounter()
        self.tf32 = False   # bf16 weights; TF32 only affects fp32 matmul

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        # Left by GemmaTokenizer's class default. The score reads logits[:, -1, :], so a
        # future transformers change flipping this default must fail loudly, not silently.
        assert_padding_side(self.tokenizer, "left", spec.alias)

        model_kwargs = {"dtype": resolve_dtype(spec.dtype, self.device)}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)
        self.model.to(self.device).eval()

        # Capital 'Y' — the card's exact token. Lowercase 'yes' is a different id.
        self.yes_loc: int = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        self.sep_tokens: list[int] = self.tokenizer(SEP, add_special_tokens=False)["input_ids"]

        # The card reads `model(...).logits[:, -1, yes_loc]`. Those logits are bf16 (~8
        # mantissa bits), so hundreds of documents in a 500-deep list quantise onto the
        # same value and the stable sort -- not the model -- decides their order.
        # `logit_yes = h . w_yes` exactly, so take the fp32 dot product of the fp32-upcast
        # hidden state with that one lm_head row. Identical quantity, full fp32 resolution,
        # and no [B, L, V] logits tensor. GemmaModel applies its final norm internally.
        self._w_yes = self.model.lm_head.weight[self.yes_loc].float().detach().contiguous()

        logger.info(
            "[%s] yes-logit reranker ready: max_length=%d, dtype=%s, device=%s, "
            "yes_loc=%d, padding_side=%s",
            spec.alias, self.max_length, spec.dtype, self.device,
            self.yes_loc, self.tokenizer.padding_side,
        )

    def _prompt_tokens(self, instruction: str) -> list[int]:
        return self.tokenizer(instruction, add_special_tokens=False)["input_ids"]

    def _pair_overhead(self, query: str) -> int:
        """bos + ``A: {query}`` + sep. What the passage must share ``max_length`` with."""
        query_len = len(
            self.tokenizer(f"A: {query}", add_special_tokens=False, verbose=False)["input_ids"]
        )
        return 1 + query_len + len(self.sep_tokens)

    def _count_truncated(self, docs: Sequence[str], query: str) -> int:
        """Passages cut by ``prepare_for_model(truncation='only_second')``.

        The suffix prompt is appended *after* truncation and so does not compete with the
        passage for ``max_length`` — but it does occupy space in the padded batch, which is
        why ``score`` folds it into the token budget separately.
        """
        available = self.max_length - self._pair_overhead(query)
        lens = self.tokenizer([f"B: {d}" for d in docs], add_special_tokens=False, verbose=False)["input_ids"]
        return sum(1 for ids in lens if len(ids) > available)

    def _build_inputs(self, query: str, batch_docs: Sequence[str], instruction: str):
        """The card's ``get_inputs``, structure preserved.

        The card calls ``tokenizer.prepare_for_model(first, second,
        truncation='only_second', add_special_tokens=False, max_length=L)``, but
        transformers 5.x removed ``prepare_for_model`` from ``GemmaTokenizer`` (both fast
        and slow). With those exact arguments that call is just: concatenate ``first`` and
        ``second``, and if the pair exceeds ``L``, trim ``second`` (the passage side) from
        its tail. Reproduced inline below so the inputs are byte-identical to the card's.
        """
        prompt_tokens = self._prompt_tokens(instruction)
        query_inputs = self.tokenizer(
            f"A: {query}",
            add_special_tokens=False,
            max_length=self.max_length * 3 // 4,
            truncation=True,
        )
        first = [self.tokenizer.bos_token_id] + query_inputs["input_ids"]
        items = []
        for doc in batch_docs:
            passage_inputs = self.tokenizer(
                f"B: {doc}", add_special_tokens=False, max_length=self.max_length, truncation=True
            )
            second = self.sep_tokens + passage_inputs["input_ids"]
            # truncation="only_second": keep the query intact, cut the passage tail.
            overflow = len(first) + len(second) - self.max_length
            if overflow > 0:
                second = second[: len(second) - overflow]
            # The instruction trails the passage; it is never truncated.
            input_ids = first + second + self.sep_tokens + prompt_tokens
            items.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})

        return self.tokenizer.pad(
            items,
            padding=True,
            max_length=self.max_length + len(self.sep_tokens) + len(prompt_tokens),
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to(self.device)

    @torch.no_grad()
    def _score_batch(self, query: str, batch_docs: Sequence[str], instruction: str) -> list[float]:
        inputs = self._build_inputs(query, batch_docs, instruction)
        # Raw 'Yes' logit at the last position. No softmax — that is the model's contract.
        # Computed as h . w_yes in fp32 rather than read off the bf16 logits; see __init__.
        hidden = self.model.model(**inputs).last_hidden_state[:, -1, :].float()
        scores = hidden @ self._w_yes.to(hidden.device)
        return scores.cpu().tolist()

    def score(self, query: str, docs: Sequence[str], instruction: str | None = None) -> list[float]:
        if instruction is None:
            raise ValueError(f"[{self.alias}] is instruction-aware; an instruction is required.")
        if not docs:
            return []

        self.truncation.observe(len(docs), self._count_truncated(docs, query))

        # bos + query + sep + the trailing instruction ride along with every document in a
        # batch, so they count against the token budget. Omitting them makes batches larger
        # than intended and provokes avoidable OOM retries.
        overhead = self._pair_overhead(query) + len(self.sep_tokens) + len(self._prompt_tokens(instruction))
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
