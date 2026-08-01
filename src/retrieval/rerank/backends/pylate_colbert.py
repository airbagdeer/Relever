"""ColBERTv2 late interaction as a reranker, via pylate's documented ``rank.rerank``.

    from pylate import models, rank
    model = models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0", device="cuda")
    q = model.encode(queries,  is_query=True)
    d = model.encode(documents, is_query=False)
    reranked = rank.rerank(documents_ids=ids, queries_embeddings=q, documents_embeddings=d)
    # -> [[{"id": ..., "score": ...}, ...]]  sorted, best first

MaxSim between a query's token embeddings and a document's is **set-independent**, so
ColBERT is a pointwise reranker and shares the pooled score cache with the cross-encoders,
despite being a different paradigm.

``document_length=512`` is ColBERT's BERT-backbone ceiling and matches
``index_colbert.py --document-length``, so this reranker sees exactly what colbert-v2's
own retrieval arm saw. That truncates ~15% of the v2 paragraphs; declared in the registry.

Requires ``.venv-colbert``: pylate pins ``transformers<=4.56.2``, which the main venv's
``transformers>=5`` cannot satisfy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .base import Reranker, TruncationCounter, enable_tf32_if_fp32, resolve_device
from reranker_registry import RerankerSpec

logger = logging.getLogger(__name__)

# Documents encoded per forward pass. 2000 docs x 512 tokens x 128 dims x fp32 would be
# ~0.5 GB of token embeddings held at once; chunking bounds it. MaxSim is per (query, doc),
# so chunk boundaries cannot change any score.
DEFAULT_ENCODE_CHUNK = 256
# Sequences per forward pass inside a chunk. pylate defaults to 32. Measured on a
# contended A100, encode time is FLAT across 32/128/256 (3.4s / 3.9s / 3.4s per 256 docs),
# so this is not the lever it looks like. Kept at 128 as a reasonable default; colbert's
# ~78s/query is dominated by encode + the ragged numpy->tensor conversion inside
# rank.rerank, and wants profiling on an uncontended GPU before anyone "optimises" it.
DEFAULT_ENCODE_BATCH = 128


class PylateColbertReranker(Reranker):
    def __init__(self, spec: RerankerSpec, device: str | None = None,
                 encode_chunk: int = DEFAULT_ENCODE_CHUNK,
                 encode_batch: int = DEFAULT_ENCODE_BATCH):
        try:
            from pylate import models as pylate_models  # noqa: PLC0415
            from pylate import rank as pylate_rank  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - venv-dependent
            raise ImportError(
                "pylate is not installed. Run this backend with .venv-colbert:\n"
                "    CUDA_VISIBLE_DEVICES=<gpu> .venv-colbert/bin/python score_pool.py "
                "--reranker colbertv2.0"
            ) from exc

        if spec.max_length is None:
            raise ValueError(f"[{spec.alias}] pylate_colbert needs max_length (= document_length).")

        self.alias = spec.alias
        self.spec = spec
        self.document_length: int = spec.max_length
        self.device = device or resolve_device()
        self.encode_chunk = encode_chunk
        self.encode_batch = encode_batch
        self.truncation = TruncationCounter()
        self.tf32 = enable_tf32_if_fp32(spec.dtype, self.device)
        self._rank = pylate_rank

        self.model = pylate_models.ColBERT(
            model_name_or_path=spec.hf_id,
            device=str(self.device),
            document_length=self.document_length,
        )
        logger.info(
            "[%s] pylate ColBERT ready: document_length=%d, device=%s",
            spec.alias, self.document_length, self.device,
        )

    def _count_truncated(self, docs: Sequence[str]) -> int:
        tok = self.model.tokenizer
        lens = tok(list(docs), add_special_tokens=False, verbose=False)["input_ids"]
        # ColBERT prepends [CLS] [D] and reserves a slot for [SEP].
        return sum(1 for ids in lens if len(ids) > self.document_length - 3)

    def score(self, query: str, docs: Sequence[str], instruction: str | None = None) -> list[float]:
        if instruction is not None:
            raise ValueError(f"[{self.alias}] has no instruction slot.")
        if not docs:
            return []

        self.truncation.observe(len(docs), self._count_truncated(docs))

        # pylate handles the [Q]/[D] markers and query padding internally via is_query.
        #
        # Encodes return NUMPY (pylate's default). Do not ask for tensors here: documents
        # have ragged token counts and `encode(convert_to_tensor=True)` tries to
        # torch.stack them ("stack expects each tensor to be equal size"). `rank.rerank`
        # is the right place -- it pad_sequences the ragged list itself.
        q_emb = self.model.encode([query], is_query=True, batch_size=1, show_progress_bar=False)

        scores = [0.0] * len(docs)
        for start in range(0, len(docs), self.encode_chunk):
            chunk = list(docs[start:start + self.encode_chunk])
            local_ids = list(range(start, start + len(chunk)))
            d_emb = self.model.encode(
                [chunk], is_query=False, batch_size=min(self.encode_batch, len(chunk)),
                show_progress_bar=False,
            )
            reranked = self._rank.rerank(
                documents_ids=[local_ids],
                queries_embeddings=q_emb,
                documents_embeddings=d_emb,
                # Explicit, because with device=None rank.rerank "uses the device of the
                # queries embeddings" -- and numpy embeddings make that the CPU.
                #
                # Do NOT assume this is what makes colbert slow. Measured on one contended
                # A100 (2758 docs, 256-doc chunks):
                #     encode 256 docs .......... 3.4-3.9s  (flat in batch_size: 32/128/256)
                #     rank.rerank device=None ... 3.18s
                #     rank.rerank device=cuda ... 3.27s    <- no gain
                # 11 chunks x ~6.7s ~= 74s/query, matching the observed 78s. The cost is
                # spread across encode and the ragged numpy->tensor conversion inside
                # rerank, not the MaxSim itself. Left explicit because it is the correct
                # contract; the real hot spot still needs profiling on an uncontended GPU.
                device=self.device,
            )
            # rank.rerank returns [[{"id": ..., "score": ...}, ...]] sorted best-first;
            # scatter back to input order — the caller relies on positional alignment.
            for entry in reranked[0]:
                scores[int(entry["id"])] = float(entry["score"])
        return scores
