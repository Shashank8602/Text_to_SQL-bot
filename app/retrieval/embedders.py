"""Embedding providers for schema retrieval (docs/DESIGN.md §6.3). Compared via eval/retrieval_eval.py.

- local  : sentence-transformers on CPU. Free, no quota; ~130 MB model download on first use.
- gemini : Gemini embedding API via LiteLLM (free tier), truncated to 768 dimensions.
"""

from __future__ import annotations

import math
import os
import time
from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    name = "local"

    def __init__(self, model: str):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from sentence_transformers import SentenceTransformer  # heavy import: only when used

        self.model_name = model
        self._model = SentenceTransformer(model, device="cpu")
        self.dim = self._model.get_sentence_embedding_dimension()
        # BGE models are trained with an instruction prefix on queries (not on documents).
        self._query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([self._query_prefix + text], normalize_embeddings=True)[0].tolist()


class GeminiEmbedder:
    name = "gemini"
    # The free tier counts every text in a batch as one request: 100 per minute. Stay under it.
    TEXTS_PER_MINUTE = 90
    RETRIES = 3

    def __init__(self, model: str, api_key: str | None, dim: int = 768):
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set to use the gemini embedder")
        self.model_name, self._api_key, self.dim = model, api_key, dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), self.TEXTS_PER_MINUTE):
            if i:
                time.sleep(61)  # next per-minute window
            out += self._embed(texts[i:i + self.TEXTS_PER_MINUTE])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import litellm
        for attempt in range(self.RETRIES + 1):
            try:
                r = litellm.embedding(model=self.model_name, input=texts, dimensions=self.dim,
                                      api_key=self._api_key, timeout=60)
                break
            except litellm.RateLimitError:
                if attempt == self.RETRIES:
                    raise
                time.sleep(61)
        # Truncated Gemini embeddings are not unit length; normalize for cosine distance.
        return [_normalize(d["embedding"]) for d in r.data]


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


@lru_cache
def get_embedder(name: str) -> Embedder:
    s = get_settings()
    if name == "local":
        return LocalEmbedder(s.local_embedding_model)
    if name == "gemini":
        return GeminiEmbedder(s.gemini_embedding_model, s.gemini_api_key)
    raise ValueError(f"Unknown embedder '{name}' (expected 'local' or 'gemini')")
