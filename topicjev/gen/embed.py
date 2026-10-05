from __future__ import annotations

import faiss
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from topicjev.backend import ModelBackend, detect_device, empty_device_cache

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer


class Embedder(ModelBackend):
    """Base class for text embedding backends."""

    def __init__(
        self,
        mname: str,
        *,
        batch_size: int = 128,
        prefix: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(mname, **kwargs)
        self.bsize: int = batch_size
        self.prefix: str = prefix
        self.save_dir: Path | None = (
            Path(kwargs.get("save_dir", None))
            if kwargs.get("save_dir", None) is not None
            else None
        )
        self.index: faiss.IndexFlatIP | None = None

    @abstractmethod
    def embed_docs(self, docs: list[str]) -> np.ndarray:
        """Embed a list of documents into a 2D numpy array of shape (n_docs, dim)."""

    def get_embeds(self, docs: list[str]) -> np.ndarray:
        """Compute embeddings for a list of documents."""
        if self.save_dir is not None and (self.save_dir / "embeds.bin").exists():
            self.index = faiss.read_index(str(self.save_dir / "embeds.bin"))
            return np.array(self.index.reconstruct_n(0, len(docs)))

        for i in tqdm(range(0, len(docs), self.bsize), desc="Embedding docs"):
            batch = docs[i : i + self.bsize]
            embeds = self.embed_docs(batch)
            if self.index is None:
                self.index = faiss.IndexFlatIP(embeds.shape[1])
            self.index.add(embeds)

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.save_dir / "embeds.bin"))

        return np.array(self.index.reconstruct_n(0, len(docs)))


class LocalEmbedder(Embedder):
    """Embeds text using local SentenceTransformer models."""

    def __init__(self, mname: str, **kwargs: Any) -> None:
        super().__init__(mname, **kwargs)
        self.dev, self.dtype = detect_device()
        self.norm_embeds: bool = kwargs.get("normalize_embeddings", True)
        self.trust_remote_code: bool = kwargs.get("trust_remote_code", True)

    def load_model(self) -> None:
        """Load SentenceTransformer model on detected device."""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.mname,
            trust_remote_code=self.trust_remote_code,
            model_kwargs={"torch_dtype": self.dtype},
        ).to(self.dev)
        self.model.eval()

    def embed_docs(self, docs: list[str]) -> np.ndarray:
        """Compute document embeddings."""
        if self.model is None:
            self.load_model()
        formatted_docs = [f"{self.prefix}{d}" for d in docs] if self.prefix else docs
        with torch.inference_mode():
            embed_batch = self.model.encode(
                formatted_docs,
                batch_size=8,
                normalize_embeddings=self.norm_embeds,
                show_progress_bar=False,
            )
        return embed_batch

    def close(self) -> None:
        """Offload embedding model to CPU and clear GPU memory."""
        if self.model is not None and hasattr(self.model, "cpu"):
            try:
                self.model.cpu()
            except Exception as e:
                print("Failed to move model to cpu during close: %s", e)
        super().close()
        empty_device_cache()
