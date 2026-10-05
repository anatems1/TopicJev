from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from topicjev.backend import (
    ModelBackend,
    detect_device,
    empty_device_cache,
    resolve_max_context_length,
)

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase


class Compressor(ModelBackend):
    """Base class for document summarization and token-level prompt compression.

    Provides device management, maximum context length resolution, and resource
    lifecycle handling across compression strategies.
    """

    def __init__(
        self,
        mname: str,
        *,
        ratio: float = 0.3,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(mname, **kwargs)
        self.ratio = ratio
        self.bsize = batch_size
        self.source_len: int = kwargs.get("source_len", 512)
        self.tok: PreTrainedTokenizerBase | Any = None
        self.dev, self.dtype = detect_device()

    @property
    def max_enc_len(self) -> int:
        """Maximum context length supported by tokenizer or model configuration."""
        return resolve_max_context_length(self.tok, self.model, self.source_len)

    @abstractmethod
    def compress(self, docs: list[str]) -> list[str]:
        """Compress or summarize a list of documents."""

    def close(self) -> None:
        """Offload model to CPU, release tokenizer, and clear GPU cache."""
        if self.model is not None and hasattr(self.model, "cpu"):
            try:
                self.model.cpu()
            except Exception as e:
                print("Failed to move model to cpu during close: %s", e)
        super().close()
        self.tok = None
        empty_device_cache()
