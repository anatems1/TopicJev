"""Token-level prompt compression backend via LLMLingua2."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from .base import Compressor

DEFAULT_FORCE_TOKENS = ["\n", "\n\n", " ", ".", ",", ":", ";", "!", "?"]


class LinguaCompressor(Compressor):
    """Performs token-level prompt pruning using LLMLingua2 without autoregressive 
    text generation `(Pan et al., 2024)`_.

    .. _(Pan et al., 2024): https://arxiv.org/abs/2403.12968
    """

    def __init__(self, mname: str, **kwargs: Any) -> None:
        super().__init__(mname, **kwargs)
        self.target_token: int = kwargs.get("target_token", -1)
        self.force_tokens: list[str] = kwargs.get("force_tokens", DEFAULT_FORCE_TOKENS)
        self.force_digits: bool = kwargs.get("force_digits", False)
        self.drop_consecutive: bool = kwargs.get("drop_consecutive", True)

    def load_model(self) -> None:
        """Initialize LLMLingua PromptCompressor."""
        from llmlingua import PromptCompressor

        self.model = PromptCompressor(
            model_name=self.mname,
            use_llmlingua2=True,
            device_map=str(self.dev) if self.dev.type == "cuda" else "cpu",
        )

    def compress(self, docs: list[str]) -> list[str]:
        """Prune input documents to the configured compression rate."""
        compr_docs: list[str] = []
        for doc in tqdm(docs, desc="Compressing docs"):
            res = self.model.compress_prompt(
                str(doc),
                rate=self.ratio,
                target_token=self.target_token,
                force_tokens=self.force_tokens,
                force_reserve_digit=self.force_digits,
                drop_consecutive=self.drop_consecutive,
            )
            compr_docs.append(res["compressed_prompt"])
        return compr_docs
