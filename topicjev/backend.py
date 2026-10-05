"""Shared model lifecycle and backend abstractions for topicjev.

Provides canonical device detection, chat-template formatting, context-length
resolution, and resource lifecycle management (load/close/context manager).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import torch


def detect_device() -> tuple[torch.device, torch.dtype]:
    """Select the optimal torch device and dtype for local inference.

    Prefers CUDA, then Apple Silicon (MPS), then CPU. Set the ``TOPICJEV_DEVICE``
    environment variable (e.g. ``cpu``, ``mps``, ``cuda:1``) to override.
    """
    override = os.environ.get("TOPICJEV_DEVICE")
    if override:
        dev = torch.device(override)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    if dev.type == "cuda":
        major, _ = torch.cuda.get_device_capability(dev)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        # MPS stays in float32 so scores match CPU runs
        dtype = torch.float32
    return dev, dtype


def empty_device_cache() -> None:
    """Release cached accelerator memory (CUDA or MPS) after a model is offloaded."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def resolve_max_context_length(
    tokenizer: Any,
    model: Any = None,
    default: int = 512,
) -> int:
    """Determine the effective maximum sequence length from tokenizer or model config."""
    n = getattr(tokenizer, "model_max_length", None)
    if n is None or n > 1_000_000:
        cfg = getattr(model, "config", None)
        n = getattr(
            cfg,
            "n_positions",
            getattr(cfg, "max_position_embeddings", default),
        )
    return int(n if n is not None else default)


def format_chat_prompt(
    tokenizer: Any,
    text: str,
    sys_prompt: str | None = None,
    *,
    chat: bool = True,
    thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """Format input text using the tokenizer's chat template if available.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        text: User query or prompt text.
        sys_prompt: Optional system prompt to prepend.
        chat: Whether to apply chat templating (if supported by tokenizer).
        thinking: Whether to keep thinking enabled for reasoning-tuned models.
        add_generation_prompt: Whether to append the generation prompt.

    Returns:
        Formatted prompt string.
    """
    if not chat or not tokenizer or not getattr(tokenizer, "chat_template", None):
        return text

    messages: list[dict[str, str]] = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": text})

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if not thinking:
        kwargs["enable_thinking"] = False

    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Checkpoint chat template does not accept enable_thinking
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


class ChatTemplateMixin:
    """Mixin providing chat-template prompt formatting for causal / chat models."""

    chat: bool = True
    thinking: bool = False

    def _wrap(self, text: str, sys_prompt: str | None = None) -> str:
        """Wrap text with the model's chat template if available."""
        tok = getattr(self, "tok", None) or getattr(self, "tokenizer", None)
        chat_enabled = getattr(self, "chat", True)
        thinking_enabled = getattr(self, "thinking", getattr(self, "reason", False))
        return format_chat_prompt(
            tokenizer=tok,
            text=text,
            sys_prompt=sys_prompt,
            chat=chat_enabled,
            thinking=thinking_enabled,
        )


class ModelBackend(ABC):
    """Base contract for model and API client lifecycles."""

    def __init__(self, mname: str | None = None, **kwargs: Any) -> None:
        self.mname = mname
        self.model: Any = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model or initialize the client."""

    def close(self) -> None:
        """Release allocated model resources."""
        self.model = None

    def __enter__(self) -> ModelBackend:
        self.load_model()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class LocalModelBackend(ModelBackend):
    """Base for local PyTorch-based models with device and memory management."""

    def __init__(self, mname: str | None = None, **kwargs: Any) -> None:
        super().__init__(mname, **kwargs)
        self.dev, self.dtype = detect_device()
        self.tok: Any = None

    def close(self) -> None:
        """Offload model to CPU and clear GPU cache if allocated."""
        if self.model is not None and hasattr(self.model, "cpu"):
            try:
                self.model.cpu()
            except Exception as e:
                print("Failed to move model to cpu during close: %s", e)
        super().close()
        self.tok = None
        empty_device_cache()


class CausalLMBackend(LocalModelBackend, ChatTemplateMixin):
    """Base for local causal language models with left-padded tokenizers."""

    def load_model(self) -> None:
        """Load causal LM and left-padded tokenizer onto target hardware device."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            self.mname,
            dtype=self.dtype,
            attn_implementation="sdpa",
        ).to(self.dev)
        self.tok = AutoTokenizer.from_pretrained(self.mname)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.eval()