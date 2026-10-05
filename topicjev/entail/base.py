from __future__ import annotations

import torch
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from topicjev.backend import (
    ModelBackend,
    detect_device,
    empty_device_cache,
    resolve_max_context_length,
)

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(slots=True)
class Pair:
    """A single (input, optional target) pair sent to a scoring model."""

    inpt: str
    targ: str | None = None


@dataclass(slots=True)
class EntailRes:
    """Entailment classification result for a single document."""

    class_id: int
    prob: float
    probs: list[float]
    other: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "class": self.class_id,
            "prob": self.prob,
            "other": self.other,
            "probs": self.probs,
        }


@dataclass
class EntailResults:
    """Aggregates batch probabilities into top-1 classification predictions.

    Applies threshold gating to assign a document to either its top predicted
    label or falls back to 'Other'.
    """

    lbls: list[str]
    multi_lbl: bool = False
    decoys: list[str] = field(default_factory=list)
    threshold: float | None = None
    _threshold: float = field(init=False)
    _otherid: int = field(init=False)

    def __post_init__(self) -> None:
        self._otherid = len(self.lbls)
        default = 0.5 if self.multi_lbl else (2.0 / max(self._otherid, 1))
        self._threshold = default if self.threshold is None else float(self.threshold)

    def compute_results(self, batch_probs: list[list[float]]) -> list[dict[str, Any]]:
        """Map per-label probability distributions to classification result dictionaries."""
        cand = list(self.lbls) + list(self.decoys)
        results: list[dict[str, Any]] = []
        for probs in batch_probs:
            order = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)
            idx = order[0]
            max_prob = probs[idx]
            in_taxonomy = idx < len(self.lbls)
            class_id = (
                idx if (max_prob > self._threshold and in_taxonomy) else self._otherid
            )
            other = None if in_taxonomy else cand[idx]
            results.append(
                EntailRes(
                    class_id=class_id,
                    prob=max_prob,
                    probs=probs[: len(self.lbls)],
                    other=other,
                ).to_dict()
            )
        return results


class Entailment(ModelBackend):
    """Base class for entailment and zero-shot classification backends.

    Orchestrates pairing, probe calibration, temperature scaling, normalization,
    and threshold gating across heterogeneous model backends.
    """

    def __init__(
        self,
        mname: str | None,
        *,
        batch_size: int = 16,
        max_tokens: int = 512,
        multi_lbl: bool = False,
        decoys: list[str] | None = None,
        probes: list[str] | None = None,
        temperature: float = 1.0,
        threshold: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(mname, **kwargs)
        self.bsize = batch_size
        self.max_tokens = max_tokens
        self.multi_lbl = multi_lbl
        self.decoys: list[str] = list(decoys) if decoys else []
        self.probes: list[str] = list(probes) if probes else []
        self.temp = temperature
        self.threshold = threshold
        self.str_name = "base"
        self.jev = False

        self.model: PreTrainedModel | Any = None
        self.tok: PreTrainedTokenizerBase | Any = None

    @abstractmethod
    def prep_pairs(self, prompts: list[str], lbls: list[str]) -> list[Pair]:
        """Format input documents and labels into scoring pairs."""

    @abstractmethod
    def batch_score(self, pairs: list[Pair]) -> list[float] | list[list[float]]:
        """Score input pairs and return raw, uncalibrated scores."""

    def entail(self, docs: list[str], lbls: list[str]) -> list[dict[str, Any]]:
        """Score documents against labels and return classification results."""
        prompts = self.probes + docs
        all_lbls = list(lbls) + list(self.decoys)
        pairs = self.prep_pairs(prompts, all_lbls)

        raw_scores = self.batch_score(pairs)
        scores = torch.as_tensor(raw_scores, dtype=torch.float32).reshape(
            len(prompts), len(all_lbls)
        )

        if self.probes:
            n = len(self.probes)
            scores = scores[n:] - scores[:n].mean(dim=0, keepdim=True)

        scores = scores / self.temp

        if not self.multi_lbl or self.jev:
            scores = torch.softmax(scores, dim=1)
        else:
            scores = torch.sigmoid(scores)

        return EntailResults(
            lbls=lbls,
            multi_lbl=self.multi_lbl,
            decoys=self.decoys,
            threshold=self.threshold,
        ).compute_results(scores.tolist())

    def close(self) -> None:
        """Release allocated model and tokenizer resources."""
        super().close()
        self.tok = None


class LocalEntailment(Entailment):
    """Entailment backend that loads local PyTorch models onto hardware devices."""

    def __init__(self, mname: str, **kwargs: Any) -> None:
        super().__init__(mname, **kwargs)
        self.dev, self.dtype = detect_device()

    @property
    def max_len(self) -> int:
        """Maximum context length supported by tokenizer or model configuration."""
        return resolve_max_context_length(self.tok, self.model, self.max_tokens)

    def close(self) -> None:
        """Offload local model and clear GPU cache."""
        if self.model is not None and hasattr(self.model, "cpu"):
            try:
                self.model.cpu()
            except Exception as e:
                print("Failed to move model to cpu during close: %s", e)
        super().close()
        empty_device_cache()


class RemoteEntailment(Entailment):
    """Entailment backend for hosted remote API services without local GPU requirements."""
