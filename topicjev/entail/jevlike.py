"""Remote and hosted question-choice entailment backends (Jev and Laya)."""

from __future__ import annotations

import os
import json
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Any

from topicjev.backend import detect_device, empty_device_cache
from topicjev.prompts import load_prompt
from topicjev.entail.base import Pair, RemoteEntailment

JEV_PROMPT = "Which topic does the text belong to?"


class JevEntail(RemoteEntailment):
    """Zero-shot classification via hosted TypeSafe System-One decision endpoints."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(mname=None, **kwargs)
        self.str_name = "jev"
        self.questions: dict[str, Any] = {}
        self.decoys = [
            *self.decoys,
            json.dumps({"Other": "None of the other categories fit this text"}),
        ]
        self.jev = True
        self.multi_lbl = False

    def load_model(self) -> None:
        """Initialize remote TypeSafe API client."""
        from typesafe_sdk import TypeSafeClient

        load_dotenv()
        self.model = TypeSafeClient(api_key=os.environ.get("TYPESAFE_API_KEY"))

    def prep_pairs(self, prompts: list[str], lbls: list[str]) -> list[Pair]:
        """Construct question criteria schema and input prompt pairs."""
        criteria: dict[str, str] = {}
        for label in lbls:
            try:
                parsed = json.loads(label)
            except json.JSONDecodeError:
                criteria[label] = label
            else:
                criteria.update(parsed)

        self.questions = {
            "topic": {
                "type": "choice",
                "instructions": JEV_PROMPT,
                "criteria": criteria,
            }
        }
        return [Pair(inpt=p) for p in prompts]

    def batch_score(self, pairs: list[Pair]) -> list[list[float]]:
        """Score candidate choices via remote endpoint calls, returning log-probabilities."""
        scores: list[list[float]] = []

        def map_probs_to_output_map(probs: dict[str, float]) -> list[float]:
            return [
                probs.get(label, 0.0) for label in self.questions["topic"]["criteria"]
            ]

        with self.model as client:
            for i in tqdm(range(len(pairs)), desc="Jev scoring"):
                resp = client.system_one(
                    state={"text": pairs[i].inpt}, questions=self.questions
                )
                probs = map_probs_to_output_map(resp.answers["topic"].probabilities)
                scores.append(torch.log(torch.tensor(probs).clamp_min(1e-12)).tolist())
        return scores


class LayaEntail(JevEntail):
    """Zero-shot classification via local Laya agent models."""

    def __init__(self, mname: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mname = mname
        self.str_name = "laya"
        self.dev, _ = detect_device()

    def load_model(self) -> None:
        """Load local Laya model on detected hardware device."""
        from laya.agent import load as laya_load

        self.model = laya_load(model_id_or_path=self.mname, device=str(self.dev))

    def batch_score(self, pairs: list[Pair]) -> list[list[float]]:
        """Score candidate choices in local batches, returning log-probabilities."""
        scores: list[list[float]] = []
        for i in tqdm(range(0, len(pairs), self.bsize), desc="Laya scoring"):
            batch: list[Pair] = pairs[i : i + self.bsize]
            reqs = [{"text": b.inpt} for b in batch]
            out = self.model.predict_batch(reqs, self.questions)
            probs = [list(o["answers"]["topic"]["probabilities"].values()) for o in out]
            scores.extend(torch.log(torch.tensor(probs).clamp_min(1e-12)).tolist())
        return scores

    def close(self) -> None:
        """Release local Laya agent resources and clear GPU memory."""
        super().close()
        empty_device_cache()
