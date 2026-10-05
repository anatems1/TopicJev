from __future__ import annotations

import torch
from tqdm import tqdm

from topicjev.backend import CausalLMBackend
from topicjev.entail.base import Pair
from topicjev.entail.goalex import GoalExEntail


class CausalEntail(CausalLMBackend, GoalExEntail):
    """Evaluates binary property verification from the last token of a causal LM forward pass.

    Requires left-padding so position index -1 corresponds to the terminal prompt token.
    Inherits model loading and chat templating from CausalLMBackend.
    """

    def __init__(self, mname: str, **kwargs) -> None:
        super().__init__(mname, **kwargs)
        self.chat: bool = kwargs.get("chat", True)
        self.thinking: bool = kwargs.get("thinking", False)
        self.min_len: int = kwargs.get("min_len", 0)
        self.num_beams: int = kwargs.get("num_beams", 1)
        self.str_name = "causal"

    def prep_pairs(self, prompts: list[str], lbls: list[str]) -> list[Pair]:
        """Format input prompts and labels into left-padded scoring pairs."""
        blank = len(
            self.tok(self._wrap(self.templ.format(text="", property=""))).input_ids
        )
        longest = max((len(self.tok(str(l)).input_ids) for l in lbls), default=0)
        room = max(1, self.max_len - blank - longest - self.res_tokens)
        ids = self.tok([str(d) for d in prompts], truncation=True, max_length=room)
        docs = self.tok.batch_decode(ids.input_ids, skip_special_tokens=True)
        return [
            Pair(self._wrap(self.templ.format(text=d, property=l)))
            for d in docs
            for l in lbls
        ]

    def batch_score(self, pairs: list[Pair]) -> list[float]:
        """Compute yes/no logit difference at terminal token position."""
        scores: list[float] = []
        with torch.no_grad():
            for i in tqdm(range(0, len(pairs), self.bsize), desc="Entailment scoring"):
                batch: list[Pair] = pairs[i : i + self.bsize]
                enc = self.tok(
                    [p.inpt for p in batch],
                    padding="longest",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.dev) for k, v in enc.items()}
                lgts = self.model(**enc).logits.float()[:, -1, :]
                scores.extend(self._yes_no_score(lgts).cpu().tolist())
        return scores
