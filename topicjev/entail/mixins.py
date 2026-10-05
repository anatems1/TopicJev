from __future__ import annotations

import torch
from typing import Any
from functools import cached_property




class YesNoLogitMixin:
    """Computes binary yes/no logit differences for next-token prediction backends.

    Used by GoalExEntail (seq2seq decoder start) and CausalEntail (causal last-token).
    Expects host instance to provide `self.tok` and `self.mname`.
    """

    tok: Any
    mname: str | None

    @cached_property
    def yn(self) -> tuple[list[int], list[int]]:
        """Identify token IDs corresponding to affirmative and negative single-token variants."""
        yes_strs = ["yes", " yes", "Yes", " Yes", "YES", " YES"]
        no_strs = ["no", " no", "No", " No", "NO", " NO"]

        def get_valid_ids(words: list[str]) -> list[int]:
            valid_ids: list[int] = []
            for w in words:
                ids = self.tok(w, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    valid_ids.append(ids[0])
            unk = getattr(self.tok, "unk_token_id", None)
            return list(set(i for i in valid_ids if i != unk))

        yes_ids = get_valid_ids(yes_strs)
        no_ids = get_valid_ids(no_strs)
        if not yes_ids or not no_ids:
            raise ValueError(
                f"{self.mname!r} tokenizer has no single-token yes/no variant in "
                f"{yes_strs + no_strs}. Next-token binary scoring requires valid single-token representations."
            )
        return yes_ids, no_ids

    def _yes_no_score(self, lgts: torch.Tensor) -> torch.Tensor:
        """Calculate logsumexp(yes) - logsumexp(no) from (batch, vocab_size) logits."""
        yes_ids, no_ids = self.yn
        comb_yes = torch.logsumexp(lgts[:, yes_ids], dim=1)
        comb_no = torch.logsumexp(lgts[:, no_ids], dim=1)
        return comb_yes - comb_no
