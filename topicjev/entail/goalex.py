from __future__ import annotations

import torch
from tqdm import tqdm

from topicjev.entail.base import Pair
from topicjev.entail.mixins import YesNoLogitMixin
from topicjev.entail.seq2seq import Seq2SeqEntail

GOALEX = """Check whether the TEXT satisfies a PROPERTY. Respond with Yes or No. When uncertain, output No.
Now complete the following example -
input: 
PROPERTY: {property}
TEXT: {text}
output:"""


class GoalExEntail(YesNoLogitMixin, Seq2SeqEntail):
    """Evaluates candidate labels as independent binary property verification questions.

    Scores the logit difference between 'Yes' and 'No' tokens at the initial decoder step
    of an encoder-decoder seq2seq model in a single forward pass without autoregressive generation.
    
    Inspired by the GoalEx approach `(Wang, Shang, and Zhong, 2023)`_. 

    .. _(Wang, Shang, and Zhong, 2023): https://arxiv.org/abs/2305.13749
    """

    def __init__(self, mname: str, **kwargs) -> None:
        super().__init__(mname, **kwargs)
        self.templ: str = kwargs.get("template", GOALEX)
        self.temp: float = kwargs.get("temperature", 0.1)
        self.res_tokens: int = kwargs.get("reserve", 8)
        self.str_name = "goalex"

    def prep_pairs(self, prompts: list[str], lbls: list[str]) -> list[Pair]:
        """Format input prompts and labels into GoalEx question pairs."""
        blank = len(self.tok(self.templ.format(text="", property="")).input_ids)
        longest = max((len(self.tok(str(l)).input_ids) for l in lbls), default=0)
        room = max(1, self.max_len - blank - longest - self.res_tokens)
        ids = self.tok([str(d) for d in prompts], truncation=True, max_length=room)
        docs = self.tok.batch_decode(ids.input_ids, skip_special_tokens=True)
        return [Pair(self.templ.format(text=d, property=l)) for d in docs for l in lbls]

    def batch_score(self, pairs: list[Pair]) -> list[float]:
        """Score pairs by extracting yes/no logits at initial decoder token position."""
        scores: list[float] = []
        start_id = getattr(self.model.config, "decoder_start_token_id", None)
        if start_id is None:
            start_id = self.tok.pad_token_id
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
                dec = torch.full(
                    (enc["input_ids"].size(0), 1),
                    start_id,
                    dtype=torch.long,
                    device=self.dev,
                )
                lgts = self.model(**enc, decoder_input_ids=dec).logits.float()[:, 0, :]
                scores.extend(self._yes_no_score(lgts).cpu().tolist())
        return scores
