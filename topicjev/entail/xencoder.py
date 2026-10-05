from __future__ import annotations

import torch
from tqdm import tqdm
from topicjev.entail.base import LocalEntailment, Pair

TOPIC_HYP = "The text discusses {}"

SPACE_FNS: dict[str, callable[[torch.Tensor, int, int, int | None], torch.Tensor]] = {
    # 3-class NLI: entailment vs (contradiction + neutral)
    "ternary_logit": lambda lgts, ent, cnt, neu: lgts[:, ent]
    - torch.logsumexp(lgts[:, [cnt, neu if neu is not None else cnt]], dim=1),
    # 2-class NLI: entailment vs contradiction
    "binary_logit": lambda lgts, ent, cnt, _: lgts[:, ent] - lgts[:, cnt],
    # 1-class NLI: raw entailment logit
    "raw": lambda lgts, ent, _, __: lgts[:, ent],
}


class XEncoderEntail(LocalEntailment):
    """Zero-shot classification using Natural Language Inference (NLI) cross-encoders.

    Pairs documents as premises and hypothesis templates ('The text discusses {label}').
    """

    def __init__(self, mname: str, **kwargs) -> None:
        super().__init__(mname, **kwargs)
        self._idx: tuple[int, int, int | None] | None = None
        self.hyp: str = kwargs.get("hyp", TOPIC_HYP)
        self.str_name = "xencoder"

    @property
    def entail_indices(self) -> tuple[int, int, int | None]:
        """Resolve indices for entailment, contradiction, and neutral output labels."""
        if self._idx is None:
            ent, cnt, neu = 2, 0, None
            id2label = getattr(self.model.config, "id2label", {})
            for i, name in id2label.items():
                low = str(name).lower()
                if low == "entailment":
                    ent = int(i)
                elif low in ("contradiction", "not_entailment"):
                    cnt = int(i)
                elif low == "neutral":
                    neu = int(i)
            self._idx = (ent, cnt, neu)
        return self._idx

    def load_model(self) -> None:
        """Load cross-encoder classification model and tokenizer."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.mname,
            dtype=self.dtype,
        ).to(self.dev)
        self.tok = AutoTokenizer.from_pretrained(self.mname)
        self.model.eval()

    def prep_pairs(self, prompts: list[str], lbls: list[str]) -> list[Pair]:
        """Format documents and hypothesis labels into cross-encoder premise-hypothesis pairs."""
        longest_hyp = max(
            (len(self.tok(self.hyp.format(l)).input_ids) for l in lbls), default=0
        )
        room = max(1, self.max_len - longest_hyp)
        ids = self.tok([str(p) for p in prompts], truncation=True, max_length=room)
        docs = self.tok.batch_decode(ids.input_ids, skip_special_tokens=True)
        return [Pair(d, self.hyp.format(l)) for d in docs for l in lbls]

    def _space_fn(
        self, neu: int | None
    ) -> callable[[torch.Tensor, int, int, int | None], torch.Tensor]:
        """Select appropriate logit scoring function based on label space and multi-label mode."""
        if not self.multi_lbl:
            fn_key = "raw"
        elif neu is not None:
            fn_key = "ternary_logit"
        else:
            fn_key = "binary_logit"
        return SPACE_FNS[fn_key]

    def batch_score(self, pairs: list[Pair]) -> list[float]:
        """Compute cross-encoder classification logits."""
        scores: list[float] = []
        ent, cnt, neu = self.entail_indices
        space_fn = self._space_fn(neu)
        with torch.no_grad():
            for i in tqdm(range(0, len(pairs), self.bsize), desc="Entailment scoring"):
                batch: list[Pair] = pairs[i : i + self.bsize]
                enc = self.tok(
                    [p.inpt for p in batch],
                    [p.targ for p in batch],
                    padding="longest",
                    truncation="only_first",
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.dev) for k, v in enc.items()}
                lgts = self.model(**enc).logits.float()
                out = space_fn(lgts, ent, cnt, neu)
                scores.extend(out.cpu().tolist())
        return scores
