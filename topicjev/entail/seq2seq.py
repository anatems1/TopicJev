from __future__ import annotations

import torch
from tqdm import tqdm
from topicjev.entail.base import LocalEntailment, Pair

CLASS_CUE = "Text: {}\nThe topic of this text is:"


class Seq2SeqEntail(LocalEntailment):
    """Scores candidate labels by their average per-token log-likelihood under teacher-forcing."""

    def __init__(self, mname: str, **kwargs) -> None:
        super().__init__(mname, **kwargs)
        self.templ: str = kwargs.get("template", CLASS_CUE)
        self.res_tokens: int = kwargs.get("reserve", 50)
        self.str_name = "seq2seq"

    def load_model(self) -> None:
        """Load encoder-decoder seq2seq model and tokenizer."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.mname,
            dtype=self.dtype,
            attn_implementation="sdpa",
        ).to(self.dev)
        self.tok = AutoTokenizer.from_pretrained(self.mname)
        self.model.eval()

    def prep_pairs(self, prompts: list[str], lbls: list[str]) -> list[Pair]:
        """Format input prompts and candidate target labels into teacher-forced pairs."""
        overhead = len(self.tok(self.templ.format("")).input_ids)
        room = max(1, self.max_len - overhead - self.res_tokens)
        ids = self.tok([str(d) for d in prompts], truncation=True, max_length=room)
        docs = self.tok.batch_decode(ids.input_ids, skip_special_tokens=True)
        return [Pair(self.templ.format(d), targ=l) for d in docs for l in lbls]

    def batch_score(self, pairs: list[Pair]) -> list[float]:
        """Compute mean target token log-probabilities under teacher forcing."""
        scores: list[float] = []
        with torch.no_grad():
            for i in tqdm(range(0, len(pairs), self.bsize), desc="Entailment scoring"):
                batch: list[Pair] = pairs[i : i + self.bsize]
                enc = self.tok(
                    [p.inpt for p in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                tgt = self.tok(
                    [p.targ for p in batch], padding=True, return_tensors="pt"
                )
                enc = {k: v.to(self.dev) for k, v in enc.items()}
                ids = tgt["input_ids"].to(self.dev)
                mask = tgt["attention_mask"].to(self.dev)
                dec_in = self.model._shift_right(ids)
                lgts = self.model(**enc, decoder_input_ids=dec_in).logits.float()
                logprobs = torch.log_softmax(lgts, dim=-1)
                top_lk = logprobs.gather(-1, ids.unsqueeze(-1)).squeeze(-1) * mask
                scores.extend(
                    (top_lk.sum(-1) / mask.sum(-1).clamp(min=1)).cpu().tolist()
                )
        return scores
