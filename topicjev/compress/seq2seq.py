from __future__ import annotations

import torch
from tqdm import tqdm
from typing import Any
from topicjev.compress.base import Compressor

INSTRUCT_S2S = "Summarize the following text in two or three plain sentences. {}"


class CNNCompressor(Compressor):
    """Summarizes text using seq2seq models fine-tuned for summarization (e.g. BART, PEGASUS).

    Target length is computed as `ratio * max_enc_len`.
    """

    def __init__(self, mname: str, **kwargs: Any) -> None:
        super().__init__(mname, **kwargs)
        self.source_len: int = kwargs.get("source_len", 512)
        self.min_len: int = kwargs.get("min_len", 30)
        self.num_beams: int = kwargs.get("num_beams", 4)

    @property
    def max_len(self) -> int:
        """Target maximum summary generation token length."""
        return int(self.max_enc_len * self.ratio)

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

    def compress(self, docs: list[str]) -> list[str]:
        """Summarize documents using beam search generation."""
        compr_docs: list[str] = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(docs), self.bsize), desc="Compressing docs"):
                batch = docs[i : i + self.bsize]
                enc = self.tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_enc_len,
                ).to(self.dev)

                outputs = self.model.generate(
                    **enc,
                    max_length=self.max_len,
                    min_length=self.min_len,
                    num_beams=self.num_beams,
                    use_cache=True,
                ).cpu()

                compr_batch = self.tok.batch_decode(outputs, skip_special_tokens=True)
                compr_docs.extend(compr_batch)
        return compr_docs


class Seq2SeqCompressor(CNNCompressor):
    """Instruction-guided seq2seq summarizer for general instruction-tuned checkpoints (e.g. FLAN-T5)."""

    def __init__(self, mname: str, **kwargs: Any) -> None:
        super().__init__(mname, **kwargs)
        self.source_len: int = kwargs.get("source_len", 1024)
        self.instruct: str = kwargs.get("instruct", INSTRUCT_S2S)

    def compress(self, docs: list[str]) -> list[str]:
        """Format input documents with instruction prefix before running seq2seq generation."""
        instruct_docs = [self.instruct.format(doc) for doc in docs]
        return super().compress(instruct_docs)
