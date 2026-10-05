from __future__ import annotations

import torch
from tqdm import tqdm
from typing import Any

from topicjev.backend import CausalLMBackend
from topicjev.compress.base import Compressor

INSTRUCT_CLM = """Summarize the text below in several declarative sentences in paragraph form. Output only the summary, with no preamble.

TEXT:
{text}

SUMMARY:
"""


class CausalCompressor(CausalLMBackend, Compressor):
    """Summarizes text using a causal language model with dynamic input-relative token budgets.

    Formats prompts using chat templates (via CausalLMBackend) and restricts generated
    token count to `ratio * prompt_length`.
    """

    def __init__(self, mname: str, **kwargs: Any) -> None:
        super().__init__(mname, **kwargs)
        self.source_len: int = kwargs.get("source_len", 2048)
        self.instruct: str = kwargs.get("instruct", INSTRUCT_CLM)
        self.chat: bool = kwargs.get("chat", True)
        self.thinking: bool = kwargs.get("thinking", False)
        self.min_len: int = kwargs.get("min_len", 0)
        self.num_beams: int = kwargs.get("num_beams", 1)

    def compress(self, docs: list[str]) -> list[str]:
        """Summarize a list of documents in batches."""
        compr_docs: list[str] = []
        i_docs = [self.instruct.format(text=doc) for doc in docs]
        wrapped = [self._wrap(doc) for doc in i_docs]

        with torch.inference_mode():
            for i in tqdm(range(0, len(wrapped), self.bsize), desc="Compressing docs"):
                batch = wrapped[i : i + self.bsize]
                enc = self.tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_enc_len,
                    add_special_tokens=not self.chat,
                ).to(self.dev)

                prompt_length = enc["input_ids"].shape[1]
                dynamic_max_len = int(prompt_length * self.ratio)
                safe_max_len = max(dynamic_max_len, self.min_len)

                outputs = self.model.generate(
                    **enc,
                    max_new_tokens=safe_max_len,
                    min_new_tokens=self.min_len,
                    num_beams=self.num_beams,
                    pad_token_id=self.tok.pad_token_id,
                    use_cache=True,
                ).cpu()

                gen_tokens = outputs[:, prompt_length:]
                compr_batch = self.tok.batch_decode(
                    gen_tokens, skip_special_tokens=True
                )
                compr_docs.extend(compr_batch)
        return compr_docs
