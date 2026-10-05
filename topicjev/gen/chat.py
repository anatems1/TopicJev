from __future__ import annotations

import re
import json
import torch
from tqdm import tqdm
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Union

from topicjev.backend import CausalLMBackend, ModelBackend, empty_device_cache

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizerBase


def clean_json(resp: str) -> dict[str, Any]:
    """Extract and parse a JSON dictionary from an LLM response string."""
    try:
        resp_text = str(resp).strip()
        if "```" in resp_text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp_text, re.DOTALL)
            if match:
                resp_text = match.group(1).strip()
            else:
                resp_text = resp_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(resp_text)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        try:
            start = resp.find("{")
            end = resp.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(resp[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
        return {}


class Generator(ModelBackend):
    """Base class for prompt-to-text generation and structured JSON extraction."""

    def __init__(
        self,
        mname: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        max_input_tokens: int = 4000,
        json_mode: bool = True,
        reason: bool = False,
        batch_size: int = 4,
        max_retries: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(mname, **kwargs)
        self.temp = temperature
        self.max_tokens = max_tokens
        self.max_inpt_tokens = max_input_tokens
        self.json_mode = json_mode
        self.reason = reason
        self.bsize = batch_size
        self.max_retries = max_retries

    @abstractmethod
    def query(
        self,
        user_prompt: str,
        sys_prompt: str = "You are a helpful AI assistant",
    ) -> Union[str, dict[str, Any]]:
        """Query the model with a single prompt and optional system instructions."""

    @abstractmethod
    def batch(
        self,
        prompt: str,
        input_docs: list[Any],
        req_keys: list[str] | None = None,
    ) -> list[Any]:
        """Execute a prompt template over a batch of input documents."""

    def run_chain(
        self,
        prompt: str,
        input_docs: list[Any],
        req_keys: list[str] | None = None,
    ) -> list[Any]:
        """Alias for `batch()` for pipeline compatibility."""
        return self.batch(prompt, input_docs, req_keys=req_keys)


class LocalGenerator(CausalLMBackend, Generator):
    """Generates text using a local causal language model on hardware devices."""

    def __init__(self, mname: str, **kwargs: Any) -> None:
        super().__init__(mname, **kwargs)
        self.tokenizer: PreTrainedTokenizerBase | Any = None

    @property
    def tok(self) -> PreTrainedTokenizerBase | Any:
        """Alias for tokenizer for consistency across local backends."""
        return self.tokenizer

    @tok.setter
    def tok(self, value: PreTrainedTokenizerBase | Any) -> None:
        self.tokenizer = value

    def load_model(self) -> None:
        """Load causal LM, left-padded tokenizer, and optionally compile for CUDA."""
        super().load_model()
        self.tokenizer = self.tok
        self.tokenizer.clean_up_tokenization_spaces = False
        if self.dev.type == "cuda":
            self.model = torch.compile(self.model, mode="default", dynamic=True)

    def batch_gen(self, prompts: list[str]) -> list[str]:
        """Generate text outputs for a list of formatted prompts."""
        res: list[str] = []
        with torch.inference_mode():
            for i in tqdm(range(0, len(prompts), self.bsize), desc="Generating"):
                batch = prompts[i : i + self.bsize]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_inpt_tokens,
                    return_tensors="pt",
                ).to(self.dev)
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": self.max_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "use_cache": True,
                }
                if self.temp > 0.0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = self.temp
                    gen_kwargs["top_k"] = 20
                    gen_kwargs["top_p"] = 0.95
                else:
                    gen_kwargs["do_sample"] = False

                outputs = self.model.generate(**enc, **gen_kwargs)
                gen_tokens = outputs[:, enc["input_ids"].shape[1] :]
                decoded = self.tokenizer.batch_decode(
                    gen_tokens, skip_special_tokens=True
                )
                res.extend(decoded)
        return res

    def query(
        self,
        user_prompt: str,
        sys_prompt: str = "You are a helpful AI assistant",
    ) -> Union[str, dict[str, Any]]:
        """Run single-prompt inference."""
        if self.model is None:
            self.load_model()
        formatted_prompt = self._wrap(user_prompt, sys_prompt)
        raw_result = self.batch_gen([formatted_prompt])[0]
        resp_text = raw_result.strip()
        return clean_json(resp_text) if self.json_mode else str(resp_text)

    def _retry_chain(
        self,
        prompt: str,
        input_docs: list[Any],
        req_keys: list[str],
    ) -> list[Any]:
        """Retry generation for inputs whose outputs fail JSON validation."""
        final_results: list[Any] = [None] * len(input_docs)
        pending_docs = [(i, doc) for i, doc in enumerate(input_docs)]

        for _ in range(self.max_retries):
            if not pending_docs:
                break

            current_batch = [doc for _, doc in pending_docs]
            indices = [i for i, _ in pending_docs]

            formatted_prompts = [
                self._wrap(prompt.format(**doc)) for doc in current_batch
            ]
            raw_results = self.batch_gen(formatted_prompts)

            failed_docs = []
            for orig_idx, doc, content in zip(indices, current_batch, raw_results):
                content = content.strip()
                try:
                    parsed = clean_json(content)
                except Exception:
                    failed_docs.append((orig_idx, doc))
                    continue

                if req_keys:
                    if isinstance(parsed, dict) and all(k in parsed for k in req_keys):
                        final_results[orig_idx] = parsed
                    else:
                        failed_docs.append((orig_idx, doc))
                else:
                    final_results[orig_idx] = parsed

            pending_docs = failed_docs

        for orig_idx, _ in pending_docs:
            if final_results[orig_idx] is None:
                final_results[orig_idx] = {}

        return final_results

    def batch(
        self,
        prompt: str,
        input_docs: list[Any],
        req_keys: list[str] | None = None,
    ) -> list[Any]:
        """Batch-generate responses across input documents."""
        if self.model is None:
            self.load_model()

        if req_keys is not None and self.json_mode:
            return self._retry_chain(prompt, input_docs, req_keys)

        formatted_prompts = [self._wrap(prompt.format(**doc)) for doc in input_docs]
        raw_results = self.batch_gen(formatted_prompts)

        if self.json_mode:
            return [clean_json(res.strip()) for res in raw_results]
        return [res.strip() for res in raw_results]

    def close(self) -> None:
        """Offload local model and clear GPU cache."""
        if self.model is not None and hasattr(self.model, "cpu"):
            try:
                self.model.cpu()
            except Exception as e:
                print("Failed to move model to cpu during close: %s", e)
        super().close()
        self.tokenizer = None
        empty_device_cache()
