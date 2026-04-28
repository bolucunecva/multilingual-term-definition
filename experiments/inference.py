import json
import re
from typing import List, Tuple, Dict, Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class Extractor:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        use_chat_template: bool = True,
        **llm_kwargs,
    ):
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=40000, 
            gpu_memory_utilization=0.90,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        self.tokenizer = (
            AutoTokenizer.from_pretrained(model_name)
            if use_chat_template
            else None
        )

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    @staticmethod
    def parse_pairs(raw: str) -> List[Tuple[str, str]]:
        """Parse strict JSON output into (term, definition) pairs."""
        if not raw or not raw.strip():
            return []

        raw = raw.strip()

        # 1. Fast path: assume strict JSON (as enforced by prompt)
        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            # 2. Fallback: recover JSON array if model slightly misbehaves
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            match = re.search(r"\[\s*{.*}\s*\]", raw, re.DOTALL)
            if not match:
                return []
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                return []

        # 3. Validate structure
        if not isinstance(items, list):
            return []

        pairs: List[Tuple[str, str]] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            term = item.get("term")
            definition = item.get("definition")

            if not isinstance(term, str) or not isinstance(definition, str):
                continue

            term = term.strip()
            definition = definition.strip()

            if term and definition:
                pairs.append((term, definition))

        return pairs
