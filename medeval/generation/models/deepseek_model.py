"""DeepSeek Cloud LLM backend (OpenAI-compatible API)."""

import logging
from typing import Dict, List, Optional, Tuple

import openai

from medeval.config import DEEPSEEK_API_KEY
from medeval.generation.models.base import BaseLLM

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekModel(BaseLLM):
    """DeepSeek Cloud API-backed LLM (OpenAI-compatible)."""

    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None):
        self._model = model
        self._client = openai.OpenAI(
            api_key=api_key or DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None,
    ) -> Tuple[str, Optional[str]]:
        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = self._client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content.strip()
            return text, None
        except Exception as e:
            error = f"DeepSeek ({self._model}) error: {e}"
            logger.error(error)
            return "", error

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Tuple[str, Optional[str]]:
        """Use OpenAI-compatible JSON mode."""
        return self.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = DeepSeekModel("deepseek-chat")
    text, err = model.generate([{"role": "user", "content": "Say hello in one word."}])
    print(f"Response: {text}, Error: {err}")
