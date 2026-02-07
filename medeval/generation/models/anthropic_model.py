"""Anthropic LLM backend (Claude 3.5 Sonnet, etc.)."""

import logging
from typing import Dict, List, Optional, Tuple

from medeval.config import ANTHROPIC_API_KEY
from medeval.generation.models.base import BaseLLM

logger = logging.getLogger(__name__)


class AnthropicModel(BaseLLM):
    """Anthropic API-backed LLM."""

    def __init__(self, model: str = "claude-sonnet-4-5-20250929", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key or ANTHROPIC_API_KEY
        self._client = None

    def _get_client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

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
            client = self._get_client()

            # Separate system message from conversation
            system_msg = ""
            chat_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    chat_messages.append(msg)

            kwargs = {
                "model": self._model,
                "messages": chat_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_msg:
                kwargs["system"] = system_msg

            response = client.messages.create(**kwargs)
            text = response.content[0].text.strip()
            return text, None
        except Exception as e:
            error = f"Anthropic ({self._model}) error: {e}"
            logger.error(error)
            return "", error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = AnthropicModel()
    text, err = model.generate([{"role": "user", "content": "Say hello in one word."}])
    print(f"Response: {text}, Error: {err}")
