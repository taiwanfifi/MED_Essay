"""Google Gemini LLM backend (Gemini 2.5 Pro, Flash, etc.)."""

import logging
from typing import Dict, List, Optional, Tuple

from medeval.config import GEMINI_API_KEY
from medeval.generation.models.base import BaseLLM

logger = logging.getLogger(__name__)


class GeminiModel(BaseLLM):
    """Google Gemini API-backed LLM."""

    def __init__(self, model: str = "gemini-2.5-pro", api_key: Optional[str] = None):
        self._model = model
        self._api_key = api_key or GEMINI_API_KEY
        self._client = None

    def _get_client(self):
        """Lazy-init Gemini client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
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
            from google.genai import types

            client = self._get_client()

            # Convert messages to Gemini format
            # Gemini uses "user" and "model" roles; combine system into first user message
            system_text = ""
            gemini_contents = []

            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg["content"]
                elif msg["role"] == "user":
                    content = msg["content"]
                    if system_text and not gemini_contents:
                        # Prepend system prompt to first user message
                        content = f"{system_text}\n\n{content}"
                        system_text = ""
                    gemini_contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=content)],
                        )
                    )
                elif msg["role"] == "assistant":
                    gemini_contents.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=msg["content"])],
                        )
                    )

            # Gemini's thinking mode consumes output tokens, so we need
            # a higher budget than the requested max_tokens
            effective_max = max(max_tokens, 8192)

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=effective_max,
            )

            # Use non-streaming for simpler handling
            response = client.models.generate_content(
                model=self._model,
                contents=gemini_contents,
                config=config,
            )

            text = response.text.strip() if response.text else ""
            return text, None

        except Exception as e:
            error = f"Gemini ({self._model}) error: {e}"
            logger.error(error)
            return "", error

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Tuple[str, Optional[str]]:
        """Generate with JSON instruction appended."""
        # Gemini supports response_mime_type for JSON, but we use the simpler
        # prompt-based approach for compatibility across model versions
        json_messages = messages.copy()
        if json_messages and json_messages[-1]["role"] == "user":
            json_messages[-1] = {
                "role": "user",
                "content": json_messages[-1]["content"]
                + "\n\nRespond with valid JSON only. No other text.",
            }
        return self.generate(json_messages, temperature=temperature, max_tokens=max_tokens)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = GeminiModel("gemini-2.5-flash")
    text, err = model.generate([{"role": "user", "content": "Say hello in one word."}])
    print(f"Response: {text}, Error: {err}")
