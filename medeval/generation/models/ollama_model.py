"""Ollama LLM backend for local models (Llama, Qwen, DeepSeek, etc.)."""

import json
import logging
from typing import Dict, List, Optional, Tuple

import requests

from medeval.config import OLLAMA_BASE_URL
from medeval.generation.models.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaModel(BaseLLM):
    """Ollama API-backed local LLM."""

    def __init__(self, model: str = "llama3.1:8b", base_url: Optional[str] = None):
        self._model = model
        self._base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")

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
            url = f"{self._base_url}/api/chat"
            payload = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if response_format:
                payload["format"] = "json"

            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            text = data.get("message", {}).get("content", "").strip()
            return text, None
        except requests.exceptions.ConnectionError:
            error = f"Ollama ({self._model}): Cannot connect to {self._base_url}. Is Ollama running?"
            logger.error(error)
            return "", error
        except Exception as e:
            error = f"Ollama ({self._model}) error: {e}"
            logger.error(error)
            return "", error

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Tuple[str, Optional[str]]:
        """Use Ollama's native JSON mode."""
        return self.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = OllamaModel("llama3.1:8b")
    text, err = model.generate([{"role": "user", "content": "Say hello in one word."}])
    print(f"Response: {text}, Error: {err}")
