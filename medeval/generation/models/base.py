"""Abstract base class for LLM backends."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract LLM interface for generation and validation."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None,
    ) -> Tuple[str, Optional[str]]:
        """Generate a response from the LLM.

        Args:
            messages: Chat messages in [{"role": "...", "content": "..."}] format
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens to generate
            response_format: Optional structured output format (e.g. JSON schema)

        Returns:
            (response_text, error) tuple
        """
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string."""
        pass

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Tuple[str, Optional[str]]:
        """Generate with JSON response format hint.

        Default implementation appends a JSON instruction; backends may override
        with native JSON mode support.
        """
        # Append JSON instruction to the last user message
        json_messages = messages.copy()
        if json_messages and json_messages[-1]["role"] == "user":
            json_messages[-1] = {
                "role": "user",
                "content": json_messages[-1]["content"]
                + "\n\nRespond with valid JSON only. No other text.",
            }
        return self.generate(json_messages, temperature=temperature, max_tokens=max_tokens)
