"""Multi-model cross-validation for generated benchmark items."""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from medeval.generation.models.base import BaseLLM
from medeval.generation.generator import GeneratedItem
from medeval.generation.prompts.m4_counterfactual import build_validation_messages

logger = logging.getLogger(__name__)


@dataclass
class ModelVerdict:
    """A single model's verdict on a generated item."""
    model_name: str
    medically_correct: Optional[bool] = None
    clinically_plausible: Optional[bool] = None
    answer_correct: Optional[bool] = None
    overall_verdict: str = ""  # accept, reject, needs_revision
    issues: str = ""
    raw_response: str = ""
    parse_error: Optional[str] = None


@dataclass
class ValidationResult:
    """Aggregated validation result from multiple models."""
    item_seed_id: str
    item_variant_type: str
    verdicts: List[ModelVerdict] = field(default_factory=list)
    consensus: bool = False
    agreement_score: float = 0.0
    needs_human_review: bool = False
    final_verdict: str = ""  # accept, reject, needs_revision

    @property
    def accept_count(self) -> int:
        return sum(1 for v in self.verdicts if v.overall_verdict == "accept")

    @property
    def reject_count(self) -> int:
        return sum(1 for v in self.verdicts if v.overall_verdict == "reject")


class MultiModelValidator:
    """Validate generated items using multiple LLMs for cross-supervision."""

    def __init__(self, models: List[BaseLLM], consensus_threshold: float = 2 / 3):
        """
        Args:
            models: At least 2 LLMs from different families for diverse validation
            consensus_threshold: Fraction of models that must agree for consensus
        """
        if len(models) < 2:
            raise ValueError("MultiModelValidator requires at least 2 models")
        self.models = models
        self.consensus_threshold = consensus_threshold

    def _parse_verdict(self, response: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Parse validation verdict from LLM response."""
        cleaned = response.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(cleaned), None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"

    def validate_m4_item(self, item: GeneratedItem, original_question: str, original_answer: str) -> ValidationResult:
        """Validate a single M4 counterfactual item using all models.

        Args:
            item: The generated counterfactual item
            original_question: Original question text
            original_answer: Original correct answer

        Returns:
            ValidationResult with all model verdicts
        """
        if not item.parsed_data:
            return ValidationResult(
                item_seed_id=item.seed_id,
                item_variant_type=item.variant_type,
                needs_human_review=True,
                final_verdict="reject",
            )

        result = ValidationResult(
            item_seed_id=item.seed_id,
            item_variant_type=item.variant_type,
        )

        messages = build_validation_messages(
            original_question=original_question,
            original_answer=original_answer,
            modified_question=item.parsed_data.get("modified_question", ""),
            new_answer=item.parsed_data.get("new_correct_answer", ""),
            clinical_reasoning=item.parsed_data.get("clinical_reasoning", ""),
            perturbation_type=item.variant_type,
        )

        for model in self.models:
            response, error = model.generate_json(messages)

            verdict = ModelVerdict(model_name=model.model_name(), raw_response=response)

            if error:
                verdict.parse_error = error
                result.verdicts.append(verdict)
                continue

            parsed, parse_err = self._parse_verdict(response)
            if parse_err:
                verdict.parse_error = parse_err
                result.verdicts.append(verdict)
                continue

            verdict.medically_correct = parsed.get("medically_correct")
            verdict.clinically_plausible = parsed.get("clinically_plausible")
            verdict.answer_correct = parsed.get("answer_correct")
            verdict.overall_verdict = parsed.get("overall_verdict", "")
            verdict.issues = parsed.get("issues", "")
            result.verdicts.append(verdict)

        # Compute consensus
        valid_verdicts = [v for v in result.verdicts if v.overall_verdict in ("accept", "reject", "needs_revision")]
        if valid_verdicts:
            result.agreement_score = result.accept_count / len(valid_verdicts)
            result.consensus = result.agreement_score >= self.consensus_threshold
            result.final_verdict = "accept" if result.consensus else "reject"
        else:
            result.needs_human_review = True
            result.final_verdict = "needs_revision"

        if not result.consensus:
            result.needs_human_review = True

        logger.info(
            f"Validation for {item.seed_id}/{item.variant_type}: "
            f"{result.accept_count}/{len(valid_verdicts)} accept, "
            f"consensus={result.consensus}"
        )

        return result

    def validate_batch(
        self,
        items: List[GeneratedItem],
        original_questions: Dict[str, str],
        original_answers: Dict[str, str],
    ) -> List[ValidationResult]:
        """Validate a batch of generated items.

        Args:
            items: List of GeneratedItem objects
            original_questions: Mapping of seed_id -> question text
            original_answers: Mapping of seed_id -> correct answer
        """
        results = []
        for i, item in enumerate(items):
            logger.info(f"[{i + 1}/{len(items)}] Validating {item.seed_id}/{item.variant_type}")
            orig_q = original_questions.get(item.seed_id, "")
            orig_a = original_answers.get(item.seed_id, "")
            result = self.validate_m4_item(item, orig_q, orig_a)
            results.append(result)
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("MultiModelValidator ready. Requires at least 2 BaseLLM instances.")
    print("Example:")
    print("  from medeval.generation.models.openai_model import OpenAIModel")
    print("  from medeval.generation.models.anthropic_model import AnthropicModel")
    print("  validator = MultiModelValidator([OpenAIModel(), AnthropicModel()])")
