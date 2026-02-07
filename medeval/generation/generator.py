"""Benchmark question generator using LLMs with structured output."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from medeval.datasets.schema import MedicalQuestion
from medeval.generation.models.base import BaseLLM
from medeval.generation.prompts import m4_counterfactual, m2_ebm_conflict, m5_ehr_noise, m7_cognitive_bias

logger = logging.getLogger(__name__)


@dataclass
class GeneratedItem:
    """A single generated benchmark item."""
    seed_id: str
    module: str  # m2, m4, m5, m7
    variant_type: str  # perturbation type, bias type, etc.
    raw_response: str
    parsed_data: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None
    generator_model: str = ""
    generation_time_ms: int = 0


@dataclass
class GenerationBatch:
    """Results of a batch generation run."""
    module: str
    total_attempted: int = 0
    total_success: int = 0
    total_parse_error: int = 0
    total_api_error: int = 0
    items: List[GeneratedItem] = field(default_factory=list)


class BenchmarkGenerator:
    """Generate benchmark questions using LLMs."""

    def __init__(self, model: BaseLLM, output_dir: Optional[Path] = None):
        self.model = model
        self.output_dir = output_dir

    def _parse_json_response(self, text: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Parse JSON from LLM response, handling markdown code blocks and common issues."""
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(cleaned), None
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract the first {...} block (handles extra text around JSON)
        import re
        match = re.search(r'\{', cleaned)
        if match:
            brace_count = 0
            start = match.start()
            for i in range(start, len(cleaned)):
                if cleaned[i] == '{':
                    brace_count += 1
                elif cleaned[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = cleaned[start:i + 1]
                        try:
                            return json.loads(candidate), None
                        except json.JSONDecodeError:
                            break

        # Fallback: try fixing common issues (trailing commas, unescaped newlines)
        try:
            fixed = re.sub(r',\s*}', '}', cleaned)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed), None
        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {e}"

    def generate_m4_counterfactual(
        self,
        seed: MedicalQuestion,
        perturbation_type: str,
    ) -> GeneratedItem:
        """Generate a single M4 counterfactual variant.

        Args:
            seed: Original question to perturb
            perturbation_type: Key from m4_counterfactual.PERTURBATION_TYPES
        """
        # Format options for prompt
        options_str = ""
        if seed.options:
            for key, val in sorted(seed.options.items()):
                options_str += f"{key}) {val}\n"

        messages = m4_counterfactual.build_generation_messages(
            question=seed.question,
            options=options_str,
            correct_answer=seed.correct_answer,
            perturbation_type=perturbation_type,
        )

        start = time.time()
        response, error = self.model.generate_json(messages)
        elapsed_ms = int((time.time() - start) * 1000)

        item = GeneratedItem(
            seed_id=seed.id,
            module="m4",
            variant_type=perturbation_type,
            raw_response=response,
            generator_model=self.model.model_name(),
            generation_time_ms=elapsed_ms,
        )

        if error:
            item.parse_error = error
            return item

        parsed, parse_err = self._parse_json_response(response)
        item.parsed_data = parsed
        item.parse_error = parse_err
        return item

    def generate_m2_ebm_conflict(self, topic: str, bias_type: str) -> GeneratedItem:
        """Generate a single M2 EBM conflict scenario."""
        messages = m2_ebm_conflict.build_generation_messages(topic, bias_type)

        start = time.time()
        response, error = self.model.generate_json(messages)
        elapsed_ms = int((time.time() - start) * 1000)

        item = GeneratedItem(
            seed_id=f"m2_{topic}_{bias_type}",
            module="m2",
            variant_type=bias_type,
            raw_response=response,
            generator_model=self.model.model_name(),
            generation_time_ms=elapsed_ms,
        )

        if error:
            item.parse_error = error
            return item

        parsed, parse_err = self._parse_json_response(response)
        item.parsed_data = parsed
        item.parse_error = parse_err
        return item

    def generate_m5_ehr_noise(
        self,
        seed: MedicalQuestion,
        noise_type: str,
        intensity: str = "moderate",
    ) -> GeneratedItem:
        """Generate a single M5 EHR noise variant."""
        messages = m5_ehr_noise.build_generation_messages(
            clean_scenario=seed.question,
            correct_answer=seed.correct_answer,
            noise_type=noise_type,
            intensity=intensity,
        )

        start = time.time()
        response, error = self.model.generate_json(messages)
        elapsed_ms = int((time.time() - start) * 1000)

        item = GeneratedItem(
            seed_id=seed.id,
            module="m5",
            variant_type=f"{noise_type}_{intensity}",
            raw_response=response,
            generator_model=self.model.model_name(),
            generation_time_ms=elapsed_ms,
        )

        if error:
            item.parse_error = error
            return item

        parsed, parse_err = self._parse_json_response(response)
        item.parsed_data = parsed
        item.parse_error = parse_err
        return item

    def generate_m7_cognitive_bias(self, bias_type: str, specialty: str) -> GeneratedItem:
        """Generate a single M7 cognitive bias test pair."""
        messages = m7_cognitive_bias.build_generation_messages(bias_type, specialty)

        start = time.time()
        response, error = self.model.generate_json(messages)
        elapsed_ms = int((time.time() - start) * 1000)

        item = GeneratedItem(
            seed_id=f"m7_{bias_type}_{specialty}",
            module="m7",
            variant_type=bias_type,
            raw_response=response,
            generator_model=self.model.model_name(),
            generation_time_ms=elapsed_ms,
        )

        if error:
            item.parse_error = error
            return item

        parsed, parse_err = self._parse_json_response(response)
        item.parsed_data = parsed
        item.parse_error = parse_err
        return item

    def generate_m4_batch(
        self,
        seeds: List[MedicalQuestion],
        perturbation_types: Optional[List[str]] = None,
    ) -> GenerationBatch:
        """Generate M4 counterfactual variants for a batch of seeds.

        Args:
            seeds: List of seed questions
            perturbation_types: Which perturbation types to apply (default: all 6)
        """
        if perturbation_types is None:
            perturbation_types = list(m4_counterfactual.PERTURBATION_TYPES.keys())

        batch = GenerationBatch(module="m4")
        total = len(seeds) * len(perturbation_types)

        for i, seed in enumerate(seeds):
            for j, ptype in enumerate(perturbation_types):
                batch.total_attempted += 1
                idx = i * len(perturbation_types) + j + 1
                logger.info(f"[{idx}/{total}] Generating M4 {ptype} for {seed.id}")

                item = self.generate_m4_counterfactual(seed, ptype)
                batch.items.append(item)

                if item.parse_error:
                    batch.total_parse_error += 1
                    logger.warning(f"  Error: {item.parse_error}")
                elif item.parsed_data:
                    batch.total_success += 1
                else:
                    batch.total_api_error += 1

        return batch

    def save_batch(self, batch: GenerationBatch, filepath: Path):
        """Save a generation batch to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "module": batch.module,
            "total_attempted": batch.total_attempted,
            "total_success": batch.total_success,
            "total_parse_error": batch.total_parse_error,
            "total_api_error": batch.total_api_error,
            "items": [
                {
                    "seed_id": item.seed_id,
                    "module": item.module,
                    "variant_type": item.variant_type,
                    "parsed_data": item.parsed_data,
                    "parse_error": item.parse_error,
                    "generator_model": item.generator_model,
                    "generation_time_ms": item.generation_time_ms,
                }
                for item in batch.items
            ],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Batch saved to {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("BenchmarkGenerator ready. Use with an LLM model instance.")
    print("Example:")
    print("  from medeval.generation.models.openai_model import OpenAIModel")
    print("  model = OpenAIModel('gpt-4o')")
    print("  gen = BenchmarkGenerator(model)")
