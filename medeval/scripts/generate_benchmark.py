#!/usr/bin/env python3
"""One-click M10 benchmark generation script.

Usage:
    # Pilot test: 10 seeds × 6 perturbation types = 60 counterfactuals
    python -m medeval.scripts.generate_benchmark --module m4 --count 10 --pilot

    # Full generation with validation
    python -m medeval.scripts.generate_benchmark --module m4 --count 400

    # M7 cognitive bias generation
    python -m medeval.scripts.generate_benchmark --module m7 --count 30
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from medeval.config import BASE_DIR
from medeval.datasets.store import QuestionStore
from medeval.generation.seed_selector import SeedSelector
from medeval.generation.generator import BenchmarkGenerator
from medeval.generation.validator import MultiModelValidator
from medeval.generation.quality_report import generate_report, save_report, print_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "results"


def init_generator_model(model_name: str = "gpt-4o"):
    """Initialize the generator LLM based on model name prefix."""
    if model_name.startswith("gpt"):
        from medeval.generation.models.openai_model import OpenAIModel
        return OpenAIModel(model_name)
    elif model_name.startswith("claude"):
        from medeval.generation.models.anthropic_model import AnthropicModel
        return AnthropicModel(model_name)
    elif model_name.startswith("gemini"):
        from medeval.generation.models.gemini_model import GeminiModel
        return GeminiModel(model_name)
    elif model_name.startswith("deepseek"):
        from medeval.generation.models.deepseek_model import DeepSeekModel
        return DeepSeekModel(model_name)
    else:
        # Everything else → Ollama (deepseek, qwen, llama, phi, gemma, etc.)
        from medeval.generation.models.ollama_model import OllamaModel
        return OllamaModel(model_name)


def init_validator_models(model_names: list):
    """Initialize validator LLMs."""
    models = []
    for name in model_names:
        models.append(init_generator_model(name))
    return models


def run_m4_generation(args):
    """Run M4 counterfactual generation pipeline."""
    print("=" * 60)
    print("M4 Counterfactual Benchmark Generation")
    print("=" * 60)

    # Step 1: Select seeds
    store = QuestionStore()
    selector = SeedSelector(store, seed=42)

    seeds = selector.select_seeds(
        n=args.count,
        dataset=args.dataset,
        split=args.split,
    )

    if not seeds:
        print("ERROR: No seed questions found. Run prepare_datasets.py first.")
        return

    summary = selector.get_selection_summary(seeds)
    print(f"\nSelected {len(seeds)} seed questions")
    print(f"Topic distribution: {summary}")

    # Step 2: Generate counterfactuals
    gen_model = init_generator_model(args.generator)
    generator = BenchmarkGenerator(gen_model)

    perturbation_types = None
    if args.pilot:
        # Pilot: only 2 perturbation types
        perturbation_types = ["pregnancy", "ckd_stage4"]
        print(f"\nPILOT MODE: Using {perturbation_types}")

    batch = generator.generate_m4_batch(seeds, perturbation_types=perturbation_types)

    # Save raw generation results
    raw_path = RESULTS_DIR / f"m4_raw_generation_{args.count}.json"
    generator.save_batch(batch, raw_path)

    print(f"\nGeneration complete: {batch.total_success}/{batch.total_attempted} succeeded")

    # Step 3: Validate (if not pilot or if --validate flag)
    validation_results = None
    if args.validate and not args.pilot:
        print("\nStarting multi-model validation...")
        validator_models = init_validator_models(args.validators.split(","))
        validator = MultiModelValidator(validator_models)

        # Build lookup maps
        orig_questions = {s.id: s.question for s in seeds}
        orig_answers = {s.id: s.correct_answer for s in seeds}

        successful_items = [item for item in batch.items if item.parsed_data]
        validation_results = validator.validate_batch(
            successful_items, orig_questions, orig_answers
        )

    # Step 4: Quality report
    report = generate_report(batch, validation_results)
    print_report(report)

    report_path = RESULTS_DIR / f"m4_quality_report_{args.count}.json"
    save_report(report, report_path)

    print(f"\nResults saved to: {RESULTS_DIR}")


def run_m7_generation(args):
    """Run M7 cognitive bias generation pipeline."""
    from medeval.generation.prompts.m7_cognitive_bias import BIAS_TYPES

    print("=" * 60)
    print("M7 Cognitive Bias Benchmark Generation")
    print("=" * 60)

    gen_model = init_generator_model(args.generator)
    generator = BenchmarkGenerator(gen_model)

    specialties = [
        "cardiology", "pharmacology", "emergency_medicine",
        "internal_medicine", "neurology", "oncology",
        "pediatrics", "psychiatry", "surgery", "radiology",
    ]

    from medeval.generation.generator import GenerationBatch

    batch = GenerationBatch(module="m7")
    total = len(BIAS_TYPES) * min(args.count, len(specialties))

    for bias_type in BIAS_TYPES:
        for specialty in specialties[: args.count]:
            batch.total_attempted += 1
            logger.info(f"Generating M7 {bias_type}/{specialty}")
            item = generator.generate_m7_cognitive_bias(bias_type, specialty)
            batch.items.append(item)

            if item.parsed_data:
                batch.total_success += 1
            elif item.parse_error:
                batch.total_parse_error += 1

    raw_path = RESULTS_DIR / f"m7_raw_generation_{args.count}.json"
    generator.save_batch(batch, raw_path)

    report = generate_report(batch)
    print_report(report)

    report_path = RESULTS_DIR / f"m7_quality_report_{args.count}.json"
    save_report(report, report_path)

    print(f"\nResults saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="MedEval-X Benchmark Generation (M10 Pipeline)")
    parser.add_argument("--module", choices=["m4", "m7", "m2", "m5"], default="m4",
                        help="Which module to generate benchmarks for")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of seed questions (m4) or items per bias type (m7)")
    parser.add_argument("--dataset", default="medqa",
                        help="Source dataset for seed selection")
    parser.add_argument("--split", default="test",
                        help="Data split for seed selection")
    parser.add_argument("--generator", default="gpt-4o",
                        help="Generator model name")
    parser.add_argument("--validators", default="claude-sonnet-4-5-20250929,gpt-4o-mini",
                        help="Comma-separated validator model names")
    parser.add_argument("--validate", action="store_true",
                        help="Run multi-model validation after generation")
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot test with reduced scope")

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.module == "m4":
        run_m4_generation(args)
    elif args.module == "m7":
        run_m7_generation(args)
    else:
        print(f"Module {args.module} generation not yet implemented.")
        print("Available: m4, m7")


if __name__ == "__main__":
    main()
