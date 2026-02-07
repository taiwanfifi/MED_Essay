#!/usr/bin/env python3
"""
Run ALL 7 models (4 cloud + 3 local) for the Aesop paper.

Safe to Ctrl+C at any time — checkpoint is saved per-scenario.
Re-run this script to resume from where you left off.

Estimated time (from scratch):
  Cloud:  gpt-4o (~15min), gpt-4o-mini (~12min), claude-sonnet-4-5 (~45min), deepseek-chat (~45min)
  Local:  llama-3.1-8b (~40min), deepseek-r1-14b (~70min), qwen3-32b (~110min)
  Total:  ~5-6 hours
"""
import json
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_optimization import (
    run_ab_test, compute_safety_metrics, compute_before_after_comparison,
    generate_figures, DATA_DIR, RESULTS_DIR, _clear_checkpoint
)

# All 7 models in order: fast cloud first, then local by size
ALL_MODELS = [
    "gpt-4o-mini",       # cloud, fastest
    "gpt-4o",            # cloud
    "deepseek-chat",     # cloud
    "claude-sonnet-4-5", # cloud, slower
    "llama-3.1-8b",      # local 8B
    "deepseek-r1-14b",   # local 14B
    "qwen3-32b",         # local 32B, slowest
]

_DONE_FILE = RESULTS_DIR / ".all_models_done.json"


def _load_done() -> list[str]:
    if _DONE_FILE.exists():
        return json.loads(_DONE_FILE.read_text())
    return []


def _save_done(done: list[str]):
    _DONE_FILE.write_text(json.dumps(done))


def _save_results(results_by_model: dict):
    """Save all results merged into the main file."""
    results_path = RESULTS_DIR / "M9_ab_test_results.json"
    merged = []
    for mid in ALL_MODELS:
        merged.extend(results_by_model.get(mid, []))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    n = len(merged)
    print(f"  [SAVE] {n} total results -> {results_path.name}")
    return merged


def _validate_results(results: list[dict], model_id: str) -> dict:
    """Check for empty/simulation responses."""
    empty = 0
    sim = 0
    real = 0
    for r in results:
        reasoning = r.get("reasoning", "")
        decision = r.get("decision", "")
        latency = r.get("latency_ms", 0)
        if not reasoning or decision.lower() == "unknown":
            empty += 1
        elif "Based on the clinical information" in reasoning[:80]:
            sim += 1
        elif latency < 100:
            sim += 1
        else:
            real += 1
    return {"real": real, "empty": empty, "sim": sim, "total": len(results)}


def main():
    # Load scenarios
    scenario_path = DATA_DIR / "M9_prior_auth_scenarios.json"
    with open(scenario_path) as f:
        scenarios = json.load(f)
    expected_per_model = len(scenarios) * 2
    print(f"Loaded {len(scenarios)} scenarios ({expected_per_model} calls per model)")
    print(f"Total expected: {len(ALL_MODELS)} models x {expected_per_model} = {len(ALL_MODELS) * expected_per_model}")

    # Check done models
    done_models = _load_done()
    results_by_model = {}

    # Load completed models from existing main file
    results_path = RESULTS_DIR / "M9_ab_test_results.json"
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        for mid in done_models:
            model_data = [r for r in existing if r.get("model_id") == mid]
            if model_data:
                results_by_model[mid] = model_data
                v = _validate_results(model_data, mid)
                print(f"  [DONE] {mid}: {v['real']} real, {v['empty']} empty, {v['sim']} sim")

    # Run remaining models
    for model_id in ALL_MODELS:
        if model_id in done_models:
            continue

        print(f"\n{'='*60}")
        print(f"Running: {model_id}")
        print(f"{'='*60}")

        results = run_ab_test(model_id, scenarios)
        results_by_model[model_id] = results

        # Validate
        v = _validate_results(results, model_id)
        print(f"  Results: {v['real']} real, {v['empty']} empty, {v['sim']} sim out of {v['total']}")

        if v["real"] == 0:
            print(f"  [WARNING] {model_id}: ALL responses are empty or simulation!")
            print(f"  Check API key / Ollama status. Skipping this model.")
            continue

        if v["empty"] + v["sim"] > v["total"] * 0.1:
            print(f"  [WARNING] {model_id}: >{10}% bad responses ({v['empty']+v['sim']}/{v['total']})")

        if len(results) < expected_per_model:
            print(f"  [PARTIAL] {model_id}: {len(results)}/{expected_per_model} — interrupted.")
            _save_results(results_by_model)
            print("Re-run this script to resume.")
            return

        # Model complete
        done_models.append(model_id)
        _save_done(done_models)
        _save_results(results_by_model)
        print(f"  [COMPLETE] {model_id} done. ({len(done_models)}/{len(ALL_MODELS)} models)")

    # All done
    if _DONE_FILE.exists():
        _DONE_FILE.unlink()

    # Clear all checkpoint files
    for mid in ALL_MODELS:
        _clear_checkpoint(mid)

    # Analyze
    print("\n" + "=" * 60)
    print("All models complete! Analyzing...")
    print("=" * 60)
    merged = _save_results(results_by_model)
    metrics = compute_safety_metrics(merged, scenarios)

    metrics_path = RESULTS_DIR / "M9_safety_metrics.csv"
    if metrics:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Saved metrics -> {metrics_path.name}")

    comparisons = compute_before_after_comparison(metrics)
    comp_path = RESULTS_DIR / "M9_before_after_comparison.csv"
    if comparisons:
        with open(comp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
            writer.writeheader()
            writer.writerows(comparisons)
        print(f"Saved comparisons -> {comp_path.name}")

    # Figures
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)
    generate_figures(metrics, comparisons)

    print("\n" + "=" * 60)
    print("ALL DONE! 7 models x 380 = 2660 real API evaluations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
