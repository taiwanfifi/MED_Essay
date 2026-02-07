#!/usr/bin/env python3
"""
Rerun ONLY local Ollama models, keeping existing cloud results intact.
Then re-analyze and regenerate figures.

Safe to Ctrl+C at any time — checkpoint is saved per-scenario.
Re-run this script to resume from where you left off.
"""
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_optimization import (
    run_ab_test, compute_safety_metrics, compute_before_after_comparison,
    generate_figures, DATA_DIR, RESULTS_DIR, _load_checkpoint
)

LOCAL_MODELS = ["llama-3.1-8b", "deepseek-r1-14b", "qwen3-32b",
                "phi4-14b", "biomistral-7b", "med42-8b"]

# Completed-model tracker (persisted so we know which are done on resume)
_DONE_FILE = RESULTS_DIR / ".local_models_done.json"


def _load_done_models() -> list[str]:
    if _DONE_FILE.exists():
        return json.loads(_DONE_FILE.read_text())
    return []


def _save_done_models(done: list[str]):
    _DONE_FILE.write_text(json.dumps(done))


def _merge_and_save(cloud_results, local_results_by_model):
    """Merge cloud + all available local results and save."""
    results_path = RESULTS_DIR / "M9_ab_test_results.json"
    merged = list(cloud_results)
    for model_id in LOCAL_MODELS:
        merged.extend(local_results_by_model.get(model_id, []))
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  [SAVE] {len(merged)} total results → {results_path.name}")
    return merged


def main():
    # 1. Load existing results
    results_path = RESULTS_DIR / "M9_ab_test_results.json"
    with open(results_path) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results")

    # 2. Keep only cloud model results (remove old local/simulation results)
    old_local_ids = {"llama-3.1-8b", "qwen-2.5-32b", "deepseek-r1-14b",
                     "biomistral-7b", "med42-v2", "med42-8b", "qwen3-32b",
                     "phi4-14b"}
    cloud_results = [r for r in all_results if r.get("model_id") not in old_local_ids]
    print(f"Kept {len(cloud_results)} cloud results")

    # 3. Load scenarios
    scenario_path = DATA_DIR / "M9_prior_auth_scenarios.json"
    with open(scenario_path) as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} scenarios")

    # 4. Check which models already finished in a previous run
    done_models = _load_done_models()
    local_results_by_model = {}

    # Load already-completed models' data from checkpoints or prior merge
    for model_id in done_models:
        # Data is already in the main results file from the last merge
        model_data = [r for r in all_results if r.get("model_id") == model_id]
        if model_data:
            local_results_by_model[model_id] = model_data
            print(f"  [DONE] {model_id}: {len(model_data)} results (completed previously)")

    # 5. Run each remaining local model
    for model_id in LOCAL_MODELS:
        if model_id in done_models:
            continue

        print(f"\n{'='*60}")
        print(f"Running A/B test: {model_id}")
        print(f"{'='*60}")

        results = run_ab_test(model_id, scenarios)
        local_results_by_model[model_id] = results
        print(f"  Got {len(results)} results for {model_id}")

        expected = len(scenarios) * 2
        if len(results) < expected:
            # Interrupted mid-model — save what we have and exit
            print(f"\n  [PARTIAL] {model_id}: {len(results)}/{expected} — saving and exiting.")
            _merge_and_save(cloud_results, local_results_by_model)
            print("\nRe-run this script to resume.")
            return

        # Model fully completed — record it
        done_models.append(model_id)
        _save_done_models(done_models)
        _merge_and_save(cloud_results, local_results_by_model)

    # 6. All done — clean up tracker
    if _DONE_FILE.exists():
        _DONE_FILE.unlink()

    # 7. Re-analyze
    merged = _merge_and_save(cloud_results, local_results_by_model)
    print("\n" + "=" * 60)
    print("Re-analyzing results...")
    print("=" * 60)
    metrics = compute_safety_metrics(merged, scenarios)

    import csv
    metrics_path = RESULTS_DIR / "M9_safety_metrics.csv"
    if metrics:
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Saved metrics to {metrics_path}")

    comparisons = compute_before_after_comparison(metrics)
    comp_path = RESULTS_DIR / "M9_before_after_comparison.csv"
    if comparisons:
        with open(comp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
            writer.writeheader()
            writer.writerows(comparisons)
        print(f"Saved comparisons to {comp_path}")

    # 8. Regenerate figures
    print("\n" + "=" * 60)
    print("Regenerating figures...")
    print("=" * 60)
    generate_figures(metrics, comparisons)

    print("\nDone! All local models re-run with real Ollama data.")


if __name__ == "__main__":
    main()
