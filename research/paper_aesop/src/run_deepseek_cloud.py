#!/usr/bin/env python3
"""Run DeepSeek-Chat cloud model (missing from original run).

Safe to Ctrl+C at any time — checkpoint is saved per-scenario.
Re-run this script to resume from where you left off.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_optimization import run_ab_test, DATA_DIR, RESULTS_DIR

MODEL_ID = "deepseek-chat"


def main():
    # Load scenarios
    scenario_path = DATA_DIR / "M9_prior_auth_scenarios.json"
    with open(scenario_path) as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} scenarios")

    # Run DeepSeek-Chat (checkpoint/resume handled inside run_ab_test)
    print(f"\n{'='*60}")
    print(f"Running A/B test: {MODEL_ID}")
    print(f"{'='*60}")
    results = run_ab_test(MODEL_ID, scenarios)
    print(f"Got {len(results)} results")

    expected = len(scenarios) * 2
    if len(results) < expected:
        print(f"\n[PARTIAL] {len(results)}/{expected} — checkpoint saved.")
        print("Re-run this script to resume.")
    else:
        print(f"\n[COMPLETE] All {len(results)} results done.")

    # Merge into main results file (replace any old deepseek-chat entries)
    results_path = RESULTS_DIR / "M9_ab_test_results.json"
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        other = [r for r in all_results if r.get("model_id") != MODEL_ID]
    else:
        other = []

    merged = other + results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {len(merged)} total results → {results_path.name}")


if __name__ == "__main__":
    main()
