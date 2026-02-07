#!/usr/bin/env python3
"""One-click script to download, convert, and store all MedEval-X datasets.

Usage:
    python -m medeval.scripts.prepare_datasets
    # or
    python medeval/scripts/prepare_datasets.py
"""

import logging
import sys
from pathlib import Path

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from medeval.datasets.downloader import (
    download_medqa,
    download_medmcqa,
    download_mmlu_med,
    download_pubmedqa,
)
from medeval.datasets.converter import (
    convert_medqa,
    convert_medmcqa,
    convert_mmlu_med,
    convert_pubmedqa,
)
from medeval.datasets.store import QuestionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_medqa(store: QuestionStore) -> int:
    """Download and store MedQA."""
    ds, err = download_medqa()
    if err:
        logger.error(f"MedQA download failed: {err}")
        return 0
    total = 0
    for split in ds:
        questions = convert_medqa(ds[split], split)
        count, err = store.insert_questions(questions)
        if err:
            logger.error(f"MedQA {split} insert failed: {err}")
        else:
            total += count
    return total


def prepare_medmcqa(store: QuestionStore) -> int:
    """Download and store MedMCQA."""
    ds, err = download_medmcqa()
    if err:
        logger.error(f"MedMCQA download failed: {err}")
        return 0
    total = 0
    for split in ds:
        questions = convert_medmcqa(ds[split], split)
        count, err = store.insert_questions(questions)
        if err:
            logger.error(f"MedMCQA {split} insert failed: {err}")
        else:
            total += count
    return total


def prepare_mmlu_med(store: QuestionStore) -> int:
    """Download and store MMLU medical subtasks."""
    subtasks, err = download_mmlu_med()
    if err:
        logger.error(f"MMLU-Med download failed: {err}")
        return 0
    total = 0
    for subtask_name, ds in subtasks.items():
        for split in ds:
            questions = convert_mmlu_med(ds[split], subtask_name, split)
            count, err = store.insert_questions(questions)
            if err:
                logger.error(f"MMLU-Med {subtask_name}/{split} insert failed: {err}")
            else:
                total += count
    return total


def prepare_pubmedqa(store: QuestionStore) -> int:
    """Download and store PubMedQA."""
    ds, err = download_pubmedqa()
    if err:
        logger.error(f"PubMedQA download failed: {err}")
        return 0
    total = 0
    for split in ds:
        questions = convert_pubmedqa(ds[split], split)
        count, err = store.insert_questions(questions)
        if err:
            logger.error(f"PubMedQA {split} insert failed: {err}")
        else:
            total += count
    return total


def main():
    """Run the full dataset preparation pipeline."""
    print("=" * 60)
    print("MedEval-X Dataset Preparation Pipeline")
    print("=" * 60)

    store = QuestionStore()

    results = {}

    # 1. MedQA
    print("\n[1/4] Preparing MedQA (USMLE)...")
    results["MedQA"] = prepare_medqa(store)

    # 2. MedMCQA
    print("\n[2/4] Preparing MedMCQA...")
    results["MedMCQA"] = prepare_medmcqa(store)

    # 3. MMLU-Med
    print("\n[3/4] Preparing MMLU-Med (6 subtasks)...")
    results["MMLU-Med"] = prepare_mmlu_med(store)

    # 4. PubMedQA
    print("\n[4/4] Preparing PubMedQA...")
    results["PubMedQA"] = prepare_pubmedqa(store)

    # Summary
    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)

    for name, count in results.items():
        status = "OK" if count > 0 else "FAILED"
        print(f"  {status} {name}: {count:,} questions loaded")

    total = store.get_total_count()
    print(f"\n  Total: {total:,} questions in medeval.db")

    # Topic distribution
    print("\nTopic distribution:")
    topics = store.count_by_topic()
    for topic, count in list(topics.items())[:20]:
        print(f"  {topic}: {count:,}")
    if len(topics) > 20:
        print(f"  ... and {len(topics) - 20} more topics")

    # Dataset × split breakdown
    print("\nDataset × Split breakdown:")
    ds_split = store.count_by_dataset_split()
    for ds_name, splits in ds_split.items():
        for split_name, count in splits.items():
            print(f"  {ds_name}/{split_name}: {count:,}")

    print(f"\nDatabase saved to: {store.db_path}")


if __name__ == "__main__":
    main()
