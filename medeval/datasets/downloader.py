"""Download datasets from HuggingFace for MedEval-X."""

import logging
from typing import Optional, Tuple

from datasets import load_dataset, DatasetDict

from medeval.config import DATASETS, MMLU_MED_SUBTASKS

logger = logging.getLogger(__name__)


def download_medqa() -> Tuple[Optional[DatasetDict], Optional[str]]:
    """Download MedQA (USMLE 4-options) from HuggingFace.

    Returns:
        (dataset, error) tuple
    """
    try:
        logger.info("Downloading MedQA (USMLE 4-options)...")
        ds = load_dataset(DATASETS["medqa"])
        logger.info(f"MedQA downloaded: {ds}")
        return ds, None
    except Exception as e:
        error = f"Failed to download MedQA: {e}"
        logger.error(error)
        return None, error


def download_medmcqa() -> Tuple[Optional[DatasetDict], Optional[str]]:
    """Download MedMCQA from HuggingFace.

    Returns:
        (dataset, error) tuple
    """
    try:
        logger.info("Downloading MedMCQA...")
        ds = load_dataset(DATASETS["medmcqa"])
        logger.info(f"MedMCQA downloaded: {ds}")
        return ds, None
    except Exception as e:
        error = f"Failed to download MedMCQA: {e}"
        logger.error(error)
        return None, error


def download_mmlu_med() -> Tuple[Optional[dict], Optional[str]]:
    """Download MMLU medical subtasks from HuggingFace.

    Returns:
        (dict of subtask_name -> dataset, error) tuple
    """
    try:
        logger.info("Downloading MMLU medical subtasks...")
        subtasks = {}
        for subtask in MMLU_MED_SUBTASKS:
            logger.info(f"  Loading MMLU subtask: {subtask}")
            ds = load_dataset(DATASETS["mmlu_med"], subtask)
            subtasks[subtask] = ds
        logger.info(f"MMLU-Med downloaded: {len(subtasks)} subtasks")
        return subtasks, None
    except Exception as e:
        error = f"Failed to download MMLU-Med: {e}"
        logger.error(error)
        return None, error


def download_pubmedqa() -> Tuple[Optional[DatasetDict], Optional[str]]:
    """Download PubMedQA (labeled subset) from HuggingFace.

    Returns:
        (dataset, error) tuple
    """
    try:
        logger.info("Downloading PubMedQA...")
        ds = load_dataset(DATASETS["pubmedqa"], "pqa_labeled")
        logger.info(f"PubMedQA downloaded: {ds}")
        return ds, None
    except Exception as e:
        error = f"Failed to download PubMedQA: {e}"
        logger.error(error)
        return None, error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing MedQA download ===")
    ds, err = download_medqa()
    if ds:
        for split in ds:
            print(f"  {split}: {len(ds[split])} examples")
    else:
        print(f"  Error: {err}")

    print("\n=== Testing MedMCQA download ===")
    ds, err = download_medmcqa()
    if ds:
        for split in ds:
            print(f"  {split}: {len(ds[split])} examples")
    else:
        print(f"  Error: {err}")

    print("\n=== Testing MMLU-Med download ===")
    subtasks, err = download_mmlu_med()
    if subtasks:
        for name, ds in subtasks.items():
            for split in ds:
                print(f"  {name}/{split}: {len(ds[split])} examples")
    else:
        print(f"  Error: {err}")

    print("\n=== Testing PubMedQA download ===")
    ds, err = download_pubmedqa()
    if ds:
        for split in ds:
            print(f"  {split}: {len(ds[split])} examples")
    else:
        print(f"  Error: {err}")
