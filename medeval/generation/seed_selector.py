"""Select seed questions from the database using stratified sampling."""

import logging
import random
from collections import defaultdict
from typing import Dict, List, Optional

from medeval.datasets.schema import MedicalQuestion
from medeval.datasets.store import QuestionStore

logger = logging.getLogger(__name__)


class SeedSelector:
    """Stratified sampling of seed questions for benchmark generation."""

    def __init__(self, store: QuestionStore, seed: int = 42):
        self.store = store
        self.rng = random.Random(seed)

    def select_seeds(
        self,
        n: int = 400,
        dataset: str = "medqa",
        split: str = "test",
        stratify_by: str = "topic",
    ) -> List[MedicalQuestion]:
        """Select n seed questions using stratified sampling.

        Args:
            n: Number of seeds to select
            dataset: Source dataset to sample from
            split: Data split to sample from
            stratify_by: Field to stratify by (topic or difficulty)

        Returns:
            List of selected MedicalQuestion objects
        """
        all_questions = self.store.get_questions(dataset=dataset, split=split)
        if not all_questions:
            logger.warning(f"No questions found for {dataset}/{split}")
            return []

        # Group by stratum
        strata: Dict[str, List[MedicalQuestion]] = defaultdict(list)
        for q in all_questions:
            key = getattr(q, stratify_by, None) or "(none)"
            strata[key].append(q)

        # Calculate proportional allocation
        total_available = len(all_questions)
        n_to_select = min(n, total_available)

        selected = []
        remaining = n_to_select

        sorted_strata = sorted(strata.items(), key=lambda x: len(x[1]), reverse=True)

        for i, (stratum_key, questions) in enumerate(sorted_strata):
            # Proportional allocation
            proportion = len(questions) / total_available
            stratum_n = round(proportion * n_to_select)

            # Last stratum gets whatever is remaining
            if i == len(sorted_strata) - 1:
                stratum_n = remaining

            stratum_n = min(stratum_n, len(questions), remaining)
            if stratum_n <= 0:
                continue

            sampled = self.rng.sample(questions, stratum_n)
            selected.extend(sampled)
            remaining -= stratum_n

            logger.info(f"  Stratum '{stratum_key}': {len(questions)} available, {stratum_n} selected")

        self.rng.shuffle(selected)
        logger.info(f"Selected {len(selected)} seed questions from {dataset}/{split} (stratified by {stratify_by})")
        return selected

    def get_selection_summary(self, seeds: List[MedicalQuestion]) -> Dict[str, int]:
        """Get a summary of selected seeds by topic."""
        summary: Dict[str, int] = defaultdict(int)
        for q in seeds:
            key = q.topic or "(none)"
            summary[key] += 1
        return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = QuestionStore()
    selector = SeedSelector(store)

    seeds = selector.select_seeds(n=50, dataset="medqa", split="test")
    print(f"\nSelected {len(seeds)} seeds")
    summary = selector.get_selection_summary(seeds)
    print(f"Topic distribution: {summary}")

    if seeds:
        print(f"\nExample seed: {seeds[0].question[:100]}...")
