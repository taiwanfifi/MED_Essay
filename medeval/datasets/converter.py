"""Convert various medical datasets into unified MedicalQuestion schema."""

import logging
from typing import List

from datasets import Dataset

from medeval.datasets.schema import MedicalQuestion

logger = logging.getLogger(__name__)

# MedMCQA cop (1-4) -> option key mapping
_COP_TO_KEY = {1: "A", 2: "B", 3: "C", 4: "D"}

# MMLU answer index (0-3) -> option key mapping
_IDX_TO_KEY = {0: "A", 1: "B", 2: "C", 3: "D"}


def convert_medqa(dataset: Dataset, split: str) -> List[MedicalQuestion]:
    """Convert MedQA dataset to unified schema.

    MedQA fields: question, options (dict), answer_idx, answer, meta_info
    """
    questions = []
    for idx, row in enumerate(dataset):
        options = row.get("options", {})
        # options may be a dict like {"A": "...", "B": "..."}
        if isinstance(options, dict):
            opts = options
        else:
            opts = None

        answer = row.get("answer_idx", row.get("answer", ""))
        meta_info = row.get("meta_info", "")
        topic = None
        if isinstance(meta_info, str) and meta_info:
            topic = meta_info
        elif isinstance(meta_info, dict):
            topic = meta_info.get("topic") or meta_info.get("subject")

        q = MedicalQuestion(
            id=f"medqa_{split}_{idx}",
            dataset="medqa",
            split=split,
            question=row["question"],
            options=opts,
            correct_answer=str(answer),
            topic=topic,
            metadata={"source": "usmle"},
        )
        questions.append(q)

    logger.info(f"MedQA {split}: converted {len(questions)} questions")
    return questions


def convert_medmcqa(dataset: Dataset, split: str) -> List[MedicalQuestion]:
    """Convert MedMCQA dataset to unified schema.

    MedMCQA fields: question, opa, opb, opc, opd, cop (1-4), subject_name, topic_name
    """
    questions = []
    for idx, row in enumerate(dataset):
        options = {
            "A": str(row.get("opa", "")),
            "B": str(row.get("opb", "")),
            "C": str(row.get("opc", "")),
            "D": str(row.get("opd", "")),
        }

        cop = row.get("cop")
        correct_answer = _COP_TO_KEY.get(cop, str(cop)) if cop is not None else ""

        subject = row.get("subject_name", "")
        topic_name = row.get("topic_name", "")
        topic = subject if subject else None

        q = MedicalQuestion(
            id=f"medmcqa_{split}_{idx}",
            dataset="medmcqa",
            split=split,
            question=row["question"],
            options=options,
            correct_answer=correct_answer,
            topic=topic,
            metadata={"subject_name": subject, "topic_name": topic_name},
        )
        questions.append(q)

    logger.info(f"MedMCQA {split}: converted {len(questions)} questions")
    return questions


def convert_mmlu_med(dataset: Dataset, subtask: str, split: str) -> List[MedicalQuestion]:
    """Convert MMLU medical subtask to unified schema.

    MMLU fields: question, choices (list), answer (0-3)
    """
    questions = []
    for idx, row in enumerate(dataset):
        choices = row.get("choices", [])
        options = {}
        for i, choice in enumerate(choices):
            key = _IDX_TO_KEY.get(i, str(i))
            options[key] = str(choice)

        answer_idx = row.get("answer")
        correct_answer = _IDX_TO_KEY.get(answer_idx, str(answer_idx))

        # Use subtask name as topic (e.g. clinical_knowledge -> Clinical Knowledge)
        topic = subtask.replace("_", " ").title()

        q = MedicalQuestion(
            id=f"mmlu_med_{subtask}_{split}_{idx}",
            dataset="mmlu_med",
            split=split,
            question=row["question"],
            options=options,
            correct_answer=correct_answer,
            topic=topic,
            metadata={"subtask": subtask},
        )
        questions.append(q)

    logger.info(f"MMLU-Med {subtask}/{split}: converted {len(questions)} questions")
    return questions


def convert_pubmedqa(dataset: Dataset, split: str) -> List[MedicalQuestion]:
    """Convert PubMedQA (labeled) to unified schema.

    PubMedQA fields: pubid, question, context, long_answer, final_decision (yes/no/maybe)
    """
    questions = []
    for idx, row in enumerate(dataset):
        context = row.get("context", {})
        long_answer = row.get("long_answer", "")
        final_decision = row.get("final_decision", "")

        # PubMedQA is yes/no/maybe, represent as options
        options = {"A": "yes", "B": "no", "C": "maybe"}
        decision_lower = str(final_decision).lower().strip()
        answer_map = {"yes": "A", "no": "B", "maybe": "C"}
        correct_answer = answer_map.get(decision_lower, decision_lower)

        q = MedicalQuestion(
            id=f"pubmedqa_{split}_{idx}",
            dataset="pubmedqa",
            split=split,
            question=row["question"],
            options=options,
            correct_answer=correct_answer,
            topic="biomedical_literature",
            metadata={
                "pubid": row.get("pubid"),
                "context": context,
                "long_answer": long_answer,
                "final_decision": final_decision,
            },
        )
        questions.append(q)

    logger.info(f"PubMedQA {split}: converted {len(questions)} questions")
    return questions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from medeval.datasets.downloader import download_medqa

    ds, err = download_medqa()
    if ds and "test" in ds:
        questions = convert_medqa(ds["test"], "test")
        print(f"\nFirst question: {questions[0]}")
        print(f"\nAs dict: {questions[0].to_dict()}")
