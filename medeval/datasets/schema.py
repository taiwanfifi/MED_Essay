"""Unified schema definition for MedEval-X medical questions."""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class MedicalQuestion:
    """Unified representation of a medical question across all datasets.

    Attributes:
        id: Global unique ID in format "{dataset}_{split}_{idx}"
        dataset: Source dataset name (medqa, medmcqa, mmlu_med, pubmedqa)
        split: Data split (train, dev, test)
        question: Full question text
        options: Answer choices as {"A": "...", "B": "...", ...} or None
        correct_answer: Correct option key (A/B/C/D) or answer text
        topic: Medical subdomain (pharmacology, anatomy, ...)
        difficulty: Difficulty level (easy, medium, hard) if available
        metadata: Dataset-specific extra fields
    """

    id: str
    dataset: str
    split: str
    question: str
    correct_answer: str
    options: Optional[Dict[str, str]] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d["options_json"] = json.dumps(d.pop("options"), ensure_ascii=False) if d.get("options") else None
        d["metadata_json"] = json.dumps(d.pop("metadata"), ensure_ascii=False)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "MedicalQuestion":
        """Reconstruct from a database row dictionary."""
        options = json.loads(d["options_json"]) if d.get("options_json") else None
        metadata = json.loads(d["metadata_json"]) if d.get("metadata_json") else {}
        return cls(
            id=d["id"],
            dataset=d["dataset"],
            split=d["split"],
            question=d["question"],
            correct_answer=d["correct_answer"],
            options=options,
            topic=d.get("topic"),
            difficulty=d.get("difficulty"),
            metadata=metadata,
        )


if __name__ == "__main__":
    # Quick test
    q = MedicalQuestion(
        id="medqa_test_0",
        dataset="medqa",
        split="test",
        question="A 65-year-old male presents with chest pain. What is the diagnosis?",
        options={"A": "PE", "B": "STEMI", "C": "Aortic dissection", "D": "Pericarditis"},
        correct_answer="B",
        topic="cardiology",
        metadata={"source": "usmle_step1"},
    )
    d = q.to_dict()
    print("to_dict:", d)
    q2 = MedicalQuestion.from_dict(d)
    print("from_dict:", q2)
    assert q.id == q2.id
    assert q.options == q2.options
    print("Schema test passed.")
