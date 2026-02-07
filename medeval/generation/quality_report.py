"""Quality report generation for M10 benchmark pipeline."""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from medeval.generation.generator import GenerationBatch
from medeval.generation.validator import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Summary statistics for a benchmark generation run."""

    module: str
    total_generated: int = 0
    format_valid: int = 0
    ai_consensus_pass: int = 0
    ai_consensus_fail: int = 0
    needs_human_review: int = 0
    human_reviewed: int = 0
    human_approved: int = 0
    final_usable: int = 0

    format_pass_rate: float = 0.0
    ai_consensus_rate: float = 0.0
    human_approval_rate: float = 0.0
    overall_pass_rate: float = 0.0

    topic_distribution: Dict[str, int] = field(default_factory=dict)
    variant_distribution: Dict[str, int] = field(default_factory=dict)
    avg_generation_time_ms: float = 0.0
    collective_hallucination_rate: float = 0.0

    model_agreement_scores: List[float] = field(default_factory=list)

    @property
    def mean_agreement(self) -> float:
        if not self.model_agreement_scores:
            return 0.0
        return sum(self.model_agreement_scores) / len(self.model_agreement_scores)


def generate_report(
    batch: GenerationBatch,
    validation_results: Optional[List[ValidationResult]] = None,
) -> QualityReport:
    """Generate a quality report from generation and validation results.

    Args:
        batch: The generation batch results
        validation_results: Optional list of validation results (if validation was run)
    """
    report = QualityReport(module=batch.module)
    report.total_generated = batch.total_attempted
    report.format_valid = batch.total_success

    if batch.total_attempted > 0:
        report.format_pass_rate = batch.total_success / batch.total_attempted

    # Variant distribution
    for item in batch.items:
        report.variant_distribution[item.variant_type] = (
            report.variant_distribution.get(item.variant_type, 0) + 1
        )

    # Average generation time
    times = [item.generation_time_ms for item in batch.items if item.generation_time_ms > 0]
    if times:
        report.avg_generation_time_ms = sum(times) / len(times)

    # Process validation results
    if validation_results:
        for vr in validation_results:
            report.model_agreement_scores.append(vr.agreement_score)

            if vr.final_verdict == "accept":
                report.ai_consensus_pass += 1
            else:
                report.ai_consensus_fail += 1

            if vr.needs_human_review:
                report.needs_human_review += 1

        valid_count = report.ai_consensus_pass + report.ai_consensus_fail
        if valid_count > 0:
            report.ai_consensus_rate = report.ai_consensus_pass / valid_count

        # Without human review, final usable = AI consensus pass
        report.final_usable = report.ai_consensus_pass
        if report.total_generated > 0:
            report.overall_pass_rate = report.final_usable / report.total_generated

    return report


def save_report(report: QualityReport, filepath: Path):
    """Save quality report to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "module": report.module,
        "total_generated": report.total_generated,
        "format_valid": report.format_valid,
        "format_pass_rate": round(report.format_pass_rate, 4),
        "ai_consensus_pass": report.ai_consensus_pass,
        "ai_consensus_fail": report.ai_consensus_fail,
        "ai_consensus_rate": round(report.ai_consensus_rate, 4),
        "needs_human_review": report.needs_human_review,
        "human_reviewed": report.human_reviewed,
        "human_approved": report.human_approved,
        "final_usable": report.final_usable,
        "overall_pass_rate": round(report.overall_pass_rate, 4),
        "mean_model_agreement": round(report.mean_agreement, 4),
        "variant_distribution": report.variant_distribution,
        "avg_generation_time_ms": round(report.avg_generation_time_ms, 1),
        "collective_hallucination_rate": round(report.collective_hallucination_rate, 4),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Quality report saved to {filepath}")


def print_report(report: QualityReport):
    """Print a formatted quality report to stdout."""
    print("\n" + "=" * 60)
    print(f"QUALITY REPORT — Module {report.module.upper()}")
    print("=" * 60)
    print(f"  Total generated:      {report.total_generated}")
    print(f"  Format valid:         {report.format_valid} ({report.format_pass_rate:.1%})")
    print(f"  AI consensus pass:    {report.ai_consensus_pass} ({report.ai_consensus_rate:.1%})")
    print(f"  AI consensus fail:    {report.ai_consensus_fail}")
    print(f"  Needs human review:   {report.needs_human_review}")
    print(f"  Final usable:         {report.final_usable} ({report.overall_pass_rate:.1%})")
    print(f"  Mean model agreement: {report.mean_agreement:.3f}")
    print(f"  Avg generation time:  {report.avg_generation_time_ms:.0f} ms")
    print()
    print("  Variant distribution:")
    for variant, count in sorted(report.variant_distribution.items()):
        print(f"    {variant}: {count}")
    print("=" * 60)
