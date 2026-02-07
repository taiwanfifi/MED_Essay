#!/usr/bin/env python3
"""
run_experiment.py — Paper Foundation Experiment
================================================
MCQ vs Open-Ended Clinical Reasoning + Verbalized Confidence Calibration

This script implements the core experiments from M1 + M6:
1. Load MedQA (USMLE 4-options) from HuggingFace
2. For each question, test in MCQ format and Open-Ended format
3. Collect verbalized confidence scores
4. Compute Option Bias and calibration metrics (ECE, SW-ECE)
5. Save results to results_foundation.json
6. Generate figures: Reliability Diagram + Option Bias Bar Chart

Usage:
    # Smoke test (20 questions)
    python run_experiment.py --smoke-test

    # Full run (all MedQA test questions)
    python run_experiment.py

    # Generate figures from existing results
    python run_experiment.py --figures-only

Requires:
    - OPENAI_API_KEY in medeval/.env (or environment)
    - pip install openai datasets matplotlib numpy scikit-learn
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for medeval imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medeval.config import OPENAI_API_KEY, DATASETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Directories ---
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# --- Safety weights for SW-ECE (from M6 theory) ---
SAFETY_WEIGHTS = {
    "Pharmacology": 3.0,
    "pharmacology": 3.0,
    "Emergency Medicine": 3.0,
    "Pediatrics": 2.5,
    "Obstetrics & Gynecology": 2.5,
    "OB/GYN": 2.5,
    "Internal Medicine": 2.0,
    "Surgery": 2.0,
    "Anatomy": 1.0,
    "Biochemistry": 1.0,
    "Physiology": 1.0,
    "Microbiology": 1.0,
    "Behavioral Science": 1.0,
    "Pathology": 1.5,
    "Psychiatry": 1.5,
}
DEFAULT_SAFETY_WEIGHT = 1.5


# =====================================================================
# Data Loading
# =====================================================================

def load_medqa_sample(n: Optional[int] = None) -> List[Dict]:
    """Load MedQA USMLE test set from HuggingFace.

    Args:
        n: Number of samples to load (None = all)

    Returns:
        List of dicts with keys: question, options, answer, answer_idx, meta_info, topic
    """
    from datasets import load_dataset

    logger.info("Loading MedQA dataset from HuggingFace...")
    ds = load_dataset(DATASETS["medqa"], split="test")
    logger.info(f"MedQA test set: {len(ds)} questions")

    samples = []
    indices = range(n) if n else range(len(ds))
    for i in indices:
        if i >= len(ds):
            break
        row = ds[i]
        # MedQA format: question, options (dict A/B/C/D), answer, answer_idx, meta_info
        question = row.get("question", "")
        options = row.get("options", {})
        answer = row.get("answer", "")
        answer_idx = row.get("answer_idx", "")
        meta_info = row.get("meta_info", "")

        # Try to extract topic from meta_info
        topic = "Unknown"
        if meta_info and isinstance(meta_info, str):
            topic = meta_info.strip()

        samples.append({
            "id": i,
            "question": question,
            "options": options if isinstance(options, dict) else {},
            "answer": answer,
            "answer_idx": answer_idx,
            "topic": topic,
        })

    logger.info(f"Loaded {len(samples)} samples")
    return samples


# =====================================================================
# Prompt Templates
# =====================================================================

MCQ_PROMPT_TEMPLATE = """Answer the following medical question by selecting the correct option (A, B, C, or D).
After your answer, state your confidence level as a percentage (0-100%).

Question: {question}

{options_text}

Format your response EXACTLY as:
Answer: [A/B/C/D]
Confidence: [X]%"""

OPENENDED_PROMPT_TEMPLATE = """Answer the following medical question. Provide your answer directly without multiple choice options.
After your answer, state your confidence level as a percentage (0-100%).

Question: {question}

Format your response EXACTLY as:
Answer: [your answer]
Confidence: [X]%"""

JUDGE_PROMPT_TEMPLATE = """You are a clinical expert judge. Compare the model's open-ended answer to the correct answer.

Question: {question}
Correct Answer: {correct_answer}
Model's Answer: {model_answer}

Judge whether the model's answer is:
(A) Clinically correct — semantically equivalent to the correct answer, clinically actionable
(B) Partially correct — right direction but imprecise, or correct concept but wrong specificity level
(C) Clinically incorrect — different from correct answer in clinically meaningful way

Respond with ONLY a JSON object:
{{"level": "A", "reasoning": "brief explanation"}}
or
{{"level": "B", "reasoning": "brief explanation"}}
or
{{"level": "C", "reasoning": "brief explanation"}}"""


def format_options(options: Dict[str, str]) -> str:
    """Format MCQ options for display."""
    lines = []
    for key in sorted(options.keys()):
        lines.append(f"{key}) {options[key]}")
    return "\n".join(lines)


# =====================================================================
# API Call Wrappers
# =====================================================================

def call_openai(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> Tuple[str, Optional[str]]:
    """Call OpenAI API with error handling and rate limiting.

    Returns:
        (response_text, error) tuple
    """
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content.strip()
        return text, None
    except openai.RateLimitError:
        logger.warning("Rate limited, waiting 30s...")
        time.sleep(30)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content.strip()
            return text, None
        except Exception as e:
            return "", f"Rate limit retry failed: {e}"
    except Exception as e:
        return "", f"OpenAI error: {e}"


# =====================================================================
# Response Parsing
# =====================================================================

def parse_mcq_response(response: str) -> Tuple[str, float]:
    """Parse MCQ response to extract answer letter and confidence.

    Returns:
        (answer_letter, confidence_pct)
    """
    answer = ""
    confidence = 50.0  # default

    # Extract answer letter
    ans_match = re.search(r"Answer:\s*\(?([A-Da-d])\)?", response)
    if ans_match:
        answer = ans_match.group(1).upper()
    else:
        # Fallback: look for standalone letter at start
        letter_match = re.search(r"^([A-Da-d])\b", response.strip())
        if letter_match:
            answer = letter_match.group(1).upper()

    # Extract confidence
    conf_match = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)\s*%?", response)
    if conf_match:
        confidence = float(conf_match.group(1))
    else:
        conf_match = re.search(r"(\d+(?:\.\d+)?)\s*%", response)
        if conf_match:
            confidence = float(conf_match.group(1))

    confidence = max(0.0, min(100.0, confidence))
    return answer, confidence


def parse_openended_response(response: str) -> Tuple[str, float]:
    """Parse open-ended response to extract answer text and confidence.

    Returns:
        (answer_text, confidence_pct)
    """
    answer = ""
    confidence = 50.0

    # Extract answer text
    ans_match = re.search(r"Answer:\s*(.+?)(?:\n|Confidence:|$)", response, re.DOTALL)
    if ans_match:
        answer = ans_match.group(1).strip()
    else:
        # Use first line as answer
        lines = response.strip().split("\n")
        answer = lines[0].strip() if lines else ""

    # Extract confidence
    conf_match = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)\s*%?", response)
    if conf_match:
        confidence = float(conf_match.group(1))
    else:
        conf_match = re.search(r"(\d+(?:\.\d+)?)\s*%", response)
        if conf_match:
            confidence = float(conf_match.group(1))

    confidence = max(0.0, min(100.0, confidence))
    return answer, confidence


def parse_judge_response(response: str) -> Tuple[str, str]:
    """Parse judge response to extract level and reasoning.

    Returns:
        (level: A/B/C, reasoning)
    """
    try:
        # Try to parse JSON
        json_match = re.search(r'\{[^{}]*"level"[^{}]*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            level = data.get("level", "C").upper()
            reasoning = data.get("reasoning", "")
            if level in ("A", "B", "C"):
                return level, reasoning
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: look for (A), (B), (C) pattern
    level_match = re.search(r"\(([ABC])\)", response)
    if level_match:
        return level_match.group(1), response

    # Fallback: look for "Level A/B/C" or just "A/B/C"
    level_match = re.search(r"(?:Level\s+)?([ABC])\b", response)
    if level_match:
        return level_match.group(1), response

    return "C", "Could not parse judge response"


# =====================================================================
# Core Experiment Loop
# =====================================================================

def run_single_question(
    sample: Dict,
    model: str = "gpt-4o",
    judge_model: str = "gpt-4o",
) -> Dict:
    """Run MCQ + Open-Ended experiment on a single question.

    Returns:
        Dict with all results for this question
    """
    question = sample["question"]
    options = sample["options"]
    correct_answer = sample["answer"]
    correct_idx = sample.get("answer_idx", "")

    result = {
        "id": sample["id"],
        "topic": sample["topic"],
        "question": question[:200],  # truncate for storage
        "correct_answer": correct_answer,
        "correct_idx": correct_idx,
    }

    # --- MCQ Format ---
    options_text = format_options(options)
    mcq_prompt = MCQ_PROMPT_TEMPLATE.format(
        question=question,
        options_text=options_text,
    )
    mcq_response, mcq_error = call_openai(mcq_prompt, model=model)
    if mcq_error:
        logger.warning(f"MCQ error for Q{sample['id']}: {mcq_error}")
        result["mcq_error"] = mcq_error
        result["mcq_correct"] = False
        result["mcq_confidence"] = 50.0
    else:
        mcq_answer, mcq_confidence = parse_mcq_response(mcq_response)
        # Check correctness: compare letter to correct_idx or answer
        mcq_correct = False
        if correct_idx and isinstance(correct_idx, str):
            mcq_correct = mcq_answer == correct_idx.upper()
        elif correct_answer:
            # Try matching answer text to option
            for key, val in options.items():
                if val.strip().lower() == correct_answer.strip().lower():
                    mcq_correct = mcq_answer == key.upper()
                    break

        result["mcq_answer"] = mcq_answer
        result["mcq_response"] = mcq_response[:300]
        result["mcq_correct"] = mcq_correct
        result["mcq_confidence"] = mcq_confidence

    # Small delay to avoid rate limits
    time.sleep(0.5)

    # --- Open-Ended Format ---
    oe_prompt = OPENENDED_PROMPT_TEMPLATE.format(question=question)
    oe_response, oe_error = call_openai(oe_prompt, model=model)
    if oe_error:
        logger.warning(f"OE error for Q{sample['id']}: {oe_error}")
        result["oe_error"] = oe_error
        result["oe_level"] = "C"
        result["oe_confidence"] = 50.0
    else:
        oe_answer, oe_confidence = parse_openended_response(oe_response)
        result["oe_answer"] = oe_answer[:200]
        result["oe_response"] = oe_response[:300]
        result["oe_confidence"] = oe_confidence

        # --- Judge the open-ended answer ---
        time.sleep(0.5)
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            model_answer=oe_answer,
        )
        judge_response, judge_error = call_openai(judge_prompt, model=judge_model)
        if judge_error:
            logger.warning(f"Judge error for Q{sample['id']}: {judge_error}")
            result["oe_level"] = "C"
            result["judge_error"] = judge_error
        else:
            level, reasoning = parse_judge_response(judge_response)
            result["oe_level"] = level
            result["judge_reasoning"] = reasoning[:200]

    return result


def run_experiment(
    samples: List[Dict],
    model: str = "gpt-4o",
    judge_model: str = "gpt-4o",
) -> List[Dict]:
    """Run the full experiment on all samples.

    Returns:
        List of result dicts
    """
    results = []
    total = len(samples)

    for i, sample in enumerate(samples):
        logger.info(f"Processing question {i+1}/{total} (ID: {sample['id']})")
        result = run_single_question(sample, model=model, judge_model=judge_model)
        results.append(result)

        # Progress report every 10 questions
        if (i + 1) % 10 == 0:
            mcq_correct = sum(1 for r in results if r.get("mcq_correct"))
            oe_a = sum(1 for r in results if r.get("oe_level") == "A")
            logger.info(
                f"  Progress: {i+1}/{total} | MCQ Acc: {mcq_correct}/{i+1} "
                f"({mcq_correct/(i+1)*100:.1f}%) | OE Level A: {oe_a}/{i+1} "
                f"({oe_a/(i+1)*100:.1f}%)"
            )

    return results


# =====================================================================
# Metrics Computation
# =====================================================================

def compute_option_bias(results: List[Dict]) -> Dict:
    """Compute Option Bias and related metrics."""
    valid = [r for r in results if "mcq_error" not in r and "oe_error" not in r]
    if not valid:
        return {"error": "No valid results"}

    n = len(valid)
    mcq_acc = sum(1 for r in valid if r["mcq_correct"]) / n
    oe_level_a = sum(1 for r in valid if r.get("oe_level") == "A") / n
    oe_level_b = sum(1 for r in valid if r.get("oe_level") == "B") / n
    oe_level_c = sum(1 for r in valid if r.get("oe_level") == "C") / n

    option_bias = mcq_acc - oe_level_a
    adjusted_oe = oe_level_a + 0.5 * oe_level_b
    adjusted_option_bias = mcq_acc - adjusted_oe
    relative_option_bias = (option_bias / mcq_acc * 100) if mcq_acc > 0 else 0

    return {
        "n_valid": n,
        "mcq_accuracy": round(mcq_acc, 4),
        "oe_level_a": round(oe_level_a, 4),
        "oe_level_b": round(oe_level_b, 4),
        "oe_level_c": round(oe_level_c, 4),
        "option_bias": round(option_bias, 4),
        "adjusted_option_bias": round(adjusted_option_bias, 4),
        "relative_option_bias_pct": round(relative_option_bias, 2),
    }


def compute_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict]]:
    """Compute Expected Calibration Error.

    Returns:
        (ece_value, list of bin dicts for plotting)
    """
    bins = []
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # include right edge in last bin
            mask = (confidences >= lo) & (confidences <= hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0, "acc": 0, "conf": 0, "gap": 0})
            continue

        acc = correctness[mask].mean()
        conf = confidences[mask].mean()
        gap = abs(acc - conf)
        ece += (n_in_bin / len(confidences)) * gap

        bins.append({
            "lo": round(float(lo), 2),
            "hi": round(float(hi), 2),
            "n": int(n_in_bin),
            "acc": round(float(acc), 4),
            "conf": round(float(conf), 4),
            "gap": round(float(gap), 4),
        })

    return round(float(ece), 4), bins


def compute_sw_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    topics: List[str],
    n_bins: int = 10,
) -> float:
    """Compute Safety-Weighted ECE."""
    weights = np.array([SAFETY_WEIGHTS.get(t, DEFAULT_SAFETY_WEIGHT) for t in topics])
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    sw_ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            continue

        bin_weights = weights[mask]
        acc = correctness[mask].mean()
        conf = confidences[mask].mean()
        gap = abs(acc - conf)
        sw_ece += (bin_weights.sum() / total_weight) * gap

    return round(float(sw_ece), 4)


def compute_brier_score(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """Compute Brier Score."""
    return round(float(np.mean((confidences - correctness) ** 2)), 4)


def compute_all_metrics(results: List[Dict]) -> Dict:
    """Compute all calibration and option bias metrics."""
    metrics = {}

    # Option Bias
    metrics["option_bias"] = compute_option_bias(results)

    # Calibration — MCQ confidence
    valid_mcq = [r for r in results if "mcq_error" not in r]
    if valid_mcq:
        mcq_conf = np.array([r["mcq_confidence"] / 100.0 for r in valid_mcq])
        mcq_correct = np.array([1.0 if r["mcq_correct"] else 0.0 for r in valid_mcq])
        topics_mcq = [r.get("topic", "Unknown") for r in valid_mcq]

        mcq_ece, mcq_bins = compute_ece(mcq_conf, mcq_correct)
        mcq_sw_ece = compute_sw_ece(mcq_conf, mcq_correct, topics_mcq)
        mcq_brier = compute_brier_score(mcq_conf, mcq_correct)

        metrics["mcq_calibration"] = {
            "ece": mcq_ece,
            "sw_ece": mcq_sw_ece,
            "brier": mcq_brier,
            "bins": mcq_bins,
            "mean_confidence": round(float(mcq_conf.mean()), 4),
            "mean_accuracy": round(float(mcq_correct.mean()), 4),
        }

    # Calibration — Open-Ended confidence
    valid_oe = [r for r in results if "oe_error" not in r]
    if valid_oe:
        oe_conf = np.array([r["oe_confidence"] / 100.0 for r in valid_oe])
        oe_correct = np.array([1.0 if r.get("oe_level") == "A" else 0.0 for r in valid_oe])
        topics_oe = [r.get("topic", "Unknown") for r in valid_oe]

        oe_ece, oe_bins = compute_ece(oe_conf, oe_correct)
        oe_sw_ece = compute_sw_ece(oe_conf, oe_correct, topics_oe)
        oe_brier = compute_brier_score(oe_conf, oe_correct)

        metrics["oe_calibration"] = {
            "ece": oe_ece,
            "sw_ece": oe_sw_ece,
            "brier": oe_brier,
            "bins": oe_bins,
            "mean_confidence": round(float(oe_conf.mean()), 4),
            "mean_accuracy": round(float(oe_correct.mean()), 4),
        }

    # Topic-level analysis
    topics_set = set(r.get("topic", "Unknown") for r in results)
    topic_metrics = {}
    for topic in sorted(topics_set):
        topic_results = [r for r in results if r.get("topic") == topic]
        if len(topic_results) < 3:
            continue
        topic_metrics[topic] = compute_option_bias(topic_results)
    metrics["topic_breakdown"] = topic_metrics

    # Overconfident-wrong cases
    overconf_wrong = []
    for r in results:
        if r.get("mcq_confidence", 0) > 80 and not r.get("mcq_correct", True):
            overconf_wrong.append({
                "id": r["id"],
                "topic": r.get("topic"),
                "confidence": r.get("mcq_confidence"),
                "question_preview": r.get("question", "")[:100],
            })
    metrics["overconfident_wrong_mcq"] = sorted(
        overconf_wrong, key=lambda x: x["confidence"], reverse=True
    )[:20]

    return metrics


# =====================================================================
# Figure Generation
# =====================================================================

def plot_reliability_diagram(bins: List[Dict], title: str, output_path: Path):
    """Generate a reliability diagram (calibration curve)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Reliability diagram
    bin_centers = []
    accs = []
    confs = []
    ns = []

    for b in bins:
        if b["n"] > 0:
            center = (b["lo"] + b["hi"]) / 2
            bin_centers.append(center)
            accs.append(b["acc"])
            confs.append(b["conf"])
            ns.append(b["n"])

    if not bin_centers:
        plt.close()
        return

    bin_centers = np.array(bin_centers)
    accs = np.array(accs)
    confs = np.array(confs)
    ns = np.array(ns)

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Bar chart showing gap
    width = 0.08
    ax1.bar(bin_centers, accs, width=width, alpha=0.7, color="#2196F3",
            edgecolor="black", linewidth=0.5, label="Accuracy")
    ax1.bar(bin_centers, confs - accs, bottom=accs, width=width, alpha=0.3,
            color="#FF5722", edgecolor="black", linewidth=0.5, label="Gap")

    ax1.set_xlabel("Confidence", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title(f"Reliability Diagram — {title}", fontsize=13)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Right: Sample distribution
    ax2.bar(bin_centers, ns, width=width, alpha=0.7, color="#4CAF50",
            edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Confidence", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Sample Distribution per Bin", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reliability diagram: {output_path}")


def plot_option_bias_bar_chart(metrics: Dict, output_path: Path):
    """Generate Option Bias bar chart comparing MCQ vs Open-Ended accuracy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ob = metrics.get("option_bias", {})
    if "error" in ob:
        plt.close()
        return

    categories = ["MCQ Accuracy", "OE Level A\n(Exact)", "OE Level B\n(Partial)", "OE Level C\n(Wrong)"]
    values = [
        ob.get("mcq_accuracy", 0),
        ob.get("oe_level_a", 0),
        ob.get("oe_level_b", 0),
        ob.get("oe_level_c", 0),
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add Option Bias annotation
    option_bias = ob.get("option_bias", 0)
    ax.annotate(
        f"Option Bias: {option_bias*100:.1f}%",
        xy=(0.5, max(values) + 0.08),
        fontsize=13, fontweight="bold", color="#D32F2F",
        ha="center",
    )

    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title("MCQ vs Open-Ended Performance (GPT-4o on MedQA)", fontsize=14)
    ax.set_ylim(0, max(values) + 0.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved option bias chart: {output_path}")


def plot_topic_option_bias(metrics: Dict, output_path: Path):
    """Generate per-topic Option Bias bar chart."""
    topic_data = metrics.get("topic_breakdown", {})
    if not topic_data:
        return

    # Filter topics with enough data
    topics = []
    biases = []
    for topic, data in sorted(topic_data.items(), key=lambda x: x[1].get("option_bias", 0), reverse=True):
        if data.get("n_valid", 0) >= 3 and "error" not in data:
            topics.append(topic[:20])  # truncate long names
            biases.append(data.get("option_bias", 0))

    if not topics:
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(topics) * 0.4)))

    colors = ["#F44336" if b > 0.2 else "#FF9800" if b > 0.1 else "#4CAF50" for b in biases]
    bars = ax.barh(topics, [b * 100 for b in biases], color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.85)

    for bar, val in zip(bars, biases):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", ha="left", va="center", fontsize=9)

    ax.set_xlabel("Option Bias (%)", fontsize=12)
    ax.set_title("Option Bias by Medical Topic", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved topic option bias chart: {output_path}")


def plot_confidence_distribution(results: List[Dict], output_path: Path):
    """Plot confidence distributions for correct vs incorrect answers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # MCQ
    mcq_correct_conf = [r["mcq_confidence"] for r in results
                        if r.get("mcq_correct") and "mcq_error" not in r]
    mcq_wrong_conf = [r["mcq_confidence"] for r in results
                      if not r.get("mcq_correct") and "mcq_error" not in r]

    if mcq_correct_conf or mcq_wrong_conf:
        ax1.hist(mcq_correct_conf, bins=20, alpha=0.6, color="#4CAF50", label="Correct", density=True)
        ax1.hist(mcq_wrong_conf, bins=20, alpha=0.6, color="#F44336", label="Incorrect", density=True)
        ax1.set_xlabel("Confidence (%)", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("MCQ: Confidence Distribution", fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Open-Ended
    oe_correct_conf = [r["oe_confidence"] for r in results
                       if r.get("oe_level") == "A" and "oe_error" not in r]
    oe_wrong_conf = [r["oe_confidence"] for r in results
                     if r.get("oe_level") in ("B", "C") and "oe_error" not in r]

    if oe_correct_conf or oe_wrong_conf:
        ax2.hist(oe_correct_conf, bins=20, alpha=0.6, color="#4CAF50", label="Level A (Correct)", density=True)
        ax2.hist(oe_wrong_conf, bins=20, alpha=0.6, color="#F44336", label="Level B/C (Wrong)", density=True)
        ax2.set_xlabel("Confidence (%)", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title("Open-Ended: Confidence Distribution", fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confidence distribution: {output_path}")


def generate_all_figures(results: List[Dict], metrics: Dict):
    """Generate all figures from results."""
    # 1. Reliability Diagram — MCQ
    mcq_cal = metrics.get("mcq_calibration", {})
    if "bins" in mcq_cal:
        plot_reliability_diagram(
            mcq_cal["bins"],
            f"MCQ (ECE={mcq_cal['ece']:.3f})",
            FIGURES_DIR / "reliability_diagram_mcq.png",
        )

    # 2. Reliability Diagram — Open-Ended
    oe_cal = metrics.get("oe_calibration", {})
    if "bins" in oe_cal:
        plot_reliability_diagram(
            oe_cal["bins"],
            f"Open-Ended (ECE={oe_cal['ece']:.3f})",
            FIGURES_DIR / "reliability_diagram_oe.png",
        )

    # 3. Option Bias Bar Chart
    plot_option_bias_bar_chart(metrics, FIGURES_DIR / "option_bias_bar_chart.png")

    # 4. Topic-level Option Bias
    plot_topic_option_bias(metrics, FIGURES_DIR / "topic_option_bias.png")

    # 5. Confidence distributions
    plot_confidence_distribution(results, FIGURES_DIR / "confidence_distribution.png")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper Foundation Experiment")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run on 20 samples only")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of samples to run (default: all)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to evaluate (default: gpt-4o)")
    parser.add_argument("--judge-model", type=str, default="gpt-4o",
                        help="Model for judging open-ended answers")
    parser.add_argument("--figures-only", action="store_true",
                        help="Generate figures from existing results_foundation.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else RESULTS_DIR / "results_foundation.json"

    # --- Figures-only mode ---
    if args.figures_only:
        if output_path.exists():
            with open(output_path) as f:
                data = json.load(f)
            results = data.get("raw_results", [])
            metrics = data.get("metrics", {})
            generate_all_figures(results, metrics)
            logger.info("Figures generated from existing results.")
        else:
            logger.error(f"Results file not found: {output_path}")
        return

    # --- Check API key ---
    if not OPENAI_API_KEY:
        logger.error(
            "OPENAI_API_KEY not found. Set it in medeval/.env or environment.\n"
            "Cannot run experiment without API access."
        )
        sys.exit(1)

    # --- Load data ---
    n_samples = 20 if args.smoke_test else args.n
    samples = load_medqa_sample(n=n_samples)

    if not samples:
        logger.error("No samples loaded. Check dataset availability.")
        sys.exit(1)

    # --- Run experiment ---
    logger.info(f"Starting experiment: {len(samples)} questions, model={args.model}")
    start_time = time.time()
    results = run_experiment(samples, model=args.model, judge_model=args.judge_model)
    elapsed = time.time() - start_time

    # --- Compute metrics ---
    metrics = compute_all_metrics(results)
    metrics["experiment_info"] = {
        "model": args.model,
        "judge_model": args.judge_model,
        "n_samples": len(samples),
        "elapsed_seconds": round(elapsed, 1),
        "smoke_test": args.smoke_test,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # --- Save results ---
    output_data = {
        "metrics": metrics,
        "raw_results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {output_path}")

    # --- Print summary ---
    ob = metrics.get("option_bias", {})
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Questions: {ob.get('n_valid', 0)}")
    print(f"MCQ Accuracy: {ob.get('mcq_accuracy', 0)*100:.1f}%")
    print(f"OE Level A (Exact): {ob.get('oe_level_a', 0)*100:.1f}%")
    print(f"OE Level B (Partial): {ob.get('oe_level_b', 0)*100:.1f}%")
    print(f"OE Level C (Wrong): {ob.get('oe_level_c', 0)*100:.1f}%")
    print(f"Option Bias: {ob.get('option_bias', 0)*100:.1f}%")
    print(f"Relative Option Bias: {ob.get('relative_option_bias_pct', 0):.1f}%")

    mcq_cal = metrics.get("mcq_calibration", {})
    oe_cal = metrics.get("oe_calibration", {})
    if mcq_cal:
        print(f"\nMCQ ECE: {mcq_cal.get('ece', 'N/A')}")
        print(f"MCQ SW-ECE: {mcq_cal.get('sw_ece', 'N/A')}")
        print(f"MCQ Brier: {mcq_cal.get('brier', 'N/A')}")
    if oe_cal:
        print(f"\nOE ECE: {oe_cal.get('ece', 'N/A')}")
        print(f"OE SW-ECE: {oe_cal.get('sw_ece', 'N/A')}")
        print(f"OE Brier: {oe_cal.get('brier', 'N/A')}")

    overconf = metrics.get("overconfident_wrong_mcq", [])
    print(f"\nOverconfident-Wrong Cases (>80% conf): {len(overconf)}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)

    # --- Generate figures ---
    generate_all_figures(results, metrics)
    logger.info("All figures generated.")


if __name__ == "__main__":
    main()
