# MedEval-X: A Multi-Dimensional Safety Evaluation Framework for Medical Large Language Models

> **Project Overview Document** — Maps how the original 11 research modules (M1–M11) were consolidated into 3 papers, and how the papers interconnect as a unified research programme.

---

## Project Identity

| Field | Value |
|-------|-------|
| **Research Group** | Prof. Yang's Lab, Taipei Medical University |
| **Domain** | Pharmacovigilance, Drug Safety, Clinical Decision Support |
| **Core Thesis** | The "Memorization–Safety Gap" — frontier LLMs pass medical exams but fail when patient safety requires deviation from memorized patterns |
| **Framework Name** | MedEval-X |
| **Total Papers** | 3 (all completed with real API data) |

---

## The Three Papers

### Paper 1: Foundation — "The MCQ Illusion"

**Title:** Beyond Multiple Choice: Calibration-Aware Evaluation Reveals Overconfident Clinical Reasoning in Large Language Models

| Field | Value |
|-------|-------|
| **Modules** | M1 + M6 + M11 |
| **Target** | JBI / IJMI |
| **Data** | n=1,273 MedQA, GPT-4o, 3,819 API calls |
| **Core Finding** | Option Bias = 31.7% — MCQ inflates accuracy by one-third |
| **Key Metric** | Safety-Weighted ECE (SW-ECE) |
| **Location** | `paper_foundation/` |

**Research question:** Do MCQ benchmarks accurately measure clinical reasoning, or does the format itself inflate performance?

**What it proves:** LLMs achieve 87.8% on MCQs but only 56.2% in open-ended format. Confidence stays at ~91% regardless — the model doesn't know it doesn't know (Calibration Paradox). 94.8% of errors are high-confidence.

---

### Paper 2: Safety — "The Blind Spot Paper"

**Title:** When AI Fails Drug Safety: Counterfactual Stress Testing Reveals Critical Blind Spots in Medical Large Language Models

| Field | Value |
|-------|-------|
| **Modules** | M4 + M5 + M8 + M3 |
| **Target** | npj Digital Medicine / JAMA Network Open |
| **Data** | 4 models × 20 scenarios × 2 = 160 API calls |
| **Core Finding** | Pediatric SCC = 0.65 — systemic blind spot |
| **Key Metric** | Safety-Critical Consistency (SCC) |
| **Location** | `paper_safety/` |

**Research question:** Do LLMs maintain drug safety accuracy when patient conditions change (pregnancy, CKD, pediatric age, polypharmacy)?

**What it proves:** All models score 100% on standard scenarios but drop 10–20% under safety perturbation. Pediatric drug safety is a critical blind spot (SCC = 0.65); fluoroquinolone contraindication fails in 75% of models. DDI detection is perfect (1.00).

---

### Paper 3: Aesop — "The Fix"

**Title:** Aesop Guardrail: Condition-Aware Instruction Chaining for Mitigating Cognitive Biases in Clinical LLM Prior Authorization Systems

| Field | Value |
|-------|-------|
| **Modules** | M9 + M7 + M2 |
| **Target** | JAMIA / Lancet Digital Health |
| **Data** | 190 scenarios × 8 models × 2 conditions = 3,040 API calls |
| **Core Finding** | False Approval Rate reduced 27.3% → 8.6% (−18.7 pp) |
| **Key Metric** | Sub-Population Safety Score (SS) |
| **Location** | `paper_aesop/` |

**Research question:** Can a model-agnostic, prompt-level intervention fix the safety failures identified in Papers 1 and 2?

**What it proves:** The 5-step CAIC protocol reduces false approval of contraindicated drugs by 18.7 pp. Smaller models (7-14B) benefit 2-3× more than large models. Sub-population safety improves across all 10 vulnerable groups.

---

## The Narrative Arc: Problem → Diagnosis → Treatment

```
Paper 1 (Foundation)          Paper 2 (Safety)           Paper 3 (Aesop)
"The MCQ Illusion"            "The Blind Spot"            "The Fix"
─────────────────            ─────────────────           ─────────────────
ESTABLISHES                   DEMONSTRATES                SOLVES
the evaluation gap            the clinical danger          the problem

MCQ inflates accuracy         Safety failures are          Structured prompting
by 31.7%. Calibration         category-specific:           reduces false approvals
collapses in open-ended       pediatric = critical         by 18.7 pp. Works
format. 94.8% of errors       blind spot. All models       across all models.
are high-confidence.           fail similarly.              Smaller models benefit
                                                           most.
        │                           │                           │
        └──────── "LLMs don't ──────┘──── "Here's how to ──────┘
                   know what              make them safer"
                   they don't
                   know"
```

---

## Module-to-Paper Mapping

| Module | Description | Paper | Role |
|--------|------------|-------|------|
| **M1** | Open-Ended Clinical Reasoning | Foundation | Core methodology (MCQ vs OE comparison) |
| **M2** | EBM Hierarchy Sensitivity | Aesop | EBM ranking in Step 3 alternatives |
| **M3** | Clinical Error Atlas | Safety | Error taxonomy for failure classification |
| **M4** | Counterfactual Stress Test | Safety | Core methodology (perturbation framework) |
| **M5** | EHR Noise Robustness | Safety | Noise layer + Confidence-Noise Paradox |
| **M6** | Calibration & Selective Prediction | Foundation | Calibration analysis (ECE, SW-ECE) |
| **M7** | Clinical Cognitive Biases | Aesop | Bias identification + anti-bias mapping |
| **M8** | Patient Safety Risk Matrix | Safety | Severity classification (WHO/NCC MERP) |
| **M9** | RxLLama Upgrade Framework | Aesop | 5-step CAIC protocol + SS metric |
| **M10** | AI-Generated Benchmarks | *(Future)* | Data generation infrastructure |
| **M11** | Multi-Model Cross-Supervision | Foundation | Multi-model consensus framework |

### Modules Not Yet Fully Published

| Module | Status | Planned Use |
|--------|--------|-------------|
| **M10a** (Generation Methodology) | Design complete | Separate methods paper on AI-generated medical benchmarks |
| **M10b** (Cross-Validation) | Overlaps M11 | Merged with M11 for consensus quality paper |
| **M10c** (Explorer Platform) | Conceptual | MedEval-X public evaluation platform |

---

## Cross-Paper Data Flow

```
Paper 1 (Foundation)                Paper 2 (Safety)                Paper 3 (Aesop)
────────────────────                ────────────────                ────────────────
MedQA n=1,273                      20 attack scenarios              190 PA scenarios
GPT-4o only                        4 frontier models                8 models (cloud+local)

Option Bias = 31.7%    ──────►     "Models memorize,
ECE_OE = 0.364         ──────►      don't reason about     ──────► "Aesop fixes the
94.8% errors are                    safety conditions"               reasoning gap"
high-confidence        ──────►     Pediatric SCC = 0.65   ──────►  FAR: 27.3% → 8.6%
                                   MemGap = 10-20%                  ΔSS = +0.153

SW-ECE (novel metric)              SCC (novel metric)               SS (novel metric)
3-tier judgment                    3-level perturbation              5-step CAIC protocol
Calibration Paradox                Memorization-Safety Gap           Smaller-Models-Benefit-More
```

### Shared Concepts Across Papers

| Concept | Paper 1 | Paper 2 | Paper 3 |
|---------|---------|---------|---------|
| **Overconfidence** | ECE = 0.364 in OE format | High-confidence safety failures | Step 4 recalibrates confidence |
| **Condition-blindness** | Format-dependent (MCQ scaffold) | Condition-dependent (pregnancy, pediatric) | Step 1-2 force condition awareness |
| **Sub-population vulnerability** | Not yet analyzed by subdomain | Pediatric = critical blind spot | 10 sub-populations evaluated |
| **Deployment threshold** | Coverage@95% | SCC ≥ 0.80 per category | SS ≥ 0.70 per sub-population |
| **Regulatory implications** | Three-tier screening framework | FDA SaMD, EU AI Act mapping | HIS/EHR integration design |

---

## Novel Contributions Summary

| Contribution | Paper | Impact |
|-------------|-------|--------|
| **Option Bias** quantification | Foundation | First systematic measurement of MCQ format inflation in medical LLMs |
| **Safety-Weighted ECE (SW-ECE)** | Foundation | First clinically-weighted calibration metric |
| **Three-tier clinical judgment (A/B/C)** | Foundation | Captures partial correctness missed by binary scoring |
| **Calibration Paradox** | Foundation | Confidence increases when accuracy drops across formats |
| **Memorization–Safety Gap** | Safety | Performance gap between memorized and safety-perturbed scenarios |
| **Safety-Critical Consistency (SCC)** | Safety | Category-stratified safety evaluation metric |
| **Conditional Inversion testing** | Safety | Systematic method for testing condition-aware reasoning |
| **Pediatric blind spot** identification | Safety | First empirical evidence of systematic pediatric safety failure |
| **Condition-Aware Instruction Chaining (CAIC)** | Aesop | 5-step debiasing protocol without model modification |
| **Sub-Population Safety Score (SS)** | Aesop | Multiplicative metric penalizing critical failures |
| **"Smaller Models Benefit More"** | Aesop | Prompt-level guardrails as equity infrastructure |
| **Anti-bias mapping** | Aesop | Each protocol step targets specific cognitive bias |

---

## Experimental Scale Summary

| Paper | Models | Scenarios | Total API Calls | Data Source |
|-------|--------|-----------|-----------------|-------------|
| Foundation | 1 (GPT-4o) | 1,273 × 2 formats | 3,819 | MedQA USMLE |
| Safety | 4 (GPT-4o, Claude, Gemini, DeepSeek) | 20 × 2 versions | 160 | Custom attack matrix |
| Aesop | 8 (5 cloud + 3 local) | 190 × 2 conditions | 3,040 | Custom PA scenarios |
| **Total** | **8 unique models** | — | **~7,019** | — |

---

## Where Each Original Module Idea Went

For researchers returning to the original M1-M11 documents, this table shows what was used, what was deferred, and what changed:

| Module | Original Scope | What Made It Into Papers | What Was Deferred |
|--------|---------------|-------------------------|-------------------|
| M1 | 6,256 questions, 3 datasets, SNOMED CT matching | 1,273 MedQA, GPT-4o judge | Multi-dataset, SNOMED CT |
| M2 | 6 bias tests × 30 scenarios × 4 debiasing conditions | EBM ranking as Step 3 design principle | Full EBM sensitivity experiment |
| M3 | 18,000 errors, 15 types × 10 specialties × 5 stages | Pattern analysis of 10 failure cases | Full 3D error atlas |
| M4 | 400 seeds × 6 perturbations = 2,800 variants | 20 Level-2 scenarios (conditional inversion) | Level 1/3 perturbations |
| M5 | 200 × 5 noise × 3 severity = 3,200 variants | Framework described, not fully evaluated | Noise × perturbation interaction |
| M6 | 4 confidence methods, Coverage@95% | Verbalized confidence only | Self-consistency, logit-based, ensemble |
| M7 | 6 biases × 30 scenarios × 4 conditions | 5 biases mapped to protocol steps | Full cognitive bias profiling experiment |
| M8 | Full risk matrix + regulatory gap analysis | Severity classification of observed failures | Collective hallucination analysis |
| M9 | 8-dimensional scorecard, PA system upgrade | 5-step CAIC + SS metric | Full 8D scorecard |
| M10 | AI benchmark generation + explorer platform | — | Entire module deferred |
| M11 | 3 consensus strategies, failure mode analysis | Conceptual framework in Discussion | Full M11 experiment |

---

## File Locations

```
research/
├── M1-open-ended-clinical-reasoning.md      ← Original module ideas
├── M2-ebm-hierarchy-sensitivity.md          ←
├── M3-clinical-error-atlas.md               ←
├── M4-counterfactual-stress-test.md         ←
├── M5-ehr-noise-robustness.md               ←
├── M6-calibration-selective-prediction.md   ←
├── M7-clinical-cognitive-biases.md          ←
├── M8-patient-safety-risk-matrix.md         ←
├── M9-rxllama-upgrade-framework.md          ←
├── M10-ai-generated-benchmark.md            ←
├── M11-multi-model-cross-supervision.md     ←
│
├── MedEval-X_integrated_overview.md         ← THIS FILE (project overview)
│
├── paper_foundation/
│   ├── paper_foundation_integrated.md       ← Integrated .md (M1+M6+M11)
│   ├── main.tex + main.pdf                  ← LaTeX paper
│   ├── run_experiment.py                    ← Experiment code
│   ├── results_foundation_full.json         ← Raw results (n=1,273)
│   └── figures/                             ← 5 figures
│
├── paper_safety/
│   ├── paper_safety_integrated.md           ← Integrated .md (M4+M5+M8+M3)
│   ├── adversarial_design.md                ← Attack hypothesis design
│   ├── aesop_guardrail_architecture.md      ← Guardrail spec (→ moved to paper_aesop)
│   ├── main.tex + main.pdf                  ← LaTeX paper
│   ├── src/run_real_stress_test.py          ← Real API pipeline
│   ├── results/                             ← JSON + CSV results
│   └── figures/                             ← 3 real figures
│
└── paper_aesop/
    ├── paper_aesop_integrated.md            ← Integrated .md (M9+M7+M2)
    ├── main.tex + main.pdf                  ← LaTeX paper
    ├── src/run_optimization.py              ← A/B test pipeline
    ├── results/                             ← JSON + CSV results
    └── figures/                             ← 6 figures
```
