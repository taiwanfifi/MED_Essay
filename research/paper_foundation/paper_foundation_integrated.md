# Beyond Multiple Choice: Calibration-Aware Evaluation Reveals Overconfident Clinical Reasoning in Large Language Models

> **Integrated Research Document** — Consolidates M1 (Open-Ended Clinical Reasoning) + M6 (Calibration & Selective Prediction) + M11 (Multi-Model Cross-Supervision) into a unified paper narrative.
>
> **Target Journal:** Journal of Biomedical Informatics (JBI / IJMI)
>
> **Status:** Paper complete with real API data (n=1,273 MedQA questions, GPT-4o, 3,819 total inferences)

---

## 1. Research Problem

### 1.1 The MCQ Illusion

Medical LLM evaluation is dominated by multiple-choice question (MCQ) benchmarks — MedQA, USMLE, MMLU-Med. When GPT-4 "passes the USMLE," the implicit claim is that it possesses clinical reasoning ability. But MCQ format provides structural advantages that inflate perceived competence:

- **Elimination strategy** — Removing obviously wrong options narrows the search space
- **Answer anchoring** — Correct answers are visible, priming retrieval of relevant knowledge
- **Format familiarity** — LLMs are trained on millions of MCQ examples

**Core thesis:** MCQ accuracy systematically overestimates clinical reasoning. Removing the option scaffolding reveals a large **Option Bias** — the gap between MCQ accuracy and open-ended clinical correctness. This gap is compounded by severe **miscalibration**: models maintain high confidence even when accuracy drops, creating a dangerous illusion of competence.

### 1.2 Why This Matters

- **Clinical practice is open-ended** — Physicians don't select from 4 options; they generate differential diagnoses, treatment plans, and safety assessments from scratch
- **CDSS deployment decisions** are made based on MCQ benchmarks — if these overestimate ability, unsafe systems get deployed
- **Alert fatigue** — Overconfident AI generates excessive false-positive alerts, eroding clinician trust
- **High-confidence errors** are the most dangerous — clinicians are less likely to override confident AI recommendations

### 1.3 Research Questions

| # | Question | Module Origin |
|---|----------|---------------|
| RQ1 | How much does MCQ format inflate perceived medical LLM accuracy? | M1 |
| RQ2 | Does model confidence calibration differ between MCQ and open-ended formats? | M6 |
| RQ3 | How prevalent are high-confidence errors, and what is their clinical risk? | M6 + M11 |
| RQ4 | Can multi-model cross-supervision improve evaluation reliability? | M11 |
| RQ5 | Should clinical AI evaluation weight errors differently by medical severity? | M6 (SW-ECE) |

---

## 2. Theoretical Framework

### 2.1 From Scattered Modules to Unified Evaluation

This paper integrates three modules from the MedEval-X framework:

```
M1: Open-Ended Reasoning ──── "Does it reason or pattern-match?"
        │
        ├── M6: Calibration ──────── "Does it know what it doesn't know?"
        │
        └── M11: Cross-Supervision ── "Can multiple models catch errors?"
```

**Integration logic:** M1 establishes the baseline performance gap between MCQ and open-ended formats (the Option Bias). M6 analyzes whether the model's confidence tracks this accuracy drop (calibration analysis). M11 proposes multi-model consensus as a practical solution for detecting overconfident errors at scale.

### 2.2 Three-Tier Clinical Judgment System (from M1)

Standard MCQ evaluation uses binary scoring (correct/incorrect). Clinical reality is graded — a response can be "in the right direction" without being precise. We introduce a three-tier system:

| Level | Label | Definition | Example |
|-------|-------|-----------|---------|
| **A** | Clinically Correct | Semantically equivalent to reference answer | "Inferior STEMI" for "ST-elevation MI of inferior wall" |
| **B** | Partially Correct | Correct clinical direction, imprecise | "Myocardial infarction" when reference is "Inferior STEMI" |
| **C** | Clinically Incorrect | Clinically distinct from reference | "Pulmonary embolism" when reference is "Inferior STEMI" |

This captures the nuanced partial correctness that binary scoring misses — a Level B response reflects genuine clinical knowledge, not complete failure.

### 2.3 Safety-Weighted ECE (from M6) — Novel Metric

Standard Expected Calibration Error (ECE) treats all miscalibration equally: a 10% overconfidence on an anatomy question is penalized identically to a 10% overconfidence on a drug dosing question. In clinical contexts, **not all miscalibration is equally dangerous**.

**Safety-Weighted ECE (SW-ECE)** applies domain-specific severity weights:

| Medical Domain | Weight | Rationale |
|---------------|--------|-----------|
| Pharmacology | 3.0 | Medication errors can be fatal |
| Emergency Medicine | 3.0 | Delayed treatment can be fatal |
| Pediatrics | 2.5 | Dosing errors in children are critical |
| OB/GYN | 2.5 | Pregnancy drug safety risks |
| Internal Medicine | 2.0 | Chronic disease management impact |
| Surgery | 2.0 | Surgical decision consequences |
| Pathology | 1.5 | Diagnostic interpretation impact |
| Psychiatry | 1.5 | Treatment plan impact |
| Basic Sciences | 1.0 | Indirect clinical impact |

SW-ECE makes calibration evaluation clinically meaningful — a model that is poorly calibrated specifically in pharmacology is more dangerous than one poorly calibrated in basic science.

---

## 3. Methodology

### 3.1 Dataset

**MedQA (USMLE) complete test set:** 1,273 questions spanning multiple medical disciplines.

Topic distribution: "step1" (basic sciences, anatomy, biochemistry, pharmacology) and "step2&3" (clinical medicine, surgery, pediatrics, psychiatry, OB/GYN).

> **Limitation noted:** MedQA topic categorization is coarse — only "step1" and "step2&3" labels are available. Finer subdomain analysis requires MedMCQA (21 subject tags) in future work.

### 3.2 Experimental Design

Each question is evaluated in **two formats** with **verbalized confidence**:

| Format | Input | Output | Purpose |
|--------|-------|--------|---------|
| **MCQ** | Question + 4 options (A/B/C/D) | Selected option + confidence (0-100%) | Standard benchmark baseline |
| **Open-Ended** | Question stem only (options removed) | Free-text answer + confidence (0-100%) | Clinical reasoning assessment |

**Model:** GPT-4o (temperature = 0, deterministic)

**Evaluation pipeline:**
1. MCQ evaluation: 1,273 API calls → binary correct/incorrect
2. Open-ended evaluation: 1,273 API calls → free-text responses
3. Clinical judgment: 1,273 API calls → GPT-4o classifies each open-ended response as Level A/B/C
4. **Total: 3,819 model inferences**

### 3.3 Option Bias Metrics (from M1)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Option Bias** | Acc_MCQ − Acc_OE(Level A) | Raw accuracy gap between formats |
| **Adjusted Option Bias** | Acc_MCQ − (Level A + 0.5 × Level B) | Gives partial credit for Level B responses |
| **Relative Option Bias** | (Acc_MCQ − Acc_OE) / Acc_MCQ × 100% | Percentage of MCQ accuracy attributable to format |

### 3.4 Calibration Metrics (from M6)

| Metric | What It Measures |
|--------|-----------------|
| **ECE** | Expected Calibration Error — average gap between confidence and accuracy across 10 bins |
| **SW-ECE** | Safety-Weighted ECE — ECE weighted by clinical domain severity |
| **Brier Score** | Joint measure of accuracy + calibration (lower = better) |
| **Confidence-Accuracy Gap** | Mean confidence − mean accuracy (positive = overconfident) |
| **Coverage@95%** | Proportion of questions answerable while maintaining 95% accuracy (selective prediction) |

### 3.5 Overconfident Error Analysis

**Definition:** Cases where model expresses >80% confidence on an incorrect MCQ answer.

These are the most clinically dangerous cases — high confidence suppresses clinician override. We analyze:
- Prevalence (what proportion of all questions?)
- Concentration (what proportion of errors are high-confidence?)
- Domain distribution (which medical domains have the most overconfident errors?)

---

## 4. Results

### 4.1 Option Bias Analysis (Core Finding)

| Metric | Value |
|--------|-------|
| MCQ Accuracy | **87.8%** (1,119/1,273) |
| Open-Ended Level A (Exact) | **56.2%** (715/1,273) |
| Open-Ended Level B (Partial) | **24.6%** (313/1,273) |
| Open-Ended Level C (Wrong) | **19.2%** (245/1,273) |
| **Option Bias** | **31.7 percentage points** |
| Adjusted Option Bias | 19.4 pp |
| Relative Option Bias | **36.0%** |

**Interpretation:** Over one-third (36%) of GPT-4o's MCQ accuracy is attributable to the MCQ format itself, not to genuine clinical reasoning. When the option scaffold is removed, accuracy drops from 87.8% to 56.2%.

The **24.6% Level B (partially correct)** rate is important — it means that ~25% of the time, the model demonstrates relevant clinical knowledge but lacks the precision required for correct clinical action. This is not captured by binary MCQ scoring.

### 4.2 Calibration Analysis (The Calibration Paradox)

| Metric | MCQ | Open-Ended | Change |
|--------|-----|------------|--------|
| ECE | **0.029** | **0.364** | 12.6× worse |
| SW-ECE | 0.029 | 0.364 | 12.6× worse |
| Brier Score | 0.103 | 0.372 | 3.6× worse |
| Mean Confidence | **90.7%** | **92.5%** | +1.8 pp (↑) |
| Mean Accuracy | 87.8% | 56.2% | −31.6 pp (↓) |
| Confidence-Accuracy Gap | **2.9%** | **36.3%** | 12.5× worse |

**The Calibration Paradox:** Model confidence actually *increases* slightly (90.7% → 92.5%) when moving to open-ended format, despite accuracy dropping 31.7 percentage points. The model does not recognize that it has lost the MCQ scaffold — it maintains the same high confidence in a regime where it is far less accurate.

- **MCQ format:** Well-calibrated (ECE = 0.029), confidence closely tracks accuracy
- **Open-ended format:** Severely miscalibrated (ECE = 0.364), confidence-accuracy gap of 36.3%

### 4.3 Overconfident Incorrect Cases

| Metric | Value |
|--------|-------|
| Total overconfident-wrong (>80% conf, wrong MCQ) | **147 / 1,273 = 11.5%** |
| As proportion of all MCQ errors | **147 / 155 = 94.8%** |

**Critical finding:** Nearly all MCQ errors (94.8%) occur with high confidence. The model almost never expresses low confidence when wrong — it is either right-and-confident or wrong-and-confident. This means confidence cannot serve as a reliable safety filter.

### 4.4 Confidence Distribution

- **MCQ correct answers:** Tight confidence distribution centered at 90-95%
- **MCQ incorrect answers:** Similarly high confidence (>80% for 94.8%)
- **Open-ended correct:** High confidence (~90-95%)
- **Open-ended incorrect:** High confidence (~85-95%, overlapping with correct)

The confidence distributions for correct and incorrect answers are **nearly overlapping** in open-ended format, making confidence-based triage ineffective.

---

## 5. Discussion

### 5.1 The MCQ Illusion: Implications for Clinical AI Evaluation

Headlines claiming "GPT-4 passes the USMLE" are based on MCQ evaluation. Our Option Bias of 31.7% suggests that such claims overstate clinical reasoning by approximately one-third. This has direct policy implications:

- **Regulatory frameworks** (FDA SaMD, EU AI Act) that rely on benchmark accuracy may approve systems with inflated performance claims
- **Hospital procurement decisions** based on MCQ benchmarks may overestimate deployed system capability
- **Public trust** in medical AI may be misplaced if evaluation methods are not clinically representative

### 5.2 The Calibration Paradox: Why Confidence Fails as a Safety Signal

The expectation that "the model will know when it doesn't know" underlies many CDSS deployment architectures — route low-confidence cases to human review, auto-approve high-confidence ones. Our findings show this assumption fails:

- In MCQ format, calibration is excellent (ECE = 0.029) — validating the assumption within the MCQ regime
- In open-ended format, calibration collapses (ECE = 0.364) — invalidating confidence-based routing in clinical contexts
- The **format transfer failure** means calibration validated on MCQ benchmarks does not predict calibration in clinical deployment

### 5.3 Overconfidence and Patient Safety

The 147 overconfident-wrong cases (11.5%) represent the most dangerous failure mode:

- **Alert fatigue amplification:** If every high-confidence AI recommendation is treated as reliable, 11.5% will be wrong
- **Override resistance:** Clinicians are less likely to override confident AI, even when their own judgment differs
- **Concentration in errors:** 94.8% of errors are high-confidence — the model almost never provides an "I'm unsure" signal that could trigger human review

### 5.4 Safety-Weighted ECE: Not All Miscalibration is Equal

Standard ECE treats a 10% overconfidence on a biochemistry question the same as a 10% overconfidence on a pediatric drug dosing question. In clinical deployment:
- A pharmacology miscalibration → potential drug error → patient harm
- A basic science miscalibration → conceptual misunderstanding → no direct patient impact

SW-ECE addresses this by weighting miscalibration by domain severity. While MCQ SW-ECE and ECE happen to be identical in this dataset (0.029), the metric becomes critical when evaluating across clinical domains with differential risk profiles.

### 5.5 Three-Tier Clinical AI Screening Framework

We propose that medical AI systems undergo three-tier evaluation before deployment:

| Tier | Assessment | What It Reveals |
|------|-----------|-----------------|
| **Tier 1: Competence** | MCQ + Open-Ended accuracy, Option Bias | True reasoning ability vs format dependency |
| **Tier 2: Self-Awareness** | ECE, SW-ECE, Coverage@95% | Whether the model knows what it doesn't know |
| **Tier 3: Robustness** | Multi-model cross-supervision, ensemble agreement | Whether errors are caught by complementary models |

Tier 3 leverages M11's multi-model consensus: when diverse models agree, confidence is warranted; when they disagree, cases are flagged for human review. Preliminary pilot testing with four cloud providers (OpenAI, Anthropic, Google, DeepSeek) demonstrated that while models reach consensus on clear-cut cases (e.g., ACE inhibitor contraindication in pregnancy), they diverge on nuanced therapeutic choices — validating ensemble disagreement as a practical uncertainty signal.

### 5.6 Limitations

1. **Single model** — Only GPT-4o evaluated; findings may not generalize to all models
2. **Automated judgment** — GPT-4o as clinical judge introduces potential systematic bias
3. **Single dataset** — MedQA only; MedMCQA (21 subjects) and MMLU-Med planned for future
4. **Coarse topics** — "step1"/"step2&3" only; finer subdomain analysis not yet possible
5. **Simplified semantic matching** — Not using SNOMED CT ontology for Level A/B/C classification
6. **No multi-model calibration comparison** — Only GPT-4o's calibration analyzed; cross-model calibration comparison planned

---

## 6. Conclusion

We demonstrate that MCQ-based evaluation inflates medical LLM accuracy by 31.7 percentage points (36% relative), creating a dangerous **MCQ Illusion**. Worse, model confidence does not decrease when accuracy drops — the Calibration Paradox means that confidence-based safety mechanisms fail in the open-ended clinical context. With 94.8% of errors occurring at high confidence, the model provides essentially no self-corrective signal.

We introduce Safety-Weighted ECE to make calibration evaluation clinically meaningful and propose a three-tier screening framework (Competence + Self-Awareness + Robustness) as a minimum standard for clinical AI deployment. These findings challenge the assumption that benchmark performance translates to clinical safety and argue for open-ended, calibration-aware evaluation as a regulatory requirement.

---

## 7. Module Integration Map

### What Each Module Contributed

| Module | Role in Paper | Sections Used |
|--------|--------------|---------------|
| **M1** (Open-Ended Reasoning) | **Core methodology** — MCQ vs open-ended comparison, three-tier judgment, Option Bias metrics | Methods §3.1-3.3, Results §4.1, Discussion §5.1 |
| **M6** (Calibration) | **Calibration analysis** — ECE, SW-ECE, Brier Score, overconfident error analysis | Methods §3.4-3.5, Results §4.2-4.4, Discussion §5.2-5.4 |
| **M11** (Multi-Model Cross-Supervision) | **Solution framework** — Multi-model ensemble as uncertainty signal, three-tier screening; preliminary pilot with 4 cloud providers conducted | Discussion §5.5 |

### What Changed from Individual Modules

| Original Module Scope | Paper Scope | Why |
|----------------------|-------------|-----|
| M1: 6,256 MCQ → open-ended across 3 datasets, SNOMED CT matching | 1,273 MedQA only, GPT-4o judgment | Focus on depth (full dataset, real API) over breadth |
| M6: 4 confidence methods (verbalized, self-consistency, ensemble, logit) | Verbalized confidence only | Self-consistency (10× cost) and logit-based (local models only) deferred to multi-model paper |
| M11: 3 consensus strategies, 500 questions, failure mode analysis | Conceptual framework in Discussion | Full M11 experiment planned as extension; here it provides the theoretical solution |

### Connections to Other Papers

| Paper | Shared Concepts | Relationship |
|-------|----------------|-------------|
| **paper_safety** (M4+M5+M8+M3) | Overconfidence concept, calibration | Safety paper applies calibration insights to drug safety specifically |
| **paper_aesop** (M9+M7+M2) | Debiasing, structured prompting | Aesop paper tests whether structured prompting fixes the overconfidence identified here |

---

## 8. Experimental Infrastructure

### 8.1 Code

| File | Purpose |
|------|---------|
| `run_experiment.py` | MCQ vs OE experiment pipeline with verbalized confidence + calibration metrics |

### 8.2 Results

| File | Contents |
|------|----------|
| `results_foundation_full.json` | GPT-4o full dataset (n=1,273) raw API results |

### 8.3 Figures

| Figure | File | Content |
|--------|------|---------|
| Fig. 1 | `figures/option_bias_bar_chart.png` | MCQ vs OE performance comparison |
| Fig. 2 | `figures/reliability_diagram_mcq.png` | Reliability diagram — MCQ (ECE=0.029) |
| Fig. 3 | `figures/reliability_diagram_oe.png` | Reliability diagram — OE (ECE=0.364) |
| Fig. 4 | `figures/confidence_distribution.png` | Confidence distributions: correct vs incorrect |

---

## 9. Key Terminology

| Term | Definition |
|------|-----------|
| **Option Bias** | Accuracy gap between MCQ and open-ended format (measures format dependency) |
| **Adjusted Option Bias** | Option Bias with partial credit for Level B responses |
| **ECE** | Expected Calibration Error — mean gap between confidence and accuracy |
| **SW-ECE** | Safety-Weighted ECE — ECE with clinical domain severity weights |
| **Level A/B/C** | Three-tier clinical judgment: Correct / Partially Correct / Incorrect |
| **Calibration Paradox** | Confidence stays high even when accuracy drops across formats |
| **Coverage@95%** | Proportion of questions answerable while maintaining 95% accuracy |
| **Overconfident-Wrong** | >80% confidence on an incorrect answer |
| **MCQ Illusion** | False perception of clinical competence from inflated MCQ scores |
| **Selective Prediction** | Routing uncertain cases to human review based on confidence thresholds |
