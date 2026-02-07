# Theory Framework: Beyond Multiple Choice — Calibration-Aware Evaluation of LLM Clinical Reasoning

## 1. Research Hypotheses

### Primary Hypothesis (H1): MCQ Format Overestimates Clinical Competence
**Statement:** Large Language Models achieve significantly higher accuracy on medical benchmark questions presented in multiple-choice format compared to open-ended format, indicating that MCQ-based evaluations systematically overestimate true clinical reasoning ability.

**Formalization:**
- Let Acc_MCQ = accuracy under standard MCQ format
- Let Acc_OE = accuracy under open-ended format (Level A clinically correct)
- H1: Acc_MCQ - Acc_OE > 0 (Option Bias > 0) across all tested models
- Expected effect size: 10-30% accuracy drop (Cohen's h > 0.20)

**Clinical Implication:** Claims such as "GPT-4 passes the USMLE" must be re-examined, as the MCQ format provides an artificial scaffolding (candidate answer set) that does not exist in real clinical practice.

### Secondary Hypothesis (H2): LLMs Exhibit Systematic Overconfidence on Incorrect Answers
**Statement:** When LLMs produce incorrect clinical answers, their self-reported confidence scores are significantly higher than warranted by their actual accuracy, particularly in safety-critical domains such as pharmacology.

**Formalization:**
- Let ECE = Expected Calibration Error
- H2a: ECE > 0.10 for all models (indicating poor calibration)
- H2b: ECE(pharmacology) > ECE(basic_sciences) (domain-specific miscalibration)
- H2c: P(confidence > 80% | incorrect) is non-trivially large (dangerous overconfidence)

**Clinical Implication:** An overconfident AI in a Clinical Decision Support System (CDSS) generates high-confidence incorrect recommendations. Clinicians experiencing time pressure or cognitive overload may trust these recommendations, leading to medical errors. Furthermore, if the system generates many high-confidence alerts that are actually wrong, clinicians develop "alert fatigue" — systematically ignoring AI recommendations, which defeats the purpose of the CDSS.

### Tertiary Hypothesis (H3): Multi-Model Cross-Supervision Can Identify Unreliable Outputs
**Statement:** Agreement among diverse LLM architectures serves as a reliable proxy for answer correctness, and disagreement signals can be used for selective prediction (abstaining from unreliable responses).

**Formalization:**
- Let Agreement(q) = proportion of models giving the same answer for question q
- H3: AUROC(Agreement, Correctness) > 0.70 (model agreement discriminates correct from incorrect)
- The ensemble-based confidence provides better calibration than individual model self-assessment

---

## 2. Theoretical Framing: The Trust Gap in Clinical AI

### 2.1 The MCQ Illusion
Current medical AI benchmarks (MedQA, MedMCQA, MMLU-Med) employ multiple-choice formats that provide the model with a closed answer space. This creates what we term the **"MCQ Illusion"** — an inflated perception of clinical competence arising from three mechanisms:

1. **Elimination Strategy:** Models can use pattern matching to eliminate implausible options, reducing the effective problem space
2. **Answer Anchoring:** Candidate answers prime relevant knowledge, facilitating retrieval that would not occur unprompted
3. **Format Familiarity:** Training data contains extensive MCQ-format content, creating distribution advantage

In real clinical practice, a physician faces: "What is the diagnosis?" — not "Which of these four options is the diagnosis?" The gap between these two tasks represents a fundamental validity threat to current evaluation paradigms.

### 2.2 The Overconfidence-Safety Paradox
Building on Yang et al.'s work on drug safety and diagnostic recommendations in CDSS, we identify a critical paradox:

**Paradox Statement:** The same property that makes LLMs appear useful (high confidence, fluent explanations) is precisely what makes them dangerous when wrong.

A poorly calibrated LLM in a CDSS context creates two failure modes:

| Failure Mode | Mechanism | Clinical Consequence |
|---|---|---|
| **Type I: Confident Error** | High confidence (>80%) on wrong answer | Clinician trusts AI → wrong treatment → patient harm |
| **Type II: Alert Fatigue** | Many confident-but-wrong alerts accumulate | Clinician learns to ignore AI → misses valid alerts → patient harm |

Both failure modes converge on the same outcome: patient safety compromise. Critically, **Type II is iatrogenic to the AI system itself** — the system's overconfidence destroys its own utility over time.

### 2.3 Safety-Weighted Calibration: A Novel Framework
Standard Expected Calibration Error (ECE) treats all miscalibration equally. However, in clinical contexts:

- A miscalibrated answer about embryology (basic science) has minimal direct patient impact
- A miscalibrated answer about drug dosing (pharmacology) can be lethal

We propose **Safety-Weighted ECE (SW-ECE)** that incorporates clinical severity:

$$SW-ECE = \sum_{b=1}^{B} \frac{\sum_{i \in b} w_i}{\sum_{i} w_i} |acc(b) - conf(b)|$$

Where safety weights w_i are assigned by medical subdomain based on potential for direct patient harm:

| Domain | Weight | Rationale |
|---|---|---|
| Pharmacology | 3.0 | Medication errors can be fatal |
| Emergency Medicine | 3.0 | Delayed treatment can be fatal |
| Pediatrics | 2.5 | Dosing errors in children are critical |
| OB/GYN | 2.5 | Pregnancy drug safety |
| Internal Medicine | 2.0 | Chronic disease management |
| Surgery | 2.0 | Surgical decision impact |
| Other Clinical | 1.5 | Default moderate weight |
| Basic Sciences | 1.0 | Indirect clinical impact |

This framework directly connects to Prof. Yang's research focus on drug safety: SW-ECE amplifies the signal from pharmacology miscalibration, which is exactly where overconfidence has the highest stakes.

---

## 3. Clinical Significance

### 3.1 Direct Impact on CDSS Deployment Decisions
This research provides quantitative tools for answering the critical deployment question: **"Can we trust this AI model for clinical use?"**

Current decision-making relies on MCQ accuracy alone:
- "GPT-4 scores 90% on USMLE" → Deploy?

Our framework adds three dimensions:
1. **Open-ended accuracy** (Option Bias) → How much of that 90% is real clinical reasoning vs. pattern matching?
2. **Calibration quality** (SW-ECE) → When it says "90% confident," is it actually right 90% of the time?
3. **Selective prediction coverage** → At 95% accuracy threshold, what proportion of clinical queries can it handle autonomously?

### 3.2 Connection to Drug Safety (楊教授研究專長)
Prof. Yang's expertise in pharmacovigilance and CDSS directly motivates this work:

- **Drug-drug interaction alerts:** If an AI system confidently predicts "no interaction" when one exists, the consequences can be fatal
- **Dosing recommendations:** Overconfident incorrect dosing in pediatrics or renal-adjusted patients represents the highest-severity failure mode
- **Diagnostic recommendations:** Prof. Yang's work on diagnostic CDSS highlights that confidence calibration is a prerequisite for safe AI-assisted diagnosis

Our SW-ECE framework operationalizes this domain expertise into a quantitative metric that can be incorporated into CDSS validation protocols.

### 3.3 Regulatory Implications
As regulatory bodies (FDA, TFDA) develop frameworks for AI/ML-based medical devices:
- MCQ accuracy alone is insufficient for demonstrating clinical safety
- Calibration metrics (especially safety-weighted variants) should be part of pre-market evaluation
- Selective prediction capability (the ability to "know what you don't know") is a key safety property

---

## 4. Discussion Framework (Draft)

### 4.1 Why MCQ Accuracy Is Necessary But Not Sufficient
We argue that MCQ benchmarks serve as a useful screening tool (necessary condition) but fail as deployment-readiness indicators (not sufficient). The Option Bias metric quantifies this gap. Models with high Option Bias demonstrate format-dependent performance — strong pattern matching within closed answer spaces but weak generative clinical reasoning.

### 4.2 The Calibration Imperative for Clinical AI
A key insight from our framework: **a well-calibrated 80%-accurate model is safer than a poorly-calibrated 90%-accurate model.**

The well-calibrated model:
- Correctly identifies its uncertain cases → routes to human review
- Maintains trust with clinicians → avoids alert fatigue
- Enables hybrid workflows → AI handles confident cases, humans handle uncertain ones

The poorly-calibrated model:
- Generates confident errors → clinicians make wrong decisions
- Erodes trust over time → clinicians ignore all AI suggestions
- Cannot support effective triage → unreliable confidence renders selective prediction useless

### 4.3 Multi-Model Supervision as Safety Net
Drawing from M11 (Multi-Model Cross-Supervision), we discuss how ensemble disagreement provides an orthogonal safety signal:
- When multiple diverse architectures disagree, the probability of error is higher
- Disagreement-based abstention outperforms single-model confidence thresholding
- This has practical implications: hospital systems could run 2-3 models simultaneously and flag disagreements for human review

### 4.4 Pharmacology: The Highest-Stakes Domain
We predict (and aim to demonstrate) that pharmacology questions exhibit the worst calibration among medical subdomains. This is because:
1. Drug names require exact recall (no partial credit)
2. Drug interactions are combinatorial (exponential knowledge space)
3. Training data may contain outdated prescribing information
4. Models confuse similarly named drugs (e.g., hydroxyzine vs hydralazine)

This finding, if confirmed, directly supports integrating calibration checks into drug-related CDSS queries as a mandatory safety feature.

### 4.5 Toward Trustworthy Clinical AI: A Screening Framework
We propose a three-tier screening framework for evaluating AI readiness for clinical deployment:

**Tier 1: Competence Screening** (MCQ + Open-Ended accuracy)
- Does the model know enough medicine? (traditional benchmarks)
- How much of its performance depends on format? (Option Bias)

**Tier 2: Self-Awareness Screening** (Calibration + Selective Prediction)
- Does the model know what it doesn't know? (ECE, SW-ECE)
- Can it safely abstain from uncertain cases? (Coverage@95%)

**Tier 3: Robustness Screening** (Cross-Model + Adversarial)
- Do diverse models agree? (ensemble concordance)
- Does it fail gracefully under stress? (adversarial probes)

Only models passing all three tiers should proceed to clinical pilot testing.

---

## 5. Key References Supporting This Framework

1. **Guo et al. (2017)** - On Calibration of Modern Neural Networks. Established that modern deep learning models are poorly calibrated, with calibration worsening as models get more accurate.

2. **Kadavath et al. (2022)** - Language Models (Mostly) Know What They Know. Showed that LLMs have some metacognitive ability but significant gaps in self-assessment.

3. **Nori et al. (2023)** - Demonstrated GPT-4's strong performance on medical benchmarks but noted limitations in real-world clinical reasoning.

4. **Singhal et al. (2023)** - Med-PaLM achieved state-of-the-art on medical QA but highlighted the gap between benchmark performance and clinical utility.

5. **Tian et al. (2023)** - Just Ask for Calibration: Demonstrated that prompting strategies can improve verbalized confidence calibration.

6. **Jin et al. (2021)** - MedQA: Established the primary benchmark dataset used in this study.

7. **Yang et al. (Prof. Yang's work)** - Research on drug safety, diagnostic recommendations, and CDSS validation at Taipei Medical University, providing the clinical motivation for safety-weighted calibration metrics.

---

## 6. Innovation Claims

1. **First systematic quantification of Option Bias across medical LLMs** using paired MCQ/open-ended evaluation with SNOMED CT semantic matching
2. **Safety-Weighted ECE (SW-ECE):** Novel calibration metric incorporating clinical severity weights by medical subdomain
3. **Three-tier clinical AI screening framework** integrating competence, self-awareness, and robustness assessment
4. **Empirical demonstration that pharmacology is the highest-risk domain** for LLM overconfidence in clinical settings
5. **Practical selective prediction guidelines** for CDSS deployment: coverage thresholds at clinically-acceptable accuracy levels
