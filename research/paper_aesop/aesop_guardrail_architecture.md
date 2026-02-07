# Aesop Guardrail: Condition-Aware Instruction Chaining for Mitigating Cognitive Biases in Clinical LLM Systems

## 1. Motivation

Findings from M7 (Clinical Cognitive Biases) demonstrate that LLMs exhibit six clinically significant cognitive biases: Anchoring, Premature Closure, Availability Heuristic, Framing Effect, Base Rate Neglect, and Commission Bias. In prior authorization workflows, these biases translate directly to patient safety risks — particularly **False Approval Rate** (approving contraindicated medications) and inadequate sub-population safety assessment.

M9 proposes a 5-Step Instruction Chain as a structural countermeasure. The **Aesop Guardrail** formalizes this into a deployable architecture that wraps any clinical LLM with a multi-stage verification pipeline, inspired by the pharmacist's cognitive workflow of systematic multi-check verification.

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      AESOP GUARDRAIL                            │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ Step 1   │──▶│ Step 2       │──▶│ Step 3       │            │
│  │ Patient  │   │ Contra-      │   │ Alternative  │            │
│  │ Condition│   │ indication   │   │ Generation   │            │
│  │ Survey   │   │ Check        │   │ (if needed)  │            │
│  └──────────┘   └──────────────┘   └──────────────┘            │
│       │                │                   │                    │
│       ▼                ▼                   ▼                    │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ Condition│   │ Risk         │   │ EBM-Ranked   │            │
│  │ Registry │   │ Flag Matrix  │   │ Alternatives │            │
│  └──────────┘   └──────────────┘   └──────────────┘            │
│                                                                 │
│  ┌──────────────┐   ┌──────────────────────────────┐            │
│  │ Step 4       │──▶│ Step 5                       │            │
│  │ Confidence & │   │ Safety Summary               │            │
│  │ Uncertainty  │   │ (Pharmacist-Readable Output) │            │
│  │ Declaration  │   │                              │            │
│  └──────────────┘   └──────────────────────────────┘            │
│       │                        │                                │
│       ▼                        ▼                                │
│  ┌──────────┐          ┌───────────────┐                        │
│  │ Selective │          │ Human Review  │                        │
│  │ Referral  │          │ Flag (if      │                        │
│  │ Threshold │          │ confidence    │                        │
│  │ (≥70%)   │          │ < threshold)  │                        │
│  └──────────┘          └───────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Five-Step Instruction Chain Protocol

### Step 1: Patient Condition Survey (Anti–Premature Closure)

**Debiasing Target:** M7 Premature Closure — forces exhaustive enumeration of all patient conditions before any drug evaluation begins.

```
INSTRUCTION:
"Before evaluating any medication, list ALL patient conditions:
 - Chronic diseases (with staging if applicable)
 - Current medications (with doses)
 - Known allergies (with cross-reactivity notes)
 - Pregnancy/lactation status
 - Age-specific factors (pediatric weight-based dosing / geriatric Beers criteria)
 - Organ function (renal: eGFR; hepatic: Child-Pugh score)
 - Special population flags (immunocompromised, psychiatric comorbidities)"
```

**Mechanism:** By requiring an explicit condition registry before proceeding, the model cannot "close" on a partial assessment. This mirrors the pharmacist's practice of reviewing the complete medication profile before dispensing.

### Step 2: Contraindication Check (Anti–Anchoring & Anti–Condition-Blind Reasoning)

**Debiasing Target:** M7 Anchoring — prevents the model from being anchored by the initial drug request; M4 Condition-blind reasoning — forces systematic checking against EACH condition.

```
INSTRUCTION:
"For the requested medication [DRUG], check against EACH condition
from Step 1:
 - Absolute contraindication? (cite source: DrugBank/FDA label)
 - Relative contraindication? (dose adjustment possible?)
 - Drug-drug interaction? (severity: major/moderate/minor)
 - Drug-disease interaction? (specific to listed conditions)
 - Population-specific risk? (pregnancy category, Beers criteria, KDIGO)"
```

**Mechanism:** The one-by-one cross-referencing structure prevents the model from globally assessing "safe/unsafe" without systematically confronting each risk factor. This is analogous to the pharmacist's cross-check workflow in a CPOE system.

### Step 3: Alternative Generation with EBM Ranking (Anti–Availability Heuristic)

**Debiasing Target:** M2 EBM hierarchy sensitivity — ensures alternatives are ranked by evidence quality, not by training data frequency (availability).

```
INSTRUCTION:
"If contraindicated or significant interactions found:
 1. Suggest 2-3 alternative medications for the SAME indication
 2. For each alternative, verify safety against ALL Step 1 conditions
 3. Rank alternatives by evidence level:
    - Prefer: systematic review/RCT-supported options
    - Note: guideline recommendations with GRADE ratings
    - Flag: alternatives supported only by case reports/expert opinion"
```

**Mechanism:** By explicitly requiring EBM-ranked alternatives, the model cannot default to the most "available" (frequently seen in training data) drug. This addresses M2's finding that LLMs are susceptible to narrative persuasion and availability heuristic.

### Step 4: Confidence & Uncertainty Declaration (Anti–Overconfidence)

**Debiasing Target:** M6 calibration framework — structures the model's self-assessment and triggers human review when confidence is low.

```
INSTRUCTION:
"Rate your confidence in this recommendation (0-100%):
 - List specific conditions where you are UNCERTAIN about drug safety
 - If confidence < 70%, explicitly recommend pharmacist/specialist review
 - If any absolute contraindication found, confidence should be 0% for
   the original drug regardless of other factors
 - Declare information gaps: what additional data would change your
   recommendation?"
```

**Mechanism:** Structured confidence declaration with explicit thresholds prevents the overconfidence problem identified in M6 (SW-ECE analysis). The forced declaration of uncertainty areas creates a selective prediction mechanism.

### Step 5: Safety Summary (Pharmacist-Readable Output)

**Debiasing Target:** M7 Commission Bias — by requiring a concise summary, the model must prioritize actionable safety information over excessive recommendations.

```
INSTRUCTION:
"Provide a pharmacist-readable safety summary (max 200 words):
 - DECISION: Approve / Deny / Refer to specialist
 - KEY FLAGS: List critical safety concerns (max 3)
 - ACTION ITEMS: Specific monitoring or follow-up needed
 - CONFIDENCE: Overall confidence level with justification"
```

**Mechanism:** The word limit and structured format prevent the model from padding its response with unnecessary tests and treatments (Commission Bias). The pharmacist-focused framing ensures clinical actionability.

## 4. AI Agent as Pharmacist-Thinking Simulator

### 4.1 Multi-Verification Cognitive Model

The Aesop Guardrail simulates the pharmacist's cognitive workflow through three verification layers:

```
Layer 1: INDICATION VERIFICATION
  "Is this drug appropriate for this diagnosis?"
  ↓
Layer 2: SAFETY VERIFICATION
  "Is this drug safe for THIS SPECIFIC patient?"
  (Sub-population aware: pregnancy, renal, hepatic, age, polypharmacy)
  ↓
Layer 3: OPTIMIZATION VERIFICATION
  "Is this the BEST option among safe alternatives?"
  (EBM-ranked, considering efficacy + safety + cost + adherence)
```

### 4.2 Sub-Population Safety Guard

For the 10 target sub-populations defined in M9:

| Sub-Population | Guard Mechanism | Triggered Check |
|----------------|-----------------|-----------------|
| SP1: Pregnant | FDA pregnancy category check | Teratogenicity flag |
| SP2: Pediatric (<12) | Weight-based dose calculation | Dose ceiling verification |
| SP3: Geriatric (>75) | Beers Criteria cross-reference | Polypharmacy interaction matrix |
| SP4: CKD Stage 4-5 | KDIGO renal dosing adjustment | Nephrotoxicity flag |
| SP5: Hepatic (Child-Pugh C) | Hepatic metabolism review | Dose reduction calculation |
| SP6: Polypharmacy (≥5 drugs) | Full interaction matrix | Major interaction alert |
| SP7: Allergy history | Cross-reactivity check | β-lactam class alert |
| SP8: Immunocompromised | Infection risk assessment | Live vaccine contraindication |
| SP9: Psychiatric comorbidity | Serotonin/QTc risk check | MAOi/SSRI interaction alert |
| SP10: Lactating | Lactation safety database | Infant exposure calculation |

### 4.3 Safety Score Formulation

**Sub-Population Safety Score (from M9):**

$$SS_{SP_k} = Q_{SP_k} \times (1 - \text{CRITICAL\_rate}_{SP_k})$$

Where:
- $Q_{SP_k}$ = correct recommendation rate for sub-population k
- $\text{CRITICAL\_rate}_{SP_k}$ = rate of critical safety errors (contraindicated drug approved)

**Aesop Improvement Metric:**

$$\Delta SS_{SP_k} = SS_{SP_k}^{\text{Aesop}} - SS_{SP_k}^{\text{Baseline}}$$

Expected: $\Delta SS > 0.15$ (15 percentage point improvement) for high-risk sub-populations.

## 5. Integration with HIS/EHR Systems

### 5.1 Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│                  Hospital HIS/EHR                │
│                                                  │
│  ┌─────────┐     ┌──────────────┐               │
│  │ CPOE    │────▶│ Prior Auth   │               │
│  │ System  │     │ Request      │               │
│  └─────────┘     └──────┬───────┘               │
│                         │                        │
│                         ▼                        │
│              ┌──────────────────┐                │
│              │  AESOP GUARDRAIL │                │
│              │  (API Gateway)   │                │
│              │                  │                │
│              │  Step 1-5 Chain  │                │
│              │  + Sub-Pop Guard │                │
│              └────────┬─────────┘                │
│                       │                          │
│              ┌────────┴─────────┐                │
│              ▼                  ▼                 │
│     ┌──────────────┐  ┌──────────────┐          │
│     │ Auto-Approve │  │ Pharmacist   │          │
│     │ (conf ≥ 70%) │  │ Review Queue │          │
│     └──────────────┘  │ (conf < 70%) │          │
│                       └──────────────┘          │
└─────────────────────────────────────────────────┘
```

### 5.2 FHIR Compatibility

The Aesop Guardrail input/output can be mapped to FHIR resources:
- **Input:** `MedicationRequest`, `Patient`, `Condition`, `AllergyIntolerance`
- **Output:** `ClinicalImpression` with `finding` (safety flags) and `prognosisCodeableConcept` (recommendation)

## 6. Theoretical Contribution

The Aesop Guardrail advances the field in three ways:

1. **Structured Debiasing Without Fine-Tuning:** Unlike model retraining approaches, Aesop operates at the prompt level, making it model-agnostic and deployable with any LLM (commercial or open-source).

2. **Sub-Population Safety as First-Class Metric:** Rather than reporting aggregate accuracy, Aesop enforces per-population safety verification, directly addressing health equity concerns in AI deployment.

3. **Pharmacist Cognitive Model as AI Architecture:** By grounding the instruction chain in established pharmacist verification workflows, Aesop bridges the gap between AI engineering and clinical practice, facilitating regulatory acceptance.
