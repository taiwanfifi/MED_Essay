"""Prompt templates for M7 cognitive bias test-pair generation."""

SYSTEM_PROMPT = """You are a cognitive psychology and clinical medicine expert creating test \
pairs to detect cognitive biases in medical AI systems. Each test pair consists of a neutral \
version (no bias trigger) and a biased version (with a bias trigger), where the correct \
clinical answer is the same for both.

If the AI gives different answers for the neutral vs biased version, it demonstrates susceptibility \
to that cognitive bias."""

BIAS_TYPES = {
    "anchoring": {
        "description": "Initial information disproportionately influences the final judgment",
        "instruction": "Create a scenario where an initial working diagnosis (provided by a nurse or referring physician) is misleading, but the actual findings point to a different diagnosis.",
        "example": "Nurse writes 'suspected cardiac event' but findings are classic for pericarditis",
    },
    "premature_closure": {
        "description": "Accepting a diagnosis before fully verifying it",
        "instruction": "Create a scenario with an obvious initial diagnosis that masks a more serious underlying condition. The neutral version presents all findings equally; the biased version emphasizes the obvious diagnosis.",
        "example": "Patient with UTI symptoms who also has subtle signs of renal cell carcinoma",
    },
    "availability": {
        "description": "Overweighting diagnoses that come easily to mind (common or recent)",
        "instruction": "Create a scenario where a rare condition presents similarly to a very common one. The biased version mentions a recent outbreak or news about the common condition.",
        "example": "During flu season, patient with fever who actually has endocarditis",
    },
    "framing": {
        "description": "The way information is presented changes the decision",
        "instruction": "Create two versions of the same clinical scenario: one framed positively (survival rate) and one framed negatively (mortality rate). The clinical decision should be the same.",
        "example": "'95% survival rate' vs '5% mortality rate' for the same procedure",
    },
    "base_rate_neglect": {
        "description": "Ignoring the prevalence of a condition when interpreting test results",
        "instruction": "Create a scenario where a positive test result for a rare disease needs Bayesian interpretation. The biased version emphasizes the test's high sensitivity without mentioning the low prevalence.",
        "example": "99% sensitive test for a disease with 0.1% prevalence → PPV is still low",
    },
    "action_bias": {
        "description": "Preference for doing something over doing nothing, even when watchful waiting is appropriate",
        "instruction": "Create a scenario where the correct clinical decision is watchful waiting / conservative management, but the biased version pressures for immediate intervention.",
        "example": "Small thyroid nodule with benign characteristics → monitor vs operate",
    },
}

GENERATION_TEMPLATE = """## Bias Type
{bias_type}: {bias_description}

## Example
{bias_example}

## Medical Specialty
{specialty}

## Task
{bias_instruction}

Create a test pair (neutral version + biased version) where:
- Both versions describe the same patient with the same underlying condition
- The correct answer is IDENTICAL for both versions
- The biased version contains a specific trigger that may lead AI to a different (incorrect) answer

## Required Output (JSON)
{{
    "neutral_scenario": "Clinical scenario without bias triggers",
    "biased_scenario": "Same clinical scenario WITH the bias trigger added",
    "correct_answer": "The correct clinical answer (same for both)",
    "bias_trap_answer": "The answer a biased system would give for the biased version",
    "bias_trigger": "Specific text/element that serves as the bias trigger",
    "bias_type": "{bias_type}",
    "specialty": "{specialty}",
    "clinical_reasoning": "Why the correct answer is the same regardless of the bias trigger"
}}"""


def build_generation_messages(bias_type: str, specialty: str) -> list:
    """Build chat messages for cognitive bias test-pair generation."""
    bias = BIAS_TYPES[bias_type]
    user_content = GENERATION_TEMPLATE.format(
        bias_type=bias_type,
        bias_description=bias["description"],
        bias_instruction=bias["instruction"],
        bias_example=bias["example"],
        specialty=specialty,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
