"""Prompt templates for M2 EBM hierarchy conflict scenario generation."""

SYSTEM_PROMPT = """You are a clinical epidemiology expert creating test scenarios that pit \
different levels of medical evidence against each other. Your goal is to create situations where \
a compelling but lower-quality evidence source conflicts with a less dramatic but higher-quality source.

Evidence hierarchy (strongest to weakest):
1. Systematic Reviews / Meta-analyses
2. Randomized Controlled Trials (RCTs)
3. Cohort Studies
4. Case-Control Studies
5. Case Series / Case Reports
6. Expert Opinion / Anecdotal Evidence"""

BIAS_TYPES = {
    "authority": {
        "description": "A renowned expert's opinion contradicts meta-analysis results",
        "instruction": "Create a scenario where a famous specialist publicly advocates for treatment B based on personal experience, while a Cochrane meta-analysis supports treatment A.",
    },
    "recency": {
        "description": "A recent small study contradicts established systematic review",
        "instruction": "Create a scenario where a newly published single RCT (n=50) contradicts a 2-year-old meta-analysis (15 RCTs, n=12,000).",
    },
    "narrative": {
        "description": "A vivid patient story contradicts statistical evidence",
        "instruction": "Create a scenario with a dramatic patient recovery story (case report) that contradicts RCT evidence on treatment efficacy.",
    },
    "sample_neglect": {
        "description": "A high-percentage small study vs low-percentage large study",
        "instruction": "Create a scenario where a small study (n=20, 90% success) contradicts a large RCT (n=5,000, 65% success) for the same treatment.",
    },
    "confirmation": {
        "description": "Evidence that confirms a common belief vs evidence that refutes it",
        "instruction": "Create a scenario where a widely-held clinical belief is supported by case series but refuted by a well-designed RCT.",
    },
    "guideline_anchor": {
        "description": "Outdated guideline recommendation vs new evidence",
        "instruction": "Create a scenario where a current clinical guideline recommends treatment A, but a new high-quality RCT shows treatment B is superior.",
    },
}

GENERATION_TEMPLATE = """## Medical Topic
{topic}

## Bias Type
{bias_type}: {bias_description}

## Task
{bias_instruction}

## Required Output (JSON)
{{
    "scenario": "A detailed clinical scenario (3-5 sentences) presenting the evidence conflict",
    "evidence_high": {{
        "type": "meta-analysis or RCT or cohort",
        "description": "Description of the higher-quality evidence",
        "conclusion": "What this evidence supports"
    }},
    "evidence_low": {{
        "type": "case report or expert opinion or small study",
        "description": "Description of the lower-quality but more compelling evidence",
        "conclusion": "What this evidence supports"
    }},
    "question": "Based on the available evidence, what treatment should be recommended?",
    "correct_answer": "The answer based on evidence hierarchy (higher-quality evidence)",
    "bias_trap_answer": "The answer a biased reasoner would choose (lower-quality evidence)",
    "clinical_reasoning": "Why the higher-quality evidence should prevail",
    "bias_type": "{bias_type}",
    "topic": "{topic}"
}}"""


def build_generation_messages(topic: str, bias_type: str) -> list:
    """Build chat messages for EBM conflict scenario generation."""
    bias = BIAS_TYPES[bias_type]
    user_content = GENERATION_TEMPLATE.format(
        topic=topic,
        bias_type=bias_type,
        bias_description=bias["description"],
        bias_instruction=bias["instruction"],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
