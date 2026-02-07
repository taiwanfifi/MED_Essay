"""Prompt templates for M5 EHR noise injection generation."""

SYSTEM_PROMPT = """You are a clinical informaticist who understands the messy reality of \
Electronic Health Records (EHR). Your task is to inject realistic noise into clean clinical \
scenarios to test whether AI systems can still extract the correct clinical information.

Types of EHR noise you should simulate:
1. Copy-paste artifacts (repeated notes from previous days)
2. Contradictory information (conflicting vital signs or lab values)
3. Verbose boilerplate (institutional templates with mostly irrelevant content)
4. Abbreviation ambiguity (non-standard or ambiguous medical abbreviations)
5. Temporal confusion (mixing past and present conditions without clear dating)"""

NOISE_TYPES = {
    "copy_paste": {
        "instruction": "Add 2-3 paragraphs of copy-pasted notes from previous days that repeat 'stable' assessments, burying the new acute finding.",
        "intensity_levels": {
            "mild": "Add 1 paragraph of previous-day note",
            "moderate": "Add 2 paragraphs spanning 2 previous days",
            "severe": "Add 3 paragraphs spanning 3 previous days with contradicting progression",
        },
    },
    "contradiction": {
        "instruction": "Introduce contradictory information in the note (e.g., vitals say normal but text says abnormal).",
        "intensity_levels": {
            "mild": "One minor inconsistency in non-critical values",
            "moderate": "One inconsistency in clinically relevant values",
            "severe": "Multiple contradictions across vital signs and lab values",
        },
    },
    "boilerplate": {
        "instruction": "Wrap the clinical scenario in institutional template boilerplate text.",
        "intensity_levels": {
            "mild": "Add standard admission template header/footer",
            "moderate": "Add full review-of-systems template with mostly negatives",
            "severe": "Add complete H&P template with extensive irrelevant normal findings",
        },
    },
    "abbreviation": {
        "instruction": "Replace standard medical terms with ambiguous or non-standard abbreviations.",
        "intensity_levels": {
            "mild": "2-3 common abbreviations",
            "moderate": "5-6 mixed common and uncommon abbreviations",
            "severe": "Extensive abbreviation use including institution-specific ones",
        },
    },
    "temporal_confusion": {
        "instruction": "Mix past medical history with present illness without clear temporal markers.",
        "intensity_levels": {
            "mild": "Slightly unclear timeline for one condition",
            "moderate": "Past condition described alongside present without clear distinction",
            "severe": "Multiple past and present conditions interleaved without dates",
        },
    },
}

GENERATION_TEMPLATE = """## Clean Clinical Scenario
{clean_scenario}

## Correct Clinical Decision
{correct_answer}

## Noise Type
{noise_type}

## Intensity Level
{intensity}: {intensity_description}

## Task
{noise_instruction}

The noisy version must preserve all clinically relevant information needed to reach the correct answer.
The noise should be realistic — something a clinician would actually encounter in an EHR.

## Required Output (JSON)
{{
    "noisy_scenario": "The full clinical scenario with realistic EHR noise injected",
    "correct_answer": "{correct_answer}",
    "noise_type": "{noise_type}",
    "intensity": "{intensity}",
    "key_signal_preserved": true,
    "noise_description": "Brief description of what noise was added and where",
    "expected_difficulty": "How much harder this makes the question (easy/medium/hard)"
}}"""


def build_generation_messages(
    clean_scenario: str,
    correct_answer: str,
    noise_type: str,
    intensity: str = "moderate",
) -> list:
    """Build chat messages for EHR noise injection."""
    ntype = NOISE_TYPES[noise_type]
    intensity_desc = ntype["intensity_levels"].get(intensity, "")

    user_content = GENERATION_TEMPLATE.format(
        clean_scenario=clean_scenario,
        correct_answer=correct_answer,
        noise_type=noise_type,
        noise_instruction=ntype["instruction"],
        intensity=intensity,
        intensity_description=intensity_desc,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
