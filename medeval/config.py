"""Unified configuration for MedEval-X."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from medeval/ directory
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# --- Paths ---
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "medeval.db"
DATA_DIR = BASE_DIR / "datasets" / "data"
RESULTS_DIR = BASE_DIR / "results"

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Model Registry ---
# 每家 Cloud 的訓練資料不同，交叉驗證時代表不同「知識來源」
# 角色: generator = 出題, validator = 審題, both = 兩者皆可
CLOUD_MODELS = {
    # --- OpenAI (trained on CommonCrawl, books, code, etc.) ---
    "gpt-4o": {"provider": "openai", "role": "both", "cost_tier": "high"},
    "gpt-4o-mini": {"provider": "openai", "role": "validator", "cost_tier": "low"},
    # --- Anthropic (Constitutional AI, RLHF, different data mix) ---
    "claude-sonnet-4-5-20250929": {"provider": "anthropic", "role": "both", "cost_tier": "medium"},
    "claude-haiku-4-5-20251001": {"provider": "anthropic", "role": "validator", "cost_tier": "low"},
    # --- Google (trained on Google's web index, distinct from OpenAI/Anthropic) ---
    "gemini-2.5-pro": {"provider": "gemini", "role": "both", "cost_tier": "medium"},
    "gemini-2.5-flash": {"provider": "gemini", "role": "validator", "cost_tier": "low"},
    # --- DeepSeek (中國語料 + 數學/推理強化, MoE architecture) ---
    "deepseek-chat": {"provider": "deepseek", "role": "both", "cost_tier": "low"},
    "deepseek-reasoner": {"provider": "deepseek", "role": "both", "cost_tier": "medium"},
}

LOCAL_MODELS = {
    # --- Ollama (各有不同訓練基礎) ---
    "deepseek-r1:14b": {"role": "both", "vram_gb": 9, "strength": "reasoning"},
    "qwen3:32b": {"role": "both", "vram_gb": 20, "strength": "multilingual"},
    "qwen3:4b": {"role": "validator", "vram_gb": 2.5, "strength": "fast"},
    "llama3.1:8b": {"role": "both", "vram_gb": 4.7, "strength": "general"},
    "phi3.5:3.8b-mini-instruct-q4_K_M": {"role": "validator", "vram_gb": 2.4, "strength": "compact"},
    "gemma3:latest": {"role": "validator", "vram_gb": 3.3, "strength": "google_open"},
}

# 預設的四模型交叉驗證組合（異家族 Cloud）
DEFAULT_CROSS_VALIDATION_MODELS = [
    "gpt-4o",           # OpenAI — CommonCrawl + 書籍
    "claude-sonnet-4-5-20250929",  # Anthropic — Constitutional AI
    "gemini-2.5-flash", # Google — Google 搜尋索引
    "deepseek-chat",    # DeepSeek — 中國語料 + 推理強化
]

# --- HuggingFace Dataset IDs ---
DATASETS = {
    "medqa": "GBaker/MedQA-USMLE-4-options",
    "medmcqa": "openlifescienceai/medmcqa",
    "mmlu_med": "cais/mmlu",
    "pubmedqa": "qiaojin/PubMedQA",
}

# MMLU medical subtasks
MMLU_MED_SUBTASKS = [
    "clinical_knowledge",
    "medical_genetics",
    "anatomy",
    "professional_medicine",
    "college_biology",
    "college_medicine",
]

# --- Generation Defaults ---
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 2048
VALIDATION_TEMPERATURE = 0.0

# --- Quality Thresholds ---
FORMAT_PASS_RATE_THRESHOLD = 0.95
AI_CONSENSUS_THRESHOLD = 2 / 3
HUMAN_KAPPA_THRESHOLD = 0.75
