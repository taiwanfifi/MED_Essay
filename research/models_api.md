# MedEval-X 模型清單

> API keys 存放在 `medeval/.env`（已加入 .gitignore，不會上傳）

---

## Pilot Test 結果 (2026-02-05)

**測試題目**：高血壓藥物選擇 + 懷孕條件
- 原始答案：A (Lisinopril) — ACE inhibitor
- 擾動：加入「懷孕第一孕期」
- 預期：ACE inhibitor 絕對禁忌，必須換藥

| 提供者 | 模型 | 速度 | 換藥？ | 推薦替代藥 | 嚴重度 |
|--------|------|------|--------|-----------|--------|
| OpenAI | gpt-4o | 9.8s | ✓ Yes | **C) Amlodipine** | critical |
| Anthropic | claude-sonnet-4.5 | 9.9s | ✓ Yes | **B) Metoprolol** | critical |
| Google | gemini-2.5-flash | 24.9s | ✓ Yes | **C) Amlodipine** | critical |
| DeepSeek | deepseek-chat | 9.7s | ✓ Yes | **B) Metoprolol** | critical |

**結論**：
- ✓ **四家一致同意**：Lisinopril 懷孕禁忌 → 必須換藥
- △ **替代藥分歧**：OpenAI+Google 選 Amlodipine，Anthropic+DeepSeek 選 Metoprolol
- → 這種「禁忌一致、替代分歧」正是需要醫師校正的案例

完整結果：`medeval/results/pilot_test_4_cloud.json`

---

## Cloud Models（四家不同訓練資料 → 交叉驗證核心）

不同家族的模型用不同的訓練語料，等於「擁有不同知識背景的專家」。
四家同時判斷一道題，共識 ≥ 3/4 才通過，分歧大的才請醫師校正。

### OpenAI

| 模型 | Model ID | 角色 | 成本 | 備註 |
|------|----------|------|------|------|
| GPT-4o | `gpt-4o` | 出題 + 審題 | $2.5/$10 per MTok | 主力生成模型 |
| GPT-4o mini | `gpt-4o-mini` | 審題 | $0.15/$0.6 per MTok | 便宜的批量驗證 |

```python
from medeval.generation.models import OpenAIModel
model = OpenAIModel("gpt-4o")
```

### Anthropic (Claude)

| 模型 | Model ID | 角色 | 成本 | 備註 |
|------|----------|------|------|------|
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | 出題 + 審題 | $3/$15 per MTok | 主力驗證模型 |
| Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | 審題 | 低 | 快速篩選 |
| Claude Opus 4.5 | `claude-opus-4-5-20251101` | 出題（困難題） | $5/$25 per MTok | 最強但最貴 |

```python
from medeval.generation.models import AnthropicModel
model = AnthropicModel("claude-sonnet-4-5-20250929")
```

### Google Gemini

| 模型 | Model ID | 角色 | 成本 | 備註 |
|------|----------|------|------|------|
| Gemini 2.5 Pro | `gemini-2.5-pro` | 出題 + 審題 | 付費 | Google 搜尋資料訓練 |
| Gemini 2.5 Flash | `gemini-2.5-flash` | 審題 | 較低 | 快速驗證 |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` | 出題 | 付費 | 最新版（preview） |

```python
from medeval.generation.models import GeminiModel
model = GeminiModel("gemini-2.5-flash")
```

### DeepSeek

| 模型 | Model ID | 角色 | 成本 | 備註 |
|------|----------|------|------|------|
| DeepSeek Chat | `deepseek-chat` | 出題 + 審題 | 低 | 中國語料 + MoE，便宜 |
| DeepSeek Reasoner | `deepseek-reasoner` | 出題 | 中 | 推理能力強 |

```python
from medeval.generation.models import DeepSeekModel
model = DeepSeekModel("deepseek-chat")
```

---

## Local Models（Ollama — 額外知識來源 + 零成本）

本機已安裝的模型。各自的訓練資料也不同（DeepSeek 用中國語料、Qwen 用阿里巴巴語料、Llama 用 Meta 語料），進一步增加多樣性。

| 模型 | Ollama ID | 大小 | 強項 | 角色 |
|------|-----------|------|------|------|
| **DeepSeek-R1 14B** | `deepseek-r1:14b` | 9 GB | 推理能力強，CoT | 出題 + 審題 |
| **Qwen3 32B** | `qwen3:32b` | 20 GB | 多語言，中文強 | 出題 + 審題 |
| **Qwen3 4B** | `qwen3:4b` | 2.5 GB | 快速 | 快速審題 |
| **Llama 3.1 8B** | `llama3.1:8b` | 4.7 GB | 通用基線 | 出題 + 審題 |
| **Phi 3.5 3.8B** | `phi3.5:3.8b-mini-instruct-q4_K_M` | 2.4 GB | 小型但聰明 | 審題 |
| **Gemma 3** | `gemma3:latest` | 3.3 GB | Google 開源版 | 審題 |

```python
from medeval.generation.models import OllamaModel
model = OllamaModel("deepseek-r1:14b")
```

---

## 預設交叉驗證組合

### 組合 A：四家 Cloud（預設，最可靠）

```
GPT-4o (OpenAI) + Claude Sonnet (Anthropic) + Gemini Flash (Google) + DeepSeek Chat
```
- 四家訓練資料完全不同
- 共識 ≥ 3/4 → 自動通過
- 分歧太大 → 標記「需人工」→ 請醫師校正

### 組合 B：三家 Cloud（省成本）

```
GPT-4o + Claude Sonnet + DeepSeek Chat
```
- 省 Gemini API 費用
- 三家就夠做交叉驗證

### 組合 C：全 Local（零成本）

```
DeepSeek-R1 14B (Ollama) + Qwen3 32B (Ollama) + Llama 3.1 8B (Ollama)
```
- 不花 API 費用
- 品質可能不如 Cloud，但可做初步篩選
- 只有分歧項再送 Cloud 驗證 → 省錢

### 組合 D：速度優先

```
GPT-4o-mini + Claude Haiku + Gemini Flash + DeepSeek Chat
```
- 全用便宜/快速模型
- 適合 pilot test 或初步篩選

---

## 使用範例

### 一鍵啟動四模型交叉驗證

```python
from medeval.generation.models import OpenAIModel, AnthropicModel, GeminiModel, DeepSeekModel
from medeval.generation.validator import MultiModelValidator

validator = MultiModelValidator([
    OpenAIModel("gpt-4o"),
    AnthropicModel("claude-sonnet-4-5-20250929"),
    GeminiModel("gemini-2.5-flash"),
    DeepSeekModel("deepseek-chat"),
])

# 驗證一批生成的題目
results = validator.validate_batch(items, orig_questions, orig_answers)
```

### 用 CLI 指定模型

```bash
# 用 GPT-4o 生成，四家 Cloud 驗證
python -m medeval.scripts.generate_benchmark \
    --module m4 --count 10 --pilot \
    --generator gpt-4o \
    --validators "claude-sonnet-4-5-20250929,gemini-2.5-flash,deepseek-chat" \
    --validate

# 用本機 DeepSeek 生成（免費），Cloud 驗證
python -m medeval.scripts.generate_benchmark \
    --module m4 --count 10 \
    --generator deepseek-r1:14b \
    --validators "gpt-4o,claude-sonnet-4-5-20250929,deepseek-chat" \
    --validate
```

---

## 安裝依賴

```bash
pip install openai anthropic google-genai requests python-dotenv datasets
```

## 為什麼多家模型交叉很重要？

```
OpenAI    → 訓練於 CommonCrawl + 書籍 + 程式碼
Anthropic → Constitutional AI + 不同的 RLHF 資料
Google    → Google 搜尋索引 + 學術文獻
DeepSeek  → 中國語料 + 數學/推理強化 (MoE)
Qwen      → 阿里巴巴語料 + 多語言
Llama     → Meta 的開放語料

每家「讀的書不同」→ 知道的東西不同 → 互相檢查更可靠
→ Pilot test 證實：四家都知道 ACE-I 懷孕禁忌（共識）
   但對替代藥有不同意見（分歧）→ 這正是醫師要校正的
```


