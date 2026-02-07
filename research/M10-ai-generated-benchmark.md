# M10: AI 生成醫學評估基準
# AI-Generated Medical Evaluation Benchmarks: A Systematic Framework for LLM-Constructed Clinical Assessments

> **層級**：基礎設施 — 支撐 M2/M4/M5/M7 的資料集建構
> **狀態**：🟡 需開發生成 pipeline
> **Phase**：Phase 1.5（與 Phase 1 同步進行）
> **子論文**：M10a（生成方法論）、M10b（多模型驗證）、M10c（基準平台）

---

## 傘型構想概述

M10 不是單一論文，而是一個有共同基礎設施的研究領域，至少可拆分為 3 篇各有獨立貢獻的論文：

```
M10 (傘型)
├── M10a: 生成方法論  → "如何用 AI 生成醫學基準題目？"
├── M10b: 多模型驗證  → "多模型監督能替代多少人工審核？"
└── M10c: 基準平台    → "可查詢的醫學 AI 評估基準系統"
```

**資料流關係：**
```
M10a (生成方法論) ──生成──→ M2/M4/M5/M7 的資料集
         │
         ▼
M10b (多模型監督) ──驗證──→ M10a 生成的品質
         │
         ▼
M10c (Explorer 平台) ──整合──→ 所有 M1-M9 + M10a/M10b 的結果
```

---

# M10a：AI 生成方法論

## 論文標題

*"Can AI Build Its Own Exam? A Systematic Framework for LLM-Generated Medical Evaluation Benchmarks"*

---

## 研究問題 (Research Problem)

MedEval-X 框架中的 M2（EBM 等級衝突）、M4（反事實擾動）、M5（EHR 雜訊注入）、M7（認知偏誤測試對）合計需要約 3,960 個精心設計的臨床情境。這些特殊格式的題目不存在於任何公開資料集中，而純人工建構在時間和成本上均不實際（假設 1 位醫師每小時寫 3 題，需 1,320 小時）。

**核心問題：**
1. AI 生成的醫學題目品質能達到人工建構的水準嗎？
2. 什麼 prompt 策略（zero-shot / few-shot / CoT / structured output）能產出最高品質的題目？
3. 生成題目的多樣性、難度分布、醫學正確性如何量化？
4. 不同題型（反事實 / 偏誤 / 雜訊注入）的生成難度是否不同？

**為什麼 AI 生成是合理的：**
- **規模化**：數小時產出數千題
- **一致性**：相同 prompt template 確保格式統一
- **可控性**：精確控制擾動類型、偏誤類型、雜訊強度
- **可重現**：固定 seed + prompt → 完全可重現

**相關先行研究：**
- Self-Instruct (Wang et al., 2023) — 用 LLM 生成指令數據
- Evol-Instruct / WizardLM (Xu et al., 2023) — 漸進式複雜化
- ARES (Saad-Falcon et al., 2024) — AI 自動評估 RAG 系統
- DynaBoard / Dynabench (Kiela et al., 2021) — 動態 benchmark 生成
- **但尚無人系統性做「醫學領域的 AI benchmark 生成方法論」**

---

## 核心方法 (Core Approach)

### 1. 五階段生成 Pipeline

```
Stage 1: 種子題目 (Seed Questions)
  └─ 從 MedQA/MedMCQA 選取 400 題作為種子
  └─ 按科別、難度、題型分層抽樣

Stage 2: AI 生成 (Generation)
  └─ Model A (GPT-4o) 根據種子題 + prompt template 生成變體
  └─ 每個種子題生成 6 個變體（M4）或對應情境（M2/M7）
  └─ 使用 structured output 確保格式一致

Stage 3: 多模型交叉驗證 (Cross-Validation)
  └─ Model B (Claude) 獨立驗證：題目合理嗎？答案正確嗎？
  └─ Model C (DeepSeek-R1) 第三方仲裁不一致的案例
  └─ 記錄三方一致性分數

Stage 4: 人工審核 (Human Review)
  └─ 分層抽樣 20-30% 的題目
  └─ 2 位臨床醫師獨立審核
  └─ 計算 Cohen's Kappa (目標 > 0.75)
  └─ 若 Kappa 不達標 → 修正 prompt → 回到 Stage 2

Stage 5: 品質報告 (Quality Report)
  └─ 報告 AI 生成通過率、人工修改率、最終可用題數
  └─ 此報告本身就是論文的重要 contribution
```

### 2. 品質保證五層機制

| 層級 | 機制 | 目的 |
|------|------|------|
| **自動-語法** | JSON schema 驗證 + 格式檢查 | 確保結構完整 |
| **自動-語義** | SNOMED CT / DrugBank 交叉比對 | 確保醫學實體存在且正確 |
| **AI-交叉** | 多模型一致性 ≥ 2/3 | 過濾明顯錯誤 |
| **人工-抽樣** | 醫師審核 20-30% | 驗證臨床合理性 |
| **統計-校驗** | 答案分布檢查、難度分布 | 確保無系統性偏差 |

---

## 實驗設計 (Experimental Design)

### 實驗 1：Prompt 策略比較

```
4 種策略 × 4 種題型 × 50 題/策略 = 800 題
策略: Zero-shot / Few-shot (3-shot) / CoT / Structured Output (JSON schema)
題型: M2 EBM 衝突 / M4 反事實 / M5 EHR 雜訊 / M7 認知偏誤

評估維度（每題由 2 位醫師評分 1-5）：
  a) 醫學正確性 (Medical Accuracy)
  b) 臨床合理性 (Clinical Plausibility)
  c) 難度適當性 (Difficulty Appropriateness)
  d) 格式合規性 (Format Compliance)
  e) 指令遵從性 (Instruction Following)
```

**統計檢驗：**
- 二因子 ANOVA：策略 × 題型對品質分數的影響
- Tukey HSD 事後比較
- 效果量：η²

### 實驗 2：Benchmark Turing Test

```
混合 100 題人工建構 + 100 題 AI 生成（最佳策略）
3 位醫師盲審：判斷每題是「人寫」還是「AI 寫」
計算：AI 生成的"通過圖靈測試"比例
統計：Chi-square test for independence
```

### 實驗 3：Evol-Instruct for Medical Benchmarks

```
借鏡 WizardLM 的漸進複雜化策略：
  Seed question → Level 1 (加細節) → Level 2 (加併發症) → Level 3 (加矛盾)
  每層驗證品質是否維持
  最終生成高難度題目的成功率
```

**分析：**
- 每層的品質分數變化曲線
- 最高可維持品質的複雜度層級
- 與 Self-Instruct 基線比較

---

## 需要的積木 (Required Building Blocks)

### 資料集
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| MedQA 種子題（分層抽樣） | 400 題 | ✅ 來源可得 | 需抽樣 |
| MedMCQA 種子題 | 200 題 | ✅ 來源可得 | 補充多樣性 |
| DrugBank 藥物資料庫 | 16,000+ | ✅ 學術免費 | 語義驗證 |
| SNOMED CT 瀏覽器 | - | ✅ 免費 | 實體驗證 |

### 模型
| 模型 | 角色 | 狀態 |
|------|------|------|
| GPT-4o | 主要生成者 | ✅ 可用 |
| Claude 3.5 Sonnet | 交叉驗證者 | ✅ 可用 |
| DeepSeek-R1 | 仲裁者 | ✅ 可用 |
| Llama 3.1 8B | 開源基線 | ✅ 可用 |

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
data/M10a_seeds_600.json                      # 種子題目
data/M10a_generated_by_strategy.json          # 4 策略 × 4 題型生成結果
data/M10a_turing_test_mixed.json              # 圖靈測試混合題目
results/M10a_strategy_comparison.csv          # 策略比較結果
results/M10a_turing_test_results.csv          # 圖靈測試結果
results/M10a_evol_instruct_quality.csv        # 漸進複雜化品質
```

### 視覺化
```
figures/M10a_strategy_quality_heatmap.png     # 策略 × 題型品質熱力圖
figures/M10a_turing_test_confusion.png        # 圖靈測試混淆矩陣
figures/M10a_evol_quality_curve.png           # 漸進複雜化品質曲線
figures/M10a_quality_dimension_radar.png      # 5 維品質雷達圖
```

### 學術表格
- Table 1: Five-Stage Generation Pipeline Overview
- Table 2: Prompt Strategy Comparison by Question Type (ANOVA Results)
- Table 3: Benchmark Turing Test Results (Detection Rate by Physician)
- Table 4: Evol-Instruct Quality Degradation by Complexity Level
- Table 5: Final Dataset Statistics (Count, Distribution, Pass Rates)

---

## 預期發現 (Expected Findings)

1. **Structured Output 表現最佳**：JSON schema 約束下的生成品質預期最高（格式合規 > 95%）
2. **題型差異顯著**：反事實（M4）生成最容易，認知偏誤（M7）最難
3. **圖靈測試通過率高**：預期 60-70% 的 AI 生成題目被誤判為人類建構
4. **Evol-Instruct 有上限**：Level 2 以上的複雜化開始出現品質下降
5. **五階段 pipeline 可重現**：相同 prompt + seed → 高度一致的輸出

---

## 醫學特有價值

1. **解決資料稀缺**：為 M2/M4/M5/M7 提供所需的特殊格式題目
2. **首個醫學 benchmark 生成方法論**：可被其他研究團隊採用
3. **開源 prompt 模板**：按題型分類的可重用模板庫
4. **品質基準線**：為未來 AI 生成醫學內容提供量化標準

**投稿目標：** ACL/EMNLP Resource Track, Nature Scientific Data

---

# M10b：多模型交叉監督驗證

## 論文標題

*"Consensus Without Humans? Multi-Model Cross-Supervision for Validating AI-Generated Clinical Questions"*

---

## 研究問題 (Research Problem)

M10a 的 pipeline 生成了數千道題目，但品質驗證仍需大量人工審核。如果多個 LLM 的共識判斷能替代部分人工審核，將大幅降低成本並提高效率。然而，LLM 可能存在「集體幻覺」——所有模型都同意但都錯了。

**核心問題：**
1. N 個 LLM 的共識判斷，能達到多接近人類醫師的判斷品質？
2. 「集體幻覺」（所有 AI 都同意但都錯了）的發生率是多少？
3. 最佳的模型組合是什麼？（同家族 vs 異家族 vs 醫學特化）
4. 最少需要幾個模型才能達到可接受的品質門檻？

---

## 核心方法 (Core Approach)

### 1. 模型組合設計

```
5 種模型組合 × 500 題 × 全量人類審核

組合 1: GPT-4o + Claude + Gemini        (異家族 Cloud)
組合 2: GPT-4o + GPT-4o-mini + o1       (同家族 Cloud)
組合 3: Llama + Qwen + DeepSeek         (異家族 Local)
組合 4: BioMistral + Med42 + OpenBioLLM (醫學特化)
組合 5: 混合最佳                         (根據前4組結果選)
```

### 2. 驗證流程

```
For each 組合:
  1. 對 500 題 AI 生成的醫學問答進行驗證
  2. 每個模型獨立判斷：
     (a) 醫學正確？(b) 臨床合理？(c) 答案正確？
  3. 計算多數決共識
  4. 2 位人類醫師全量審核（黃金標準）
  5. 計算 vs 人類：Precision, Recall, F1
  6. 分析「集體幻覺率」= 所有模型同意但人類判錯的比率
```

### 3. N-Model Scaling 實驗

```
測試 2 模型 / 3 模型 / 4 模型 / 5 模型 的品質如何變化
繪製 N-model vs Quality 曲線
找到收益遞減的拐點 → 推薦最具成本效益的模型數
```

---

## 實驗設計 (Experimental Design)

### 實驗 1：5 種組合 vs 人類醫師

**對每組 500 題：**
- 計算 Precision / Recall / F1（以人類判斷為 ground truth）
- 計算 Cohen's Kappa（AI 共識 vs 人類）
- 按醫學子領域拆解表現差異

### 實驗 2：集體幻覺分析

**定義：所有模型一致同意且錯誤 = 集體幻覺**

**分析維度：**
- 集體幻覺率（overall 和 per-domain）
- 幻覺主題分類（哪些醫學領域最容易觸發）
- 錯誤嚴重度分布（Level 1-4, 參照 M8）
- 與單模型幻覺率的比較

### 實驗 3：N-Model Scaling Law

```
For N in [2, 3, 4, 5]:
  For each possible N-model subset:
    Compute quality metrics vs human gold standard
  Average across subsets

Plot: N vs {F1, Precision, Recall, Collective Hallucination Rate}
Identify: optimal N for cost-quality tradeoff
```

### 實驗 4：模型家族效應

**分析同家族 vs 異家族組合的差異：**
- 假設：同家族模型共享訓練偏差 → 集體幻覺率更高
- 假設：異家族組合的多樣性 → 更好的錯誤偵測

---

## 需要的積木 (Required Building Blocks)

### 資料集
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| M10a 生成的題目 | 500 題 | 🟡 需 M10a 產出 | 分層抽樣 |
| 人類醫師審核 | 500 × 2 人 | ❌ 需安排 | 黃金標準 |

### 模型
| 模型 | 組合 | 狀態 |
|------|------|------|
| GPT-4o | 1, 2, 5 | ✅ 可用 |
| GPT-4o-mini | 2 | ✅ 可用 |
| o1 | 2 | ✅ 可用 |
| Claude 3.5 Sonnet | 1, 5 | ✅ 可用 |
| Gemini Pro | 1 | ✅ 可用 |
| Llama 3.1 8B | 3 | ✅ 可用 |
| Qwen 2.5 32B | 3 | ✅ 可用 |
| DeepSeek-R1 14B | 3 | ✅ 可用 |
| BioMistral-7B | 4 | ✅ 已有 |
| Med42-v2 | 4 | ❌ 需下載 |
| OpenBioLLM | 4 | ❌ 需下載 |

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
results/M10b_combination_metrics.csv          # 5 種組合的 P/R/F1
results/M10b_collective_hallucination.json    # 集體幻覺案例
results/M10b_n_model_scaling.csv              # N-model 品質曲線
results/M10b_family_effect.csv                # 家族效應分析
```

### 視覺化
```
figures/M10b_combination_f1_barplot.png        # 5 種組合 F1 比較
figures/M10b_n_model_scaling_curve.png         # N vs Quality 曲線
figures/M10b_hallucination_by_domain.png       # 集體幻覺分布
figures/M10b_family_vs_diverse.png             # 同家族 vs 異家族
```

### 學術表格
- Table 1: Five Model Combination Configurations
- Table 2: Validation Quality Metrics by Combination (P/R/F1/Kappa)
- Table 3: Collective Hallucination Rate by Medical Domain
- Table 4: N-Model Scaling Results
- Table 5: Same-Family vs Cross-Family Comparison

---

## 預期發現 (Expected Findings)

1. **異家族 Cloud 組合最佳**：GPT-4o + Claude + Gemini 預期 F1 > 0.85
2. **集體幻覺率 < 5%**：但在藥物交互作用領域可能更高
3. **3 模型是最佳平衡點**：2 模型不足，4-5 模型收益遞減
4. **同家族幻覺率更高**：驗證多樣性假說
5. **醫學特化模型在特定領域勝出**：但整體不如大型通用模型組合

**投稿目標：** AAAI, NeurIPS, EMNLP

---

# M10c：MedEval-X Explorer 平台

## 論文標題

*"MedEval-X Explorer: A Queryable Platform for Multi-Dimensional Medical LLM Evaluation"*

---

## 研究問題 (Research Problem)

M1-M9 的研究產出了大量的題目、結果和分析數據，但目前缺乏統一的查詢和互動介面。研究者需要能按多種維度（科別、難度、模組、模型）檢索題目與結果。

**核心問題：**
1. 如何讓醫學 AI 評估基準可查詢、可互動、可擴展？
2. 能否整合 Text2SQL 讓使用者用自然語言查詢基準結果？
3. 能否成為醫學 LLM 評估的標準平台？

---

## 核心方法 (Core Approach)

### 1. 系統架構

```
MedEval-X Explorer
├── 題目瀏覽器 (Question Browser)
│   ├── 按科別篩選 (Cardiology, Pharmacology, ...)
│   ├── 按難度篩選 (Easy / Medium / Hard)
│   ├── 按題型篩選 (Diagnosis / Treatment / Mechanism)
│   ├── 按模組篩選 (M1 原始題 / M4 反事實變體 / M7 偏誤對)
│   └── 全文搜尋
│
├── 結果分析器 (Result Analyzer)
│   ├── 選擇模型 → 查看該模型在各維度的表現
│   ├── 選擇題目 → 查看各模型對該題的回答
│   ├── 錯誤模式查詢 → M3 錯誤圖譜的互動版
│   └── 校準曲線互動圖
│
└── API 存取
    ├── GET /questions?specialty=pharmacology&module=M4
    ├── GET /results?model=gpt-4o&dataset=medqa
    ├── GET /errors?type=contraindication&severity=4
    └── POST /evaluate (提交新模型結果)
```

### 2. 技術實現

- **後端：** FastAPI（已有經驗）
- **資料庫：** SQLite/PostgreSQL（題目 + 結果 + 元數據）
- **前端：** Streamlit 或 Gradio（快速原型）
- **查詢：** 整合現有 Text2SQL 系統，讓使用者用自然語言查詢基準結果

### 3. 差異化定位

| 現有標準/平台 | 定位 | MedEval-X 的差異化 |
|-------------|------|-------------------|
| **HELM** (Stanford) | 通用 LLM 評估 | MedEval-X 專注醫學，有臨床嚴重度加權 |
| **lm-eval-harness** (EleutherAI) | 技術框架 | MedEval-X 提供醫學特有指標 |
| **OpenCompass** | 中文 LLM 評估 | MedEval-X 聚焦安全性維度 |
| **MultiMedQA** (Google) | 醫學 QA 合集 | MedEval-X 加入反事實、偏誤、校準等多維度 |
| **MedBench** | 中文醫學評估 | MedEval-X 是多語言、多維度、safety-first |

---

## 預期產出 (Expected Outputs)

### 系統交付物
```
medeval-x-explorer/
├── backend/                # FastAPI 後端
├── frontend/               # Streamlit 前端
├── api/                    # RESTful API
├── text2sql/               # 自然語言查詢整合
└── docs/                   # API 文件 + 使用指南
```

### 學術產出
- 系統描述論文（JMIR / JBI）
- 線上 demo + 開源代碼
- API 文件 + 使用範例

---

## 標準化路徑

```
Phase A: 學術發表（2025-2026）
  └─ M1-M9 逐步發表 → 建立學術信譽
  └─ M10 方法論論文 → 建立資料集生成規範
  └─ 所有代碼與資料開源 → 建立社群基礎

Phase B: 工具化（2026-2027）
  └─ 發佈 medeval-x Python package
  └─ 提供 CLI: medeval-x evaluate --model gpt-4o --modules M1,M4,M6
  └─ 建立 leaderboard 網站

Phase C: 社群治理（2027+）
  └─ 向 AMIA / IMIA 提交為推薦評估框架
  └─ 與 CDISC/BRIDG 連結（已有基礎）
  └─ 建立 Working Group，邀請多機構參與
  └─ 向 FDA/TFDA 提出作為 AI 醫療器材審查參考
```

**投稿目標：** JMIR, JBI, Nature Methods（若影響力夠大）

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M10a (生成方法論) | ← 上游 | M10b 驗證 M10a 的輸出品質 |
| M10b (多模型監督) | ← 上游 | M10c 整合 M10b 的驗證結果 |
| M1-M9 (所有研究) | ← 上游 | M10c 是所有研究的整合平台 |
| Text2SQL 系統 | ↔ 技術共享 | 復用現有 NL→SQL 技術 |
| M9 (RxLLama) | → 交付物 | M10c 可作為 M9 的交付物之一 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Wang, Y., et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions. *ACL 2023*.
- Xu, C., et al. (2023). WizardLM: Empowering Large Language Models to Follow Complex Instructions. *arXiv:2304.12244*.
- Saad-Falcon, J., et al. (2024). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *NAACL 2024*.
- Kiela, D., et al. (2021). Dynabench: Rethinking Benchmarking in NLP. *NAACL 2021*.
- Liang, P., et al. (2023). Holistic Evaluation of Language Models. *Transactions on Machine Learning Research*. [HELM]
- Singhal, K., et al. (2023). Towards Expert-Level Medical Question Answering with Large Language Models. *arXiv:2305.09617*. [Med-PaLM 2]

### 內部文件
- `research/M4-counterfactual-stress-test.md` — 反事實擾動題型設計
- `research/M7-clinical-cognitive-biases.md` — 認知偏誤題型設計
- `國科會_RxLLama/關聯資料/text2sql/` — Text2SQL 技術基礎
