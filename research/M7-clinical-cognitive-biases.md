# M7: 臨床認知偏誤
# Clinical Cognitive Biases in LLMs: Do AI Systems Inherit the Diagnostic Pitfalls of Human Clinicians?

> **層級**：Layer 4 — 行為分析
> **財經對應**：I2 (Behavioral Finance Biases in LLMs)
> **狀態**：🟡 Partially Ready — 需建構 180 個臨床情境
> **Phase**：Phase 2（核心貢獻）

---

## 研究問題 (Research Problem)

Croskerry (2002) 系統性地記錄了急診醫師在臨床推理中常見的 30+ 種認知偏誤。Kahneman (2011) 的 System 1/2 理論提供了解釋框架：快速直覺（System 1）在時間壓力下容易產生偏誤，而緩慢分析（System 2）可以修正但需要認知努力。

**LLM 是否表現出類似的臨床認知偏誤？** 這個問題既有理論意義（LLM 的推理是否結構性地類似人類 System 1？），也有實踐意義（如果 LLM 有錨定偏誤，醫師使用 AI 建議時可能被進一步錨定）。

**與 M2 的區別：**
- M2 聚焦 **EBM 證據等級偏誤**（能否區分高/低品質證據）
- M7 聚焦 **臨床推理過程中的認知偏誤**（推理本身是否有系統性偏差）
- M2 是 epistemological（關於知識品質的判斷），M7 是 cognitive（關於推理過程的偏差）

**具體未知：**
1. 在 6 種臨床認知偏誤中，LLM 最容易受哪些影響？
2. LLM 的偏誤輪廓與人類醫師（文獻報告）是否相似？
3. Chain-of-Thought 是放大還是衰減偏誤？
4. 針對性的 debiasing prompt 能否有效降低偏誤？

---

## 核心方法 (Core Approach)

### 1. 六種臨床認知偏誤 (Six Clinical Cognitive Biases)

#### Bias 1: Anchoring（錨定效應）

**定義：** 過度依賴最先接收到的資訊（initial impression），即使後續資訊指向不同方向。

**臨床情境設計：**
```
Anchoring condition:
"A 55-year-old male is brought to the ED. The triage nurse documents
'likely cardiac event' based on initial complaint of chest pain.

On your examination, you find: chest pain is pleuritic (worse with
breathing), fever 38.3°C, friction rub on auscultation, diffuse
ST-elevation on ECG (saddle-shaped), recent URI one week ago.

What is the most likely diagnosis?"

Non-anchored control:
"A 55-year-old male presents to the ED with chest pain that is pleuritic,
fever 38.3°C, friction rub on auscultation, diffuse ST-elevation on ECG
(saddle-shaped), recent URI one week ago.

What is the most likely diagnosis?"
```

**正確答案：** Acute pericarditis（不是 MI）
**Bias 測試：** 「triage nurse documents likely cardiac event」是否錨定模型走向 MI？

#### Bias 2: Premature Closure（過早關閉）

**定義：** 一旦找到一個「夠好」的解釋就停止搜索，忽略未被解釋的異常發現。

**臨床情境設計：**
```
Premature closure condition:
"A 45-year-old female presents with fatigue, weight gain, and cold intolerance.
TSH is 12 mIU/L (elevated), confirming hypothyroidism.

However, you also note: persistent hypercalcemia (Ca 11.8 mg/dL),
elevated PTH (95 pg/mL), and a neck mass on palpation.

What is your complete assessment?"

Control:
"A 45-year-old female presents with fatigue, weight gain, cold intolerance,
persistent hypercalcemia (Ca 11.8 mg/dL), elevated PTH (95 pg/mL),
elevated TSH (12 mIU/L), and a neck mass on palpation.

What is your complete assessment?"
```

**正確答案：** Hypothyroidism AND primary hyperparathyroidism（可能 MEN1）
**Bias 測試：** 確認甲狀腺低下後，是否忽略高鈣血症和副甲狀腺過高？

#### Bias 3: Availability Heuristic（可得性啟發）

**定義：** 最近看到或最容易想到的診斷影響判斷，而非基於實際可能性。

**臨床情境設計：**
```
Availability condition:
"During a COVID-19 surge, you evaluate a 30-year-old female with fever,
cough, and myalgias for 3 days. Your last 15 patients today were all
COVID-positive.

Additional findings: petechial rash on lower extremities, platelet count
45,000/μL, hemoglobin 9.5 g/dL, elevated LDH.

What is the most likely diagnosis?"

Control (no priming):
"A 30-year-old female presents with fever, cough, myalgias for 3 days,
petechial rash on lower extremities, platelet count 45,000/μL,
hemoglobin 9.5 g/dL, elevated LDH.

What is the most likely diagnosis?"
```

**正確答案：** TTP/HUS 或其他血液疾病（不是 COVID-19）
**Bias 測試：** COVID surge 的 priming 是否讓模型忽略血液學發現而直接歸因為 COVID？

#### Bias 4: Framing Effect（框架效應）

**定義：** 相同資訊的不同呈現方式影響決策。

**臨床情境設計：**
```
Positive frame:
"A new chemotherapy regimen has a 70% five-year survival rate for Stage III
colon cancer. The patient asks for your recommendation."

Negative frame:
"A new chemotherapy regimen has a 30% five-year mortality rate for Stage III
colon cancer. The patient asks for your recommendation."

Question: "Would you recommend this treatment? Explain your reasoning."
```

**正確答案：** 建議應相同（70% survival = 30% mortality，是同一資訊）
**Bias 測試：** 正面 vs 負面框架是否改變模型的推薦語氣和建議？

#### Bias 5: Base Rate Neglect（基礎率忽略）

**定義：** 忽略疾病的先驗概率（prevalence），過度受檢驗結果影響。

**臨床情境設計：**
```
"A 22-year-old healthy female college student with no risk factors presents
for routine screening. She has no symptoms, no family history, no travel
history, and no exposures.

A screening test for Disease X (prevalence 0.1% in this population) comes
back positive. The test has sensitivity 95% and specificity 95%.

What is the probability that she actually has Disease X?
Should you start treatment based on this result?"
```

**正確答案：** PPV ≈ 1.9%（用 Bayes' theorem），不應僅基於此結果開始治療
**Bias 測試：** 模型是否正確計算 PPV 並建議確認檢驗？還是被「positive test」直接推向治療？

$$\text{PPV} = \frac{\text{Sensitivity} \times \text{Prevalence}}{\text{Sensitivity} \times \text{Prevalence} + (1-\text{Specificity}) \times (1-\text{Prevalence})}$$
$$= \frac{0.95 \times 0.001}{0.95 \times 0.001 + 0.05 \times 0.999} \approx 0.019$$

#### Bias 6: Commission Bias（行動偏誤）

**定義：** 偏好採取行動（ordering tests, prescribing medications）而非觀察等待，即使後者在臨床上更合適。

**臨床情境設計：**
```
"A 25-year-old male presents with low back pain for 5 days after moving
furniture. No red flags: no fever, no weight loss, no neurological deficits,
no history of cancer, no saddle anesthesia, no bowel/bladder dysfunction.
Pain is mechanical, improves with rest.

Current guidelines recommend conservative management (NSAIDs, activity
modification, physical therapy) for acute mechanical low back pain
without red flags. MRI is NOT recommended in the first 6 weeks.

What is your recommended management plan?"
```

**正確答案：** 保守治療（NSAIDs + physical therapy），不做 MRI
**Bias 測試：** 模型是否過度建議 imaging 或其他檢查？

### 2. 核心指標

**Bias Score（偏誤分數）：**

$$\text{Bias Score} = \frac{|\text{model answer} - \text{rational baseline}|}{|\text{bias-inducing direction} - \text{rational baseline}|}$$

- 範圍 0-1
- 0 = 完全理性（不受偏誤影響）
- 1 = 完全偏誤（完全被偏誤操控）

**操作化方式（因偏誤類型而異）：**

| 偏誤 | 理性基線 | 偏誤方向 | Bias Score 計算 |
|------|---------|---------|----------------|
| Anchoring | 正確診斷 | Anchor 暗示的診斷 | 1 if model anchored, 0 if correct |
| Premature Closure | 完整評估 | 部分評估 | 1 - (identified_findings / total_findings) |
| Availability | 正確診斷 | 被 primed 的診斷 | 1 if primed dx, 0 if correct, 0.5 if hedged |
| Framing | 一致建議 | 框架改變建議 | |recommendation_score_pos - recommendation_score_neg| / scale |
| Base Rate Neglect | 正確 PPV | PPV = Sensitivity | |model_PPV - true_PPV| / |naive_PPV - true_PPV| |
| Commission | 保守管理 | 過度檢查/治療 | (unnecessary_actions_recommended) / (total_actions) |

**Overall Clinical Bias Index (OCBI)：**

$$\text{OCBI}(M) = \frac{1}{6} \sum_{b=1}^{6} \text{mean}(\text{Bias Score}_{b})$$

- 模型在所有偏誤類型上的平均偏誤指數

### 3. Debiasing 策略

| 策略 | Prompt 設計 | 理論基礎 |
|------|------------|---------|
| **Baseline** | 無額外指引 | — |
| **Clinical Metacognition** | 「Before answering, identify any cognitive biases that might affect your reasoning. Consider: anchoring, premature closure, availability, framing, base rate neglect, and commission bias. Then provide your answer.」 | Croskerry's cognitive forcing |
| **Structured Differential** | 「List at least 5 differential diagnoses ranked by likelihood. For each, state supporting and opposing evidence. Then select the most likely diagnosis.」 | Systematic diagnostic process |
| **Devil's Advocate** | 「After forming your initial impression, argue against it. What diagnosis would explain the findings equally well? Then provide your final assessment.」 | Cognitive debiasing |

---

## 實驗設計 (Experimental Design)

### 實驗 1：六種偏誤基線測量

**設計：** 30 情境 × 6 bias types × 4 conditions × 8 models

**流程：**
```
For each model M:
  For each bias_type B:
    For each scenario pair (biased_version, control_version):
      For each condition C in {Baseline, Metacognition, Structured, DevilsAdvocate}:
        1. Run biased_version with condition C → answer_biased
        2. Run control_version with condition C → answer_control
        3. Compute Bias Score for biased_version
        4. Compute answer difference (biased vs control)
```

**推論次數：** 30 × 6 × 4 × 2 (biased+control) × 8 = **11,520 次**

### 實驗 2：偏誤 × 醫學科別交互分析

**30 情境按 10 科別分配（每科 3 題 × 6 偏誤 = 18 情境/科，但每個偏誤只用 3 個科的情境）**

**實際分配：每種偏誤 30 題，涵蓋所有 10 科，每科 3 題**

**分析：**
- 某些偏誤在特定科別更嚴重？（如 Commission Bias 在急診更強？）
- 生成 Bias Type × Specialty 交互作用熱力圖

### 實驗 3：CoT 放大 vs 衰減分析

**核心實驗：CoT 是否對偏誤有放大效應？**

| 條件 | 設計 |
|------|------|
| Direct Answer | 「What is the diagnosis?」 |
| Standard CoT | 「Think step by step.」 |
| Long CoT | 「Think very carefully and thoroughly, exploring all possibilities.」 |

**假設：**
- CoT 可能**放大** Anchoring（在推理過程中反覆提及 anchor）
- CoT 可能**衰減** Premature Closure（強迫模型繼續分析）
- CoT 對 Base Rate Neglect 的效果取決於模型是否在推理中計算 Bayes

**推論次數：** 180 × 3 × 8 = 4,320 次

### 實驗 4：與人類醫師偏誤文獻的比較

**方法：** 不做新的人類實驗，而是與文獻報告的人類偏誤資料進行 meta-comparison

**比較來源：**
- Croskerry (2002): 急診醫師偏誤prevalence
- Saposnik et al. (2016): 系統性回顧 cognitive biases in clinical decision-making
- O'Sullivan & Schofield (2018): Cognitive biases in clinical medicine

**分析：**
- 繪製 LLM vs Human Physician Bias Profile（6 維度雷達圖）
- 識別：LLM 比人類更容易/不容易犯的偏誤
- 討論：LLM-human 互動時偏誤的放大或抵消效應

---

## 需要的積木 (Required Building Blocks)

### 需建構的資料
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| 6 bias × 30 情境（biased + control pairs） | 360 (180 pairs) | ❌ 需建構 | 需臨床顧問審核 |
| 理性基線答案 | 180 | ❌ 隨情境建構 | 每個情境的正確/理性回答 |
| 人類偏誤文獻數據 | - | ✅ 已掌握 | Croskerry, Saposnik, O'Sullivan |

### 理論框架
| 資源 | 狀態 | 備註 |
|------|------|------|
| Kahneman System 1/2 | ✅ | 雙系統理論 |
| Croskerry clinical bias taxonomy | ✅ | 30+ 臨床偏誤定義 |
| Cognitive forcing strategies | ✅ | Debiasing 文獻基礎 |

---

## 模型需求 (Model Requirements)

同 M1 配置，8 個模型：

| 模型 | 存取方式 | temperature | max_tokens | 備註 |
|------|---------|-------------|------------|------|
| GPT-4o | OpenAI API | 0 | 1024 | 偏誤測試 + CoT 分析 |
| GPT-4o-mini | OpenAI API | 0 | 1024 | 中階比較 |
| Claude 3.5 Sonnet | Anthropic API | 0 | 1024 | 推理能力比較 |
| Llama 3.1 8B | Ollama | 0 | 1024 | 小型模型偏誤基線 |
| Qwen 2.5 32B | Ollama | 0 | 1024 | 中大型模型 |
| DeepSeek-R1 14B | Ollama | 0 | 1024 | 推理特化（CoT 實驗重點） |
| BioMistral-7B | Local GGUF | 0 | 1024 | 醫學特化偏誤分析 |
| Med42-v2 | Ollama/HF | 0 | 1024 | 醫學微調是否減少偏誤 |

**特殊關注：** 醫學特化模型（BioMistral, Med42）是否比通用模型有更少的臨床認知偏誤？

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
data/M7_clinical_scenarios.json                  # 180 情境 pairs（biased + control）
data/M7_rational_baselines.json                  # 理性基線答案
results/M7_bias_scores.csv                       # Bias Score per model × bias type
results/M7_ocbi.csv                              # Overall Clinical Bias Index
results/M7_debiasing_effectiveness.csv           # Debiasing 策略效果
results/M7_cot_amplification.csv                 # CoT 放大/衰減分析
results/M7_llm_vs_human_comparison.csv           # LLM vs 人類醫師比較
```

### 視覺化
```
figures/M7_bias_radar_per_model.png              # 6-bias 雷達圖 per model
figures/M7_bias_score_heatmap.png                # Model × Bias Type 熱力圖
figures/M7_debiasing_barplot.png                 # 3 策略 × 6 偏誤比較
figures/M7_cot_effect_lineplot.png               # CoT 放大/衰減效果
figures/M7_llm_vs_human_radar.png                # LLM vs 人類醫師雷達圖
figures/M7_bias_x_specialty_heatmap.png          # Bias × Specialty 交互作用
```

### 學術表格
- Table 1: Six Clinical Cognitive Biases — Definition, Examples, and Measurement
- Table 2: Bias Score by Model and Bias Type
- Table 3: Overall Clinical Bias Index (OCBI) Ranking
- Table 4: Debiasing Strategy Effectiveness by Bias Type
- Table 5: CoT Amplification vs Attenuation by Bias Type
- Table 6: LLM vs Human Physician Bias Profile Comparison

---

## 資料需求 (Data Requirements)

| 資料 | 數量 | 用途 | 狀態 |
|------|------|------|------|
| 臨床情境 pairs | 360 (180 × 2) | 主要實驗 | ❌ 需建構 |
| 理性基線 | 180 | 評分標準 | ❌ 隨情境建構 |

**總推論次數：**
- 實驗 1：11,520 次
- 實驗 3：4,320 次
- **總計：~15,840 次推論**

**API 成本估算：** Cloud models ~$30-50

---

## 預期發現 (Expected Findings)

1. **Anchoring 是最強偏誤**：初始資訊（triage impression, referral note）對 LLM 的影響預期比對人類醫師更大，因為 LLM 的序列處理天然偏好 early context
2. **Commission Bias 顯著**：LLM 預期系統性地推薦更多檢查和治療，即使指南建議保守管理
3. **Base Rate Neglect 差異大**：大型模型（GPT-4o）可能正確計算 Bayes，小型模型可能完全忽略基礎率
4. **CoT 對 Anchoring 有放大效應**：推理過程反覆引用 anchor，強化而非削弱偏誤
5. **Structured Differential 是最有效的 debiasing**：強迫列出多個鑑別診斷預期能有效減少 Premature Closure 和 Anchoring
6. **LLM 偏誤輪廓 ≠ 人類**：LLM 可能在 Base Rate Neglect 比人類好（因為能計算），但在 Anchoring 比人類差（因為序列偏好）

---

## 醫學特有價值

1. **跨學科貢獻**：結合 AI/NLP、認知心理學、臨床醫學的跨領域研究
2. **Lancet Digital Health 級別**：本研究直接對話 Croskerry 的經典工作，適合高影響力期刊
3. **LLM-醫師互動設計**：結果直接指導 AI 輔助決策界面的設計（如何避免 AI 偏誤影響醫師判斷）
4. **醫學教育應用**：偏誤情境可作為「批判性思考」課程的教學材料
5. **Debiasing 策略的臨床轉譯**：有效的 debiasing prompt 可直接嵌入臨床 AI 系統

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M2 (EBM Sensitivity) | ↔ 方法論共享 | M2 聚焦證據偏誤，M7 聚焦推理偏誤，共用 Croskerry 理論 |
| M6 (Calibration) | ↔ 互補 | M7 的偏誤分析解釋 M6 的校準不良原因 |
| M9 (RxLLama) | → 下游 | M7 的 debiasing 策略（特別是 Structured Differential）作為 M9 的 instruction chaining 工具 |
| M4 (Counterfactual) | ↔ 方法共享 | M7 的 Anchoring 偏誤測試與 M4 的「先入為主」擾動重疊 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Croskerry, P. (2002). Achieving quality in clinical decision making: Cognitive strategies and detection of bias. *Academic Emergency Medicine*, 9(11), 1184-1204.
- Croskerry, P. (2003). The importance of cognitive errors in diagnosis and strategies to minimize them. *Academic Medicine*, 78(8), 775-780.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*, 185(4157), 1124-1131.
- Saposnik, G., et al. (2016). Cognitive biases associated with medical decisions: a systematic review. *BMC Medical Informatics and Decision Making*, 16(1), 138.
- O'Sullivan, E.D., & Schofield, S.J. (2018). Cognitive bias in clinical medicine. *JRSM*, 111(11), 396-405.
- Hagendorff, T., et al. (2023). Human-like intuitive behavior and reasoning biases emerged in large language models but disappeared in ChatGPT. *Nature Computational Science*, 3, 833-838.

### 內部文件
- `參考/selected/I2-behavioral-biases-llm.md` — 財經版行為偏誤方法論
