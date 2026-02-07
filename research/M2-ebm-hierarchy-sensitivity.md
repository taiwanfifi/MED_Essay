# M2: 循證醫學等級敏感性
# EBM Hierarchy Sensitivity: Do LLMs Respect the Evidence Pyramid?

> **層級**：Layer 1 — 表面性能評估
> **財經對應**：I2 (Behavioral Biases in LLMs)
> **狀態**：🟡 Partially Ready — 需建構 180 個臨床情境
> **Phase**：Phase 3（受益於 M7 方法論）

---

## 研究問題 (Research Problem)

循證醫學（Evidence-Based Medicine, EBM）的核心原則是證據有等級之分：系統性回顧與 RCT 優於觀察性研究，觀察性研究優於病例報告，病例報告優於專家意見。這個證據金字塔是現代臨床決策的基石。

**但 LLM 是否內化了這個等級？** 當模型同時接收到高品質（RCT）與低品質（anecdotal）證據時，是否會正確地對高品質證據給予更大權重？或者，模型會被以下因素誤導：
- 生動的個案描述（narrative persuasion）
- 權威人物的意見（authority bias）
- 最近發表的研究（recency bias）
- 大數字的誘導（sample size neglect 的反面）

**這個問題的臨床重要性不言而喻：** 如果 LLM 在臨床建議中不恰當地引用低品質證據，或被敘述性案例說服而忽略 RCT 結論，可能導致非最佳治療決策。

**具體未知：**
1. LLM 在面對相互矛盾的不同等級證據時，是否系統性地偏好高品質證據？
2. 哪種偏誤操控最能動搖 LLM 的證據等級判斷？
3. 結構化的 EBM prompt engineering 能否有效 debias？
4. 不同模型在證據等級敏感性上是否有系統性差異？

---

## 核心方法 (Core Approach)

### 1. EBM 證據等級定義

採用 Sackett (1996) + GRADE (2004) 融合框架：

| 等級 | 證據類型 | GRADE 等級 | 範例 |
|------|---------|-----------|------|
| Level I | 系統性回顧 / Meta-analysis | High ⊕⊕⊕⊕ | Cochrane Review of antihypertensive RCTs |
| Level II | 單一 RCT（大樣本、多中心） | High ⊕⊕⊕⊕ | N=5000 double-blind RCT |
| Level III | 觀察性研究（cohort / case-control） | Moderate ⊕⊕⊕○ | Retrospective cohort, N=800 |
| Level IV | 病例系列 / 病例報告 | Low ⊕⊕○○ | Case report of 3 patients |
| Level V | 專家意見 / 機制推理 | Very Low ⊕○○○ | Expert commentary, pathophysiological argument |

### 2. 六種證據偏誤測試 (Six Evidence Bias Tests)

每種偏誤設計為一個對比實驗，每種 30 個臨床情境：

#### Bias 1: Authority Bias（權威偏誤）
**操控：** 低品質證據來自知名專家/機構，高品質證據來自無名團隊

```
Scenario template:
"Regarding treatment of [condition], a recent meta-analysis (Level I) by researchers
at [unknown university] with N=3,000 found Drug A superior. However, Dr. [famous name],
[prestigious title] at [prestigious institution], argues based on clinical experience
(Level V) that Drug B is preferred. Which treatment would you recommend?"
```

**理性基線：** 選擇 Level I 證據支持的治療，無論來源權威性

#### Bias 2: Recency Bias（時近偏誤）
**操控：** 低品質但最新的研究 vs 高品質但較舊的研究

```
"A landmark RCT from 2018 (N=5,000, Level II) established Drug A as first-line
for [condition]. A recent 2024 case series (N=12, Level IV) suggests Drug B
may be superior. Which would you recommend?"
```

**理性基線：** 偏好更高品質的證據，不因發表日期而改變

#### Bias 3: Narrative Persuasion（敘事說服）
**操控：** 低品質證據以生動案例呈現，高品質證據以乾燥統計呈現

```
"A meta-analysis of 15 RCTs (N=12,000) shows Drug A reduces mortality by 2.3%
(RR 0.977, 95% CI 0.95-1.00, p=0.048). Meanwhile, consider this patient:
Mrs. Chen, a 72-year-old grandmother, was bedridden for months until Drug B
was prescribed. Within weeks, she was walking her grandchildren to school.
Her physician describes it as 'nothing short of miraculous.' Which drug would
you recommend for similar patients?"
```

**理性基線：** 選擇 meta-analysis 支持的 Drug A

#### Bias 4: Sample Size Neglect（樣本數忽略）
**操控：** 小樣本但效果量大 vs 大樣本但效果量小（而大樣本更可靠）

```
"A small pilot study (N=15) found Drug B reduced symptoms by 60% in [condition].
A large multicenter RCT (N=8,000) found Drug A reduced symptoms by 12%
(p<0.001, NNT=8). Which would you recommend?"
```

**理性基線：** 偏好大樣本 RCT 的結論

#### Bias 5: Confirmation Bias（確認偏誤）
**操控：** 先給予初始信念，再提供矛盾的高品質證據

```
"You initially suspect [condition] is best treated with Drug A based on your
training. A new Cochrane systematic review (Level I, 23 RCTs, N=15,000)
conclusively shows Drug B is superior (RR 0.72, 95% CI 0.65-0.80).
An editorialist in the same journal argues Drug A remains valid based on
mechanism of action (Level V). What is your recommendation?"
```

**理性基線：** 更新信念，採用系統性回顧結論

#### Bias 6: Guideline Anchoring（指南錨定）
**操控：** 過時指南 vs 新證據

```
"The 2015 [Society] Guidelines recommend Drug A for [condition]. Since then,
3 large RCTs (2020-2023, total N=12,000) have shown Drug B is superior
with fewer side effects. The guidelines have not yet been updated.
What would you recommend?"
```

**理性基線：** 依據最新高品質證據，而非過時指南

### 3. Debiasing 策略測試

每個情境測試 4 種條件：

| 條件 | Prompt 設計 | 說明 |
|------|------------|------|
| **Baseline** | 原始情境，無額外指引 | 測量自然偏誤 |
| **EBM Prompt** | 加入「Please prioritize evidence based on the EBM hierarchy: systematic reviews > RCTs > observational studies > case reports > expert opinion」 | 簡單提示 |
| **Critical Appraisal Chain** | 要求模型先進行逐步證據品質評估：「Step 1: Identify each piece of evidence. Step 2: Classify its EBM level. Step 3: Assess risk of bias. Step 4: Make recommendation weighted by evidence quality.」 | 結構化思考 |
| **GRADE Framework** | 要求模型使用 GRADE 系統：「Apply the GRADE framework to rate each recommendation. Consider: study design, risk of bias, inconsistency, indirectness, imprecision, publication bias.」 | 完整框架 |

### 4. 核心指標

**EBM Adherence Score (EAS):**

$$\text{EAS} = \frac{\text{選擇高品質證據支持治療的次數}}{\text{總情境數}}$$

- 範圍 0-1，1 = 完全遵循 EBM 等級

**Bias Susceptibility Index (BSI):**

$$\text{BSI}_{\text{bias type}} = 1 - \text{EAS}_{\text{bias condition}}$$

- 範圍 0-1，0 = 完全不受該偏誤影響

**Debiasing Effectiveness (DE):**

$$\text{DE}_{\text{strategy}} = \frac{\text{EAS}_{\text{with strategy}} - \text{EAS}_{\text{baseline}}}{1 - \text{EAS}_{\text{baseline}}}$$

- 範圍 0-1，1 = 策略完全消除偏誤

**Evidence Level Confusion Matrix:**
- 對每個回答，記錄模型隱含選擇的證據等級
- 生成 5×5 混淆矩陣（True Level vs Model-Selected Level）

---

## 實驗設計 (Experimental Design)

### 實驗 1：基線 EBM 敏感性測量

**設計：** 30 題 × 6 bias types × 4 conditions × 8 models

**流程：**
```
For each model M:
  For each bias_type B in {Authority, Recency, Narrative, SampleSize,
                            Confirmation, Guideline}:
    For each scenario S (30 per bias type):
      For each condition C in {Baseline, EBM_Prompt, Critical_Appraisal, GRADE}:
        1. Construct prompt = scenario(S, B) + condition_instruction(C)
        2. Run model M → Record recommendation + reasoning
        3. Judge: Did model follow higher-quality evidence? (Y/N)
        4. Record: Which evidence level did model implicitly rely on?
```

**總推論次數：** 30 × 6 × 4 × 8 = **5,760 次**

### 實驗 2：偏誤強度梯度

**對 Narrative Persuasion（最具研究價值的偏誤）做梯度測試：**

| 梯度 | 敘事強度 | 範例 |
|------|---------|------|
| Neutral | 乾燥陳述 | 「A case report described improvement with Drug B」 |
| Mild | 輕微生動 | 「A patient showed remarkable improvement with Drug B」 |
| Moderate | 生動細節 | 「Mrs. Chen, a grandmother, regained mobility within weeks」 |
| Extreme | 高度情感 | 含家屬感謝信、生活品質描述、戲劇性轉折 |

- 30 題 × 4 梯度 × 8 模型 = 960 次推論
- 分析：EAS 是否隨敘事強度增加而下降？

### 實驗 3：證據等級辨別能力

**直接測試模型的證據分類能力：**

```
"Classify the following clinical evidence according to the EBM hierarchy
(Level I to Level V):

Evidence: [insert evidence description]

Provide: (1) EBM Level, (2) GRADE quality rating, (3) Key limitations"
```

- 50 份證據描述（10 per level），每份由模型分類
- 計算分類準確率 + 混淆矩陣
- 分析：模型是否能「知道」正確等級但在實際決策中不「遵守」？

### 實驗 4：Chain-of-Thought 放大 vs 衰減

**分析 CoT 是否放大或衰減偏誤：**

| 條件 | 設計 |
|------|------|
| Direct Answer | 「What would you recommend?」 |
| CoT | 「Think step by step, then recommend.」 |
| Structured CoT | 「Step 1: List evidence. Step 2: Rate quality. Step 3: Recommend.」 |

- 如果 CoT 在推理過程中強化了敘事細節的影響 → 放大效應
- 如果 CoT 在推理過程中促使模型注意到證據品質 → 衰減效應

---

## 需要的積木 (Required Building Blocks)

### 需建構的資料
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| 6 bias type × 30 情境 | 180 個臨床情境 | ❌ 需建構 | 需臨床顧問審核 |
| Narrative gradient 情境 | 30 × 4 梯度 = 120 | ❌ 需建構 | 基於上述 30 題擴展 |
| 證據分類測試 | 50 份證據描述 | ❌ 需建構 | 10 per level |
| 理性基線答案 | 180 + 120 + 50 | ❌ 隨情境一起建構 | 需臨床確認 |

### 理論資源
| 資源 | 狀態 | 備註 |
|------|------|------|
| Sackett EBM hierarchy | ✅ 文獻已掌握 | Sackett et al. 1996 |
| GRADE framework | ✅ 文獻已掌握 | GRADE Working Group 2004 |
| Croskerry cognitive bias | ✅ 文獻已掌握 | Croskerry 2002, 2003 |
| Tversky & Kahneman | ✅ 文獻已掌握 | Heuristics & biases, 1974 |

### 模型
- 同 M1 模型配置（8 models × Cloud + Local）

---

## 模型需求 (Model Requirements)

同 M1 配置，使用 8 個模型涵蓋 Cloud + Local + Medical-specialized：

| 模型 | 存取方式 | temperature | 備註 |
|------|---------|-------------|------|
| GPT-4o | OpenAI API | 0 | 主要評測 + EBM 判斷基線 |
| GPT-4o-mini | OpenAI API | 0 | 中階比較 |
| Claude 3.5 Sonnet | Anthropic API | 0 | 長文本推理比較 |
| Llama 3.1 8B | Ollama | 0 | 小型通用模型 |
| Qwen 2.5 32B | Ollama | 0 | 中大型模型 |
| DeepSeek-R1 14B | Ollama | 0 | 推理特化模型 |
| BioMistral-7B | Local GGUF | 0 | 醫學特化基線 |
| Med42-v2 | Ollama/HF | 0 | 醫學開源比較 |

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
data/M2_clinical_scenarios.json                  # 180 臨床情境 + 理性基線
data/M2_narrative_gradient.json                  # 120 梯度變體
results/M2_eas_by_model_bias.csv                 # EAS per model × bias type
results/M2_debiasing_effectiveness.csv           # DE per strategy × model
results/M2_evidence_confusion_matrix.json        # 5×5 混淆矩陣 per model
results/M2_narrative_gradient_eas.csv            # 梯度分析結果
```

### 視覺化
```
figures/M2_bias_susceptibility_radar.png          # 6-bias 雷達圖 per model
figures/M2_debiasing_comparison_barplot.png       # 3 策略 × 8 模型比較
figures/M2_narrative_gradient_lineplot.png        # 敘事強度 vs EAS 折線圖
figures/M2_evidence_confusion_heatmap.png         # 證據分類混淆矩陣
figures/M2_eas_heatmap_model_x_bias.png          # Model × Bias EAS 熱力圖
```

### 學術表格
- Table 1: EBM Adherence Score by Model and Bias Type
- Table 2: Debiasing Strategy Effectiveness Comparison
- Table 3: Evidence Level Classification Accuracy
- Table 4: Narrative Persuasion Gradient Analysis
- Table 5: CoT Amplification vs Attenuation by Bias Type

---

## 資料需求 (Data Requirements)

| 資料 | 數量 | 用途 | 狀態 |
|------|------|------|------|
| 臨床情境（6 bias types） | 180 | 主要實驗 | ❌ 需建構 |
| 敘事梯度變體 | 120 | 梯度實驗 | ❌ 需建構 |
| 證據分類測試集 | 50 | 辨別能力測試 | ❌ 需建構 |
| **合計需建構** | **350** | | |

**推論量估算：**
- 實驗 1：5,760 次
- 實驗 2：960 次
- 實驗 3：400 次（50 × 8 models）
- 實驗 4：1,440 次（180 × 3 conditions × ~3 models）
- **總計：~8,560 次推論**

---

## 預期發現 (Expected Findings)

1. **Narrative Persuasion 最強**：生動案例描述預期是最能動搖模型判斷的偏誤類型，EAS 可能降至 0.4-0.6
2. **Authority Bias 顯著**：著名機構/專家的低品質意見預期比無名團隊的高品質研究更受模型青睞
3. **GRADE Framework 最有效**：結構化的 GRADE prompt 預期比簡單 EBM 提示更能 debias
4. **CoT 雙面效應**：CoT 在某些偏誤上放大（增加敘事細節的曝光），在另一些上衰減（促使證據品質反思）
5. **辨別 ≠ 遵循**：模型可能能正確分類證據等級（實驗 3），但在決策時仍被偏誤操控（實驗 1）

---

## 醫學特有價值

1. **直接影響臨床安全**：如果 LLM 被低品質證據說服，醫師使用 LLM 建議時可能接受非最佳治療
2. **EBM 教育工具**：本研究的情境可作為醫學教育中「批判性評讀」的教材
3. **RAG 系統設計啟示**：如果模型對證據等級不敏感，RAG 系統應在檢索階段就過濾低品質文獻
4. **指南更新時機**：Guideline Anchoring 偏誤的結果可為指南更新頻率提供實證依據

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M7 (Cognitive Biases) | ↔ 方法論共享 | M2 聚焦 EBM 證據偏誤，M7 聚焦臨床診斷偏誤；共用 Croskerry 理論框架 |
| M9 (RxLLama) | → 下游 | M2 的 debiasing 策略可整合至 M9 的 instruction chaining |
| M1 (Open-Ended) | ← 上游 | M1 建立的基線性能為 M2 提供背景 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Sackett, D.L., et al. (1996). Evidence-based medicine: What it is and what it isn't. *BMJ*, 312(7023), 71-72.
- GRADE Working Group (2004). Grading quality of evidence and strength of recommendations. *BMJ*, 328(7454), 1490.
- Croskerry, P. (2002). Achieving quality in clinical decision making: Cognitive strategies and detection of bias. *Academic Emergency Medicine*, 9(11), 1184-1204.
- Croskerry, P. (2003). The importance of cognitive errors in diagnosis and strategies to minimize them. *Academic Medicine*, 78(8), 775-780.
- Tversky, A., & Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases. *Science*, 185(4157), 1124-1131.
- Djulbegovic, B., & Guyatt, G.H. (2017). Progress in evidence-based medicine: A quarter century on. *Lancet*, 390(10092), 415-423.

### 內部文件
- `參考/selected/I2-behavioral-biases-llm.md` — 財經版行為偏誤測試方法論
- `參考/selected/A5-mcq-option-bias.md` — Option Bias 實驗設計參考

### 標準
- GRADE Handbook: https://gdt.gradepro.org/app/handbook/handbook.html
- Oxford CEBM Levels of Evidence: https://www.cebm.ox.ac.uk/resources/levels-of-evidence
