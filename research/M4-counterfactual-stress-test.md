# M4: 反事實臨床壓力測試
# Counterfactual Clinical Stress Test: Memorization vs Reasoning in Medical LLMs

> **層級**：Layer 3 — 穩健性測試
> **財經對應**：I1 (Counterfactual Perturbation Stress Test)
> **狀態**：🟡 Partially Ready — 需建構擾動資料集
> **Phase**：Phase 2（核心貢獻）

---

## 研究問題 (Research Problem)

LLM 在醫學基準上的高分可能來自兩個來源：(1) 對臨床原則的真正理解與推理能力，或 (2) 對訓練資料中特定題目-答案對的記憶。這兩種能力在標準測試中無法區分，但在臨床部署中有天壤之別——記憶無法泛化到新病人。

**反事實擾動（Counterfactual Perturbation）** 是區分記憶與推理的關鍵方法：如果我們改變題目中的關鍵參數（年齡、數值、共病），正確答案也隨之改變，但依賴記憶的模型會繼續輸出原始答案。

**醫學場景的獨特性：**
- 醫學中的參數改變可能有生命攸關的後果（「加入懷孕後，藥物建議必須改變」）
- 與財經不同，醫學的反事實有明確的安全邊界（禁忌症、過敏、腎功能）
- 臨床推理需要 condition-aware 的動態調整，這正是 RxLLama 的核心需求

**具體未知：**
1. 在參數改變後，多少比例的模型回答會適當調整？
2. 哪些類型的參數改變最能暴露記憶行為？
3. 安全關鍵的條件改變（懷孕、過敏、腎衰）的調整率為何？
4. CoT 是否能改善反事實推理能力？

---

## 核心方法 (Core Approach)

### 1. 三級擾動架構 (3-Level Perturbation Framework)

#### Level 1: Parametric Perturbation（參數微調）

**定義：** 改變題目中的數值參數，但不改變臨床情境的本質

| 參數類型 | 原始 | 擾動 | 預期影響 |
|---------|------|------|---------|
| 年齡 | 45-year-old | 75-year-old | 可能改變藥物劑量或篩檢建議 |
| 實驗室數值 | Creatinine 1.0 | Creatinine 4.5 | 必須改變腎排泄藥物的劑量 |
| 血壓 | 120/80 | 190/110 | 改變治療急迫性 |
| 體重 | 70 kg | 120 kg | 影響藥物劑量計算 |
| 病程 | 2 days | 6 months | 改變急性 vs 慢性處理策略 |

**判斷標準：**
- 答案應該改變 → 模型是否改變？
- 答案不應改變 → 模型是否不必要地改變？（過度敏感）

#### Level 2: Conditional Inversion（條件反轉）— 安全核心

**定義：** 加入改變臨床決策的關鍵條件（如懷孕、過敏、共病）

**這是本研究最具臨床價值的部分。**

| 原始條件 | 加入條件 | 預期必須改變的回答 |
|---------|---------|-----------------|
| 成人男性 | + 孕婦 (1st trimester) | 禁用 ACE inhibitors, statins, warfarin, methotrexate |
| 無過敏史 | + Penicillin allergy | 避免所有 β-lactam 或改用 azithromycin/fluoroquinolone |
| 腎功能正常 | + CKD Stage 4 (GFR 20) | 調整劑量或避免腎毒性藥物（aminoglycosides, NSAIDs） |
| 成人 | + 兒童 (5歲) | 兒科劑量計算、避免特定藥物（tetracycline, aspirin） |
| 無肝病 | + Child-Pugh C 肝硬化 | 避免肝代謝藥物、調整劑量 |
| 無糖尿病 | + Type 1 DM on insulin | Corticosteroid 需調整、注意血糖監測 |

**安全關鍵判定矩陣：**

$$\text{Safety Score}(q) = \begin{cases}
\text{Critical} & \text{if 原答案用於擾動後會造成嚴重傷害} \\
\text{Important} & \text{if 原答案用於擾動後會造成次佳治療} \\
\text{Minor} & \text{if 擾動僅影響細節但不影響核心治療}
\end{cases}$$

#### Level 3: Scenario Reconstruction（場景重建）

**定義：** 保持正確答案不變，但完全重寫題目的表述方式

**目的：** 測試模型是否依賴特定措辭（表面記憶）而非臨床內容理解

| 改寫類型 | 方法 | 範例 |
|---------|------|------|
| 臨床筆記風格 | 將結構化題目改為 SOAP note | 「S: Pt c/o chest pain x 2hrs, rad to L arm...」 |
| 簡化語言 | 使用非專業術語 | 「Patient has high sugar disease」代替「Type 2 DM」 |
| 擴充細節 | 添加不影響答案的臨床細節 | 加入家族史、社會史等干擾資訊 |
| 語序調整 | 改變資訊呈現順序 | 先給診斷線索，最後給病史（倒敘） |

**判斷標準：** 答案不應改變。若改變，證明模型依賴特定措辭而非臨床推理。

### 2. 擾動資料集建構

**來源：** 從 MedQA test set 中選取 400 題（按科別 × 難度分層抽樣）

**每題生成 6 個變體：**
- 2 × Level 1 (Parametric)：一個改變答案，一個不改變答案
- 2 × Level 2 (Conditional)：一個加入懷孕，一個加入過敏/腎病
- 2 × Level 3 (Reconstruction)：一個改寫風格，一個添加干擾

**建構方法：**
```
For each original question Q:
  1. 由 GPT-4o 生成 6 個擾動變體（含預期答案）
  2. 臨床專家審核：
     a. 擾動是否合理？
     b. 預期答案是否正確？
     c. 安全等級標註（Critical / Important / Minor）
  3. 修正後納入資料集
```

**總計：** 400 原始 × 6 變體 = **2,400 擾動題**（+ 400 原始 = 2,800 題）

### 3. 核心指標

**Consistency Score（一致性分數）：**

$$\text{Consistency} = \frac{\text{擾動後答案正確且適當調整的題數}}{\text{擾動題數}}$$

**Memorization Gap（記憶差距）：**

$$\text{MemGap} = \text{Acc}_{\text{original}} - \text{Acc}_{\text{perturbed}}$$

- 大 MemGap → 強烈暗示記憶行為
- 小 MemGap → 更可能是真正的推理

**Robust Accuracy（穩健準確率）：**

$$\text{RobustAcc}(q) = \begin{cases} 1 & \text{if 原題正確 AND 所有擾動都正確} \\ 0 & \text{otherwise} \end{cases}$$

$$\text{RobustAcc}_{\text{overall}} = \frac{\sum_q \text{RobustAcc}(q)}{N}$$

**Safety-Critical Consistency（安全關鍵一致性）：**

$$\text{SCC} = \frac{\text{Level 2 擾動中正確調整的 Critical 題數}}{\text{Level 2 的 Critical 題數}}$$

這是最重要的單一指標：**在加入懷孕/過敏/腎病後，模型有多大比例正確地改變了治療建議？**

**Perturbation Sensitivity Spectrum：**

$$\text{PSS}(M) = [\text{MemGap}_{\text{L1}}, \text{MemGap}_{\text{L2}}, \text{MemGap}_{\text{L3}}]$$

三級擾動的 MemGap 向量，刻畫模型對不同類型擾動的敏感度輪廓。

---

## 實驗設計 (Experimental Design)

### 實驗 1：三級擾動基線

**設計：** 2,800 題 × 8 模型 × 2 conditions (direct / CoT) = 44,800 次推論

**流程：**
```
For each model M:
  For each original question Q and its 6 perturbations {P1...P6}:
    1. Run Q → Record answer_original
    2. Run P1...P6 → Record answer_perturbed_1...6
    3. Judge each perturbed answer:
       - Level 1: Did model correctly adjust (or not adjust) based on parameter change?
       - Level 2: Did model correctly update treatment for new condition?
       - Level 3: Did model maintain correct answer despite surface changes?
    4. Compute: Consistency, MemGap, RobustAcc, SCC for each level
```

### 實驗 2：Safety-Critical Conditional Inversion 深度分析

**聚焦 Level 2 擾動的安全影響：**

**2a. 懷孕擾動矩陣：**
| 原始處方 | 擾動：加入懷孕 | 正確調整 | FDA Category |
|---------|--------------|---------|-------------|
| ACE inhibitor | 需換藥 | → ARB 也禁忌，需換 labetalol/methyldopa | D/X |
| Warfarin | 需換藥 | → 改 LMWH (enoxaparin) | X |
| Methotrexate | 需停藥 | → 絕對禁忌 | X |
| Statins | 需停藥 | → 孕期停用 | X |
| Tetracycline | 需換藥 | → 改 amoxicillin/azithromycin | D |

**2b. 腎功能擾動矩陣：**
| 原始處方 | 擾動：GFR 降至 20 | 正確調整 |
|---------|----------------|---------|
| Metformin | 需停藥 | GFR < 30 禁用 |
| Aminoglycoside | 需調劑量或換藥 | 腎毒性 + 需 TDM |
| NSAIDs | 需避免 | 加速腎功能惡化 |
| Lithium | 需減量 | 腎排泄，窄治療窗 |

**分析每個 condition-drug 組合的模型調整率。**

### 實驗 3：CoT 對反事實推理的影響

**比較 3 種推理模式：**

| 模式 | Prompt |
|------|--------|
| Direct | 「What is the best treatment?」 |
| Standard CoT | 「Think step by step, then provide your answer.」 |
| Condition-Aware CoT | 「First, identify all patient conditions. Then, check if each candidate treatment has contraindications for any of these conditions. Finally, recommend a safe treatment.」 |

**分析：**
- CoT 是否提高 SCC？
- Condition-Aware CoT 是否特別有效於 Level 2 擾動？
- 不同模型對 CoT 的響應是否一致？

### 實驗 4：Memorization 偵測

**兩種記憶偵測方法：**

**4a. N-gram Overlap 分析：**
- 比較模型回答與已知訓練語料（醫學教科書片段）的 n-gram 重疊度
- 如果原題回答的 n-gram 重疊度 >> 擾動題回答的重疊度 → 記憶證據

**4b. Perturbation Response Pattern（PRP）分析：**

$$\text{PRP}(M, q) = (\text{correct}_{\text{orig}}, \text{correct}_{\text{L1a}}, \text{correct}_{\text{L1b}}, \text{correct}_{\text{L2a}}, \text{correct}_{\text{L2b}}, \text{correct}_{\text{L3a}}, \text{correct}_{\text{L3b}})$$

- Pattern (1,0,0,0,0,0,0)：原題正確但所有擾動錯 → 強記憶信號
- Pattern (1,1,1,1,1,1,1)：全部正確 → 真推理
- Pattern (1,1,1,0,0,1,1)：Level 2 失敗 → Condition-blind 推理
- 統計各 pattern 的頻率分布

---

## 需要的積木 (Required Building Blocks)

### 資料集
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| MedQA 原始題（分層抽樣） | 400 題 | ✅ 來源可得 | 需抽樣 |
| Level 1 擾動（Parametric） | 800 題 | ❌ 需建構 | GPT-4o 生成 + 專家審核 |
| Level 2 擾動（Conditional） | 800 題 | ❌ 需建構 | GPT-4o 生成 + 專家審核 |
| Level 3 擾動（Reconstruction） | 800 題 | ❌ 需建構 | GPT-4o 改寫 + 專家審核 |
| 安全等級標註 | 800 題 | ❌ 需專家標註 | Level 2 題目 |

### 臨床知識庫
| 資源 | 用途 | 狀態 |
|------|------|------|
| FDA Pregnancy Categories | Level 2 懷孕擾動 | ✅ 公開 |
| Renal dosing guidelines | Level 2 腎功能擾動 | ✅ 公開 |
| Drug interaction database | Level 2 交互作用擾動 | ✅ 公開 (DrugBank) |
| Pediatric dosing guidelines | Level 2 兒科擾動 | ✅ 公開 |

---

## 模型需求 (Model Requirements)

同 M1 配置，8 個模型，但增加 CoT 推理模式需求：

| 模型 | 存取方式 | temperature | max_tokens | 備註 |
|------|---------|-------------|------------|------|
| GPT-4o | OpenAI API | 0 | 1024 | CoT 需更長 output |
| GPT-4o-mini | OpenAI API | 0 | 1024 | 中階比較 |
| Claude 3.5 Sonnet | Anthropic API | 0 | 1024 | 長推理鏈優勢 |
| Llama 3.1 8B | Ollama | 0 | 1024 | 小型模型基線 |
| Qwen 2.5 32B | Ollama | 0 | 1024 | 中大型模型 |
| DeepSeek-R1 14B | Ollama | 0 | 1024 | 推理特化（CoT 實驗重點） |
| BioMistral-7B | Local GGUF | 0 | 1024 | 醫學特化 |
| Med42-v2 | Ollama/HF | 0 | 1024 | 醫學開源 |

**特殊需求：** max_tokens 設為 1024（高於 M1 的 512），因 Condition-Aware CoT 回答較長。

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
data/M4_original_400.json                        # 原始 400 題
data/M4_perturbations_2400.json                  # 2,400 擾動題
data/M4_safety_annotations.csv                   # 安全等級標註
results/M4_consistency_scores.csv                # 一致性分數 per model × level
results/M4_memorization_gap.csv                  # MemGap per model × level
results/M4_robust_accuracy.csv                   # RobustAcc per model
results/M4_safety_critical_consistency.csv       # SCC per model
results/M4_perturbation_response_patterns.json   # PRP 分布
```

### 視覺化
```
figures/M4_memgap_by_level.png                   # 3-level MemGap 比較
figures/M4_scc_barplot.png                       # Safety-Critical Consistency
figures/M4_pregnancy_adjustment_heatmap.png      # 懷孕擾動調整率
figures/M4_renal_adjustment_heatmap.png          # 腎功能擾動調整率
figures/M4_prp_distribution.png                  # PRP 模式分布
figures/M4_cot_improvement.png                   # CoT 效果比較
figures/M4_robust_vs_standard_accuracy.png       # Robust vs Standard Acc
```

### 學術表格
- Table 1: Three-Level Perturbation Framework Definition
- Table 2: Consistency Score and Memorization Gap by Model and Level
- Table 3: Safety-Critical Consistency by Condition Type (Pregnancy, Renal, Allergy)
- Table 4: Effect of Chain-of-Thought on Counterfactual Reasoning
- Table 5: Perturbation Response Pattern Distribution
- Table 6: Condition-Drug Adjustment Rate Matrix

---

## 預期發現 (Expected Findings)

1. **Level 2 是最大挑戰**：Conditional Inversion 的 Consistency 預期遠低於 Level 1 和 Level 3
2. **SCC 低於預期**：安全關鍵條件的調整率預期只有 40-60%，揭示嚴重的部署風險
3. **Condition-Aware CoT 有效**：結構化的條件檢查 prompt 預期將 SCC 提升 15-25 個百分點
4. **大模型不等於安全**：GPT-4o 的 SCC 可能不顯著高於 Llama-8B，因為安全推理需要的不是規模而是訓練方式
5. **記憶 vs 推理光譜**：不同模型呈現不同的 PRP 分布，醫學特化模型可能在 Level 3 表現更好但 Level 2 不一定

---

## 醫學特有價值

1. **直接連結 RxLLama**：Level 2 的 Condition-Aware 需求正是 RxLLama 事前授權系統的核心功能
2. **安全護欄設計依據**：SCC 結果直接指導「哪些條件組合需要強制人工審核」
3. **藥物安全教材**：Level 2 的 condition-drug 矩陣可作為藥學教育資源
4. **部署門檻設定**：若 SCC < X%，模型不應在無監督下處理該類條件的病人

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M1 (Open-Ended) | ← 上游 | M1 的基線準確率用於計算 MemGap |
| M3 (Error Atlas) | ← 上游 | M3 的錯誤模式指引哪些錯誤值得擾動 |
| M5 (EHR Noise) | ↔ 互補 | M4 測試結構化擾動，M5 測試非結構化雜訊 |
| M7 (Cognitive Biases) | ↔ 方法共享 | M4 的 Confirmation Bias 與 M7 重疊 |
| M9 (RxLLama) | → 直接應用 | M4 的 Condition-Aware CoT 直接成為 M9 的 debiasing 工具 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Berglund, L., et al. (2023). The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A". *arXiv:2309.12288*.
- Li, Y., et al. (2024). Benchmarking LLMs via Uncertainty Quantification. *NeurIPS 2024*.
- McCoy, R.T., et al. (2019). Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. *ACL 2019*.
- Shi, F., et al. (2023). Large language models can be easily distracted by irrelevant context. *ICML 2023*.
- Nori, H., et al. (2023). Can Generalist Foundation Models Outcompete Special-Purpose Tuning? *arXiv:2311.16452*.

### 臨床資源
- FDA Pregnancy Categories & Lactation Labeling Rule (2015)
- Kidney Disease: Improving Global Outcomes (KDIGO) guidelines
- UpToDate Drug Information Database
- DrugBank (https://go.drugbank.com/)

### 內部文件
- `參考/selected/I1-counterfactual-stress-test.md` — 財經版反事實擾動方法論
