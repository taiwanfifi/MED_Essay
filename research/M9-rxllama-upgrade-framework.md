# M9: RxLLama 與事前授權系統升級
# Upgrading RxLLama & Prior Authorization: Applying the MedEval-X Toolkit to Real Clinical AI Systems

> **層級**：Layer 6 — 整合應用
> **財經對應**：整合 A1 + D1 + E1 + I1（全部工具包的應用）
> **狀態**：⚪ Conceptual — 需要 M1-M8 方法論工具包
> **Phase**：Phase 3（政策與整合）

---

## 研究問題 (Research Problem)

RxLLama 是國科會計畫中的藥物推薦 LLM 系統，其核心任務是在事前授權（Prior Authorization）流程中輔助臨床決策。目前系統使用單一的準確率分數（如 60/85/95）來評估性能，但 M1-M8 的研究揭示了這種單一維度評估的根本不足。

**本研究的核心問題：**
用 M1-M8 的多維度評估工具包重新評估和升級 RxLLama 系統，從「一個分數」轉變為「多維度 safety-aware 評估框架」。

**四個升級方向：**
1. **多維度評估取代單一分數**：60/85/95 → 多維度計分卡
2. **Sub-Population Q-value**：通用分數 → 特定族群的專用分數（孕婦、兒科、腎病）
3. **事前授權對抗性測試**：標準測試 → EHR 雜訊和邊界案例的壓力測試
4. **Condition-Aware Instruction Chaining**：簡單 prompt → 結構化的條件感知指令鏈

**從方法論角度，M9 回答的是：**
> 「我們在 M1-M8 中開發的評估框架和發現，能否實際改善現有的臨床 AI 系統？」

---

## 核心方法 (Core Approach)

### 1. 升級 1：多維度計分卡 (Multi-Dimensional Scorecard)

**現狀：** RxLLama 的評估使用單一 Q-value（例如 Q=85），代表整體推薦品質。

**問題：** Q=85 可能掩蓋了嚴重的子維度缺陷：
- 整體 85% 正確，但藥理學禁忌症辨識只有 60%
- 整體校準良好，但在罕見疾病上極度過度自信
- 準確率高，但面對 EHR 雜訊時大幅下降

**升級後的計分卡維度：**

| 維度 | 來源 | 指標 | 說明 |
|------|------|------|------|
| D1: 基礎準確率 | M1 | Acc_MCQ, Acc_OpenEnded | 兩種格式的準確率 |
| D2: Option Bias | M1 | Option Bias, Relative OB | MCQ 依賴度 |
| D3: 錯誤嚴重度 | M3 | Severity Distribution | 錯誤的臨床後果分布 |
| D4: 穩健性 | M4, M5 | RobustAcc, SCC, NSI | 擾動/雜訊穩健性 |
| D5: 校準品質 | M6 | ECE, SW-ECE, Coverage@95% | 信心可靠度 |
| D6: 認知偏誤 | M7 | OCBI | 偏誤指數 |
| D7: 安全風險 | M8 | CRITICAL_cases, CH_rate | 高風險案例率 |
| D8: EBM 遵循 | M2 | EAS | 證據等級敏感度 |

**計分卡視覺化：** 8 維雷達圖，每個模型一張

**綜合分數（可選）：**

$$\text{MedEval-X Score} = \sum_{d=1}^{8} w_d \cdot \text{normalized}(D_d)$$

安全導向權重：
- D4 (穩健性): $w = 2.0$
- D5 (校準): $w = 2.0$
- D7 (安全): $w = 3.0$
- D1 (準確率): $w = 1.0$
- 其他: $w = 1.5$

### 2. 升級 2：Sub-Population Q-value

**核心概念：** 通用的 Q=85 在特定族群可能是 Q=95 或 Q=45。臨床上，最重要的是最脆弱族群的 Q-value。

**10 個目標族群 (Sub-Populations)：**

| 編號 | 族群 | 臨床特殊性 | 主要風險 |
|------|------|-----------|---------|
| SP1 | 孕婦 | FDA pregnancy categories | 致畸性藥物 |
| SP2 | 兒科 (< 12歲) | Weight-based dosing | 劑量計算錯誤 |
| SP3 | 老年 (> 75歲) | Polypharmacy, renal decline | 交互作用、蓄積 |
| SP4 | CKD Stage 4-5 | Renal dosing adjustment | 腎毒性、蓄積 |
| SP5 | 肝硬化 (Child-Pugh C) | Hepatic metabolism impaired | 肝代謝藥物 |
| SP6 | 多重用藥 (≥ 5 drugs) | Drug interactions | 交互作用矩陣 |
| SP7 | 過敏史 | Cross-reactivity | β-lactam 交叉過敏 |
| SP8 | 免疫抑制 | Immunocompromised | 感染風險、疫苗禁忌 |
| SP9 | 精神科共病 | Psychiatric medications | MAOi, SSRI 交互作用 |
| SP10 | 哺乳中 | Lactation drug safety | 乳汁分泌藥物 |

**Sub-Population Q-value 計算：**

$$Q_{\text{SP}_k} = \frac{\text{Correct recommendations for SP}_k}{\text{Total recommendations for SP}_k}$$

**Sub-Population Safety Score：**

$$\text{SS}_{\text{SP}_k} = Q_{\text{SP}_k} \times (1 - \text{CRITICAL rate}_{\text{SP}_k})$$

- 乘以 (1 - CRITICAL rate) 確保安全關鍵錯誤被嚴厲懲罰

### 3. 升級 3：事前授權對抗性測試 (Adversarial Prior Authorization Testing)

**事前授權流程模擬：**

```
輸入：
  - Patient demographics (age, sex, weight, conditions)
  - Current medications
  - Requested medication
  - Clinical indication
  - EHR notes (with realistic noise)

LLM 任務：
  1. 判斷 requested medication 是否合適
  2. 識別禁忌症
  3. 識別藥物交互作用
  4. 建議替代方案（如不合適）
  5. 提供信心估計

評估：
  - 禁忌症辨識率 (Contraindication Detection Rate)
  - 交互作用辨識率 (Interaction Detection Rate)
  - 替代方案品質 (Alternative Quality Score)
  - 假陽性率 (False Rejection Rate — 不必要的拒絕)
  - 假陰性率 (False Approval Rate — 應拒絕但通過)
```

**3 種測試條件：**

| 條件 | 說明 | 來源 |
|------|------|------|
| Clean | 乾淨、完整、無矛盾的患者資訊 | 基線 |
| EHR Noisy | M5 的 5 種雜訊注入 | M5 方法論 |
| Adversarial | M4 的 Level 2 條件反轉 | M4 方法論 |

### 4. 升級 4：Condition-Aware Instruction Chaining

**現狀：** RxLLama 使用簡單的 prompt 進行推薦。

**升級：** 基於 M4 和 M7 的發現，設計結構化的 instruction chain：

**Instruction Chain Protocol：**
```
Step 1 — Patient Condition Survey:
"List ALL patient conditions, including:
 - Chronic diseases
 - Current medications
 - Allergies
 - Pregnancy/lactation status
 - Age-specific considerations (pediatric/geriatric)
 - Organ function (renal GFR, hepatic Child-Pugh)"

Step 2 — Contraindication Check:
"For the requested medication [drug], check against EACH condition
listed in Step 1:
 - Is there an absolute contraindication?
 - Is there a relative contraindication?
 - Is dose adjustment needed?
 - List the specific interaction or contraindication."

Step 3 — Alternative Generation (if contraindicated):
"If the medication is contraindicated, suggest alternatives that:
 - Treat the same indication
 - Are safe for ALL listed conditions
 - Have the best evidence level (prioritize RCT-supported options)"

Step 4 — Confidence & Uncertainty Declaration:
"Rate your confidence in this recommendation (0-100%).
List any conditions where you are uncertain about drug safety.
Recommend specialist consultation if confidence < 70%."

Step 5 — Safety Summary:
"Provide a one-paragraph safety summary that a pharmacist can
quickly review, highlighting any flags."
```

**Debiasing 機制：**
- Step 1 防止 M7 的 Premature Closure（強迫列出所有條件）
- Step 2 防止 M4 的 Condition-blind 推理（逐一檢查禁忌症）
- Step 3 利用 M2 的 EBM 原則（優先 RCT-supported 替代方案）
- Step 4 利用 M6 的校準框架（結構化信心聲明）

---

## 實驗設計 (Experimental Design)

### 實驗 1：多維度計分卡驗證

**設計：** 用 M1-M8 的指標為 8 個模型生成完整計分卡

**流程：**
```
For each model M:
  1. 收集 M1-M8 的所有指標
  2. 歸一化至 0-100 分
  3. 計算 8 維計分卡
  4. 計算加權綜合分數
  5. 生成雷達圖
  6. 排名：Overall vs 各維度排名

分析：
  - 模型在哪些維度差異最大？
  - 綜合排名 vs 單一準確率排名是否不同？
  - 安全加權排名是否顛覆傳統排名？
```

### 實驗 2：Sub-Population Q-value 測量

**設計：** 10 個族群 × 20 個測試案例 = 200 題

**測試案例設計：**
```
For each sub-population SP_k (10):
  Design 20 prior authorization scenarios:
    - 10 where medication IS appropriate for SP_k
    - 10 where medication is CONTRAINDICATED for SP_k

  Each scenario includes:
    - Patient profile (matching SP_k characteristics)
    - Requested medication
    - Clinical indication
    - Expected decision (approve/deny)
    - Expected reasoning
```

**推論次數：** 200 × 8 models × 2 conditions (direct, chained) = **3,200 次**

### 實驗 3：EHR 雜訊對事前授權的影響

**設計：** 200 題 × 3 conditions (Clean / Noisy / Adversarial) × 8 models

**流程：**
```
For each question Q (200):
  For each condition C in {Clean, EHR_Noisy, Adversarial}:
    For each model M:
      1. Run prior auth simulation
      2. Record: decision, reasoning, confidence
      3. Evaluate: correct decision? identified contraindications?

  Compute:
    - Decision accuracy per condition
    - Contraindication detection rate per condition
    - False approval rate per condition (most important for safety)
```

**推論次數：** 200 × 3 × 8 = **4,800 次**

### 實驗 4：Condition-Aware Instruction Chaining 效果

**比較 3 種推理模式：**

| 模式 | 說明 |
|------|------|
| Simple Prompt | 「Is this medication appropriate for this patient?」 |
| Standard CoT | 「Think step by step about whether this medication is appropriate.」 |
| Instruction Chain | 5-Step Condition-Aware Protocol（見上方） |

**設計：** 200 題 × 3 modes × 8 models = **4,800 次**

**分析：**
- Instruction Chaining vs Simple Prompt 的準確率提升
- Instruction Chaining 對 Sub-Population 的特別效果
- Instruction Chaining 的延遲成本（更長的推論時間）

### 實驗 5：Before vs After 系統評估

**用計分卡比較「升級前」和「升級後」的系統表現：**

```
Before (Baseline):
  - Simple prompt
  - Single Q-value
  - No noise robustness testing
  - No sub-population analysis

After (Upgraded):
  - Instruction Chain prompt
  - 8-dimension scorecard
  - Sub-population Q-values
  - EHR noise + adversarial tested
  - Selective prediction with confidence threshold

Generate: Before vs After 比較表
```

---

## 需要的積木 (Required Building Blocks)

### M1-M8 方法論工具包
| 來源 | 使用方式 | 狀態 |
|------|---------|------|
| M1 (Open-Ended) | D1 + D2 計分卡維度 | ❌ 待 M1 |
| M2 (EBM Sensitivity) | D8 計分卡維度 | ❌ 待 M2 |
| M3 (Error Atlas) | D3 計分卡維度 | ❌ 待 M3 |
| M4 (Counterfactual) | D4 穩健性 + Adversarial 測試 | ❌ 待 M4 |
| M5 (EHR Noise) | D4 穩健性 + Noisy 測試 | ❌ 待 M5 |
| M6 (Calibration) | D5 校準 + 選擇性預測 | ❌ 待 M6 |
| M7 (Cognitive Biases) | D6 偏誤 + Debiasing 策略 | ❌ 待 M7 |
| M8 (Safety Matrix) | D7 安全 + 部署標準 | ❌ 待 M8 |

### 新建構的資料
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| Prior Auth scenarios (10 SP × 20) | 200 | ❌ 需建構 | 需藥師審核 |
| Contraindication ground truth | 200 | ❌ 需建構 | 基於 DrugBank/UpToDate |
| EHR noisy variants | 200 | ❌ 需建構 | 使用 M5 方法 |
| Adversarial variants | 200 | ❌ 需建構 | 使用 M4 方法 |

### 藥物資料庫
| 資源 | 用途 | 狀態 |
|------|------|------|
| DrugBank | 禁忌症、交互作用 ground truth | ✅ 公開 |
| FDA Pregnancy/Lactation Labels | 孕婦/哺乳安全分級 | ✅ 公開 |
| KDIGO Renal Dosing | CKD 劑量調整 | ✅ 指南可得 |
| Beers Criteria | 老年用藥安全 | ✅ AGS 2023 |
| Lexicomp / UpToDate | 綜合藥物資訊 | 🟡 需訂閱 |

---

## 模型需求 (Model Requirements)

M9 使用全部 8 個模型進行多維度評估：

| 模型 | 存取方式 | temperature | max_tokens | 備註 |
|------|---------|-------------|------------|------|
| GPT-4o | OpenAI API | 0 | 2048 | 5-Step Chain 需長 output |
| GPT-4o-mini | OpenAI API | 0 | 2048 | 成本效益比較 |
| Claude 3.5 Sonnet | Anthropic API | 0 | 2048 | 長指令鏈優勢 |
| Llama 3.1 8B | Ollama | 0 | 2048 | 小型模型（Chain 受益最大？） |
| Qwen 2.5 32B | Ollama | 0 | 2048 | 中大型模型 |
| DeepSeek-R1 14B | Ollama | 0 | 2048 | 推理特化 |
| BioMistral-7B | Local GGUF | 0 | 2048 | 現有 RAG 系統模型 |
| Med42-v2 | Ollama/HF | 0 | 2048 | 醫學開源 |

**特殊需求：** max_tokens 設為 2048（最高），因 5-Step Instruction Chain 的完整回答包含條件列表、逐一禁忌症檢查、替代方案、信心聲明和安全摘要。

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
data/M9_prior_auth_scenarios.json                # 200 事前授權情境
data/M9_subpopulation_definitions.json           # 10 族群定義
results/M9_multidim_scorecard.csv                # 8 模型 × 8 維度計分卡
results/M9_subpop_qvalues.csv                    # Q-value per model × SP
results/M9_prior_auth_results.csv                # 事前授權測試結果
results/M9_instruction_chain_effect.csv          # Instruction Chaining 效果
results/M9_before_after_comparison.csv           # 升級前後比較
```

### 視覺化
```
figures/M9_scorecard_radar_per_model.png          # 8 模型雷達圖
figures/M9_subpop_qvalue_heatmap.png             # Model × Sub-Population Q-value
figures/M9_prior_auth_accuracy_by_condition.png  # 3 條件下的事前授權準確率
figures/M9_instruction_chain_improvement.png     # Instruction Chain 提升幅度
figures/M9_before_after_spider.png               # 升級前後蜘蛛圖比較
figures/M9_safety_score_by_subpop.png            # Safety Score per sub-population
```

### 學術表格
- Table 1: Multi-Dimensional Scorecard — 8 Models × 8 Dimensions
- Table 2: Sub-Population Q-values by Model and Population
- Table 3: Prior Authorization Accuracy under Clean / Noisy / Adversarial Conditions
- Table 4: Instruction Chain Protocol — Step-by-Step Design
- Table 5: Instruction Chain vs Simple Prompt — Performance Comparison
- Table 6: Before vs After System Upgrade — Comprehensive Comparison
- Table 7: Sub-Population Safety Scores (Q × (1 - CRITICAL rate))

---

## 資料需求 (Data Requirements)

| 資料 | 數量 | 用途 | 狀態 |
|------|------|------|------|
| Prior auth scenarios | 200 | 主要測試 | ❌ 需建構 |
| Noisy variants | 200 | EHR 雜訊測試 | ❌ 需建構 |
| Adversarial variants | 200 | 對抗性測試 | ❌ 需建構 |
| M1-M8 指標資料 | varies | 計分卡輸入 | ❌ 待各 M 完成 |

**推論次數：**
- 實驗 2: 3,200
- 實驗 3: 4,800
- 實驗 4: 4,800
- **總計：~12,800 次**

---

## 預期發現 (Expected Findings)

1. **綜合排名 ≠ 準確率排名**：安全加權的多維度計分卡預期會改變模型排名，某些「高準確率」模型在安全維度表現較差
2. **Sub-Population Q-value 差異巨大**：通用 Q=85 可能在孕婦族群降至 Q=55，在無特殊條件的成人維持 Q=92
3. **Instruction Chaining 顯著提升**：5-Step Protocol 預期在 Sub-Population Safety Score 上提升 15-25 個百分點
4. **EHR 雜訊對事前授權影響嚴重**：False Approval Rate 在雜訊條件下預期增加 10-20 個百分點
5. **小型模型受益最大**：Instruction Chaining 對小型模型的提升幅度 > 大型模型，因為結構化指令彌補了模型能力的不足

---

## 醫學特有價值

1. **直接服務國科會計畫**：M9 的結果直接升級 RxLLama 系統
2. **Sub-Population Safety 概念**：引入「最脆弱族群的品質決定系統品質」的安全哲學
3. **事前授權 AI 測試標準**：為 prior authorization AI 系統建立首個對抗性測試標準
4. **Instruction Chaining 作為安全機制**：提供不需要 fine-tuning 的安全提升方法
5. **計分卡可推廣**：多維度計分卡框架可推廣至任何臨床 AI 系統的評估

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M1-M8 (全部) | ← 全部上游 | M9 整合所有構想的方法論和發現 |
| Medical-RAG 系統 | ↔ 直接應用 | M9 的升級可直接應用於現有 RAG 系統 |
| Text2SQL 系統 | ↔ 潛在應用 | M9 的穩健性測試方法可延伸至 SQL 查詢系統 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Singhal, K., et al. (2023). Towards Expert-Level Medical Question Answering with Large Language Models. *arXiv:2305.09617*. [Med-PaLM 2]
- Nori, H., et al. (2023). Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine. *arXiv:2311.16452*.
- Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS 2022*.
- Khot, T., et al. (2023). Decomposed Prompting: A Modular Approach for Solving Complex Tasks. *ICLR 2023*.
- AGS (2023). American Geriatrics Society 2023 Updated Beers Criteria for Potentially Inappropriate Medication Use in Older Adults.

### 臨床資源
- DrugBank (https://go.drugbank.com/)
- KDIGO Clinical Practice Guidelines
- FDA Drug Labeling (DailyMed)
- UpToDate (https://www.uptodate.com/)

### 內部文件
- `國科會_RxLLama/` — RxLLama 現有系統
- `國科會_RxLLama/Medical-RAG-using-Bio-Mistral-7B-main/` — 現有 RAG 系統
- `國科會_RxLLama/關聯資料/text2sql/` — Text2SQL 系統
- 全部 `參考/selected/` — 財經研究方法論參考
