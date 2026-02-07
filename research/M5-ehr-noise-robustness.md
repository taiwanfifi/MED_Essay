# M5: 電子病歷雜訊穩健性
# EHR Noise Robustness: Can Medical LLMs Handle Real-World Clinical Documentation Chaos?

> **層級**：Layer 3 — 穩健性測試
> **財經對應**：I3 (Red Herrings)，但為醫學獨有問題
> **狀態**：⚪ Conceptual — 需建構完整情境資料集
> **Phase**：Phase 4（延伸，最具獨創性）

---

## 研究問題 (Research Problem)

現有醫學 LLM 基準使用的都是乾淨、結構化、精心編寫的臨床情境。但真實世界的電子病歷（EHR）充滿了雜訊：

- **Copy-paste redundancy（複製貼上冗餘）：** Tsou et al. (2017) 發現 EHR 中高達 82% 的內容是複製貼上的，同一段病史可能在不同筆記中重複出現 5-10 次
- **Conflicting provider assessments（醫師間評估矛盾）：** 不同班別的醫師在 progress note 中可能寫出矛盾的評估
- **Medication list discrepancy（用藥清單不一致）：** 用藥和解（medication reconciliation）經常不完整，病歷中可能存在多份不一致的用藥清單
- **Irrelevant clinical detail（無關臨床細節）：** 大量非關鍵的社會史、家族史、系統回顧等資訊
- **Temporal ambiguity（時序模糊）：** 病歷的時序經常不清楚，「previous」「recent」「prior」缺乏具體日期

**核心問題：** 當 LLM 接收到這些真實世界的雜訊時，推理能力下降多少？它會被矛盾資訊誤導嗎？能從冗餘資訊中提取關鍵線索嗎？

**這是醫學獨有的問題。** 財經領域的「red herring」實驗（I3）提供了方法論靈感，但 EHR 雜訊的類型、影響方式和臨床後果與財經領域截然不同。

---

## 核心方法 (Core Approach)

### 1. 五種臨床雜訊類型 (Five Clinical Noise Types)

#### Noise Type 1: Copy-Paste Redundancy（複製貼上冗餘）

**操控：** 在乾淨的臨床情境前後插入重複的病史段落

```
原始（乾淨）：
"65M with HTN, DM2, presents with acute chest pain..."

擾動（redundant）：
"Assessment from 3 days ago: 65M with HTN, DM2, no acute complaints, continue
current medications.
Assessment from 2 days ago: 65M with HTN, DM2, no acute complaints, stable.
Assessment from 1 day ago: 65M with HTN, DM2, no acute complaints, vitals stable.
Current assessment: 65M with HTN, DM2, presents with acute chest pain..."
```

**關鍵測試：** 模型是否能從冗餘的「stable」描述中識別出最新的「acute chest pain」？

**3 個冗餘等級：**
- Mild：1 份歷史筆記前綴（~50% 額外文字）
- Moderate：3 份歷史筆記前綴（~150% 額外文字）
- Severe：5 份歷史筆記前綴（~300% 額外文字）

#### Noise Type 2: Conflicting Provider Assessments（醫師間評估矛盾）

**操控：** 在情境中加入與正確評估矛盾的其他醫師意見

```
原始：
"Patient presents with classic signs of appendicitis: RLQ pain, rebound
tenderness, fever 38.5°C, WBC 15,000."

擾動：
"ED physician assessment: Likely acute appendicitis, recommend surgical consult.
Internal medicine overnight note: Symptoms more consistent with mesenteric
lymphadenitis, suggest conservative management and repeat labs.
Surgical consult: Agree with ED assessment, classic presentation for appendicitis.
Please provide your assessment and recommended management."
```

**關鍵測試：** 模型是否受到矛盾意見的影響而改變正確判斷？

#### Noise Type 3: Medication List Discrepancy（用藥清單不一致）

**操控：** 提供多份不一致的用藥清單

```
"Admission medication list: Metformin 1000mg BID, Lisinopril 10mg daily,
Atorvastatin 40mg daily.

Pharmacy reconciliation note: Metformin 500mg BID, Lisinopril 20mg daily,
Atorvastatin 40mg daily, Aspirin 81mg daily (not on admission list).

Outpatient pharmacy records: Metformin 1000mg BID, Lisinopril 10mg daily,
Atorvastatin 20mg daily, Amlodipine 5mg daily (not on other lists).

Based on this patient's medications, what drug interactions should you
be concerned about?"
```

**關鍵測試：** 模型如何處理不一致的用藥資訊？是否識別出不一致？選擇哪份清單？

#### Noise Type 4: Irrelevant Clinical Detail（無關臨床細節）

**操控：** 在關鍵臨床資訊周圍插入大量不影響答案的細節

```
原始（簡潔）：
"45F with new-onset seizure, no prior history. CT head shows ring-enhancing
lesion in right temporal lobe. What is the most likely diagnosis?"

擾動（detail-heavy）：
"45F, works as a high school teacher, married with 2 children ages 12 and 15,
lives in suburban Taipei. She enjoys hiking on weekends and has a cat named Mimi.
Family history significant for father with hypertension (managed with amlodipine)
and mother with osteoarthritis. Social history: occasional wine with dinner,
never smoked, no drug use. Immunizations up to date including COVID-19 boosters.
Last dental visit 6 months ago, no issues. Allergies: seasonal rhinitis managed
with cetirizine PRN. Presents with new-onset seizure, witnessed by husband who
reports tonic-clonic activity lasting approximately 2 minutes. No prior seizure
history. Postictal confusion lasted 15 minutes. ROS: denies headache, vision
changes, weight loss, night sweats, fever, cough. CT head shows ring-enhancing
lesion in right temporal lobe. What is the most likely diagnosis?"
```

**3 個細節等級：**
- Mild：+30% 額外文字（少量社會史/家族史）
- Moderate：+100% 額外文字（完整的無關 ROS、社會史、家族史）
- Severe：+250% 額外文字（含過去就醫紀錄、過敏清單、免疫紀錄等）

#### Noise Type 5: Temporal Ambiguity（時序模糊）

**操控：** 使用模糊的時間描述替代精確日期

```
原始（清楚時序）：
"Patient started on warfarin on Jan 1. INR checked on Jan 7 was 1.2.
Dose increased. INR checked on Jan 14 was 3.8. What should you do?"

擾動（模糊時序）：
"Patient was started on warfarin recently. A follow-up INR was subtherapeutic.
Dose was adjusted. A subsequent INR was found to be supratherapeutic.
What should you do?"
```

**關鍵測試：** 時序模糊是否改變模型的處置建議？是否遺漏與時間相關的臨床判斷？

### 2. 核心指標

**Distraction Rate（分心率）：**

$$\text{Distraction Rate} = \frac{\text{加入雜訊後答錯的題目（原本答對的）}}{\text{原本答對的題目}}$$

- 衡量雜訊導致正確答案「翻車」的比例

**Noise Sensitivity Index（雜訊敏感度指數）：**

$$\text{NSI} = 1 - \frac{\text{Acc}_{\text{noisy}}}{\text{Acc}_{\text{clean}}}$$

- 範圍 0-1，0 = 完全不受雜訊影響

**Per-Noise-Type NSI 向量：**

$$\text{NSI Vector}(M) = [\text{NSI}_{\text{CopyPaste}}, \text{NSI}_{\text{Conflict}}, \text{NSI}_{\text{MedList}}, \text{NSI}_{\text{Irrelevant}}, \text{NSI}_{\text{Temporal}}]$$

**Conflict Resolution Score（衝突解決分數）：**

$$\text{CRS} = \frac{\text{正確識別並解決矛盾的題目}}{\text{包含矛盾的題目}}$$

- 特別用於 Noise Type 2（矛盾評估）和 Type 3（用藥不一致）

**Signal-to-Noise Extraction（信噪比提取能力）：**

$$\text{SNE} = \frac{\text{Acc}_{\text{severe noise}} - \text{Acc}_{\text{random baseline}}}{\text{Acc}_{\text{clean}} - \text{Acc}_{\text{random baseline}}}$$

- 衡量模型在最嚴重雜訊下保留了多少有效推理能力

---

## 實驗設計 (Experimental Design)

### 實驗 1：五種雜訊基線測試

**設計：** 200 乾淨題 × 5 noise types × 1 noise level = 1,000 擾動題 + 200 原始 = 1,200 題

**流程：**
```
For each model M in {8 models}:
  For each clean question Q (200):
    1. Run Q (clean) → Record answer_clean, correct_clean
    For each noise_type N (5):
      2. Run Q_noisy(N, moderate) → Record answer_noisy, correct_noisy
  Compute:
    - Distraction Rate per noise type
    - NSI per noise type
    - CRS (for Noise Types 2, 3)
    - NSI Vector
```

**推論次數：** 1,200 × 8 = **9,600 次**

### 實驗 2：雜訊強度梯度

**對 Noise Type 1（Copy-Paste）和 Type 4（Irrelevant Detail）做 3 級梯度：**

| 梯度 | Type 1: Copy-Paste | Type 4: Irrelevant Detail |
|------|-------------------|--------------------------|
| Mild | +1 歷史筆記 (~50% extra) | +30% extra text |
| Moderate | +3 歷史筆記 (~150% extra) | +100% extra text |
| Severe | +5 歷史筆記 (~300% extra) | +250% extra text |

**設計：** 200 題 × 2 noise types × 3 levels = 1,200 題
**推論次數：** 1,200 × 8 = 9,600 次

**分析：**
- 繪製 Noise Level vs Accuracy 曲線（是否線性下降？有無 cliff effect？）
- 識別「穩健性閾值」：多少雜訊量開始顯著影響性能？

### 實驗 3：雜訊組合效應

**在真實 EHR 中，多種雜訊同時存在。測試組合效應：**

| 組合 | 說明 |
|------|------|
| Type 1 + 4 | 冗餘 + 無關細節（最常見的 EHR 雜訊組合） |
| Type 2 + 3 | 矛盾評估 + 用藥不一致（最危險的組合） |
| Type 1 + 2 + 4 | 冗餘 + 矛盾 + 細節（接近真實 EHR） |
| 全部 5 種 | 「Full EHR simulation」（最嚴苛測試） |

**設計：** 200 題 × 4 組合 = 800 題
**推論次數：** 800 × 8 = 6,400 次

**分析：**
- 組合效應是否 > 個別效應之和？（交互作用檢驗）
- 哪個組合最具破壞力？

### 實驗 4：RAG 系統的 EHR 雜訊穩健性

**將 EHR 雜訊測試應用於 RAG 系統（連結現有 Medical-RAG）：**

```
設計：
1. 將乾淨版和雜訊版的臨床情境都作為 context 輸入 RAG
2. 觀察：
   a. 雜訊是否影響 retrieval quality？（是否檢索到不同的文件？）
   b. 雜訊是否影響 generation quality？（相同文件下是否給出不同答案？）
3. Ablation：雜訊影響主要來自 retrieval 還是 generation？
```

---

## 需要的積木 (Required Building Blocks)

### 需建構的資料
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| 200 乾淨臨床情境 | 200 | ❌ 需建構 | 涵蓋 10 科別 × 20 題 |
| 5 noise type 變體 | 1,000 | ❌ 需建構 | GPT-4o 生成 + 臨床審核 |
| 梯度變體 | 1,200 | ❌ 需建構 | 2 types × 3 levels × 200 |
| 組合變體 | 800 | ❌ 需建構 | 4 組合 × 200 |
| **合計** | **~3,200** | | |

### 理論資源
| 資源 | 狀態 | 備註 |
|------|------|------|
| Tsou et al. 2017 (EHR copy-paste) | ✅ 已閱讀 | 82% copy-paste 率數據 |
| Hammond et al. 2019 (EHR usability) | ✅ 可取得 | ONC 報告 |
| Singh et al. 2014 (Diagnostic errors from EHR) | ✅ 可取得 | EHR 對錯誤的影響 |
| Weir et al. 2020 (Note bloat) | ✅ 可取得 | EHR note bloat 量化 |

### 基礎設施
| 資源 | 用途 | 狀態 |
|------|------|------|
| Medical-RAG 系統 | 實驗 4 | ✅ 現有系統可用 |
| Qdrant + BioMistral | RAG 基礎設施 | ✅ 已部署 |

---

## 模型需求 (Model Requirements)

同 M1 配置，8 個模型。EHR 雜訊實驗的特殊考量：

| 模型 | 存取方式 | temperature | context window | 備註 |
|------|---------|-------------|---------------|------|
| GPT-4o | OpenAI API | 0 | 128K | 長 EHR 不成問題 |
| GPT-4o-mini | OpenAI API | 0 | 128K | 成本效益比較 |
| Claude 3.5 Sonnet | Anthropic API | 0 | 200K | 超長 context 優勢 |
| Llama 3.1 8B | Ollama | 0 | 8K-128K | **Severe noise 可能超出 context** |
| Qwen 2.5 32B | Ollama | 0 | 32K-128K | 中大型模型 |
| DeepSeek-R1 14B | Ollama | 0 | 32K-128K | 推理特化 |
| BioMistral-7B | Local GGUF | 0 | 4K-8K | **context window 限制最嚴格** |
| Med42-v2 | Ollama/HF | 0 | 8K | 醫學開源 |

**特殊考量：** Severe noise 擾動可增加 2-3× 輸入長度。BioMistral-7B 的 4K context window 在 Severe 條件下可能截斷輸入，需記錄並分析 truncation 對結果的影響。

**RAG 系統（實驗 4）：**
| 組件 | 配置 | 狀態 |
|------|------|------|
| BioMistral-7B | 現有 GGUF model | ✅ 已部署 |
| PubMedBERT embeddings | NeuML/pubmedbert-base-embeddings | ✅ 已部署 |
| Qdrant | localhost:6333 | ✅ 已部署 |

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
data/M5_clean_scenarios.json                     # 200 乾淨情境
data/M5_noisy_variants.json                      # 所有雜訊變體
results/M5_distraction_rate.csv                  # 分心率 per model × noise type
results/M5_nsi_vectors.csv                       # NSI 向量 per model
results/M5_conflict_resolution.csv               # CRS per model
results/M5_gradient_analysis.csv                 # 梯度分析
results/M5_combination_effects.csv               # 組合效應
results/M5_rag_noise_impact.csv                  # RAG 雜訊影響
```

### 視覺化
```
figures/M5_nsi_radar.png                         # 5-noise NSI 雷達圖 per model
figures/M5_gradient_curve.png                    # 雜訊強度 vs 準確率曲線
figures/M5_combination_interaction.png           # 組合效應交互作用圖
figures/M5_noise_type_comparison.png             # 5 種雜訊影響比較
figures/M5_rag_retrieval_vs_generation.png       # RAG ablation 結果
figures/M5_model_robustness_ranking.png          # 模型穩健性排名
```

### 學術表格
- Table 1: Five Clinical Noise Types — Definition and Examples
- Table 2: Distraction Rate by Model and Noise Type
- Table 3: Noise Sensitivity Index Vectors
- Table 4: Noise Intensity Gradient Analysis
- Table 5: Combination Effect Interaction Analysis
- Table 6: RAG System Noise Robustness (Retrieval vs Generation Impact)

---

## 資料需求 (Data Requirements)

| 資料 | 數量 | 用途 | 狀態 |
|------|------|------|------|
| 乾淨臨床情境 | 200 | 基礎題目 | ❌ 需建構 |
| 雜訊變體（全部） | ~3,200 | 實驗 1-4 | ❌ 需建構 |
| **合計推論次數** | **~25,600** | 3,200 × 8 models | |

**API 成本估算：** Cloud models ~$40-80, 主要為 local inference

---

## 預期發現 (Expected Findings)

1. **矛盾型雜訊最具破壞力**：Type 2（Conflicting Assessments）預期導致最高的 Distraction Rate（30-50%）
2. **Copy-Paste 冗餘有 cliff effect**：在 Moderate 級別前影響不大，Severe 級別突然下降（因為 context window 被冗餘占滿）
3. **無關細節影響小模型更多**：小型模型更容易被 irrelevant detail 分散注意力
4. **組合效應超過個別之和**：多種雜訊同時存在的影響 > 個別雜訊之和（交互效應）
5. **RAG retrieval 受雜訊影響顯著**：雜訊主要影響 retrieval quality（因為 embedding 受到干擾），而非 generation

---

## 醫學特有價值

1. **醫學獨有的研究貢獻**：EHR 雜訊問題在其他 AI 領域不存在，本研究填補了 LLM 評估的重要空白
2. **直接影響臨床部署**：任何在真實醫院部署的 LLM 都必須處理 EHR 雜訊，本研究為部署可行性提供實證
3. **EHR 設計啟示**：研究結果可為 EHR 系統設計（如自動去冗餘、矛盾偵測）提供數據支持
4. **RAG 系統設計**：M5 的結果直接指導 Medical-RAG 系統的 pre-processing 改進
5. **與 BRIDG 標準的連結**：EHR 雜訊問題的解決方案可參考 BRIDG 的結構化臨床數據模型

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M4 (Counterfactual) | ↔ 互補 | M4 測試結構化擾動，M5 測試非結構化雜訊 |
| M9 (RxLLama) | → 下游 | M5 的結果指導 M9 的事前授權系統對 EHR 雜訊的處理 |
| Medical-RAG | ↔ 直接應用 | 實驗 4 直接使用現有 RAG 系統 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Tsou, A.Y., et al. (2017). Safe practices for copy and paste in the EHR. *Applied Clinical Informatics*, 8(1), 12-34.
- Hammond, K.W., et al. (2003). Are electronic medical records trustworthy? Observations on copying, pasting, and duplication. *AMIA Annual Symposium*.
- Singh, H., et al. (2013). Types and origins of diagnostic errors in primary care settings. *JAMA Internal Medicine*, 173(6), 418-425.
- Weir, C.R., et al. (2020). The role of information technology in reducing diagnostic errors. *Patient Safety Network (PSNet)*.
- Rule, A., et al. (2021). Length and redundancy of outpatient progress notes across a decade at an academic medical center. *JAMA Network Open*, 4(7).
- Shi, F., et al. (2023). Large language models can be easily distracted by irrelevant context. *ICML 2023*.

### 內部文件
- `參考/selected/` — I3 Red Herring 方法論參考
- `國科會_RxLLama/Medical-RAG-using-Bio-Mistral-7B-main/` — 現有 RAG 系統
