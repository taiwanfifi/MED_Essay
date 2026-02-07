# M1: 開放式臨床推理基準
# Open-Ended Clinical Reasoning Benchmark: Quantifying Option Bias in Medical LLMs

> **層級**：Layer 1 — 表面性能評估
> **財經對應**：A1 (Open-Ended Numerical) + A5 (MCQ Option Bias)
> **狀態**：🟢 Ready — 資料集公開可得，實驗設計明確
> **Phase**：Phase 1（立即可做）

---

## 研究問題 (Research Problem)

醫學 LLM 的能力評估幾乎完全建立在多選題（MCQ）格式之上。MedQA、MedMCQA、MMLU-Med 等主流基準均提供 4-5 個選項，模型只需從中擇一。這引發一個根本性的方法論問題：**模型展現的究竟是真正的臨床推理能力，還是在候選答案中進行模式匹配與排除法的能力？**

在真實臨床場景中，醫師面對的是開放式問題：「這個病人可能是什麼診斷？」「應該開什麼藥？」「下一步檢查做什麼？」沒有 ABCDE 選項供選擇。如果 LLM 在失去選項拐杖後表現大幅下降，我們對其臨床能力的評估就存在系統性高估。

**具體未知：**
1. 去掉選項後，各模型在不同醫學領域的準確率下降幅度為何？
2. Option bias 是否隨模型規模增大而減少？
3. 不同醫學子領域（藥理學 vs 解剖學 vs 臨床推理）的 option bias 是否有系統性差異？
4. 開放式格式下，模型的錯誤模式是否與 MCQ 格式不同？

**臨床重要性：**
若開放式臨床推理能力遠低於 MCQ 表現，則現有「GPT-4 通過美國醫師執照考試」等宣稱需要重新審視，臨床部署決策也需更加謹慎。

---

## 核心方法 (Core Approach)

### 1. MCQ → Open-Ended 轉換協定

將現有 MCQ 題目轉換為開放式格式：

**原始 MCQ 格式：**
```
Q: A 65-year-old male presents with sudden-onset chest pain radiating to the left arm,
   diaphoresis, and shortness of breath. ECG shows ST-elevation in leads II, III, aVF.
   What is the most likely diagnosis?
   A) Pulmonary embolism
   B) Acute inferior STEMI
   C) Aortic dissection
   D) Pericarditis
```

**轉換後 Open-Ended 格式：**
```
Q: A 65-year-old male presents with sudden-onset chest pain radiating to the left arm,
   diaphoresis, and shortness of breath. ECG shows ST-elevation in leads II, III, aVF.
   What is the most likely diagnosis? Provide your diagnosis directly.
```

**轉換規則：**
- 移除所有選項（A/B/C/D/E）
- 保留完整題幹不做任何修改
- 添加開放式指令：「Provide your answer directly」或「State your diagnosis/treatment/next step」
- 不提供任何提示或答案格式限制

### 2. 三層判斷機制 (Three-Tier Judgment System)

MCQ 的自動評分是二元的（選對/選錯），但開放式回答需要更細緻的判斷：

| 層級 | 定義 | 範例 |
|------|------|------|
| **Level A：臨床正確 (Clinically Correct)** | 與標準答案語義等同，臨床上可接受 | 答案：Inferior STEMI → 正確 |
| **Level B：部分正確 (Partially Correct)** | 方向正確但不夠精確，或包含正確答案但附帶錯誤資訊 | 答案：Myocardial infarction（正確但不夠精確，未指出 inferior） |
| **Level C：臨床錯誤 (Clinically Incorrect)** | 與標準答案臨床意義不同，可能導致錯誤處置 | 答案：Pulmonary embolism → 錯誤 |

**自動判斷管線：**

```
Step 1: SNOMED CT 語義匹配
  - 將模型回答與標準答案映射至 SNOMED CT 概念
  - 計算語義距離（共同祖先、階層距離）
  - 距離 ≤ 2 → Level A 候選
  - 距離 3-5 → Level B 候選
  - 距離 > 5 → Level C 候選

Step 2: GPT-4o 臨床判斷（作為仲裁者）
  - 輸入：題目 + 標準答案 + 模型回答 + SNOMED 匹配結果
  - Prompt: "As a clinical expert, judge whether this answer is
    (A) clinically correct and actionable,
    (B) partially correct but imprecise, or
    (C) clinically incorrect and potentially harmful.
    Provide reasoning."
  - 使用 structured output 確保格式一致

Step 3: 人工驗證（抽樣）
  - 隨機抽取 200 題（按 Level A/B/C 分層抽樣）
  - 2 位臨床醫師獨立判斷
  - 計算 Cohen's Kappa（目標 > 0.70）
  - 若 Kappa < 0.65，修正 GPT-4o prompt 後重新標註
```

### 3. Option Bias 量化

**核心指標：**

$$\text{Option Bias} = \text{Acc}_{\text{MCQ}} - \text{Acc}_{\text{Open-Ended}}$$

其中：
- $\text{Acc}_{\text{MCQ}}$ = 模型在原始 MCQ 格式下的準確率
- $\text{Acc}_{\text{Open-Ended}}$ = 模型在開放式格式下 Level A 的比例

**進階指標：**

$$\text{Adjusted Option Bias} = \text{Acc}_{\text{MCQ}} - (\text{Level A} + 0.5 \times \text{Level B})$$

給予部分正確回答 50% 權重，更公平地反映開放式推理能力。

$$\text{Relative Option Bias} = \frac{\text{Acc}_{\text{MCQ}} - \text{Acc}_{\text{Open-Ended}}}{\text{Acc}_{\text{MCQ}}} \times 100\%$$

表示 MCQ 表現中有多少比例來自選項的「拐杖效應」。

---

## 實驗設計 (Experimental Design)

### 實驗 1：MCQ vs Open-Ended 準確率比較

**設計：**
- 每個模型對每道題分別在 MCQ 和 Open-Ended 兩種格式下作答
- 所有模型使用 temperature=0，確保確定性輸出
- 記錄完整回答文本供後續分析

**流程：**
```
For each model M in {GPT-4o, GPT-4o-mini, Claude 3.5, Llama3.1-8B, Qwen2.5-32B,
                      DeepSeek-R1-14B, BioMistral-7B, Med42-v2}:
  For each question Q in dataset:
    1. Run Q in MCQ format → Record answer_MCQ, correct_MCQ
    2. Run Q in Open-Ended format → Record answer_OE, full_text_OE
    3. Judge answer_OE via Three-Tier System → Level A/B/C
  Compute:
    - Acc_MCQ per dataset per topic
    - Acc_OE (Level A only) per dataset per topic
    - Adjusted_Acc_OE (Level A + 0.5 * Level B)
    - Option_Bias = Acc_MCQ - Acc_OE
```

**統計檢驗：**
- McNemar's test：對每對 (MCQ_correct, OE_correct) 進行配對檢驗
- 效果量：Cohen's h for proportions
- 多重比較校正：Bonferroni correction（8 models × 3 datasets = 24 comparisons）

### 實驗 2：Option Bias 跨領域分析

**按醫學子領域拆解 Option Bias：**

| 資料集 | 子領域分類 |
|--------|-----------|
| MedQA (USMLE) | Anatomy, Biochemistry, Pharmacology, Pathology, Microbiology, Behavioral Science, Physiology, Internal Medicine, Surgery, Pediatrics, OB/GYN, Psychiatry |
| MedMCQA | Anatomy, Physiology, Biochemistry, Pharmacology, Pathology, Microbiology, Forensic Medicine, Community Medicine, Ophthalmology, ENT, Radiology, Orthopedics, Surgery, Medicine, OB/GYN, Pediatrics, Dermatology, Psychiatry, Anesthesia |
| MMLU-Med | Clinical Knowledge, Medical Genetics, Anatomy, Professional Medicine, College Biology, College Medicine |

**分析：**
- 計算每個子領域的 Option Bias
- 生成 Option Bias 熱力圖（Model × Medical Topic）
- 識別 bias 最高/最低的子領域

**假設檢驗：**
- H1：藥理學（需要精確藥名回憶）的 Option Bias > 病理學（概念推理為主）
- H2：臨床推理題的 Option Bias < 記憶型知識題
- H3：Option Bias 隨模型規模增大而減少

### 實驗 3：Option Bias vs 模型規模關係

**模型規模梯度：**
```
3.8B (Phi-3.5) → 7B (BioMistral) → 8B (Llama3.1) → 14B (DeepSeek-R1)
→ 32B (Qwen2.5) → ~200B (GPT-4o) → ~200B (Claude 3.5)
```

**分析：**
- 繪製 Model Size (log scale) vs Option Bias 散點圖
- 擬合對數回歸：$\text{Option Bias} = a \cdot \ln(\text{params}) + b$
- 計算 R² 判斷規模效應的解釋力
- 分別對醫學特化模型（BioMistral, Med42）和通用模型做比較

### 實驗 4：開放式回答的錯誤模式分析

**分析 Level B 和 Level C 回答的錯誤類型：**

| 錯誤類型 | 定義 | 範例 |
|---------|------|------|
| 精確度不足 (Imprecision) | 方向正確但概念層級過高 | 「Heart attack」而非「Inferior STEMI」 |
| 替代診斷 (Alternative Dx) | 鑑別診斷清單中的其他項目 | 將 STEMI 答成 Pericarditis |
| 幻覺 (Hallucination) | 產出不存在的醫學概念 | 虛構藥名或疾病名 |
| 過度解讀 (Over-interpretation) | 添加題目未給的資訊後推論 | 假設檢驗結果後給出過度具體的診斷 |
| 拒絕作答 (Refusal) | 表示無法確定或需要更多資訊 | 「I cannot determine without more information」 |

**生成：**
- 錯誤類型分布長條圖（per model）
- MCQ 錯誤 vs Open-Ended 錯誤 Sankey 圖（追蹤同一題在兩種格式下的表現遷移）

### 實驗 5：SNOMED CT 語義距離分析

**量化開放式回答與標準答案的語義距離：**

$$\text{Semantic Distance}(a, b) = \text{shortest\_path}(\text{SNOMED}(a), \text{SNOMED}(b))$$

**分析：**
- Level A/B/C 的平均語義距離分布
- 語義距離 vs 臨床嚴重度交叉分析
- 不同模型的語義距離分布比較（violin plot）

---

## 需要的積木 (Required Building Blocks)

### 資料集
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| MedQA (USMLE) test set | 1,273 題 | ✅ 公開可得 | Jin et al. 2021, HuggingFace |
| MedMCQA test set | 4,183 題 | ✅ 公開可得 | Pal et al. 2022, HuggingFace |
| MMLU-Med (6 subtasks) | ~800 題 | ✅ 公開可得 | Hendrycks et al. 2021 |
| SNOMED CT Browser | - | ✅ 免費瀏覽版 | 語義匹配用 |
| UMLS Metathesaurus | - | ✅ 需申請帳號 | 概念映射備用 |

### 模型
| 模型 | 存取方式 | 狀態 |
|------|---------|------|
| GPT-4o | OpenAI API | ✅ 可用 |
| GPT-4o-mini | OpenAI API | ✅ 可用 |
| Claude 3.5 Sonnet | Anthropic API | ✅ 可用 |
| Llama 3.1 8B | Ollama local | ✅ 可用 |
| Qwen 2.5 32B | Ollama local | ✅ 可用 |
| DeepSeek-R1 14B | Ollama local | ✅ 可用 |
| BioMistral-7B | Local GGUF | ✅ 已有（RAG 系統） |
| Med42-v2 | Ollama/HF | ❌ 需下載 |

### 工具
| 工具 | 用途 | 狀態 |
|------|------|------|
| Python + pandas | 資料處理 | ✅ |
| matplotlib + seaborn | 視覺化 | ✅ |
| scikit-learn | 統計檢驗 | ✅ |
| SNOMED CT API / pymedtermino | 語義匹配 | ❌ 需設定 |

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
results/M1_mcq_vs_openended_accuracy.json      # 全模型 × 全資料集準確率
results/M1_three_tier_distribution.json          # Level A/B/C 分布
results/M1_option_bias_by_topic.csv              # 子領域 Option Bias 矩陣
results/M1_error_type_distribution.json          # 錯誤類型分布
results/M1_snomed_semantic_distance.csv          # 語義距離資料
```

### 視覺化
```
figures/M1_option_bias_heatmap.png               # Model × Topic Option Bias 熱力圖
figures/M1_accuracy_comparison_barplot.png        # MCQ vs Open-Ended 並列長條圖
figures/M1_error_migration_sankey.png             # MCQ→OE 錯誤遷移 Sankey 圖
figures/M1_model_size_vs_bias.png                # 模型規模 vs Option Bias 散點圖
figures/M1_semantic_distance_violin.png           # 語義距離 violin plot
```

### 學術表格
- Table 1: MCQ vs Open-Ended Accuracy by Model and Dataset
- Table 2: Option Bias by Medical Subdomain (Top 10 highest/lowest)
- Table 3: Three-Tier Judgment Distribution (Level A/B/C) by Model
- Table 4: Error Type Distribution in Open-Ended Responses
- Table 5: Inter-rater Agreement (Cohen's Kappa) for Human Validation

---

## 資料需求 (Data Requirements)

| 資料集 | 題數 | 用途 | 格式 | 狀態 |
|--------|------|------|------|------|
| MedQA USMLE Test | 1,273 | 主要基準 | JSON (question, options, answer, meta) | ✅ 已就緒 |
| MedMCQA Test | 4,183 | 大規模基準 | JSON (question, opa-opd, cop, subject) | ✅ 已就緒 |
| MMLU-Med (6 tasks) | ~800 | 補充基準 | CSV (question, A, B, C, D, answer) | ✅ 已就緒 |
| **合計** | **~6,256** | | | |

**推論量估算：**
- 每題 2 次推論（MCQ + Open-Ended）× 8 模型 = 16 次 / 題
- 總推論次數：6,256 × 16 = **~100,096 次**
- 三層判斷（GPT-4o）：6,256 × 8 = **~50,048 次**（僅 Open-Ended 需要判斷）
- API 成本估算：Cloud models ~$80-150, Judge calls ~$50-80

---

## 模型需求 (Model Requirements)

### Cloud Models
| 模型 | API | temperature | max_tokens | 備註 |
|------|-----|-------------|------------|------|
| GPT-4o (gpt-4o-2024-08-06) | OpenAI | 0 | 512 | 主要評測 |
| GPT-4o-mini | OpenAI | 0 | 512 | 中階比較 |
| Claude 3.5 Sonnet | Anthropic | 0 | 512 | 頂級比較 |

### Local Models (Ollama)
| 模型 | VRAM 需求 | temperature | 備註 |
|------|-----------|-------------|------|
| llama3.1:8b | ~6GB | 0 | 通用基線 |
| qwen2.5:32b | ~20GB | 0 | 中大型模型 |
| deepseek-r1:14b | ~10GB | 0 | 推理特化 |
| phi3.5:3.8b | ~3GB | 0 | 小型模型下限 |

### Medical-Specialized
| 模型 | 來源 | 備註 |
|------|------|------|
| BioMistral-7B (Q4_K_M) | Local GGUF | 現有 RAG 系統模型 |
| Med42-v2-8B | Ollama | 需額外下載 |

---

## 預期發現 (Expected Findings)

1. **Option Bias 普遍存在**：預期所有模型在開放式格式下準確率下降 10-30%，小型模型下降更多
2. **子領域差異顯著**：藥理學（需回憶精確藥名）的 Option Bias 預期 > 病理學（概念推理）
3. **規模效應**：大型模型的 Option Bias 較小，但醫學特化模型可能打破此趨勢
4. **Level B 比例可觀**：預期 15-25% 的回答為「部分正確」，表明 MCQ 的二元評分大幅低估模型的臨床相關知識
5. **幻覺率低但危險**：開放式格式下預期 2-5% 的回答包含醫學幻覺，這是 MCQ 格式無法偵測的安全隱患

---

## 醫學特有價值

1. **臨床現實度**：真實臨床場景無選項，本研究評估更接近部署場景的能力
2. **SNOMED CT 整合**：首次在 LLM 醫學基準中引入標準化醫學本體作為語義匹配工具
3. **三層判斷**：比 MCQ 二元評分更能捕捉「知道方向但不夠精確」的臨床價值
4. **部署決策支持**：為「模型是否適合臨床部署」提供比 MCQ 分數更可靠的依據
5. **跨科別分析**：識別哪些醫學領域最容易被 MCQ 高估，指導優先改進方向

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M3 (Error Atlas) | → 下游 | M1 的開放式錯誤直接輸入 M3 的錯誤分類 |
| M6 (Calibration) | ↔ 共用資料 | M1 和 M6 使用相同底層資料集 |
| M4 (Counterfactual) | → 下游 | M1 建立的基線用於 M4 的擾動比較 |
| M9 (RxLLama) | → 下游 | M1 的 Option Bias 結果指導 M9 的評估重設計 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Jin, Q., et al. (2021). What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams. *Applied Sciences*. [MedQA]
- Pal, A., et al. (2022). MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical Domain Question Answering. *CHIL 2022*. [MedMCQA]
- Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR 2021*. [MMLU]
- Nori, H., et al. (2023). Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine. *arXiv:2311.16452*. [GPT-4 Medical]
- Singhal, K., et al. (2023). Large Language Models Encode Clinical Knowledge. *Nature*. [Med-PaLM]
- Labrak, Y., et al. (2024). BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains. *arXiv:2402.10373*.

### 內部文件
- `參考/selected/A1-open-ended-numerical.md` — 財經版開放式推理基準設計
- `參考/selected/A5-mcq-option-bias.md` — MCQ Option Bias 量化方法論

### 標準
- SNOMED CT International (2024). SNOMED CT Browser. https://browser.ihtsdotools.org/
- UMLS Metathesaurus. https://www.nlm.nih.gov/research/umls/
