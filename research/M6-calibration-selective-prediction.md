# M6: 臨床 LLM 信心校準
# Calibration & Selective Prediction for Clinical LLMs: How Much Can We Trust AI Confidence?

> **層級**：Layer 4 — 行為分析
> **財經對應**：D1 (Calibration & Selective Prediction)
> **狀態**：🟢 Ready — 純統計分析，使用現有 benchmark
> **Phase**：Phase 1（立即可做，門檻最低）

---

## 研究問題 (Research Problem)

臨床 AI 部署的核心問題不僅是「模型答對多少」，更是「模型知不知道自己什麼時候會答錯」。一個 80% 準確率的模型，如果能完美識別自己會答錯的 20%（並拒絕回答），就比一個 90% 準確率但無法辨別自身錯誤的模型更適合臨床使用。

**校準（Calibration）** 衡量的是：模型表達 70% 信心時，是否真的有 70% 的概率是正確的？

**選擇性預測（Selective Prediction）** 回答的是：如果模型只回答它有信心的問題，能達到多高的準確率？要達到 95% 準確率（臨床可接受門檻），模型需要拒絕多少比例的問題？

**這兩個問題對臨床部署至關重要：**
1. 過度校準（under-confident）→ 過多問題轉交人工，系統效率低
2. 校準不足（over-confident）→ 高信心的錯誤答案 → 病患安全風險
3. 安全關鍵領域（藥理學、急診）的校準是否比一般領域更差？

**醫學特殊性：** 不是所有錯誤都一樣嚴重。藥理學錯誤可能致命，解剖學知識缺口通常不影響處置。因此需要 **Safety-Weighted ECE**：藥理學和急診的校準誤差應有更高的權重。

---

## 核心方法 (Core Approach)

### 1. 四種信心估計方法 (Four Confidence Estimation Methods)

#### Method 1: Verbalized Confidence（語言化信心）

**Prompt 設計：**
```
Answer the following medical question. After your answer, state your confidence
level as a percentage (0-100%).

Question: [question]

Format:
Answer: [your answer]
Confidence: [X]%
```

**優點：** 簡單、通用、不需 logit access
**缺點：** 模型可能不誠實、受 prompt 設計影響

#### Method 2: Self-Consistency（自我一致性）

**方法：**
```
For each question:
  1. Run model k=10 times with temperature=0.7
  2. Collect 10 answers: {a_1, a_2, ..., a_10}
  3. Confidence = frequency of most common answer / k
     e.g., if 7/10 runs give same answer → confidence = 70%
```

**優點：** 不依賴模型自報、理論基礎強（Wang et al., 2023）
**缺點：** 計算成本高（10× per question）、受 temperature 影響

#### Method 3: Multi-Model Ensemble（多模型集成）

**方法：**
```
For each question:
  1. Run n models (e.g., 4 models) with temperature=0
  2. Collect n answers
  3. Confidence = agreement rate among models
     e.g., if 3/4 models agree → confidence = 75%
```

**選用模型組合：**
- Ensemble A（大型）：GPT-4o + Claude 3.5 + Qwen-32B + DeepSeek-R1-14B
- Ensemble B（小型）：Llama-8B + BioMistral-7B + Phi-3.5 + Med42

#### Method 4: Logit-based Confidence（基於 Logit 的信心）

**方法（僅限 local models with logit access）：**
```
For each question:
  1. Extract logit/probability for the chosen answer token
  2. Confidence = softmax probability of selected answer
  3. For MCQ: confidence = P(selected_option)
  4. For open-ended: confidence = geometric mean of token probabilities
```

**適用模型：** Ollama local models（Llama, Qwen, DeepSeek, Phi, BioMistral）
**不適用：** Cloud models（GPT-4o, Claude — 無 logit access）

### 2. 校準指標 (Calibration Metrics)

#### Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |\text{acc}(b) - \text{conf}(b)|$$

- 將預測信心分成 B=10 個等寬 bin
- $n_b$ = 第 b 個 bin 中的樣本數
- $\text{acc}(b)$ = 第 b 個 bin 的實際準確率
- $\text{conf}(b)$ = 第 b 個 bin 的平均信心
- 範圍 0-1，0 = 完美校準

#### Maximum Calibration Error (MCE)

$$\text{MCE} = \max_{b \in \{1,...,B\}} |\text{acc}(b) - \text{conf}(b)|$$

- 最差的單一 bin，衡量最嚴重的校準偏差

#### Brier Score

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (\text{conf}_i - \text{correct}_i)^2$$

- $\text{correct}_i \in \{0, 1\}$
- 範圍 0-1，0 = 完美
- 同時懲罰不準確和校準不良

#### Safety-Weighted ECE（本研究創新指標）

$$\text{SW-ECE} = \sum_{b=1}^{B} \frac{\sum_{i \in b} w_i}{\sum_{i} w_i} |\text{acc}(b) - \text{conf}(b)|$$

其中安全權重 $w_i$ 按醫學子領域設定：

| 子領域 | 安全權重 $w$ | 理由 |
|--------|-------------|------|
| 藥理學 (Pharmacology) | 3.0 | 用藥錯誤可能致命 |
| 急診醫學 (Emergency Med) | 3.0 | 延誤處置可能致命 |
| 內科 (Internal Medicine) | 2.0 | 慢性病管理影響大 |
| 外科 (Surgery) | 2.0 | 手術決策影響大 |
| 小兒科 (Pediatrics) | 2.5 | 兒童劑量計算關鍵 |
| 婦產科 (OB/GYN) | 2.5 | 孕期用藥安全 |
| 基礎醫學 (Basic Sciences) | 1.0 | 通常不直接影響處置 |
| 其他 | 1.5 | 預設中等權重 |

### 3. 選擇性預測框架 (Selective Prediction Framework)

**核心問題：** 在 X% 的準確率門檻下，模型能回答多少比例的問題？

**Coverage-Accuracy Tradeoff：**

$$\text{Coverage}(\tau) = \frac{|\{i : \text{conf}_i \geq \tau\}|}{N}$$

$$\text{Accuracy}(\tau) = \frac{|\{i : \text{conf}_i \geq \tau \wedge \text{correct}_i = 1\}|}{|\{i : \text{conf}_i \geq \tau\}|}$$

- $\tau$ = 信心門檻
- 繪製 Coverage vs Accuracy 曲線
- 找到 Accuracy = 95% 時的 Coverage → 臨床部署的實用指標

**AUROC for Confidence：**

$$\text{AUROC} = P(\text{conf}(\text{correct}) > \text{conf}(\text{incorrect}))$$

- 衡量信心是否能有效區分正確和錯誤回答
- 高 AUROC → 信心可作為可靠的品質過濾器

---

## 實驗設計 (Experimental Design)

### 實驗 1：校準評估（主要實驗）

**設計：** 4 methods × 8 models × 3 datasets

**流程：**
```
For each model M in {8 models}:
  For each dataset D in {MedQA, MedMCQA, MMLU-Med}:
    For each question Q in D:
      1. Method 1 (Verbalized): Run with confidence prompt → conf_verb
      2. Method 2 (Self-Consistency): Run 10× at temp=0.7 → conf_sc
      3. Method 3 (Ensemble): Run 4-model ensemble → conf_ens
      4. Method 4 (Logit): Extract logprobs (if available) → conf_logit
    Compute per method:
      - ECE, MCE, Brier Score
      - SW-ECE
      - Reliability Diagram (10-bin)
      - AUROC
```

**推論次數：**
- Method 1: 6,256 × 8 = 50,048
- Method 2: 6,256 × 10 × 8 = 500,480（最大成本項）
- Method 3: 6,256 × 4 (or 8) = 25,024 (or 50,048)
- Method 4: included in Method 1 runs for local models
- **總計：~575,000+ 次推論**

**成本控制策略：**
- Method 2 可先對 MedQA (1,273) 做完整 10 runs，MedMCQA 用 k=5
- Cloud models 只做 Method 1 + Method 3
- Local models 做全部 4 methods

### 實驗 2：子領域校準分析

**按醫學子領域拆解校準指標：**

```
For each model × method:
  For each medical subtopic T:
    Compute ECE(T), SW-ECE(T), AUROC(T)
  Generate: Topic × Model ECE 熱力圖
  Identify: 校準最差的子領域
```

**假設檢驗：**
- H1：藥理學的 ECE > 基礎醫學的 ECE（藥理學更容易過度自信）
- H2：困難子領域的過度自信更嚴重
- H3：醫學特化模型在醫學子領域的校準比通用模型好

### 實驗 3：Coverage-Accuracy Tradeoff

**核心臨床部署指標：**

```
For each model × method:
  1. Sort questions by confidence (descending)
  2. For τ in [0.0, 0.05, 0.10, ..., 1.0]:
     Compute Coverage(τ) and Accuracy(τ)
  3. Plot Coverage-Accuracy curve
  4. Find τ* where Accuracy(τ*) = 0.95
  5. Report Coverage(τ*) = "at 95% accuracy, model can answer X% of questions"
```

**臨床解讀示例：**
- 「GPT-4o 在 95% 準確率門檻下可自動回答 MedQA 中 62% 的問題」
- 「BioMistral-7B 在相同門檻下只能自動回答 28% 的問題」
- → 直接指導部署決策

### 實驗 4：信心方法比較

**比較 4 種信心估計方法的品質：**

| 評比維度 | 衡量方式 |
|---------|---------|
| 校準品質 | ECE, MCE, Brier Score 排名 |
| 區分能力 | AUROC 排名 |
| 選擇性預測效能 | Coverage@95% Accuracy |
| 計算成本 | API calls / inference time |
| 可用性 | 需要 logit access? |

**Recommendation Matrix：** 根據使用場景推薦最佳方法
- 臨床部署（real-time）→ 優先考慮 Method 1 或 4
- 離線批次評估 → Method 2 提供最可靠的信心
- 多模型可用 → Method 3

### 實驗 5：「過度自信且錯誤」案例分析

**提取最危險的案例：High Confidence + Wrong Answer**

```
Dangerous cases = {q : conf(q) > 0.8 AND correct(q) = 0}

For each dangerous case:
  1. Record: question, model answer, correct answer, confidence, topic
  2. Categorize: why was the model overconfident?
     a. Plausible but wrong (close distractor)
     b. Knowledge gap masked by fluency
     c. Systematic misconception
     d. Outdated knowledge with high certainty
  3. Assess clinical severity (4-level from M8)
```

**這些案例直接輸入 M8（Patient Safety Risk Matrix）**

---

## 需要的積木 (Required Building Blocks)

### 資料集
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| MedQA USMLE Test | 1,273 | ✅ 已就緒 | 主要資料集 |
| MedMCQA Test | 4,183 | ✅ 已就緒 | 大規模測試 |
| MMLU-Med (6 tasks) | ~800 | ✅ 已就緒 | 補充資料集 |
| PubMedQA | 1,000 | ✅ 公開可得 | yes/no/maybe 格式，校準天然適合 |

### 模型（含 logit 需求）
| 模型 | Logit Access | Methods 可用 | 狀態 |
|------|-------------|-------------|------|
| GPT-4o | ❌ | 1, 2, 3 | ✅ |
| GPT-4o-mini | ❌ | 1, 2, 3 | ✅ |
| Claude 3.5 | ❌ | 1, 2, 3 | ✅ |
| Llama 3.1 8B | ✅ (Ollama logprobs) | 1, 2, 3, 4 | ✅ |
| Qwen 2.5 32B | ✅ | 1, 2, 3, 4 | ✅ |
| DeepSeek-R1 14B | ✅ | 1, 2, 3, 4 | ✅ |
| BioMistral-7B | ✅ (llama.cpp) | 1, 2, 3, 4 | ✅ |
| Med42-v2 | ✅ | 1, 2, 3, 4 | ❌ 需下載 |

### 工具
| 工具 | 用途 | 狀態 |
|------|------|------|
| netcal (Python) | 校準分析庫 | ❌ 需安裝 |
| scikit-learn | AUROC, reliability diagram | ✅ |
| matplotlib + seaborn | 視覺化 | ✅ |

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
results/M6_calibration_metrics.csv               # ECE, MCE, Brier per model×method×dataset
results/M6_swece_by_topic.csv                    # SW-ECE per model×topic
results/M6_coverage_accuracy.csv                 # Coverage-Accuracy curve data
results/M6_auroc.csv                             # AUROC per model×method
results/M6_overconfident_wrong_cases.json        # High-conf wrong answer cases
results/M6_method_comparison.csv                 # 4-method comparison table
```

### 視覺化
```
figures/M6_reliability_diagrams/                  # 8 models × 4 methods = 32 diagrams
figures/M6_ece_heatmap_model_x_topic.png         # Model × Topic ECE 熱力圖
figures/M6_coverage_accuracy_curves.png          # Coverage-Accuracy overlay
figures/M6_swece_vs_ece_comparison.png           # SW-ECE vs standard ECE
figures/M6_method_comparison_radar.png           # 4-method 比較雷達圖
figures/M6_overconfident_distribution.png        # 過度自信案例分布
```

### 學術表格
- Table 1: Calibration Metrics (ECE, MCE, Brier) by Model, Method, and Dataset
- Table 2: Safety-Weighted ECE by Medical Subdomain
- Table 3: Coverage at 95% Accuracy Threshold by Model and Method
- Table 4: AUROC for Confidence-Correctness Discrimination
- Table 5: Confidence Estimation Method Comparison (Quality, Cost, Applicability)
- Table 6: High-Confidence Wrong Answer Analysis (Top 20 Cases)

---

## 預期發現 (Expected Findings)

1. **所有模型都過度自信**：ECE 預期 > 0.10（理想值 0），顯示系統性的 over-confidence
2. **藥理學校準最差**：藥理學子領域的 ECE 預期最高，SW-ECE 更加突出差異
3. **Self-Consistency 校準最佳**：Method 2 的 ECE 預期最低，但計算成本最高
4. **Coverage 差異大**：GPT-4o 在 95% 準確率下可能覆蓋 50-65%，BioMistral 可能只有 20-30%
5. **Verbalized ≠ True Confidence**：模型自報信心與基於 logit 的信心可能差異顯著
6. **過度自信錯誤集中在特定領域**：高信心+錯誤的案例預期集中在「模型似乎知道但實際過時」的知識領域

---

## 醫學特有價值

1. **部署門檻設定**：Coverage@95% 直接回答「部署這個模型，多少比例的臨床問題可以自動回答？」
2. **Safety-Weighted ECE 創新**：首次在 LLM 校準研究中引入臨床嚴重度加權
3. **風險分流依據**：校準良好的模型可用於低風險問題自動回答，高風險問題轉交人工
4. **直接連結 M8**：過度自信且錯誤的案例直接輸入 M8 的風險矩陣
5. **RAG 系統信心整合**：校準結果可指導現有 Medical-RAG 系統是否/如何展示信心分數

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M1 (Open-Ended) | ↔ 共用資料集 | M1 和 M6 使用相同底層 benchmark |
| M3 (Error Atlas) | ← 提供信心 | M6 的信心數據補充 M3 的「E2 虛假確信」分析 |
| M8 (Safety Matrix) | → 直接下游 | M6 的過度自信案例直接輸入 M8 |
| M9 (RxLLama) | → 下游 | M6 的校準方法應用於 M9 的評估重設計 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Guo, C., et al. (2017). On calibration of modern neural networks. *ICML 2017*.
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. *arXiv:2207.05221*.
- Wang, X., et al. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*.
- Naeini, M.P., et al. (2015). Obtaining well calibrated probabilities using Bayesian binning into quantiles. *AAAI 2015*.
- Nori, H., et al. (2023). Can Generalist Foundation Models Outcompete Special-Purpose Tuning? *arXiv:2311.16452*.
- Tian, K., et al. (2023). Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models with optimized prompting. *arXiv:2305.14975*.
- Singhal, K., et al. (2023). Large Language Models Encode Clinical Knowledge. *Nature*.

### 內部文件
- `參考/selected/D1-calibration-selective-prediction.md` — 財經版校準分析方法論

### 工具
- netcal: https://github.com/EFS-OpenSource/calibration-framework
- Ollama logprobs API: https://github.com/ollama/ollama/blob/main/docs/api.md
