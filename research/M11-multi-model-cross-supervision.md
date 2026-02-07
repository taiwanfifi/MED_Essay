# M11: 多模型交叉監督
# Multi-Model Cross-Supervision for Medical AI Quality Assurance

> **層級**：基礎設施 — 品質保證機制
> **狀態**：🟡 需開發驗證 pipeline
> **Phase**：Phase 1.5（與 M10a 同步）
> **注意**：本構想與 M10b 有高度重疊，可合併或獨立發展

---

## 研究問題 (Research Problem)

醫學 AI 系統的輸出品質驗證高度依賴人類專家審核，但這在規模化時不實際。「多模型交叉監督」提出一種替代方案：讓多個不同架構、不同訓練數據的 LLM 互相審查，利用模型間的「觀點多樣性」來偵測錯誤。

這與 M10b 的差異在於：M10b 聚焦於「驗證 AI 生成的題目」，而 M11 更廣泛地探討「多模型監督作為通用品質保證機制」在醫學 AI 中的適用性。

**具體未知：**
1. 多模型共識能否作為可靠的品質代理指標（quality proxy）？
2. 在什麼條件下，多模型共識會系統性失敗？
3. 不同的共識策略（多數決 / 加權投票 / 辯論式）哪個最優？
4. 多模型監督的成本效益比 vs 人類審核為何？

**臨床重要性：**
若多模型監督能提供可靠的品質保證，將大幅降低醫學 AI 系統部署前的驗證成本，加速從研究到臨床的轉化。

---

## 核心方法 (Core Approach)

### 1. 三種共識策略

| 策略 | 方法 | 優點 | 缺點 |
|------|------|------|------|
| **多數決 (Majority Vote)** | N 模型投票，取多數 | 簡單、透明 | 不考慮模型能力差異 |
| **加權投票 (Weighted Vote)** | 按歷史準確率加權 | 考慮模型強弱 | 需要校準數據 |
| **辯論式 (Debate)** | 模型輪流提出論據並反駁 | 可暴露推理過程 | 計算成本高 |

### 2. 監督應用場景

```
場景 A: 題目品質驗證（對應 M10a/M10b）
  └─ AI 生成的題目是否醫學正確？

場景 B: 模型回答審查（對應 M1-M9）
  └─ 被評測模型的回答是否正確？

場景 C: 安全護欄（對應 M8/M9）
  └─ 即時偵測潛在危險回答

場景 D: 校準驗證（對應 M6）
  └─ 模型信心度是否合理？
```

### 3. 辯論式監督協定

```
Round 0: 每個模型獨立回答
Round 1: 每個模型看到其他模型的回答，提出支持/反對論據
Round 2: 每個模型根據論據更新判斷
Round 3: 若仍無共識，標記為「需人工審核」

記錄：
- 每輪的一致性分數變化
- 模型「改變主意」的頻率和方向
- 最終共識 vs 人類判斷的一致性
```

---

## 實驗設計 (Experimental Design)

### 實驗 1：三種共識策略比較

**設計：** 500 題 × 3 策略 × 5 模型組合

**指標：**
- 與人類判斷的 F1
- 成本（API calls / compute time）
- 「需人工」的比例（越低越好，但不能犧牲品質）

### 實驗 2：失敗模式分析

**刻意包含已知的陷阱題（100 題）：**
- 常見醫學迷思
- 最新指引更新（模型可能沒有）
- 罕見疾病
- 文化特異性題目

**分析：**
- 哪些類型最容易觸發集體幻覺？
- 辯論式是否能降低集體幻覺率？

### 實驗 3：成本效益分析

```
方案 A: 100% 人工審核         → 成本高、品質高
方案 B: 100% 多模型監督       → 成本低、品質中
方案 C: 多模型篩選 + 20% 人工 → 成本中、品質中高
方案 D: 辯論式 + 10% 人工     → 成本中、品質？

比較每方案的：
- 每題成本
- 品質（vs 100% 人工）
- 處理速度
- Scalability
```

---

## 需要的積木 (Required Building Blocks)

### 資料集
| 資源 | 規模 | 狀態 | 備註 |
|------|------|------|------|
| MedQA test set（已有正確答案） | 500 題 | ✅ 公開可得 | 黃金標準 |
| AI 生成題目（來自 M10a） | 500 題 | 🟡 需 M10a | 無標準答案 |
| 陷阱題 | 100 題 | ❌ 需建構 | 已知困難題 |

### 模型
同 M10b 配置，5 種組合共需 11 個模型。

---

## 預期產出 (Expected Outputs)

### 代碼產出
```
results/M11_consensus_strategy_comparison.csv  # 3 策略比較
results/M11_failure_mode_analysis.json         # 失敗模式
results/M11_cost_benefit_analysis.csv          # 成本效益
results/M11_debate_transcripts.json            # 辯論過程記錄
```

### 視覺化
```
figures/M11_strategy_f1_barplot.png            # 策略 F1 比較
figures/M11_cost_quality_frontier.png          # 成本-品質前沿
figures/M11_debate_consensus_trajectory.png    # 辯論過程一致性變化
figures/M11_failure_topic_heatmap.png          # 失敗主題熱力圖
```

### 學術表格
- Table 1: Three Consensus Strategy Definitions
- Table 2: Strategy Comparison Results (F1, Cost, Scalability)
- Table 3: Collective Hallucination Analysis by Topic
- Table 4: Cost-Benefit Analysis of Four QA Approaches
- Table 5: Debate Protocol Convergence Statistics

---

## 預期發現 (Expected Findings)

1. **辯論式最佳但最貴**：F1 最高，但 API 成本 3-5 倍
2. **加權投票是最佳平衡**：接近辯論式品質，成本遠低
3. **方案 C（多模型 + 20% 人工）最實用**：品質達 95% 人工水準，成本降 60%
4. **最新指引是最大盲點**：模型知識截止日期後的更新是集體幻覺主因
5. **3 模型異家族組合足夠**：與 M10b 結論一致

---

## 醫學特有價值

1. **降低驗證成本**：讓更多研究團隊能負擔大規模醫學 AI 評估
2. **即時安全監控**：多模型共識可作為即時安全護欄
3. **知識更新偵測**：辯論過程可揭示哪些醫學知識需要更新
4. **可擴展至其他醫學 AI 任務**：不限於 QA，可用於摘要、翻譯、報告生成

---

## 與 M10b 的關係

| 維度 | M10b | M11 |
|------|------|-----|
| **焦點** | 驗證 AI 生成的題目 | 通用品質保證機制 |
| **範圍** | 題目驗證 | 題目 + 回答 + 安全 + 校準 |
| **策略** | 多數決為主 | 多數決 + 加權 + 辯論 |
| **實驗** | 5 組合比較 + N-scaling | 3 策略比較 + 成本效益 |
| **投稿** | AAAI/NeurIPS/EMNLP | AMIA/JAMIA/JMIR |

**建議：** 若資源有限，M10b 和 M11 可合併為一篇更完整的論文。若分開，M10b 偏方法論（CS 會議），M11 偏應用（醫學資訊學期刊）。

---

## 可合併的點子 (Related Ideas)

| 相關構想 | 關係 | 說明 |
|---------|------|------|
| M10a (生成方法論) | ← 上游 | M11 驗證 M10a 的輸出 |
| M10b (多模型驗證) | ↔ 高度重疊 | 可合併或分投不同venue |
| M6 (校準) | ↔ 應用場景 | M11 的監督可用於校準驗證 |
| M8 (安全矩陣) | → 應用 | M11 可作為安全護欄 |
| M9 (RxLLama) | → 應用 | M11 可整合進 RxLLama 系統 |

---

## 來源筆記 (References & Sources)

### 學術文獻
- Du, Y., et al. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate. *arXiv:2305.14325*.
- Liang, T., et al. (2023). Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. *arXiv:2305.19118*.
- Chan, C., et al. (2023). ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate. *arXiv:2308.07201*.
- Cohen, R., et al. (2023). LM vs LM: Detecting Factual Errors via Cross-Examination. *EMNLP 2023*.
- Singhal, K., et al. (2023). Large Language Models Encode Clinical Knowledge. *Nature*. [Med-PaLM]

### 內部文件
- `research/M10-ai-generated-benchmark.md` — M10 傘型構想
- `research/M6-calibration-selective-prediction.md` — 校準方法論
- `research/M8-patient-safety-risk-matrix.md` — 安全分級框架
