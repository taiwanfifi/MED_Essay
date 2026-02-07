# MedEval-X：醫療 AI 研究構想庫
# MedEval-X: A Systematic Evaluation Framework for Medical LLMs

> **機構**：臺北醫學大學 醫學資料工程
> **定位**：Inference-only 評估框架 — 不依賴 fine-tuning，全部使用 RAG、prompting、evaluation framework、adversarial testing
> **方法論來源**：借鏡財經 AI 研究（`參考/selected/`）的 contrastive design、theory grounding、risk-aware metrics、adversarial testing
> **醫學核心**：以臨床決策、病患安全、循證醫學為三大支柱

---

## 洋蔥剝皮敘事 (Story Arc)

本研究計畫採用六層遞進結構，從表面性能逐步深入至系統整合：

```
Layer 1 — 表象：LLM 在醫學推理上真的有那麼強嗎？
    │
    │   M1: 去掉選項後，醫學推理能力下降多少？
    │   M2: 模型是否尊重 EBM 證據等級？
    │
    ▼
Layer 2 — 診斷：它們到底錯在哪裡？怎麼錯的？
    │
    │   M3: 3D 臨床錯誤分類學（錯誤類型 × 科別 × 推理階段）
    │
    ▼
Layer 3 — 穩健性：是真的推理還是背題？能處理真實病歷的雜訊嗎？
    │
    │   M4: 反事實擾動 → 記憶 vs 推理
    │   M5: 真實 EHR 雜訊（copy-paste、衝突、時序模糊）
    │
    ▼
Layer 4 — 行為：它有自信嗎？信心可靠嗎？有認知偏誤嗎？
    │
    │   M6: 校準分析 + 選擇性預測（部署門檻）
    │   M7: 6 種臨床認知偏誤系統性測試
    │
    ▼
Layer 5 — 安全：對病人安全有什麼影響？法規怎麼管？
    │
    │   M8: 風險矩陣 + 集體幻覺分析 + 法規對應
    │
    ▼
Layer 6 — 整合：如何用這些發現升級現有系統？
    │
    │   M9: RxLLama 與事前授權系統多維度升級
    │
    ▼
    完整的臨床 AI 評估生態系統
```

---

## 研究構想總覽

| ID | 標題 | 層級 | 財經對應 | 核心問題 | 資料量 |
|----|------|------|----------|----------|--------|
| M1 | 開放式臨床推理基準 | L1 表象 | A1+A5 | 去掉選項後推理下降多少？ | ~6,300 題 |
| M2 | 循證醫學等級敏感性 | L1 表象 | I2 | 模型是否尊重證據等級？ | 180 情境 |
| M3 | 臨床 AI 錯誤圖譜 | L2 診斷 | E1 | 錯誤的 3D 分類學 | ~2,000 錯誤樣本 |
| M4 | 反事實臨床壓力測試 | L3 穩健性 | I1 | 推理 vs 記憶 | 2,400 題 |
| M5 | 電子病歷雜訊穩健性 | L3 穩健性 | I3 | 真實 EHR 雜訊耐受度 | 1,200 runs |
| M6 | 臨床 LLM 信心校準 | L4 行為 | D1 | 部署門檻：能自動回答多少%？ | ~6,300 題 |
| M7 | 臨床認知偏誤 | L4 行為 | I2 | LLM 有醫師的認知偏誤嗎？ | 180 情境 |
| M8 | 病患安全風險矩陣 | L5 安全 | D4 | 高信心+錯誤的臨床後果？ | M6 衍生 |
| M9 | RxLLama 升級框架 | L6 整合 | 全部 | 如何用 M1-M8 升級現有系統？ | 200 題 |

---

## 數據流 (Data Flow Between Papers)

```
M1 ──accuracy data──→ M3 (錯誤分類的輸入)
M3 ──error patterns──→ M4 (哪些錯誤來自記憶？)
M6 ──calibration data──→ M8 (過度自信錯誤分析)
M7 ──bias data──→ M9 (Condition-Aware Chaining 作為 debiasing)
M1-M8 ──all metrics──→ M9 (現有系統綜合評估)
```

**共享資源：**
- M1 與 M6 共用相同底層資料集（MedQA、MedMCQA、MMLU-Med）
- M2 與 M7 共用認知偏誤理論框架（Croskerry / Kahneman）
- M3 的錯誤分類結果直接輸入 M4 的擾動設計
- M6 的校準分析直接輸入 M8 的風險矩陣

---

## 執行優先順序

### Phase 1 — 立即可做（門檻最低）
| 構想 | 理由 |
|------|------|
| **M6** (Calibration) | 純統計分析，不需建構新資料集，使用現有 benchmark + API |
| **M1** (Open-Ended) | 實驗設計清晰，重用 MedQA/MedMCQA/MMLU-Med 公開資料集 |

### Phase 2 — 核心貢獻
| 構想 | 理由 |
|------|------|
| **M3** (Error Atlas) | 需要 Phase 1 的資料作為錯誤樣本來源 |
| **M4** (Counterfactual) | 需要建構 perturbation dataset |
| **M7** (Cognitive Biases) | 需要設計 180 個臨床情境 |

### Phase 3 — 政策與整合
| 構想 | 理由 |
|------|------|
| **M8** (Safety/Regulatory) | 需要 M6 校準資料 |
| **M9** (RxLLama Upgrade) | 需要 M1-M8 方法論工具包 |
| **M2** (EBM Sensitivity) | 獨立但受益於 M7 方法論 |

### Phase 4 — 延伸
| 構想 | 理由 |
|------|------|
| **M5** (EHR Noise) | 最具獨創性，醫學獨有，但需要最多情境建構 |

---

## 發表目標

| 構想 | 目標期刊/會議 | 論文類型 | 預計貢獻 |
|------|-------------|----------|----------|
| M1 | ACL/EMNLP Clinical NLP Workshop | Benchmark 論文 | 開放式醫學推理基準 + option bias 量化 |
| M2 | JAMIA / Journal of Biomedical Informatics | 醫學資訊學期刊 | EBM 證據等級敏感性框架 |
| M3 | Nature Scientific Data / ACL | Resource 論文 | 3D 臨床錯誤分類學 |
| M4 | NeurIPS / ICLR AI Safety Workshop | 方法論論文 | 反事實臨床壓力測試協定 |
| M5 | JAMIA / AMIA Annual Symposium | 臨床資訊學 | EHR 雜訊穩健性基準 |
| M6 | AAAI / AISTATS | 統計分析論文 | 臨床 LLM 校準框架 |
| M7 | Lancet Digital Health | 跨領域論文 | LLM 臨床認知偏誤圖譜 |
| M8 | npj Digital Medicine | 政策論文 | 病患安全風險矩陣 + 法規建議 |
| M9 | JBI / JMIR Medical Informatics | 系統論文 | RxLLama 多維度升級框架 |

---

## 共用模型配置

### Cloud Models（透過 API）
| 模型 | 用途 | 備註 |
|------|------|------|
| GPT-4o | 主要評測對象 + M3 自動分類器 | temperature=0 |
| GPT-4o-mini | 中階模型比較 | 成本效益比較 |
| Claude 3.5 Sonnet | 頂級推理能力比較 | 長文本優勢 |

### Local Models（透過 Ollama）
| 模型 | 參數量 | 用途 |
|------|--------|------|
| Llama 3.1 8B | 8B | 小型模型基線 |
| Qwen 2.5 32B | 32B | 中大型模型 |
| DeepSeek-R1 14B | 14B | 推理特化模型 |
| Phi-3.5 3.8B | 3.8B | 小型模型下限 |

### 醫學特化模型
| 模型 | 來源 | 用途 |
|------|------|------|
| BioMistral-7B | 現有 RAG 系統 | 醫學特化基線 |
| Med42-v2 | Llama 3 醫學微調 | 醫學開源比較 |
| OpenBioLLM | Llama 3 生物醫學 | 領域特化分析 |

---

## 理論基礎

本研究計畫植根於以下理論框架：

1. **循證醫學 (Evidence-Based Medicine)**
   - Sackett et al. (1996) EBM 證據等級金字塔
   - GRADE Working Group (2004) 證據品質評估框架

2. **臨床認知偏誤**
   - Croskerry (2002) 急診醫學認知偏誤分類
   - Kahneman (2011) System 1/2 雙系統理論
   - Tversky & Kahneman (1974) 啟發式與偏誤

3. **AI 安全與校準**
   - Guo et al. (2017) 深度學習校準
   - Kadavath et al. (2022) LLM self-evaluation
   - Nori et al. (2023) GPT-4 醫學能力評估

4. **臨床資訊學**
   - Tsou et al. (2017) EHR copy-paste 問題
   - Bates & Gawande (2003) 醫療錯誤與資訊系統
   - FDA (2021) Software as Medical Device 框架

5. **LLM 評估方法論**
   - Singhal et al. (2023) Med-PaLM 多維度評估
   - Jin et al. (2021) MedQA 基準設計
   - Nori et al. (2023) GPT-4 醫學推理系統性評估

---

## 資料夾結構

```
research/
├── README.md                                    ← 本文件
├── M1-open-ended-clinical-reasoning.md          # Layer 1: 表面
├── M2-ebm-hierarchy-sensitivity.md              # Layer 1: 表面
├── M3-clinical-error-atlas.md                   # Layer 2: 診斷
├── M4-counterfactual-stress-test.md             # Layer 3: 穩健性
├── M5-ehr-noise-robustness.md                   # Layer 3: 穩健性
├── M6-calibration-selective-prediction.md       # Layer 4: 行為
├── M7-clinical-cognitive-biases.md              # Layer 4: 行為
├── M8-patient-safety-risk-matrix.md             # Layer 5: 安全/政策
└── M9-rxllama-upgrade-framework.md              # Layer 6: 整合應用
```

---

## 方法論原則

1. **Inference-Only**：全部實驗不依賴 fine-tuning，確保可重現性與公平性
2. **Contrastive Design**：每個實驗都有 baseline vs intervention 的對比設計
3. **Multi-Model**：每個實驗至少測試 6 個模型（涵蓋 Cloud + Local + Medical-specialized）
4. **Safety-First**：所有指標都考慮臨床嚴重度加權
5. **Reproducible**：固定 temperature=0、記錄 model version、公開 prompt template
6. **Theory-Grounded**：每個構想都有明確的理論基礎文獻

---

## 與現有系統的關係

本研究計畫與 TMU 現有系統的連結：

| 現有系統 | 相關構想 | 連結方式 |
|---------|---------|---------|
| Medical-RAG (BioMistral) | M1, M6, M9 | RAG 系統作為評測對象之一 |
| Text2SQL | M9 | SQL 查詢的穩健性測試 |
| BRIDG Standards | M3, M8 | 臨床資料模型的錯誤分類對應 |
| RxLLama（國科會計畫） | M4, M9 | Condition-Aware 需求的直接驗證 |
