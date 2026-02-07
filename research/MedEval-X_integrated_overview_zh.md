# MedEval-X：醫療大型語言模型多維度安全評估框架 (MedEval-X: A Multi-Dimensional Safety Evaluation Framework for Medical Large Language Models)

> **專案總覽文件** — 描述原始 11 個研究模組（M1–M11）如何整合為 3 篇論文，以及這些論文如何作為統一研究計畫相互連結。

---

## 專案基本資訊

| 欄位 | 內容 |
|------|------|
| **研究團隊** | 楊教授實驗室，臺北醫學大學 |
| **研究領域** | 藥物警戒、用藥安全、臨床決策支援 |
| **核心論點** | 「記憶–安全落差」（Memorization–Safety Gap）— 前沿 LLM 能通過醫學考試，但在病人安全需要偏離記憶模式時卻會失敗 |
| **框架名稱** | MedEval-X |
| **論文總數** | 3 篇（皆已使用真實 API 資料完成） |

---

## 三篇論文

### 論文一：基礎篇 —「選擇題的幻覺」

**標題：** Beyond Multiple Choice: Calibration-Aware Evaluation Reveals Overconfident Clinical Reasoning in Large Language Models

| 欄位 | 內容 |
|------|------|
| **涵蓋模組** | M1 + M6 + M11 |
| **目標期刊** | JBI / IJMI |
| **資料規模** | n=1,273 MedQA，GPT-4o，3,819 次 API 呼叫 |
| **核心發現** | Option Bias = 31.7% — MCQ 將準確率膨脹了三分之一 |
| **關鍵指標** | Safety-Weighted ECE (SW-ECE) |
| **檔案位置** | `paper_foundation/` |

**研究問題：** MCQ 基準測試是否能準確衡量臨床推理能力，還是題目格式本身就會膨脹表現？

**研究證實：** LLM 在 MCQ 上達到 87.8% 的準確率，但在開放式作答格式下僅有 56.2%。信心度在兩種格式下都維持在約 91% — 模型不知道自己不知道（Calibration Paradox，校準悖論）。94.8% 的錯誤屬於高信心度錯誤。

---

### 論文二：安全篇 —「盲點論文」

**標題：** When AI Fails Drug Safety: Counterfactual Stress Testing Reveals Critical Blind Spots in Medical Large Language Models

| 欄位 | 內容 |
|------|------|
| **涵蓋模組** | M4 + M5 + M8 + M3 |
| **目標期刊** | npj Digital Medicine / JAMA Network Open |
| **資料規模** | 4 個模型 × 20 個情境 × 2 = 160 次 API 呼叫 |
| **核心發現** | 兒科 SCC = 0.65 — 系統性盲點 |
| **關鍵指標** | Safety-Critical Consistency (SCC) |
| **檔案位置** | `paper_safety/` |

**研究問題：** 當病患條件改變（懷孕、慢性腎臟病、兒童年齡、多重用藥）時，LLM 是否能維持用藥安全的準確性？

**研究證實：** 所有模型在標準情境下得分 100%，但在安全擾動條件下下降 10–20%。兒科用藥安全是一個關鍵盲點（SCC = 0.65）；fluoroquinolone（氟喹諾酮）禁忌症在 75% 的模型中判斷失敗。DDI（藥物交互作用）偵測則表現完美（1.00）。

---

### 論文三：Aesop 篇 —「解決方案」

**標題：** Aesop Guardrail: Condition-Aware Instruction Chaining for Mitigating Cognitive Biases in Clinical LLM Prior Authorization Systems

| 欄位 | 內容 |
|------|------|
| **涵蓋模組** | M9 + M7 + M2 |
| **目標期刊** | JAMIA / Lancet Digital Health |
| **資料規模** | 190 個情境 × 8 個模型 × 2 種條件 = 3,040 次 API 呼叫 |
| **核心發現** | False Approval Rate（錯誤核准率）從 27.3% 降至 8.6%（減少 18.7 個百分點） |
| **關鍵指標** | Sub-Population Safety Score (SS) |
| **檔案位置** | `paper_aesop/` |

**研究問題：** 一個不需修改模型本身、僅在提示層級進行的干預方法，能否修正論文一和論文二中發現的安全缺陷？

**研究證實：** 5 步驟 CAIC 協議將禁忌藥物的錯誤核准率降低了 18.7 個百分點。小型模型（7-14B）的受益程度是大型模型的 2-3 倍。所有 10 個脆弱族群的 Sub-Population Safety 均獲得改善。

---

## 敘事主軸：問題 → 診斷 → 治療

```
Paper 1 (Foundation)          Paper 2 (Safety)           Paper 3 (Aesop)
"The MCQ Illusion"            "The Blind Spot"            "The Fix"
─────────────────            ─────────────────           ─────────────────
建立                          展示                         解決
評估落差                       臨床危險                     問題

MCQ 將準確率膨脹              安全失敗是                    結構化提示
31.7%。校準在開放式            特定類別的：                  將錯誤核准率
格式下崩潰。94.8%              兒科 = 關鍵盲點。             降低 18.7 pp。
的錯誤是高信心度的。            所有模型都有類似              適用於所有模型。
                              的失敗模式。                  小型模型受益最多。

        │                           │                           │
        └──────── "LLM 不知道 ──────┘──── "以下是讓它們 ─────────┘
                   自己不知道              更安全的方法"
                   什麼"
```

---

## 模組與論文對應關係

| 模組 | 描述 | 論文 | 角色 |
|------|------|------|------|
| **M1** | 開放式臨床推理 | Foundation | 核心方法論（MCQ vs OE 比較） |
| **M2** | 實證醫學層級敏感度 | Aesop | 第 3 步替代方案中的 EBM 排序 |
| **M3** | 臨床錯誤圖譜 | Safety | 失敗分類的錯誤分類法 |
| **M4** | 反事實壓力測試 | Safety | 核心方法論（擾動框架） |
| **M5** | 電子病歷雜訊穩健性 | Safety | 雜訊層 + 信心度–雜訊悖論 |
| **M6** | 校準與選擇性預測 | Foundation | 校準分析（ECE, SW-ECE） |
| **M7** | 臨床認知偏誤 | Aesop | 偏誤識別 + 反偏誤對應 |
| **M8** | 病人安全風險矩陣 | Safety | 嚴重程度分類（WHO/NCC MERP） |
| **M9** | RxLLama 升級框架 | Aesop | 5 步驟 CAIC 協議 + SS 指標 |
| **M10** | AI 生成基準測試 | *（未來）* | 資料生成基礎建設 |
| **M11** | 多模型交叉監督 | Foundation | 多模型共識框架 |

### 尚未完整發表的模組

| 模組 | 狀態 | 計畫用途 |
|------|------|----------|
| **M10a**（生成方法論） | 設計完成 | 獨立的 AI 生成醫療基準測試方法論文 |
| **M10b**（交叉驗證） | 與 M11 重疊 | 已合併至 M11 的共識品質論文 |
| **M10c**（Explorer 平台） | 概念階段 | MedEval-X 公開評估平台 |

---

## 跨論文資料流

```
Paper 1 (Foundation)                Paper 2 (Safety)                Paper 3 (Aesop)
────────────────────                ────────────────                ────────────────
MedQA n=1,273                      20 個攻擊情境                    190 個 PA 情境
僅 GPT-4o                          4 個前沿模型                     8 個模型（雲端+本地）

Option Bias = 31.7%    ──────►     「模型靠記憶，
ECE_OE = 0.364         ──────►      而非針對安全條件     ──────► 「Aesop 修正了
94.8% 的錯誤是                      進行推理」                      推理落差」
高信心度的             ──────►     兒科 SCC = 0.65      ──────►  FAR: 27.3% → 8.6%
                                   MemGap = 10-20%                  ΔSS = +0.153

SW-ECE（新穎指標）                  SCC（新穎指標）                   SS（新穎指標）
三層級判斷                          三層級擾動                        5 步驟 CAIC 協議
Calibration Paradox                 Memorization-Safety Gap           小型模型受益更多
```

### 跨論文共享概念

| 概念 | 論文一 | 論文二 | 論文三 |
|------|--------|--------|--------|
| **過度自信** | 開放式格式 ECE = 0.364 | 高信心度的安全失敗 | 第 4 步重新校準信心度 |
| **條件盲視** | 格式依賴（MCQ 結構輔助） | 條件依賴（懷孕、兒科） | 第 1-2 步強制條件感知 |
| **次族群脆弱性** | 尚未按子領域分析 | 兒科 = 關鍵盲點 | 評估 10 個次族群 |
| **部署門檻** | Coverage@95% | 各類別 SCC ≥ 0.80 | 各次族群 SS ≥ 0.70 |
| **法規意涵** | 三層級篩選框架 | FDA SaMD、EU AI Act 對應 | HIS/EHR 整合設計 |

---

## 新穎貢獻摘要

| 貢獻 | 論文 | 影響 |
|------|------|------|
| **Option Bias** 量化 | Foundation | 首次系統性測量醫療 LLM 中 MCQ 格式的膨脹效應 |
| **Safety-Weighted ECE (SW-ECE)** | Foundation | 首個臨床加權校準指標 |
| **三層級臨床判斷（A/B/C）** | Foundation | 捕捉二元評分遺漏的部分正確性 |
| **Calibration Paradox** | Foundation | 跨格式時準確率下降但信心度反而上升 |
| **Memorization–Safety Gap** | Safety | 記憶情境與安全擾動情境之間的表現落差 |
| **Safety-Critical Consistency (SCC)** | Safety | 類別分層的安全評估指標 |
| **Conditional Inversion 測試** | Safety | 條件感知推理的系統化測試方法 |
| **兒科盲點**識別 | Safety | 首個系統性兒科安全失敗的實證證據 |
| **Condition-Aware Instruction Chaining (CAIC)** | Aesop | 無需修改模型的 5 步驟去偏誤協議 |
| **Sub-Population Safety Score (SS)** | Aesop | 對關鍵失敗進行乘法懲罰的指標 |
| **「小型模型受益更多」** | Aesop | 提示層級護欄作為公平性基礎建設 |
| **Anti-bias 對應** | Aesop | 協議的每個步驟對應特定的認知偏誤 |

---

## 實驗規模摘要

| 論文 | 模型數 | 情境數 | API 呼叫總數 | 資料來源 |
|------|--------|--------|-------------|----------|
| Foundation | 1（GPT-4o） | 1,273 × 2 種格式 | 3,819 | MedQA USMLE |
| Safety | 4（GPT-4o, Claude, Gemini, DeepSeek） | 20 × 2 版本 | 160 | 自建攻擊矩陣 |
| Aesop | 8（5 雲端 + 3 本地） | 190 × 2 條件 | 3,040 | 自建 PA 情境 |
| **合計** | **8 個不同模型** | — | **約 7,019** | — |

---

## 各原始模組的去向

對於想回顧原始 M1-M11 文件的研究者，下表說明了哪些內容被使用、哪些被延後、以及哪些有所變更：

| 模組 | 原始範圍 | 納入論文的部分 | 延後的部分 |
|------|----------|---------------|-----------|
| M1 | 6,256 題、3 個資料集、SNOMED CT 配對 | 1,273 題 MedQA、GPT-4o 判定 | 多資料集、SNOMED CT |
| M2 | 6 種偏誤測試 × 30 情境 × 4 去偏誤條件 | EBM 排序作為第 3 步設計原則 | 完整 EBM 敏感度實驗 |
| M3 | 18,000 個錯誤、15 類型 × 10 專科 × 5 階段 | 10 個失敗案例的模式分析 | 完整 3D 錯誤圖譜 |
| M4 | 400 種子 × 6 種擾動 = 2,800 變體 | 20 個 Level-2 情境（條件反轉） | Level 1/3 擾動 |
| M5 | 200 × 5 種雜訊 × 3 嚴重度 = 3,200 變體 | 描述框架，未完整評估 | 雜訊 × 擾動交互作用 |
| M6 | 4 種信心度方法、Coverage@95% | 僅使用語言化信心度 | 自我一致性、logit-based、集成方法 |
| M7 | 6 種偏誤 × 30 情境 × 4 條件 | 5 種偏誤對應至協議步驟 | 完整認知偏誤剖析實驗 |
| M8 | 完整風險矩陣 + 法規落差分析 | 觀察到之失敗的嚴重度分類 | 集體幻覺分析 |
| M9 | 8 維度計分卡、PA 系統升級 | 5 步驟 CAIC + SS 指標 | 完整 8D 計分卡 |
| M10 | AI 基準生成 + Explorer 平台 | — | 整個模組延後 |
| M11 | 3 種共識策略、失敗模式分析 | Discussion 中的概念框架 | 完整 M11 實驗 |

---

## 檔案位置

```
research/
├── M1-open-ended-clinical-reasoning.md      ← 原始模組構想
├── M2-ebm-hierarchy-sensitivity.md          ←
├── M3-clinical-error-atlas.md               ←
├── M4-counterfactual-stress-test.md         ←
├── M5-ehr-noise-robustness.md               ←
├── M6-calibration-selective-prediction.md   ←
├── M7-clinical-cognitive-biases.md          ←
├── M8-patient-safety-risk-matrix.md         ←
├── M9-rxllama-upgrade-framework.md          ←
├── M10-ai-generated-benchmark.md            ←
├── M11-multi-model-cross-supervision.md     ←
│
├── MedEval-X_integrated_overview.md         ← 英文原版（專案總覽）
├── MedEval-X_integrated_overview_zh.md      ← 本檔案（繁體中文翻譯）
│
├── paper_foundation/
│   ├── paper_foundation_integrated.md       ← 整合版 .md（M1+M6+M11）
│   ├── main.tex + main.pdf                  ← LaTeX 論文
│   ├── run_experiment.py                    ← 實驗程式碼
│   ├── results_foundation_full.json         ← 原始結果（n=1,273）
│   └── figures/                             ← 5 張圖表
│
├── paper_safety/
│   ├── paper_safety_integrated.md           ← 整合版 .md（M4+M5+M8+M3）
│   ├── adversarial_design.md                ← 攻擊假說設計
│   ├── aesop_guardrail_architecture.md      ← 護欄規格（→ 已移至 paper_aesop）
│   ├── main.tex + main.pdf                  ← LaTeX 論文
│   ├── src/run_real_stress_test.py          ← 真實 API 管線
│   ├── results/                             ← JSON + CSV 結果
│   └── figures/                             ← 3 張真實圖表
│
└── paper_aesop/
    ├── paper_aesop_integrated.md            ← 整合版 .md（M9+M7+M2）
    ├── main.tex + main.pdf                  ← LaTeX 論文
    ├── src/run_optimization.py              ← A/B 測試管線
    ├── results/                             ← JSON + CSV 結果
    └── figures/                             ← 6 張圖表
```
