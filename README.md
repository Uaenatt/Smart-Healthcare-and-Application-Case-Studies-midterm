# 智慧醫療與應用 — 期中專題

以多標籤機器學習同時預測 `Heart_Disease`、`Diabetes`、`Stroke` 三種慢性疾病。
完整的研究報告（Introduction / Methods / Results / Discussion）請參考 [`report.md`](report.md)。

## 專案結構

```
.
├── data/                        資料集
│   ├── train_dataset.csv        助教提供的 1,600 筆訓練資料（80%）
│   └── test.csv                 助教 2026-05-07 釋出的測試集（400 筆，僅特徵）
│
├── docs/                        作業文件
│   ├── case presentation.pdf    作業說明投影片
│   └── addiontal_description.txt 助教補充說明
│
├── src/                         主要腳本
│   ├── train.py                 訓練：5-fold CV + 最佳模型重訓
│   └── predict.py               推論：載入模型對測試集產生預測
│
├── outputs/                     模型成品
│   ├── model.joblib             最終模型（BR + Random Forest）
│   └── predictions.csv          test.csv 的預測結果
│
├── figures/                     ROC / PR / 特徵重要度 / 模型比較 圖表
├── results/                     交叉驗證指標、特徵重要度 CSV、最佳設定 JSON
│
├── report.md                    研究報告（Intro / Methods / Results / Discussion）
├── README.md                    本檔案
├── requirements.txt             Python 依賴
└── .gitignore
```

## 環境需求

- Python 3.10 以上
- 套件：`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`joblib`

```bash
pip install -r requirements.txt
```

## 使用方式

所有指令請於專案根目錄執行。

### 1. 重新訓練模型（5-fold 交叉驗證 + 重新擬合最佳模型）

```bash
python src/train.py
```

產出：

- `outputs/model.joblib` — 最終模型（Binary Relevance + Random Forest）
- `results/cv_metrics.csv`、`results/cv_summary_mean.csv` — 每折與平均指標
- `results/best_config.json` — 被選中的最佳 strategy × base
- `results/feature_importance_gini.csv` — 每個標籤的特徵重要度
- `figures/` — ROC / PR / 模型比較 / 特徵重要度 圖表

### 2. 對測試集進行預測

助教已於 **2026-05-07 10:00** 釋出測試集（`data/test.csv`，400 筆，僅特徵）。直接執行：

```bash
python src/predict.py
# 或指定路徑
python src/predict.py data/test.csv outputs/predictions.csv
```

輸出 `predictions.csv` 欄位：

```
Patient_ID,
Heart_Disease, Diabetes, Stroke,                         # 0/1 預測
Heart_Disease_proba, Diabetes_proba, Stroke_proba        # 機率值
```

## 模型與方法簡述

- **多標籤策略比較**：`Binary Relevance (BR)` vs `Classifier Chains (CC)`
- **基礎分類器**：Logistic Regression、Random Forest、Gradient Boosting
- **評估方法**：5-fold stratified cross-validation（以三標籤組合做分層）
- **評估指標**：per-label AUC、Macro/Micro AUC、Hamming Loss、Macro/Micro F1、Average Precision
- **最終模型**：`BR + Random Forest`（Macro AUC = 1.000，Hamming Loss = 0.003）

詳細設計、結果表格與討論請見 [`report.md`](report.md)。

## 測試集預測結果

助教已釋出之 `data/test.csv`（400 筆）經 BR + RF 模型推論後，預測陽性數與訓練集盛行率對照如下（完整輸出於 `outputs/predictions.csv`）：

| 標籤 | 訓練盛行率 | Test 預測陽性率 | 預測陽性數 |
|---|---|---|---|
| `Heart_Disease` | 8.25% | 7.25% | 29 |
| `Diabetes` | 13.06% | 13.25% | 53 |
| `Stroke` | 5.38% | 4.00% | 16 |

預測信心度高：96–98% 的樣本機率落在 ≤0.05 或 ≥0.5，灰色地帶 (機率 ∈ [0.3, 0.7]) 僅 3–5 筆。由於 test.csv 不含 ground truth，最終 AUC / Hamming 仍以助教評分為準。

## 重要提醒

1. `Gender` 在原始 CSV 中為字串 `"Male"` / `"Female"`，訓練與推論時皆會自動編碼為 `1` / `0`；直接丟入 `src/predict.py` 無需額外處理。
2. `Patient_ID` 僅作識別，**不會**進入模型特徵。
3. 隨機種子固定為 `42`，所有結果可重現。
4. 本資料集由於具有強烈規則性（見 `report.md` §3.4、§4），樹模型容易達到近乎完美的表現；正式測試集成績仍以助教評分為準。

## 關鍵日期

| 日期 | 事件 | 狀態 |
|------|------|------|
| 2026-05-07 10:00 | 助教釋出測試集（僅特徵，400 筆，存為 `data/test.csv`） | ✅ 已完成 |
| 2026-05-08 | 完成模型推論，產出 `outputs/predictions.csv` | ✅ 已完成 |
| 2026-05-14 | 課堂報告 | ⏳ 待進行 |

## 聯絡

- 授課教師：Albert C. Yang, M.D., Ph.D. （accyang@gmail.com）
- 助教：Yu-Chieh Chen （jessica20020310@gmail.com）
