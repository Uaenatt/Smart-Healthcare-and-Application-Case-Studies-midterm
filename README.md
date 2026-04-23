# 智慧醫療與應用 — 期中專題

以多標籤機器學習同時預測 `Heart_Disease`、`Diabetes`、`Stroke` 三種慢性疾病。
完整的研究報告（Introduction / Methods / Results / Discussion）請參考 [`report.md`](report.md)。

## 專案結構

```
.
├── train_dataset.csv            助教提供的 1,600 筆訓練資料（80%）
├── proxy_test_dataset.csv       由 Kaggle 原始 2,000 筆中，扣除訓練集後剩下的 400 筆
│                                 （正式 test_dataset.csv 尚未釋出前的暫代測試集）
├── case presentation.pdf        作業說明投影片
├── addiontal_description.txt    助教補充說明
├── CLAUDE.md                    專案開發規範
│
├── train.py                     訓練腳本：5-fold CV + 最佳模型重訓
├── predict.py                   推論腳本：讀取 model.joblib 對測試集產生預測
├── report.md                    研究報告（Intro / Methods / Results / Discussion）
├── requirements.txt             Python 依賴
│
├── model.joblib                 最終模型（BR + Random Forest）
├── figures/                     ROC / PR / 特徵重要度 / 模型比較 圖表
└── results/                     交叉驗證指標、特徵重要度 CSV、最佳設定 JSON
```

## 環境需求

- Python 3.10 以上
- 套件：`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`joblib`

```bash
pip install -r requirements.txt
```

## 使用方式

### 1. 重新訓練模型（5-fold 交叉驗證 + 重新擬合最佳模型）

```bash
python train.py
```

產出：

- `model.joblib` — 最終模型（Binary Relevance + Random Forest）
- `results/cv_metrics.csv`、`results/cv_summary_mean.csv` — 每折與平均指標
- `results/best_config.json` — 被選中的最佳 strategy × base
- `results/feature_importance_gini.csv` — 每個標籤的特徵重要度
- `figures/` — ROC / PR / 模型比較 / 特徵重要度 圖表

### 2. 對測試集進行預測

正式測試集 `test_dataset.csv` 將於 **2026-05-07 10:00** 由助教釋出。屆時將檔案放在此資料夾下再執行：

```bash
python predict.py
# 或指定路徑
python predict.py test_dataset.csv predictions.csv
```

在正式測試集尚未釋出前，也可用暫代測試集驗證：

```bash
python predict.py proxy_test_dataset.csv results/proxy_predictions.csv
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

## 重要提醒

1. `Gender` 在原始 CSV 中為字串 `"Male"` / `"Female"`，訓練與推論時皆會自動編碼為 `1` / `0`；直接丟入 `predict.py` 無需額外處理。
2. `Patient_ID` 僅作識別，**不會**進入模型特徵。
3. 隨機種子固定為 `42`，所有結果可重現。
4. 本資料集由於具有強烈規則性（見 `report.md` §3.4、§4），樹模型容易達到近乎完美的表現；正式 `test_dataset.csv` 釋出後以其結果為準。

## 關鍵日期

| 日期 | 事件 |
|------|------|
| 2026-05-07 10:00 | 助教釋出 `test_dataset.csv`（僅特徵，400 筆） |
| 2026-05-14 | 課堂報告 |

## 聯絡

- 授課教師：Albert C. Yang, M.D., Ph.D. （accyang@gmail.com）
- 助教：Yu-Chieh Chen （jessica20020310@gmail.com）
