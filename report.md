# 使用臨床資料同時預測 Heart Disease、Diabetes 與 Stroke
### 智慧醫療與應用 期中報告

授課教師：Albert C. Yang, M.D., Ph.D.  
助教：Yu-Chieh Chen  
報告日期：2026-05-14

---

## 1. Introduction

心血管疾病、糖尿病與中風是臨床上常見且彼此高度相關的慢性疾病。這三類疾病共享許多風險因子，例如年齡、肥胖、高血壓、高血糖、膽固醇異常與生活型態等，因此同一位病人同時具有多種疾病風險並不罕見。若能建立一個模型，同時預測 `Heart_Disease`、`Diabetes` 與 `Stroke` 三個標籤，會比各自獨立建立單一疾病模型更符合臨床使用情境。

本專題依據助教提供的 `train_dataset.csv`，以多標籤分類（Multi-label classification）方式建立機器學習模型，目標是同時預測以下三個二元標籤：

- `Heart_Disease`
- `Diabetes`
- `Stroke`

本研究比較兩種多標籤處理策略：

- `Binary Relevance (BR)`
- `Classifier Chains (CC)`

並搭配三種基礎分類器：

- `Logistic Regression (LR)`
- `Random Forest (RF)`
- `Gradient Boosting (GB)`

模型評估以 5-fold cross-validation 為主，並額外利用原始 Kaggle 資料集中「未出現在助教訓練集中的 400 筆資料」建立一份暫代測試集 `proxy_test_dataset.csv`，作為正式 `test_dataset.csv` 尚未釋出前的內部驗證依據。

---

## 2. Methods

### 2.1 資料來源與欄位

作業指定資料來源為 Kaggle 的 `rafi003/healthcare-disease-prediction-dataset`，原始檔案為 `healthcare_disease_prediction_2000.csv`，共 2000 筆資料。助教提供的 `train_dataset.csv` 共 1600 筆，包含 11 個特徵欄位與 3 個目標欄位。

特徵欄位如下：

- `Age`
- `Gender`
- `BMI`
- `Blood_Pressure_Systolic`
- `Blood_Pressure_Diastolic`
- `Cholesterol`
- `Glucose_Level`
- `Smoking`
- `Alcohol_Intake`
- `Physical_Activity`
- `Family_History`

目標欄位如下：

- `Heart_Disease`
- `Diabetes`
- `Stroke`

其中 `Patient_ID` 僅作為病患識別碼，不納入模型訓練。`Gender` 在原始 CSV 中為字串 `Male` / `Female`，在模型中明確編碼為 `1` / `0`。

### 2.2 前處理

本研究將連續型變數與二元變數分開處理：

- 連續型變數：`Age`、`BMI`、`Blood_Pressure_Systolic`、`Blood_Pressure_Diastolic`、`Cholesterol`、`Glucose_Level`
- 二元變數：`Gender`、`Smoking`、`Alcohol_Intake`、`Physical_Activity`、`Family_History`

連續型變數使用 `StandardScaler` 標準化，二元變數則直接保留。整體前處理與分類器共同包裝於同一個 `Pipeline` 中，以避免資料洩漏。

### 2.3 多標籤建模策略

本研究比較兩種多標籤問題處理方法：

- `Binary Relevance (BR)`：將三個標籤視為三個彼此獨立的二元分類問題，分別建立模型。
- `Classifier Chains (CC)`：將前一個標籤的預測結果作為後一個標籤的額外輸入，以捕捉標籤間可能的依存關係。

每一種策略再分別搭配三種基礎分類器：

- `Logistic Regression`
- `Random Forest`
- `Gradient Boosting`

因此共比較 6 組模型：

- `BR + LR`
- `CC + LR`
- `BR + RF`
- `CC + RF`
- `BR + GB`
- `CC + GB`

### 2.4 評估方式

本研究使用 5-fold stratified cross-validation。由於本題為多標籤問題，分層依據為三個標籤的組合，以盡量維持各 fold 的標籤結構一致。

評估指標包括：

- 每個標籤的 `ROC AUC`
- `Macro AUC`
- `Micro AUC`
- `Hamming Loss`
- `Macro F1`
- `Micro F1`
- `Average Precision (AP)`

模型選擇原則為：

1. 以 `Macro AUC` 最高者為優先
2. 若 `Macro AUC` 相同，則以 `Hamming Loss` 較低者為佳

### 2.5 關於正式測試集尚未釋出時的處理

在專題進行過程中，我們確認助教提供的 `train_dataset.csv` 雖然來自原始 2000 筆資料，但**並不是原始檔案前 1600 筆的直接切分**。因此若直接把原始資料「最後 400 筆」拿來測試，會與目前訓練集出現重疊，造成資料洩漏。

經比對後發現：

- 原始資料總筆數：`2000`
- 助教訓練集：`1600`
- 原始檔最後 400 筆中，有 `310` 筆其實已經出現在目前訓練集

因此，我們改採更合理的方式：  
從原始 2000 筆資料中，找出**完全沒有出現在目前 `train_dataset.csv`** 的資料，共 `400` 筆，另存為：

- `proxy_test_dataset.csv`

此檔案的意義是「暫代正式 `test_dataset.csv` 的代理測試集」，僅用於正式測試集尚未釋出前的內部驗證，避免與助教未來發布的正式測試資料混淆。

---

## 3. Results

### 3.1 交叉驗證結果

六組模型的 5-fold cross-validation 平均結果如下：

| Strategy | Base | Macro AUC | AUC Heart | AUC Diabetes | AUC Stroke | Hamming Loss | Macro F1 |
|----------|------|-----------|-----------|--------------|------------|--------------|----------|
| BR | LR | 0.9792 | 0.9731 | 0.9717 | 0.9927 | 0.0829 | 0.6700 |
| CC | LR | 0.9793 | 0.9731 | 0.9720 | 0.9929 | 0.0844 | 0.6631 |
| BR | GB | 0.9975 | 0.9926 | 1.0000 | 1.0000 | 0.0004 | 0.9974 |
| CC | GB | 0.9975 | 0.9926 | 1.0000 | 1.0000 | 0.0004 | 0.9974 |
| BR | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0029 | 0.9773 |
| CC | RF | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0035 | 0.9715 |

依照 `Macro AUC` 最高且 `Hamming Loss` 較低的原則，最終選定模型為：

- `Binary Relevance + Random Forest (BR + RF)`

其設定已保存於 `model.joblib`。

### 3.2 特徵重要度

以最終的 `BR + RF` 模型來看，各疾病最重要的特徵如下：

- `Heart_Disease`：`Cholesterol`、`Blood_Pressure_Systolic`、`Age`
- `Diabetes`：`BMI`、`Glucose_Level`
- `Stroke`：`Blood_Pressure_Systolic`、`Age`、`Smoking`

此結果在 `Gini importance` 與 `Permutation importance` 兩種分析方式下大致一致，顯示模型主要依賴上述少數幾個關鍵特徵完成預測。

### 3.3 proxy_test_dataset 的測試結果

在正式 `test_dataset.csv` 尚未釋出前，我們以 `proxy_test_dataset.csv` 作為暫代測試集進行驗證。此資料集共有 `400` 筆，且與目前 `train_dataset.csv` 的 `Patient_ID` 重疊數為 `0`。

使用目前儲存的最終模型 `BR + RF` 在 `proxy_test_dataset.csv` 上測得：

- `Macro AUC = 1.000`
- `Heart_Disease AUC = 1.000`
- `Diabetes AUC = 1.000`
- `Stroke AUC = 1.000`
- `Average Precision` 三個標籤皆為 `1.000`
- `Hamming Loss = 0.000`
- `Macro F1 = 1.000`
- `Micro F1 = 1.000`
- `Subset Accuracy = 1.000`

該暫代測試集中的真實陽性數為：

- `Heart_Disease = 29`
- `Diabetes = 53`
- `Stroke = 16`

模型預測陽性數與真實值完全一致。

### 3.4 對結果的初步判讀

不論是 cross-validation 還是 `proxy_test_dataset.csv` 的測試結果，樹模型都達到幾乎完美甚至完全完美的表現。這表示此資料集極可能不是高雜訊的真實臨床資料，而更接近可由少數規則決定的合成資料，或至少具有非常強的規則性。

---

## 4. Discussion

### 4.1 為何 Random Forest 表現最好

從結果可見，`Random Forest` 與 `Gradient Boosting` 都明顯優於 `Logistic Regression`。這代表標籤與特徵之間的關係可能並非單純線性，而是包含明確門檻、交互作用或規則式結構。樹模型特別擅長擬合這類資料，因此在本題中能夠得到極高表現。

### 4.2 為何 BR 與 CC 差異不大

`Classifier Chains` 的設計目的是利用標籤之間的依賴關係提升表現，但本研究中 `CC` 並未明顯優於 `BR`。這表示三個標籤雖然在醫學上可能相關，但在這份資料中，大部分可預測訊號已經充分存在於原始特徵中，因此額外串接標籤資訊所帶來的幫助有限。

### 4.3 關於 proxy_test_dataset 的使用限制

雖然 `proxy_test_dataset.csv` 與目前的 `train_dataset.csv` 沒有重複病患，且可用於正式測試集尚未釋出前的內部驗證，但它仍然來自同一份 Kaggle 原始資料集，因此不能完全等同於助教未來釋出的正式 `test_dataset.csv`。正式成績與最終報告結論，仍應以助教提供的 `test_dataset.csv` 為準。

### 4.4 目前可得的結論

截至目前為止，本專題可以得到以下結論：

- 本題適合以多標籤分類處理，並可合理比較 `BR` 與 `CC`
- 最佳模型為 `BR + RF`
- 最關鍵的臨床特徵包括 `Cholesterol`、`Blood_Pressure_Systolic`、`Age`、`BMI`、`Glucose_Level` 與 `Smoking`
- 在交叉驗證與 `proxy_test_dataset.csv` 上，模型皆達到極高甚至完美的表現
- 此結果顯示資料可能具有強烈規則性，正式測試仍須等待助教公布的 `test_dataset.csv`

---

## Reproducibility

安裝需求：

```bash
pip install -r requirements.txt
```

訓練模型：

```bash
python train.py
```

使用正式或暫代測試集進行推論：

```bash
python predict.py proxy_test_dataset.csv predictions.csv
```

目前主要檔案如下：

- `train_dataset.csv`：助教提供的 1600 筆訓練資料
- `proxy_test_dataset.csv`：由原始 Kaggle 2000 筆資料中篩出、未出現在訓練集的 400 筆暫代測試資料
- `train.py`：模型訓練與交叉驗證程式
- `predict.py`：模型推論程式
- `model.joblib`：最終模型
- `results/`：交叉驗證與特徵重要度結果
- `figures/`：圖表輸出

Random seed：`42`
