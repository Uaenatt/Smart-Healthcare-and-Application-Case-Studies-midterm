"""
predict.py — applies outputs/model.joblib to a test CSV and writes predictions.

Usage (run from project root):
    python src/predict.py                          # data/test.csv -> outputs/predictions.csv
    python src/predict.py path/to/test.csv         # custom test CSV
    python src/predict.py path/to/test.csv out.csv

Output CSV columns:
    Patient_ID, Heart_Disease, Diabetes, Stroke,
    Heart_Disease_proba, Diabetes_proba, Stroke_proba

Compatible with Binary Relevance (MultiOutputClassifier) or Classifier Chain
models saved by train.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
TARGETS = ["Heart_Disease", "Diabetes", "Stroke"]


def predict_proba(pipeline, X: pd.DataFrame) -> np.ndarray:
    Xt = pipeline.named_steps["pre"].transform(X)
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, MultiOutputClassifier):
        return np.column_stack([est.predict_proba(Xt)[:, 1] for est in clf.estimators_])
    if isinstance(clf, ClassifierChain):
        return clf.predict_proba(Xt)
    raise TypeError(f"Unsupported classifier: {type(clf).__name__}")


def main(argv: list[str]) -> int:
    test_path = Path(argv[1]) if len(argv) > 1 else DATA_DIR / "test.csv"
    out_path = Path(argv[2]) if len(argv) > 2 else OUT_DIR / "predictions.csv"
    if not test_path.exists():
        print(f"ERROR: {test_path} not found. "
              "Place the TA-provided test CSV under ./data/ "
              "(or pass its path as the first argument).")
        return 1

    artifact = joblib.load(OUT_DIR / "model.joblib")
    pipeline = artifact["pipeline"]
    features = artifact["features"]
    print(f"Loaded model: strategy={artifact['strategy']} base={artifact['base']}")

    df = pd.read_csv(test_path)
    id_col = df["Patient_ID"] if "Patient_ID" in df.columns else pd.Series(
        range(len(df)), name="Patient_ID")
    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).astype(int)

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"ERROR: missing required columns in test CSV: {missing}")
        return 1

    X = df[features]
    probs = predict_proba(pipeline, X)
    preds = (probs >= 0.5).astype(int)

    out = pd.DataFrame({"Patient_ID": id_col.values})
    for i, t in enumerate(TARGETS):
        out[t] = preds[:, i]
    for i, t in enumerate(TARGETS):
        out[f"{t}_proba"] = probs[:, i].round(6)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out)} rows.")
    print("Predicted positives:",
          {t: int(out[t].sum()) for t in TARGETS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
