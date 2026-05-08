"""
train.py — trains the final multi-label classifier for Heart_Disease, Diabetes, Stroke.

Run:
    python train.py

Produces:
    model.joblib         — final fitted sklearn Pipeline (preprocessor + multi-label classifier)
    results/cv_metrics.csv, results/cv_summary_mean.csv, results/best_config.json
    figures/*            — ROC, PR, model comparison, feature importance plots

Strategy selection: Binary Relevance vs Classifier Chains, each with Logistic
Regression / Random Forest / Gradient Boosting base learners, evaluated with
5-fold stratified CV on the joint label combination. The configuration with
the best mean macro-AUC (with Hamming Loss as tie-breaker) is retrained on
the full training set and saved.

See report.md for the methodology and discussion.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
OUT_DIR = ROOT / "outputs"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

TARGETS = ["Heart_Disease", "Diabetes", "Stroke"]
NUM = ["Age", "BMI", "Blood_Pressure_Systolic", "Blood_Pressure_Diastolic",
       "Cholesterol", "Glucose_Level"]
BINF = ["Gender", "Smoking", "Alcohol_Intake", "Physical_Activity", "Family_History"]
FEATURES = NUM + BINF
RNG = 42


def load_train(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).astype(int)
    return df[FEATURES].copy(), df[TARGETS].astype(int).copy()


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("bin", "passthrough", BINF),
    ])


def base_learners() -> dict[str, object]:
    return {
        "LR": LogisticRegression(max_iter=2000, class_weight="balanced",
                                 solver="lbfgs", random_state=RNG),
        "RF": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     random_state=RNG, n_jobs=-1),
        "GB": GradientBoostingClassifier(random_state=RNG),
    }


def strategy(name: str, base):
    if name == "BR":
        return MultiOutputClassifier(clone(base), n_jobs=-1)
    if name == "CC":
        return ClassifierChain(clone(base), order=[0, 1, 2], random_state=RNG)
    raise ValueError(name)


def pipeline(strat) -> Pipeline:
    return Pipeline([("pre", make_preprocessor()), ("clf", strat)])


def proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Xt = model.named_steps["pre"].transform(X)
    clf = model.named_steps["clf"]
    if isinstance(clf, MultiOutputClassifier):
        return np.column_stack([est.predict_proba(Xt)[:, 1] for est in clf.estimators_])
    if isinstance(clf, ClassifierChain):
        return clf.predict_proba(Xt)
    raise TypeError(type(clf))


def score_fold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    aucs = [roc_auc_score(y_true[:, i], y_prob[:, i]) for i in range(3)]
    aps = [average_precision_score(y_true[:, i], y_prob[:, i]) for i in range(3)]
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "auc_heart": aucs[0], "auc_diabetes": aucs[1], "auc_stroke": aucs[2],
        "auc_macro": float(np.mean(aucs)),
        "auc_micro": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "ap_heart": aps[0], "ap_diabetes": aps[1], "ap_stroke": aps[2],
        "hamming": float(hamming_loss(y_true, y_hat)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_hat, average="micro", zero_division=0)),
    }


def cross_validate(X: pd.DataFrame, Y: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run 5-fold stratified CV across all (strategy, base) combos."""
    key = Y["Heart_Disease"].astype(str) + Y["Diabetes"].astype(str) + Y["Stroke"].astype(str)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

    rows = []
    oof: dict[tuple[str, str], np.ndarray] = {}
    for bname, base in base_learners().items():
        for sname in ("BR", "CC"):
            oof[(sname, bname)] = np.zeros_like(Y.values, dtype=float)
    Y_np = Y.values

    for fold, (tr, te) in enumerate(skf.split(X, key)):
        for bname, base in base_learners().items():
            for sname in ("BR", "CC"):
                model = pipeline(strategy(sname, base))
                model.fit(X.iloc[tr], Y.iloc[tr].values)
                p = proba(model, X.iloc[te])
                oof[(sname, bname)][te] = p
                m = score_fold(Y.iloc[te].values, p)
                m.update({"strategy": sname, "base": bname, "fold": fold})
                rows.append(m)

    cv = pd.DataFrame(rows)
    cv.to_csv(RES_DIR / "cv_metrics.csv", index=False)
    agg = cv.groupby(["strategy", "base"]).mean(numeric_only=True).round(4)
    agg.drop(columns=["fold"], errors="ignore").to_csv(RES_DIR / "cv_summary_mean.csv")
    return cv, oof


def plot_curves(Y_np: np.ndarray, prob: np.ndarray, tag: str):
    for i, t in enumerate(TARGETS):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fpr, tpr, _ = roc_curve(Y_np[:, i], prob[:, i])
        ax.plot(fpr, tpr, label=f"{t} (AUC={roc_auc_score(Y_np[:, i], prob[:, i]):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set(xlabel="FPR", ylabel="TPR", title=f"ROC — {tag} (OOF)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"roc_{t}.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4.5))
        pr, rc, _ = precision_recall_curve(Y_np[:, i], prob[:, i])
        ap = average_precision_score(Y_np[:, i], prob[:, i])
        ax.plot(rc, pr, label=f"{t} (AP={ap:.3f})")
        ax.axhline(Y_np[:, i].mean(), ls="--", color="grey", alpha=0.5,
                   label=f"baseline={Y_np[:, i].mean():.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision–Recall — {tag} (OOF)")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"pr_{t}.png", dpi=140)
        plt.close(fig)


def plot_model_comparison(cv: pd.DataFrame):
    flat = cv.groupby(["strategy", "base"]).mean(numeric_only=True).reset_index()
    flat["label"] = flat["strategy"] + "+" + flat["base"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(flat["label"], flat["auc_macro"], color="#4F81BD")
    axes[0].set_ylabel("Macro AUC")
    axes[0].set_title("Macro AUC (higher is better)")
    for i, v in enumerate(flat["auc_macro"]):
        axes[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(flat["label"], flat["hamming"], color="#C0504D")
    axes[1].set_ylabel("Hamming Loss")
    axes[1].set_title("Hamming Loss (lower is better)")
    for i, v in enumerate(flat["hamming"]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Multi-label strategy × base learner — 5-fold CV")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=140)
    plt.close(fig)


def feature_importance_plots(model: Pipeline, X: pd.DataFrame, Y: pd.DataFrame):
    clf = model.named_steps["clf"]
    if not isinstance(clf, MultiOutputClassifier):
        return
    rows = []
    for t, est in zip(TARGETS, clf.estimators_):
        if hasattr(est, "feature_importances_"):
            for name, imp in zip(FEATURES, est.feature_importances_):
                rows.append({"target": t, "feature": name, "importance": float(imp)})
    if not rows:
        return
    gi = pd.DataFrame(rows)
    gi.to_csv(RES_DIR / "feature_importance_gini.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, t in zip(axes, TARGETS):
        d = gi[gi["target"] == t].sort_values("importance")
        ax.barh(d["feature"], d["importance"], color="#4F81BD")
        ax.set_title(f"{t} — Gini importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feat_importance_gini.png", dpi=140)
    plt.close(fig)


def main():
    X, Y = load_train(DATA_DIR / "train_dataset.csv")
    print(f"Loaded {len(X)} rows, {len(FEATURES)} features, {len(TARGETS)} labels")
    print("Prevalence:", Y.mean().round(4).to_dict())

    cv, oof = cross_validate(X, Y)
    agg = cv.groupby(["strategy", "base"]).mean(numeric_only=True)
    print("\n=== CV summary (mean) ===")
    print(agg[["auc_macro", "auc_heart", "auc_diabetes", "auc_stroke",
               "hamming", "f1_macro"]].round(4))

    # Pick by macro-AUC, break ties by lower Hamming
    ranked = agg.sort_values(["auc_macro", "hamming"], ascending=[False, True])
    best_key = ranked.index[0]
    print(f"\nBest: strategy={best_key[0]}  base={best_key[1]}")
    (RES_DIR / "best_config.json").write_text(json.dumps({
        "strategy": best_key[0], "base": best_key[1],
        "auc_macro": float(agg.loc[best_key, "auc_macro"]),
        "hamming": float(agg.loc[best_key, "hamming"]),
        "per_label_auc": {
            "Heart_Disease": float(agg.loc[best_key, "auc_heart"]),
            "Diabetes": float(agg.loc[best_key, "auc_diabetes"]),
            "Stroke": float(agg.loc[best_key, "auc_stroke"]),
        },
    }, indent=2))

    plot_curves(Y.values, oof[best_key], tag=f"{best_key[0]}+{best_key[1]}")
    plot_model_comparison(cv)

    # Refit on full data, save
    final = pipeline(strategy(best_key[0], base_learners()[best_key[1]]))
    final.fit(X, Y.values)
    joblib.dump({
        "pipeline": final, "features": FEATURES, "targets": TARGETS,
        "strategy": best_key[0], "base": best_key[1],
    }, OUT_DIR / "model.joblib")
    print(f"Saved model.joblib (strategy={best_key[0]}, base={best_key[1]})")

    feature_importance_plots(final, X, Y)


if __name__ == "__main__":
    main()
