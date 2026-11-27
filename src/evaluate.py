"""Evaluation utilities: nested cross-validation and metric reporting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report, confusion_matrix, roc_auc_score)


def metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def print_report(title: str, y_true, y_prob, threshold: float = 0.5) -> dict:
    m = metrics(y_true, y_prob, threshold)
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n--- {title} ---")
    print(f"n = {m['n']}  |  observed FTD prevalence = {m['prevalence']:.1%}")
    print(f"ROC-AUC {m['roc_auc']:.4f}   PR-AUC {m['pr_auc']:.4f}   Brier {m['brier']:.4f}")
    print(classification_report(y_true, y_pred, target_names=["AD", "FTD"], digits=3))
    print("Confusion matrix (rows = true AD/FTD):")
    print(confusion_matrix(y_true, y_pred))
    return m


def nested_cv(make_estimator, x: pd.DataFrame, y: pd.Series, outer_splits: int = 5, seed: int = 42) -> dict:
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    rows = []
    for k, (tr, te) in enumerate(outer.split(x, y), start=1):
        est = make_estimator()
        est.fit(x.iloc[tr], y.iloc[tr])
        prob = est.predict_proba(x.iloc[te])[:, 1]
        m = metrics(y.iloc[te], prob)
        m["fold"] = k
        rows.append(m)
        print(f"  outer fold {k}/{outer_splits}: "
              f"ROC-AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}  Brier {m['brier']:.4f}")
    df = pd.DataFrame(rows)
    summary = {f"{c}_mean": float(df[c].mean()) for c in ["roc_auc", "pr_auc", "brier"]}
    summary.update({f"{c}_sd": float(df[c].std(ddof=1)) for c in ["roc_auc", "pr_auc", "brier"]})
    summary["folds"] = rows
    print(f"  nested CV: ROC-AUC {summary['roc_auc_mean']:.4f} (SD {summary['roc_auc_sd']:.4f})  "
          f"Brier {summary['brier_mean']:.4f} (SD {summary['brier_sd']:.4f})")
    return summary