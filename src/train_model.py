import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data, split_features_and_target
from src.preprocess import get_preprocessor, perform_train_test_split
from src.evaluate import nested_cv, print_report

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"

# Inner search grid. Kept smaller for the nested loop, where it is refitted
# once per outer fold, than for the single final fit.
L1_RATIOS = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


def make_pipeline(cs: int = 20, inner_cv: int = 10, l1_ratios=None) -> Pipeline:
    model = LogisticRegressionCV(
        cv=inner_cv,
        Cs=cs,
        solver="saga",
        l1_ratios=l1_ratios if l1_ratios is not None else L1_RATIOS,
        scoring="neg_log_loss",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    return Pipeline([("preprocess", get_preprocessor()), ("model", model)])


def coefficient_table(fitted_pipeline: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    coefs = fitted_pipeline.named_steps["model"].coef_[0]
    return (pd.DataFrame({"Feature": feature_names,
                          "Log_Odds_Coef": coefs,
                          "Odds_Ratio": np.exp(coefs)})
            .sort_values("Log_Odds_Coef", key=abs, ascending=False))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results = {}

    # internal cohort
    df = load_data(str(ROOT / "data" / "mock9.parquet"))
    x, y = split_features_and_target(df, target_col="dx_ftd")
    valid = y.dropna().index
    x, y = x.loc[valid], y.loc[valid]
    x_train, x_test, y_train, y_test = perform_train_test_split(x, y)

    # 1. nested cross-validation
    print("\n=== Nested cross-validation (training set only) ===")
    results["nested_cv"] = nested_cv(
        lambda: make_pipeline(cs=10, inner_cv=5, l1_ratios=[0.1, 0.5, 0.9]),
        x_train, y_train, outer_splits=5)

    # 2. final fit, then calibration
    print("\n=== Final fit on the full training set ===")
    base = make_pipeline().fit(x_train, y_train)
    fitted = base.named_steps["model"]
    print(f"selected C = {fitted.C_[0]:.4f}   selected l1_ratio = {fitted.l1_ratio_[0]:.4f}")
    results["selected_C"] = float(fitted.C_[0])
    results["selected_l1_ratio"] = float(fitted.l1_ratio_[0])

    # scaling 
    print("fitting calibrator (Platt / sigmoid, 5-fold)...")
    calibrated = CalibratedClassifierCV(make_pipeline(cs=10, inner_cv=5, l1_ratios=[0.1, 0.5, 0.9]), method="sigmoid", cv=5)
    calibrated.fit(x_train, y_train)

    # 3. internal test set
    results["internal_uncalibrated"] = print_report("Internal held-out test set - UNCALIBRATED", y_test, base.predict_proba(x_test)[:, 1])
    results["internal_calibrated"] = print_report("Internal held-out test set - CALIBRATED", y_test, calibrated.predict_proba(x_test)[:, 1])

    print("\nCoefficients (final uncalibrated model):")
    coefs = coefficient_table(base, x_train.columns.tolist())
    print(coefs.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    coefs.to_csv(OUT / "coefficients.csv", index=False)

    # 4. external cohort
    ext_path = ROOT / "data" / "mock_external.parquet"
    if not ext_path.exists():
        print(f"\n[skip] {ext_path} not found - run `python data/generate_data.py` first.")
    else:
        print("\n=== Simulated external cohort (declared shift) ===")
        print("Shift vs development cohort: FTD prevalence 22% -> 35%; mean age -3 y; "
              "p-tau217 x1.25 and NfL x0.85 (different assay platform); volumes x1.02 "
              "(different scanner); UDS v3 only, so MMSE is entirely absent; fluid-biomarker "
              "missingness 25% -> 45%.")
        ext = load_data(str(ext_path))
        xe, ye = split_features_and_target(ext, target_col="dx_ftd")
        xe = xe[x_train.columns]
        valid_e = ye.dropna().index
        xe, ye = xe.loc[valid_e], ye.loc[valid_e]
        results["external_uncalibrated"] = print_report("Simulated external cohort - UNCALIBRATED", ye, base.predict_proba(xe)[:, 1])
        results["external_calibrated"] = print_report("Simulated external cohort - CALIBRATED", ye, calibrated.predict_proba(xe)[:, 1])

    with open(OUT / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nMetrics written to {OUT / 'metrics.json'}")
    print("Reminder: synthetic data throughout. These numbers are not diagnostic results.")


if __name__ == "__main__":
    main()

