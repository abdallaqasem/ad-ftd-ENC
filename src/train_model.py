import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from preprocess import get_preprocessor, perform_train_test_split
from sklearn.metrics import(average_precision_score, brier_score_loss, classification_report, confusion_matrix, roc_auc_score)

def train_elastic_net(x_train: pd.DataFrame, y_train: pd.Series):
    """
    Trains a Logistic Regression model with Elastic Net using a fully data-driven 
    cross-validation search across the entire l1_ratio and regularization spectrum.
    """
    # Span the entire valid l1_ratio spectrum (from near-ridge to near-lasso) 
    # so the data can empirically find the exact optimal balance point.
    
    data_driven_l1_ratios = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    
    model = LogisticRegressionCV(
        cv=10,                 # 10-fold cross-validation for maximum data rigor
        Cs=20,                 # Let the algorithm test 20 data-derived regularization strengths automatically
        solver='saga',
        l1_ratios=data_driven_l1_ratios,
        scoring='neg_log_loss',     
        class_weight = 'balanced',
        max_iter=5000,         # Generous ceiling to guarantee algorithmic convergence
        random_state=42        # Ensures mathematical reproducibility
    )

    pipeline = Pipeline([
        ("preprocess", get_preprocessor()),
        ("model", model),
    ])

    pipeline.fit(x_train, y_train)
    fitted_model = pipeline.named_steps["model"]
    # Output the empirically discovered parameters
    print(f"Empirically optimized C: {fitted_model.C_[0]:.4f}")
    print(f"Empirically optimized l1_ratio (L1/L2 mix): {fitted_model.l1_ratio_[0]:.4f}")
    print("Elastic Net model optimized purely from data patterns.")
    return pipeline


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series):
    """Evaluates the trained model on the test set using standard classification metrics."""
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    
    print(f"\nROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nPR-AUC: {average_precision_score(y_test, y_prob):.4f}")
    print(f"\nBrier Score: {brier_score_loss(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def features_weights(pipeline, feature_names: list[str]):
    coefs = pipeline.named_steps["model"].coef_[0]
    weights_df = pd.DataFrame({
        "Feature": feature_names, 
        "Log_Odds_Coef": coefs,
        "Odds_Ratio": np.exp(coefs),
    }).sort_values(by="Log_Odds_Coef", key=abs, ascending=False)
    print("Feature Weights:")
    print(weights_df.to_string(index=False))


if __name__ == "__main__":
    # Import your pipeline steps
    from data_loader import load_data, split_features_and_target
    from preprocess import perform_train_test_split
    
    # 1. Load and clean
    df = load_data()
    x, y = split_features_and_target(df, target_col="dx_ftd")
    valid_idx = y.dropna().index
    x, y = x.loc[valid_idx], y.loc[valid_idx]
    
    # 2. Split, Impute, and Scale
    x_train, x_test, y_train, y_test = perform_train_test_split(x, y)
    
    # 3. Train and Evaluate
    model = train_elastic_net(x_train, y_train)
    evaluate_model(model, x_test, y_test)
    features_weights(model, x_train.columns.tolist())

    #########################
    # Replication on an independent draw from the SAME generator.
    # This is a leakage and stability check, NOT external validation:
    # the data-generating process is identical, only the seed differs.
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from data.generate_data import generate

    out_dir = _Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    replicate_path = out_dir / "replicate_data.parquet"

    data = generate(100000, seed=99)
    data.to_parquet(replicate_path, engine="pyarrow", index=False)

    df = load_data(str(replicate_path))
    x, y = split_features_and_target(df, target_col="dx_ftd")
    valid_idx = y.dropna().index
    x, y = x.loc[valid_idx], y.loc[valid_idx]

    # evaluate
    evaluate_model(model, x, y)
    features_weights(model, x_train.columns.tolist())

  

