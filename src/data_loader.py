import pandas as pd
from pathlib import Path

def load_data(filepath: str = "data/mock9.parquet") -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please run generate_mock_data.py first."
        )
    
    df = pd.read_parquet(path)
    print(f"Successfully loaded dataset from {path}. Shape: {df.shape}")
    return df

def split_features_and_target(df: pd.DataFrame, target_col: str = "dx_ftd"):
    """ Splits the dataframe into feature matrix X and target vector y. """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")
    
    # Drop identifiers or non-feature columns if they exist in mock data
    drop_cols = [target_col, "ID"] if "ID" in df.columns else [target_col]
    
    x = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target_col]
    
    return x, y

if __name__ == "__main__":
    df = load_data()
    x, y = split_features_and_target(df)
    print(f"Features shape (x): {x.shape}")
    print(f"Target shape (y): {y.shape}")
    print("\nTarget class distribution:")
    print(y.value_counts())
    print(x)
    print(y)
