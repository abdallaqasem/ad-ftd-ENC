import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def perform_train_test_split(x: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """ Splits features and target into stratified train and test sets."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Training set shape: {x_train.shape}, Test set shape: {x_test.shape}")
    return x_train, x_test, y_train, y_test

def get_preprocessor() -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=5)),
    ])
    

if __name__ == "__main__":
    from data_loader import load_data, split_features_and_target
    
    df = load_data()
    x, y = split_features_and_target(df, target_col="dx_ftd")

    # remove null diagnosis
    valid_idx = y.dropna().index
    x = x.loc[valid_idx]
    y = y.loc[valid_idx]
    
    x_train, x_test, y_train, y_test = perform_train_test_split(x, y)
    
    preprocessor = get_preprocessor()
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    
    print("\nProcessed Training Features Preview:")
    print(x_train_processed.head(3))
