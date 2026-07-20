import pickle
import numpy as np
import pandas as pd

def load_pickle(filepath):
    """Loads a pickle file supporting both standard pickle and Pandas DataFrame serialization."""
    try:
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        print("Artifact successfully loaded.")
        return payload
    except FileNotFoundError:
        print(f"Could not find the file at {filepath}. Please verify the path.")
        raise

def prepare_data(df, target_prefix='target__', id_col='user_id'):
    print(f"DataFrame columns: {df.columns.tolist()}")
    target_cols = [c for c in df.columns if c.lower().startswith(target_prefix.lower())]

    if not target_cols:
        print(f"No target columns found with prefix '{target_prefix}'.")
        raise ValueError(f"No target columns found with prefix '{target_prefix}' in columns: {df.columns.tolist()}")

    feature_cols = [c for c in df.columns if c not in target_cols and c != id_col]
    print(f"Found target columns: {target_cols}")
    print(f"Feature columns count: {len(feature_cols)}")

    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0).values
    y = df[target_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    return X, y, feature_cols, target_cols