"""Feature engineering for network flow anomaly detection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

NUMERIC_FEATURES: List[str] = [
    "packet_count", "byte_count", "duration_sec", "packets_per_sec",
    "bytes_per_sec", "syn_count", "fin_count", "rst_count", "unique_dst_ports",
]

def prepare_features(df, feature_cols=None, log_transform=True):
    cols = feature_cols or NUMERIC_FEATURES
    available = [c for c in cols if c in df.columns]
    X = df[available].copy().fillna(0)
    if log_transform:
        for c in available:
            if X[c].min() >= 0:
                X[c] = np.log1p(X[c].astype(float))
    return X, available
