"""Anomaly detection engines."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass, field
from .features import prepare_features
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

@dataclass
class DetectionResult:
    labels: np.ndarray
    scores: np.ndarray
    method: str
    details: Dict[str, Any] = field(default_factory=dict)
    @property
    def n_anomalies(self):
        return int(self.labels.sum())

class StatisticalDetector:
    def __init__(self, threshold=4.5, min_features=2):
        self.threshold, self.min_features = threshold, min_features
        self.medians_ = self.mads_ = None
        self.feature_names_ = []
    def fit(self, df, feature_cols=None):
        X, names = prepare_features(df, feature_cols)
        self.feature_names_ = names
        self.medians_ = X.median()
        self.mads_ = (X - self.medians_).abs().median().replace(0, 1e-9)
        return self
    def predict(self, df):
        X, _ = prepare_features(df, self.feature_names_)
        abs_z = (0.6745 * (X - self.medians_) / self.mads_).abs()
        scores = abs_z.max(axis=1).values
        labels = ((abs_z > self.threshold).sum(axis=1).values >= self.min_features).astype(int)
        return DetectionResult(labels=labels, scores=scores, method="statistical")
    def fit_predict(self, df, feature_cols=None):
        return self.fit(df, feature_cols).predict(df)

class IsolationForestDetector:
    def __init__(self, contamination=0.05, n_estimators=200, random_state=42):
        if not HAS_SKLEARN: raise ImportError("pip install scikit-learn")
        self.contamination, self.n_estimators, self.random_state = contamination, n_estimators, random_state
        self.model_ = self.scaler_ = None
        self.feature_names_ = []
    def fit(self, df, feature_cols=None):
        X, names = prepare_features(df, feature_cols)
        self.feature_names_ = names
        self.scaler_ = StandardScaler()
        self.model_ = IsolationForest(contamination=self.contamination, n_estimators=self.n_estimators, random_state=self.random_state)
        self.model_.fit(self.scaler_.fit_transform(X))
        return self
    def predict(self, df):
        X, _ = prepare_features(df, self.feature_names_)
        Xs = self.scaler_.transform(X)
        labels = (self.model_.predict(Xs) == -1).astype(int)
        scores = -self.model_.decision_function(Xs)
        return DetectionResult(labels=labels, scores=scores, method="isolation_forest")
    def fit_predict(self, df, feature_cols=None):
        return self.fit(df, feature_cols).predict(df)

class RuleBasedDetector:
    def __init__(self):
        self.rules = [
            ("port_scan", lambda r: r.get("unique_dst_ports", 0) >= 30, 0.8),
            ("syn_flood", lambda r: r.get("syn_count", 0) >= 50, 0.9),
            ("rst_storm", lambda r: r.get("rst_count", 0) >= 20, 0.7),
            ("high_rate", lambda r: r.get("packets_per_sec", 0) >= 2000, 0.85),
            ("large_transfer", lambda r: r.get("byte_count", 0) >= 5_000_000, 0.75),
            ("suspicious_port", lambda r: r.get("dst_port", 0) in {31337, 4444, 6667, 1337, 65535}, 0.6),
            ("short_burst", lambda r: r.get("duration_sec", 999) < 1 and r.get("packet_count", 0) > 500, 0.8),
        ]
    def predict(self, df):
        labels = np.zeros(len(df), dtype=int)
        scores = np.zeros(len(df), dtype=float)
        for i, row in enumerate(df.to_dict("records")):
            sev = 0.0
            for _, cond, s in self.rules:
                try:
                    if cond(row): sev = max(sev, s)
                except Exception: pass
            if sev > 0: labels[i], scores[i] = 1, sev
        return DetectionResult(labels=labels, scores=scores, method="rule_based")
    def fit(self, df=None, **kw): return self
    def fit_predict(self, df, **kw): return self.predict(df)

def _norm(a):
    a = np.asarray(a, dtype=float)
    mn, mx = a.min(), a.max()
    return np.zeros_like(a) if mx-mn < 1e-12 else (a-mn)/(mx-mn)

class HybridDetector:
    def __init__(self, contamination=0.05, z_threshold=3.5, min_votes=2):
        self.stat = StatisticalDetector(threshold=z_threshold)
        self.iforest = IsolationForestDetector(contamination=contamination) if HAS_SKLEARN else None
        self.rules = RuleBasedDetector()
        self.min_votes = min_votes
        self._fitted = False
    def fit(self, df, feature_cols=None):
        self.stat.fit(df, feature_cols)
        if self.iforest: self.iforest.fit(df, feature_cols)
        self.rules.fit(df)
        self._fitted = True
        return self
    def predict(self, df):
        r_s, r_r = self.stat.predict(df), self.rules.predict(df)
        if self.iforest:
            r_i = self.iforest.predict(df)
            votes = r_s.labels + r_i.labels + r_r.labels
            scores = 0.35*_norm(r_s.scores) + 0.40*_norm(r_i.scores) + 0.25*r_r.scores
            if_a = int(r_i.labels.sum())
        else:
            votes = r_s.labels + r_r.labels
            scores = 0.6*_norm(r_s.scores) + 0.4*r_r.scores
            if_a = 0
        em = self.min_votes if self.iforest else min(self.min_votes, 2)
        return DetectionResult(labels=(votes >= em).astype(int), scores=scores, method="hybrid",
            details={"stat_anomalies": int(r_s.labels.sum()), "iforest_anomalies": if_a,
                     "rule_anomalies": int(r_r.labels.sum()), "sklearn_available": self.iforest is not None})
    def fit_predict(self, df, feature_cols=None):
        return self.fit(df, feature_cols).predict(df)

def get_detector(name, **kwargs):
    name = name.lower().replace("-", "_").replace(" ", "_")
    if name in ("stat", "statistical"): return StatisticalDetector(**kwargs)
    if name in ("iforest", "isolation_forest"):
        if not HAS_SKLEARN: raise ImportError("pip install scikit-learn")
        return IsolationForestDetector(**kwargs)
    if name in ("rule", "rule_based"): return RuleBasedDetector()
    if name in ("hybrid", "ensemble"): return HybridDetector(**kwargs)
    raise ValueError(f"Unknown detector: {name}")
