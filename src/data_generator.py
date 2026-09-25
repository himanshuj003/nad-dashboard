"""Synthetic network traffic data generator with injected anomalies."""
from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def _random_ip(rng, private=True):
    if private: return f"192.168.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
    return f"{rng.integers(1, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"

def generate_normal_flows(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime.now() - timedelta(hours=2)
    protocols = rng.choice(["TCP", "UDP", "ICMP"], size=n_samples, p=[0.75, 0.20, 0.05])
    common = [80, 443, 22, 53, 25, 110, 143, 993, 995, 3306, 5432, 8080]
    dst_ports = np.where(rng.random(n_samples) < 0.6, rng.choice(common, size=n_samples),
                         rng.choice(common + list(range(1024, 5000)), size=n_samples))
    packet_count = rng.lognormal(3.0, 1.2, n_samples).astype(int).clip(1, 5000)
    byte_count = (packet_count * rng.uniform(40, 1400, n_samples)).astype(int)
    duration = rng.exponential(2.5, n_samples).clip(0.01, 120)
    return pd.DataFrame({
        "timestamp": [start + timedelta(seconds=float(x)) for x in np.cumsum(rng.exponential(0.5, n_samples))],
        "src_ip": [_random_ip(rng) for _ in range(n_samples)],
        "dst_ip": [_random_ip(rng, False) for _ in range(n_samples)],
        "src_port": rng.integers(1024, 65535, n_samples), "dst_port": dst_ports, "protocol": protocols,
        "packet_count": packet_count, "byte_count": byte_count, "duration_sec": np.round(duration, 3),
        "packets_per_sec": np.round(packet_count/duration, 2), "bytes_per_sec": np.round(byte_count/duration, 2),
        "syn_count": rng.integers(0, 3, n_samples), "fin_count": rng.integers(0, 3, n_samples),
        "rst_count": rng.integers(0, 2, n_samples), "unique_dst_ports": rng.integers(1, 5, n_samples),
        "is_anomaly": np.zeros(n_samples, dtype=int),
    })

def inject_anomalies(df, anomaly_ratio=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n_anom = max(1, int(len(df) * anomaly_ratio))
    indices = rng.choice(len(df), size=n_anom, replace=False)
    df = df.copy()
    types = rng.choice(["port_scan", "ddos_flood", "exfiltration", "rare_port", "rst_storm"], size=n_anom)
    for idx, t in zip(indices, types):
        if t == "port_scan":
            df.loc[idx, "unique_dst_ports"] = rng.integers(50, 200)
            df.loc[idx, "packet_count"] = rng.integers(50, 300)
            df.loc[idx, "byte_count"] = df.loc[idx, "packet_count"] * rng.integers(40, 80)
            df.loc[idx, "duration_sec"] = rng.uniform(0.5, 5)
        elif t == "ddos_flood":
            df.loc[idx, "packet_count"] = rng.integers(5000, 50000)
            df.loc[idx, "byte_count"] = df.loc[idx, "packet_count"] * rng.integers(60, 200)
            df.loc[idx, "duration_sec"] = rng.uniform(0.1, 2)
            df.loc[idx, "syn_count"] = rng.integers(100, 1000)
        elif t == "exfiltration":
            df.loc[idx, "byte_count"] = rng.integers(5_000_000, 50_000_000)
            df.loc[idx, "packet_count"] = rng.integers(2000, 20000)
            df.loc[idx, "duration_sec"] = rng.uniform(30, 300)
        elif t == "rare_port":
            df.loc[idx, "dst_port"] = rng.choice([31337, 4444, 6667, 12345, 1337])
        else:
            df.loc[idx, "rst_count"] = rng.integers(50, 500)
            df.loc[idx, "packet_count"] = rng.integers(100, 1000)
        dur = max(df.loc[idx, "duration_sec"], 0.01)
        df.loc[idx, "packets_per_sec"] = round(df.loc[idx, "packet_count"] / dur, 2)
        df.loc[idx, "bytes_per_sec"] = round(df.loc[idx, "byte_count"] / dur, 2)
        df.loc[idx, "is_anomaly"] = 1
    return df

def generate_dataset(n_samples=5000, anomaly_ratio=0.05, seed=42, save_path=None):
    df = inject_anomalies(generate_normal_flows(n_samples, seed), anomaly_ratio, seed+1)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if save_path:
        df.to_csv(save_path, index=False)
    return df
