import os
import csv
from datetime import datetime
from pathlib import Path
from typing import List

METRICS_DIR = os.getenv("METRICS_DIR", "/app/data/metrics")
METRICS_FILE = os.path.join(METRICS_DIR, "queries.csv")

FIELDS = [
    "timestamp",
    "prompt_version",
    "question",
    "answer",
    "latency_s",
    "answer_length",
    "num_sources",
    "sources",
]

def _ensure_file():
    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()

def log_query(
    question: str,
    answer: str,
    latency_s: float,
    sources: List[str],
    prompt_version: str,
):
    _ensure_file()
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt_version": prompt_version,
        "question": question,
        "answer": answer,
        "latency_s": round(float(latency_s), 4),
        "answer_length": len(answer or ""),
        "num_sources": len(sources or []),
        "sources": "|".join(sources or []),
    }
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)

def load_metrics():
    import pandas as pd
    if not os.path.exists(METRICS_FILE):
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(METRICS_FILE)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
