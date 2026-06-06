"""Wysyła przykładowe requesty z inference_sample.csv do lokalnego serwisu."""

import argparse
from pathlib import Path

import pandas as pd
import requests

SERVICE_URL = "http://127.0.0.1:8000/predict"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INFERENCE_PATH = DATA_DIR / "inference_sample.csv"
TRAINING_PATH = DATA_DIR / "training_data.csv"


def build_payload(row: pd.Series) -> dict:
    payload = row.to_dict()
    for key in ("user_id", "listing_id"):
        payload[key] = int(payload[key])
    if "Y" in payload and pd.isna(payload["Y"]):
        payload.pop("Y", None)
    elif "Y" in payload:
        payload["Y"] = int(payload["Y"])
    return payload


def main(n_requests: int, with_labels: int) -> None:
    frames = [pd.read_csv(INFERENCE_PATH).head(n_requests)]
    if with_labels > 0:
        labeled = pd.read_csv(TRAINING_PATH).sample(with_labels, random_state=42)
        frames.append(labeled)
    sample = pd.concat(frames, ignore_index=True)

    for _, row in sample.iterrows():
        payload = build_payload(row)
        response = requests.post(SERVICE_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        print(
            f"user={result['user_id']} listing={result['listing_id']} "
            f"variant={result['model_variant']} prob={result['booking_probability']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Liczba requestów z inference_sample.csv")
    parser.add_argument(
        "--with-labels",
        type=int,
        default=30,
        help="Dodatkowa liczba requestów z training_data.csv (z polem Y)",
    )
    args = parser.parse_args()
    main(args.n, args.with_labels)
