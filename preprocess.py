from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from underthesea import word_tokenize


ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    parser = argparse.ArgumentParser(description="Preprocess Vietnamese rescue comments for PhoBERT training.")
    parser.add_argument("--input-csv", type=Path, default=data_dir / "labeled_comments.csv")
    parser.add_argument("--train-csv", type=Path, default=data_dir / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=data_dir / "val.csv")
    parser.add_argument("--test-csv", type=Path, default=data_dir / "test.csv")
    parser.add_argument("--class-weights-json", type=Path, default=data_dir / "class_weights.json")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = ZERO_WIDTH_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def segment_text(text: str) -> str:
    if not text:
        return ""
    segmented = word_tokenize(text, format="text")
    return WHITESPACE_PATTERN.sub(" ", segmented).strip()


def prepare_dataframe(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["confidence"] = df["confidence"].astype(float)

    df["normalized_text"] = df["text"].map(normalize_text)
    df = df[df["normalized_text"].str.len() > 0].copy()
    df["segmented_text"] = df["normalized_text"].map(segment_text)
    df["model_input_text"] = df["segmented_text"].where(df["segmented_text"].str.len() > 0, df["normalized_text"])
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_class_weights(labels: pd.Series) -> dict[str, float]:
    classes = np.array(sorted(labels.unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {str(int(label)): float(weight) for label, weight in zip(classes, weights)}


def save_split(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "text", "normalized_text", "segmented_text", "model_input_text", "label", "confidence"]
    df.to_csv(output_path, index=False, encoding="utf-8-sig", columns=columns)


def save_class_weights(class_weights: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(class_weights, ensure_ascii=False, indent=2), encoding="utf-8")


def describe_split(name: str, df: pd.DataFrame) -> None:
    distribution = df["label"].value_counts().sort_index().to_dict()
    print(f"{name}: rows={len(df)}, distribution={distribution}")


def main() -> None:
    args = parse_args()
    df = prepare_dataframe(args.input_csv)
    train_df, val_df, test_df = stratified_split(df, args.random_state)
    class_weights = compute_class_weights(df["label"])

    save_split(train_df, args.train_csv)
    save_split(val_df, args.val_csv)
    save_split(test_df, args.test_csv)
    save_class_weights(class_weights, args.class_weights_json)

    print(f"Total cleaned rows: {len(df)}")
    print(f"Class weights: {class_weights}")
    describe_split("train", train_df)
    describe_split("val", val_df)
    describe_split("test", test_df)
    print("Sample segmented text:")
    for sample in train_df["model_input_text"].head(3):
        print(f"- {sample[:200]}")


if __name__ == "__main__":
    main()
