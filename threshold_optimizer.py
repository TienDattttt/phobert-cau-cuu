from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

try:
    from peft import AutoPeftModelForSequenceClassification
except ImportError:  # pragma: no cover - optional dependency
    AutoPeftModelForSequenceClassification = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


TEXT_COLUMNS = ("model_input_text", "segmented_text", "normalized_text", "text")
DEFAULT_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


class EncodedCommentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        text_column = next((column for column in TEXT_COLUMNS if column in frame.columns), None)
        if text_column is None:
            raise ValueError("Expected one of model_input_text/segmented_text/normalized_text/text in validation CSV.")

        texts = frame[text_column].fillna("").astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    results_dir = project_dir / "results"

    parser = argparse.ArgumentParser(description="Optimize decision threshold for the Vietnamese emergency classifier.")
    parser.add_argument("--model-dir", type=Path, default=project_dir / "saved_model")
    parser.add_argument("--val-csv", type=Path, default=data_dir / "val.csv")
    parser.add_argument("--results-dir", type=Path, default=results_dir)
    parser.add_argument("--config-json", type=Path, default=project_dir / "config.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--min-precision", type=float, default=0.60)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_thresholds(threshold_text: str) -> list[float]:
    thresholds = [float(chunk.strip()) for chunk in threshold_text.split(",") if chunk.strip()]
    if not thresholds:
        raise ValueError("At least one threshold must be provided.")
    return thresholds


def load_validation_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Validation CSV not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "label" not in frame.columns:
        raise ValueError(f"Missing required 'label' column in {path}")
    frame["label"] = frame["label"].astype(int)
    return frame


def load_model_and_tokenizer(model_dir: Path, device: torch.device) -> tuple[Any, Any, str]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token

    local_adapter_config = model_dir / "adapter_config.json"
    if local_adapter_config.exists():
        if AutoPeftModelForSequenceClassification is None:
            raise RuntimeError("PEFT adapter checkpoint detected but `peft` is not installed.")
        model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir)
        load_mode = "peft-local"
    elif AutoPeftModelForSequenceClassification is not None:
        try:
            model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir)
            load_mode = "peft-auto"
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            load_mode = "transformers"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        load_mode = "transformers"

    model.to(device)
    model.eval()
    return model, tokenizer, load_mode


def predict_probabilities(
    frame: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    dataset = EncodedCommentDataset(frame, tokenizer, max_length)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            probabilities.append(probs)

    if not probabilities:
        return np.array([], dtype=np.float32)
    return np.concatenate(probabilities).astype(np.float32)


def evaluate_threshold(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    preds = (probabilities >= threshold).astype(int)
    false_negatives = int(((labels == 1) & (preds == 0)).sum())
    false_positives = int(((labels == 0) & (preds == 1)).sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def evaluate_thresholds(labels: np.ndarray, probabilities: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = [evaluate_threshold(labels, probabilities, threshold) for threshold in thresholds]
    return pd.DataFrame(rows)


def select_optimal_threshold(metrics_frame: pd.DataFrame, min_recall: float, min_precision: float) -> tuple[pd.Series, str]:
    eligible = metrics_frame[
        (metrics_frame["recall"] >= min_recall)
        & (metrics_frame["precision"] > min_precision)
    ].copy()
    if not eligible.empty:
        chosen = eligible.sort_values(by=["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
        return chosen, "meets_constraints"

    precision_only = metrics_frame[metrics_frame["precision"] > min_precision].copy()
    if not precision_only.empty:
        chosen = precision_only.sort_values(by=["recall", "f1", "threshold"], ascending=[False, False, True]).iloc[0]
        return chosen, "best_recall_with_precision_guard"

    chosen = metrics_frame.sort_values(by=["recall", "f1", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return chosen, "best_available_fallback"


def save_pr_curve(labels: np.ndarray, probabilities: np.ndarray, output_path: Path) -> None:
    precision_values, recall_values, _ = precision_recall_curve(labels, probabilities)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall_values, precision_values, color="#d94841", linewidth=2.2)
    axis.set_title("Precision-Recall Curve")
    axis.set_xlabel("Recall (cau_cuu)")
    axis.set_ylabel("Precision (cau_cuu)")
    axis.grid(alpha=0.25)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_config(config_json: Path, chosen: pd.Series, selection_reason: str) -> None:
    payload = {
        "optimal_threshold": round(float(chosen["threshold"]), 4),
        "expected_recall": round(float(chosen["recall"]), 4),
        "expected_precision": round(float(chosen["precision"]), 4),
        "selection_reason": selection_reason,
    }
    config_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.config_json.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    frame = load_validation_split(args.val_csv)
    labels = frame["label"].to_numpy(dtype=np.int64)
    model, tokenizer, load_mode = load_model_and_tokenizer(args.model_dir, device)
    probabilities = predict_probabilities(
        frame,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    metrics_frame = evaluate_thresholds(labels, probabilities, thresholds)
    chosen, selection_reason = select_optimal_threshold(metrics_frame, args.min_recall, args.min_precision)

    metrics_csv = args.results_dir / "threshold_metrics.csv"
    pr_curve_png = args.results_dir / "pr_curve.png"
    metrics_frame.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    save_pr_curve(labels, probabilities, pr_curve_png)
    save_config(args.config_json, chosen, selection_reason)

    display_frame = metrics_frame.copy()
    for column in ("threshold", "precision", "recall", "f1"):
        display_frame[column] = display_frame[column].map(lambda value: f"{float(value):.4f}")

    print(f"Model load mode: {load_mode}")
    print(f"Validation rows: {len(frame)}")
    print(display_frame.to_string(index=False))
    print()
    print(f"Optimal threshold: {float(chosen['threshold']):.4f}")
    print(f"Expected recall: {float(chosen['recall']):.4f}")
    print(f"Expected precision: {float(chosen['precision']):.4f}")
    print(f"Selection reason: {selection_reason}")
    print(f"Saved PR curve: {pr_curve_png}")
    print(f"Saved threshold metrics: {metrics_csv}")
    print(f"Saved config: {args.config_json}")


if __name__ == "__main__":
    main()
