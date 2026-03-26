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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

try:
    from peft import AutoPeftModelForSequenceClassification
except ImportError:  # pragma: no cover - environment dependent
    AutoPeftModelForSequenceClassification = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ID2LABEL = {0: "khong_cau_cuu", 1: "cau_cuu"}
LABEL_DISPLAY = [ID2LABEL[0], ID2LABEL[1]]
TEXT_COLUMNS = ("model_input_text", "segmented_text", "normalized_text", "text")


class EncodedCommentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        text_column = next((column for column in TEXT_COLUMNS if column in frame.columns), None)
        if text_column is None:
            raise ValueError("Expected one of model_input_text/segmented_text/normalized_text/text in evaluation CSV.")

        self.ids = frame["id"].fillna("").astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.raw_text = frame.get("text", frame[text_column]).fillna("").astype(str).tolist()
        texts = frame[text_column].fillna("").astype(str).tolist()
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

    parser = argparse.ArgumentParser(description="Evaluate PhoBERT emergency comment classifier on validation and test splits.")
    parser.add_argument("--model-dir", type=Path, default=project_dir / "saved_model")
    parser.add_argument("--val-csv", type=Path, default=data_dir / "val.csv")
    parser.add_argument("--test-csv", type=Path, default=data_dir / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=results_dir)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "label" not in frame.columns:
        raise ValueError(f"Missing required 'label' column in {path}")
    frame["label"] = frame["label"].astype(int)
    if "confidence" in frame.columns:
        frame["confidence"] = frame["confidence"].astype(float)
    return frame


def load_model_and_tokenizer(model_dir: Path, device: torch.device) -> tuple[Any, Any]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists() and AutoPeftModelForSequenceClassification is not None:
        model = AutoPeftModelForSequenceClassification.from_pretrained(model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.eval()
    model.to(device)
    return model, tokenizer


def predict_probabilities(
    frame: pd.DataFrame,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
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


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    preds = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_cau_cuu": float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)),
        "recall_cau_cuu": float(recall_score(labels, preds, average="binary", pos_label=1, zero_division=0)),
        "precision_cau_cuu": float(precision_score(labels, preds, average="binary", pos_label=1, zero_division=0)),
        "positive_predictions": int(preds.sum()),
    }


def build_threshold_candidates(probabilities: np.ndarray, minimum: float, maximum: float, step: float) -> np.ndarray:
    grid = np.arange(minimum, maximum + (step / 2), step, dtype=np.float32)
    merged = np.concatenate([grid, probabilities.astype(np.float32)])
    clipped = np.clip(merged, 1e-6, 1 - 1e-6)
    return np.unique(np.round(clipped, 6))


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    target_recall: float,
    threshold_override: float | None,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, pd.DataFrame, str]:
    if threshold_override is not None:
        metrics = [calculate_metrics(labels, probabilities, threshold_override)]
        return float(threshold_override), pd.DataFrame(metrics), "manual"

    candidates = build_threshold_candidates(probabilities, threshold_min, threshold_max, threshold_step)
    search_frame = pd.DataFrame(calculate_metrics(labels, probabilities, float(threshold)) for threshold in candidates)

    eligible = search_frame[search_frame["recall_cau_cuu"] >= target_recall].copy()
    if not eligible.empty:
        chosen = (
            eligible.sort_values(
                by=["f1_cau_cuu", "accuracy", "precision_cau_cuu", "threshold"],
                ascending=[False, False, False, False],
            )
            .iloc[0]
        )
        return float(chosen["threshold"]), search_frame, "target_recall"

    fallback = (
        search_frame.sort_values(
            by=["recall_cau_cuu", "f1_cau_cuu", "accuracy", "threshold"],
            ascending=[False, False, False, False],
        )
        .iloc[0]
    )
    return float(fallback["threshold"]), search_frame, "best_available_recall"


def save_confusion_matrix_image(matrix: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(LABEL_DISPLAY)),
        yticks=np.arange(len(LABEL_DISPLAY)),
        xticklabels=LABEL_DISPLAY,
        yticklabels=LABEL_DISPLAY,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(matrix[row, column], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_test_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    preds = (probabilities >= threshold).astype(int)
    result = frame.copy()
    result["prob_cau_cuu"] = probabilities
    result["prob_khong_cau_cuu"] = 1.0 - probabilities
    result["predicted_label"] = preds
    result["true_label_name"] = result["label"].map(ID2LABEL)
    result["predicted_label_name"] = result["predicted_label"].map(ID2LABEL)
    result["confidence"] = np.where(result["predicted_label"] == 1, result["prob_cau_cuu"], result["prob_khong_cau_cuu"])
    result["threshold"] = float(threshold)
    result["is_emergency"] = result["predicted_label"].eq(1)
    result["is_correct"] = result["predicted_label"].eq(result["label"])

    preferred_columns = [
        "id",
        "text",
        "model_input_text",
        "label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "prob_cau_cuu",
        "prob_khong_cau_cuu",
        "confidence",
        "threshold",
        "is_emergency",
        "is_correct",
    ]
    columns = [column for column in preferred_columns if column in result.columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig", columns=columns)


def print_target_table(metrics: dict[str, float | int]) -> None:
    recall_value = float(metrics["recall_cau_cuu"])
    f1_value = float(metrics["f1_cau_cuu"])
    accuracy_value = float(metrics["accuracy"])

    print("=" * 32)
    print("TARGET:")
    print("Recall (cau_cuu):  >= 0.85  <- QUAN TRONG NHAT")
    print("F1 (cau_cuu):      >= 0.80")
    print("Accuracy:          >= 0.82")
    print("=" * 32)
    print("RESULT:")
    print(f"Recall (cau_cuu):  {recall_value:.4f}  {'PASS' if recall_value >= 0.85 else 'FAIL'}")
    print(f"F1 (cau_cuu):      {f1_value:.4f}  {'PASS' if f1_value >= 0.80 else 'FAIL'}")
    print(f"Accuracy:          {accuracy_value:.4f}  {'PASS' if accuracy_value >= 0.82 else 'FAIL'}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.model_dir, device)

    val_frame = load_split(args.val_csv)
    test_frame = load_split(args.test_csv)

    val_probabilities = predict_probabilities(
        val_frame,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    test_probabilities = predict_probabilities(
        test_frame,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    val_labels = val_frame["label"].to_numpy(dtype=np.int64)
    test_labels = test_frame["label"].to_numpy(dtype=np.int64)

    threshold, threshold_search, threshold_source = select_threshold(
        labels=val_labels,
        probabilities=val_probabilities,
        target_recall=args.target_recall,
        threshold_override=args.threshold,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
    )

    threshold_search_path = args.output_dir / "validation_threshold_search.csv"
    threshold_search.to_csv(threshold_search_path, index=False, encoding="utf-8-sig")

    val_metrics = calculate_metrics(val_labels, val_probabilities, threshold)
    test_metrics = calculate_metrics(test_labels, test_probabilities, threshold)
    test_preds = (test_probabilities >= threshold).astype(int)

    report = classification_report(
        test_labels,
        test_preds,
        labels=[0, 1],
        target_names=LABEL_DISPLAY,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(test_labels, test_preds, labels=[0, 1])

    predictions_path = args.output_dir / "test_predictions.csv"
    confusion_path = args.output_dir / "confusion_matrix.png"
    summary_path = args.output_dir / "evaluation_summary.json"

    save_test_predictions(test_frame, test_probabilities, threshold, predictions_path)
    save_confusion_matrix_image(matrix, confusion_path)

    summary = {
        "model_dir": str(args.model_dir),
        "device": str(device),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "target_recall": args.target_recall,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": matrix.tolist(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model directory: {args.model_dir}")
    print(f"Device: {device}")
    print(f"Selected threshold: {threshold:.4f} ({threshold_source})")
    print(f"Validation threshold search saved: {threshold_search_path}")
    print("Validation metrics at selected threshold:")
    print(json.dumps(val_metrics, ensure_ascii=False, indent=2))
    print()
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)
    print_target_table(test_metrics)
    print(f"Saved test predictions: {predictions_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
