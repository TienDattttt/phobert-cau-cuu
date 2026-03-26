from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

try:
    from peft import AutoPeftModelForSequenceClassification
except ImportError:  # pragma: no cover - optional dependency
    AutoPeftModelForSequenceClassification = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

try:
    from underthesea import word_tokenize
except ImportError:  # pragma: no cover - optional dependency
    word_tokenize = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:\d[ .-]?){8,10}\d")
GPS_PATTERN = re.compile(r"[-+]?\d{1,3}[.,]\d+")
ADDRESS_HINT_PATTERN = re.compile(
    r"\b(thon|thon\.|to|to\.|doi|xom|ap|xa|phuong|quan|huyen|duong|hem|ngo|khu pho|kp|doi\s+\d+)\b",
    re.IGNORECASE,
)
SAFE_HINTS = (
    "da duoc cuu",
    "da an toan",
    "cam on",
    "duoc cuu hom qua",
)
RESCUE_TERMS = (
    "cuu",
    "sos",
    "ai cuu",
    "can cuu",
    "can giup",
    "cuu ho",
    "khan cap",
)
TRAP_TERMS = (
    "ket",
    "mac ket",
    "khong thoat",
    "thoat khong duoc",
    "co lap",
)
FLOOD_TERMS = (
    "ngap",
    "nuoc len",
    "nuoc dang",
    "ngap toi",
    "ngap sau",
    "toi mai",
    "toi nguc",
)
REQUEST_TERMS = (
    "can ca no",
    "can xuong",
    "can thuyen",
    "can cuu ho",
)
VULNERABLE_TERMS = (
    "nguoi gia",
    "tre em",
    "em be",
    "mot minh",
    "ba bau",
)


class BatchCommentDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    def __len__(self) -> int:
        return len(next(iter(self.encodings.values()))) if self.encodings else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Filter Vietnamese SOS comments from a Facebook post JSON export.")
    parser.add_argument("--json-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=project_dir / "saved_model")
    parser.add_argument("--config-json", type=Path, default=project_dir / "config.json")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_comment_id(post_id: str, source: str, indices: Iterable[int]) -> str:
    suffix = "-".join(f"{index:04d}" for index in indices)
    return f"{post_id}-{source}-{suffix}"


def parse_reaction_count(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def flatten_comments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    post_id = str(payload.get("post_id") or payload.get("post_info", {}).get("post_id") or "unknown_post")
    comments = payload.get("comments") or []
    rows: list[dict[str, Any]] = []

    def visit(items: list[dict[str, Any]], lineage: list[int], parent_author: str | None, parent_source: str) -> None:
        for offset, item in enumerate(items, start=1):
            current_lineage = [*lineage, offset]
            source = parent_source if not lineage else "reply"
            author = str(item.get("author") or "").strip()
            text = item.get("text")
            row = {
                "id": make_comment_id(post_id, source, current_lineage),
                "text": "" if text is None else str(text),
                "author": author,
                "timestamp": str(item.get("timestamp") or ""),
                "reaction_count": parse_reaction_count(item.get("reaction_count")),
                "source": source,
                "parent_author": parent_author,
                "post_id": post_id,
            }
            rows.append(row)

            replies = item.get("replies") or []
            if replies:
                visit(replies, current_lineage, author or None, "reply")

    visit(comments, [], None, "top")
    return rows


def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = ZERO_WIDTH_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


def normalize_for_rules(text: str) -> str:
    normalized = normalize_whitespace(text).lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return WHITESPACE_PATTERN.sub(" ", ascii_text).strip()


def preprocess_for_model(text: str) -> str:
    normalized = normalize_whitespace(text)
    if not normalized:
        return ""
    if word_tokenize is None:
        return normalized
    segmented = word_tokenize(normalized, format="text")
    return normalize_whitespace(segmented)


def is_effectively_empty(text: str) -> bool:
    if not text or not text.strip():
        return True
    without_urls = URL_PATTERN.sub(" ", text)
    without_symbols = re.sub(r"[\W_]+", "", without_urls, flags=re.UNICODE)
    return not without_symbols.strip()


def should_override_to_emergency(text: str, prob_cau_cuu: float) -> bool:
    lowered = normalize_for_rules(text)
    if any(token in lowered for token in SAFE_HINTS):
        return False

    has_rescue = any(token in lowered for token in RESCUE_TERMS)
    has_trap = any(token in lowered for token in TRAP_TERMS)
    has_flood = any(token in lowered for token in FLOOD_TERMS)
    has_request = any(token in lowered for token in REQUEST_TERMS)
    has_vulnerable = any(token in lowered for token in VULNERABLE_TERMS)
    has_phone = bool(PHONE_PATTERN.search(lowered))

    strong_pattern = (
        has_request
        or (has_rescue and has_trap)
        or (has_rescue and has_flood)
        or (has_trap and has_flood)
        or (has_phone and (has_rescue or has_flood or has_trap))
        or (has_vulnerable and (has_rescue or has_flood or has_trap))
    )
    return strong_pattern and prob_cau_cuu >= 0.20


def load_threshold(config_json: Path, threshold_override: float | None) -> float:
    if threshold_override is not None:
        return float(threshold_override)
    if config_json.exists():
        payload = json.loads(config_json.read_text(encoding="utf-8"))
        if "optimal_threshold" in payload:
            return float(payload["optimal_threshold"])
    return 0.5


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


def predict_batch_probabilities(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> list[float]:
    if not texts:
        return []

    dataset = BatchCommentDataset(texts, tokenizer, max_length)
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            probabilities.extend(float(value) for value in probs)
    return probabilities


def chunked(items: list[dict[str, Any]], chunk_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def serialize_output(rows: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_file.suffix.lower()

    if suffix == ".csv":
        frame = pd.DataFrame(rows)
        frame.to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    output_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_cau_cuu_comments(
    json_file: str,
    output_file: str,
    threshold: float = None,
    min_confidence: float = None,
    batch_size: int = 32,
    model_dir: str | None = None,
    config_json: str | None = None,
    max_length: int = 256,
    device: str = "auto",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    project_dir = Path(__file__).resolve().parent
    json_path = Path(json_file)
    output_path = Path(output_file)
    model_path = Path(model_dir) if model_dir is not None else project_dir / "saved_model"
    config_path = Path(config_json) if config_json is not None else project_dir / "config.json"
    resolved_threshold = load_threshold(config_path, threshold)
    resolved_device = resolve_device(device)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    flattened = flatten_comments(payload)
    total_comments = len(flattened)

    model, tokenizer, load_mode = load_model_and_tokenizer(model_path, resolved_device)

    kept_rows: list[dict[str, Any]] = []
    processed_comments = 0
    inferable_comments = 0
    iterator = chunked(flattened, batch_size)
    if tqdm is not None:
        iterator = tqdm(list(iterator), total=(total_comments + batch_size - 1) // batch_size, desc="Filtering comments")
    else:
        iterator = chunked(flattened, batch_size)

    for batch in iterator:
        prepared_items: list[tuple[dict[str, Any], str]] = []
        for row in batch:
            raw_text = row["text"]
            if is_effectively_empty(raw_text):
                continue
            prepared_text = preprocess_for_model(raw_text)
            if not prepared_text:
                continue
            prepared_items.append((row, prepared_text))

        probabilities = predict_batch_probabilities(
            [item[1] for item in prepared_items],
            tokenizer=tokenizer,
            model=model,
            device=resolved_device,
            max_length=max_length,
            batch_size=batch_size,
        )

        inferable_comments += len(prepared_items)
        for (row, _prepared_text), probability in zip(prepared_items, probabilities):
            probability = float(probability)
            is_positive = probability >= resolved_threshold or should_override_to_emergency(row["text"], probability)
            if not is_positive:
                continue
            if min_confidence is not None and probability < float(min_confidence):
                continue
            kept_rows.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "author": row["author"],
                    "timestamp": row["timestamp"],
                    "reaction_count": row["reaction_count"],
                    "confidence": round(probability, 4),
                    "source": row["source"],
                    "parent_author": row["parent_author"],
                    "post_id": row["post_id"],
                }
            )

        processed_comments += len(batch)
        if processed_comments % 100 == 0 or processed_comments == total_comments:
            print(f"Processed {processed_comments}/{total_comments} comments")

    kept_rows.sort(key=lambda item: (-float(item["confidence"]), -int(item["reaction_count"]), str(item["timestamp"])))
    serialize_output(kept_rows, output_path)

    avg_confidence = sum(float(row["confidence"]) for row in kept_rows) / len(kept_rows) if kept_rows else 0.0
    processing_time_seconds = time.perf_counter() - started_at
    summary = {
        "total_comments": total_comments,
        "cau_cuu_count": len(kept_rows),
        "filter_rate": round((total_comments - len(kept_rows)) / total_comments, 4) if total_comments else 0.0,
        "avg_confidence": round(avg_confidence, 4),
        "output_file": str(output_path),
        "processing_time_seconds": round(processing_time_seconds, 2),
        "threshold": round(resolved_threshold, 4),
        "inferable_comments": inferable_comments,
        "model_load_mode": load_mode,
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = filter_cau_cuu_comments(
        json_file=str(args.json_file),
        output_file=str(args.output_file),
        threshold=args.threshold,
        min_confidence=args.min_confidence,
        batch_size=args.batch_size,
        model_dir=str(args.model_dir),
        config_json=str(args.config_json),
        max_length=args.max_length,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
