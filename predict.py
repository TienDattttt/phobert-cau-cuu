from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import AutoPeftModelForSequenceClassification
except ImportError:  # pragma: no cover - optional dependency
    AutoPeftModelForSequenceClassification = None

try:
    from underthesea import word_tokenize
except ImportError:  # pragma: no cover - optional dependency
    word_tokenize = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_SOURCE = PROJECT_DIR / "saved_model"
DEFAULT_SUMMARY_JSON = PROJECT_DIR / "results_final" / "evaluation_summary.json"
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")
PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:\d[ .-]?){8,10}\d")
ID2LABEL = {0: "khong_cau_cuu", 1: "cau_cuu"}
SOS_HINTS = (
    "cứu",
    "cuu",
    "sos",
    "kẹt",
    "ket",
    "mắc kẹt",
    "mac ket",
    "khẩn cấp",
    "khan cap",
    "ngập",
    "ngap",
    "không thoát",
    "khong thoat",
    "cần giúp",
    "can giup",
    "cần cứu",
    "can cuu",
)
SAFE_HINTS = (
    "đã được cứu",
    "da duoc cuu",
    "đã an toàn",
    "da an toan",
    "cảm ơn",
    "cam on",
)
RESCUE_TERMS = (
    "cứu",
    "cuu",
    "sos",
    "ai cứu",
    "ai cuu",
    "cần cứu",
    "can cuu",
    "cần giúp",
    "can giup",
    "cứu hộ",
    "cuu ho",
    "khẩn cấp",
    "khan cap",
)
TRAP_TERMS = (
    "kẹt",
    "ket",
    "mắc kẹt",
    "mac ket",
    "không thoát",
    "khong thoat",
    "thoát không được",
    "thoat khong duoc",
    "cô lập",
    "co lap",
)
FLOOD_TERMS = (
    "ngập",
    "ngap",
    "nước lên",
    "nuoc len",
    "nước dâng",
    "nuoc dang",
    "ngập tới",
    "ngap toi",
    "ngập sâu",
    "ngap sau",
    "tới mái",
    "toi mai",
    "tới ngực",
    "toi nguc",
)
REQUEST_TERMS = (
    "cần ca nô",
    "can ca no",
    "cần xuồng",
    "can xuong",
    "cần thuyền",
    "can thuyen",
    "cần cứu hộ",
    "can cuu ho",
)
VULNERABLE_TERMS = (
    "người già",
    "nguoi gia",
    "trẻ em",
    "tre em",
    "em bé",
    "em be",
    "một mình",
    "mot minh",
    "bà bầu",
    "ba bau",
)
DEFAULT_DEMO_TEXTS = [
    "Cứu với, nhà em đang ngập gần tới mái, còn 2 người già và 1 em bé bị kẹt ở Vĩnh Trung, số 0912345678",
    "SOS, nước lên rất nhanh, gia đình tôi không thoát ra được, cần ca nô gấp",
    "Mọi người chia sẻ tin bão giúp để ai cũng biết đường tránh nhé",
    "Nhà em đã được cứu rồi, cả nhà hiện an toàn, cảm ơn mọi người nhiều",
    "Khu này đang mưa lớn và có nguy cơ ngập, bà con chú ý theo dõi thông báo",
    "Mẹ em ở một mình, nước ngập tới ngực, ai cứu giúp với ạ",
]


@dataclass
class PredictionConfig:
    model_source: str
    summary_json: Path | None
    threshold: float
    max_length: int
    device: torch.device


class CauCuuPredictor:
    def __init__(self, config: PredictionConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_source, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token or self.tokenizer.unk_token
        self.model, self.model_load_mode = self._load_model(config.model_source)
        self.model.to(config.device)
        self.model.eval()

    def _load_model(self, model_source: str) -> tuple[Any, str]:
        source_path = Path(model_source)
        local_adapter_config = source_path / "adapter_config.json"

        if source_path.exists() and local_adapter_config.exists():
            if AutoPeftModelForSequenceClassification is None:
                raise RuntimeError(
                    "This checkpoint contains PEFT adapter files but `peft` is not installed. Install `peft` before inference."
                )
            model = AutoPeftModelForSequenceClassification.from_pretrained(model_source)
            return model, "peft-local"

        if AutoPeftModelForSequenceClassification is not None:
            try:
                model = AutoPeftModelForSequenceClassification.from_pretrained(model_source)
                return model, "peft-auto"
            except Exception:
                pass

        model = AutoModelForSequenceClassification.from_pretrained(model_source)
        return model, "transformers"

    def preprocess(self, text: str) -> str:
        normalized = unicodedata.normalize("NFC", text or "")
        normalized = ZERO_WIDTH_PATTERN.sub(" ", normalized)
        normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
        if not normalized:
            return ""
        if word_tokenize is None:
            return normalized
        segmented = word_tokenize(normalized, format="text")
        return WHITESPACE_PATTERN.sub(" ", segmented).strip()

    def predict_cau_cuu(self, text: str) -> dict[str, Any]:
        clean_text = self.preprocess(text)
        if not clean_text:
            return {
                "label": "khong_cau_cuu",
                "confidence": 0.0,
                "is_emergency": False,
                "warning": "Input text is empty.",
            }

        encoded = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        prob_cau_cuu = float(probabilities[1].item())
        model_positive = prob_cau_cuu >= self.config.threshold
        heuristic_override = self._should_override_to_emergency(text=text, prob_cau_cuu=prob_cau_cuu)
        is_emergency = model_positive or heuristic_override
        label = "cau_cuu" if is_emergency else "khong_cau_cuu"
        confidence = prob_cau_cuu if label == "cau_cuu" else 1.0 - prob_cau_cuu
        warning = self._build_warning(
            text=text,
            prob_cau_cuu=prob_cau_cuu,
            model_positive=model_positive,
            heuristic_override=heuristic_override,
        )

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "is_emergency": is_emergency,
            "warning": warning,
        }

    def _should_override_to_emergency(self, text: str, prob_cau_cuu: float) -> bool:
        lowered = self._normalize_for_rules(text)
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

    def _normalize_for_rules(self, text: str) -> str:
        lowered = unicodedata.normalize("NFC", text or "").lower()
        lowered = ZERO_WIDTH_PATTERN.sub(" ", lowered)
        lowered = WHITESPACE_PATTERN.sub(" ", lowered).strip()
        return lowered

    def _build_warning(self, text: str, prob_cau_cuu: float, model_positive: bool, heuristic_override: bool) -> str:
        lowered = self._normalize_for_rules(text)
        contains_sos = any(token in lowered for token in SOS_HINTS)
        contains_safe = any(token in lowered for token in SAFE_HINTS)

        if heuristic_override and not model_positive:
            return "Heuristic emergency override triggered because the text contains strong SOS signals. Manual verification should happen immediately."
        if model_positive and prob_cau_cuu >= 0.85:
            return "High-priority rescue signal detected. Manual verification should happen immediately."
        if model_positive:
            return "Possible emergency detected. Please verify location and contact details quickly."
        if contains_safe:
            return "Comment mentions safety or successful rescue; treat as update rather than active SOS."
        if contains_sos and not model_positive:
            return "Rescue-related wording detected but below decision threshold; manual review is recommended."
        return "No immediate rescue signal detected."


_PREDICTOR: CauCuuPredictor | None = None


def load_threshold(summary_json: Path | None, threshold_override: float | None) -> float:
    if threshold_override is not None:
        return float(threshold_override)
    if summary_json and summary_json.exists():
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
        if "threshold" in payload:
            return float(payload["threshold"])
    return 0.5


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predictor(
    model_source: str | Path = DEFAULT_MODEL_SOURCE,
    summary_json: str | Path | None = DEFAULT_SUMMARY_JSON,
    threshold: float | None = None,
    max_length: int = 256,
    device: str = "auto",
) -> CauCuuPredictor:
    global _PREDICTOR

    model_source_str = str(model_source)
    summary_path = Path(summary_json) if summary_json is not None else None
    resolved_threshold = load_threshold(summary_path, threshold)
    resolved_device = resolve_device(device)

    needs_reload = (
        _PREDICTOR is None
        or _PREDICTOR.config.model_source != model_source_str
        or _PREDICTOR.config.threshold != resolved_threshold
        or _PREDICTOR.config.max_length != max_length
        or str(_PREDICTOR.config.device) != str(resolved_device)
    )

    if needs_reload:
        config = PredictionConfig(
            model_source=model_source_str,
            summary_json=summary_path,
            threshold=resolved_threshold,
            max_length=max_length,
            device=resolved_device,
        )
        _PREDICTOR = CauCuuPredictor(config)
    return _PREDICTOR


def predict_cau_cuu(text: str) -> dict[str, Any]:
    predictor = get_predictor()
    return predictor.predict_cau_cuu(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the Vietnamese emergency comment classifier.")
    parser.add_argument("--model-source", default=str(DEFAULT_MODEL_SOURCE))
    parser.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = get_predictor(
        model_source=args.model_source,
        summary_json=args.summary_json,
        threshold=args.threshold,
        max_length=args.max_length,
        device=args.device,
    )

    texts = args.text or DEFAULT_DEMO_TEXTS
    if args.demo and not args.text:
        texts = DEFAULT_DEMO_TEXTS

    print(
        json.dumps(
            {
                "model_source": predictor.config.model_source,
                "threshold": predictor.config.threshold,
                "device": str(predictor.config.device),
                "samples": len(texts),
                "model_load_mode": predictor.model_load_mode,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    for index, text in enumerate(texts, start=1):
        result = predictor.predict_cau_cuu(text)
        payload = {
            "index": index,
            "text": text,
            **result,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
