from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from upload_json_filter import filter_uploaded_json_with_predictor

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


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
LOCAL_MODEL_FALLBACK = PROJECT_DIR / "saved_model" / "checkpoint-171"
DEFAULT_MODEL_SOURCE = str(LOCAL_MODEL_FALLBACK if LOCAL_MODEL_FALLBACK.exists() else "dat201204/phobert-vi-caucu-classifier")
DEFAULT_SUMMARY_JSON = PROJECT_DIR / "results_final" / "evaluation_summary.json"
DEFAULT_CONFIG_JSON = PROJECT_DIR / "config.json"
DEFAULT_BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.4
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")
PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:\d[ .-]?){8,10}\d")
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
SINGLE_EXAMPLES = [
    ["Cứu với, nhà em đang ngập gần tới mái, còn 2 người già và 1 em bé bị kẹt ở Vĩnh Trung, số 0912345678"],
    ["SOS, nước lên rất nhanh, gia đình tôi không thoát ra được, cần ca nô gấp"],
    ["Nhà em đã được cứu rồi, cả nhà hiện an toàn, cảm ơn mọi người nhiều"],
    ["Mọi người chia sẻ tin bão giúp để ai cũng biết đường tránh nhé"],
]
BATCH_EXAMPLE = """Cứu với, nhà em đang ngập sâu và có người già bị kẹt\nSOS, nước lên rất nhanh, gia đình tôi không thoát ra được, cần ca nô gấp\nNhà em đã được cứu rồi, cả nhà hiện an toàn\nMọi người chia sẻ tin bão giúp để ai cũng biết đường tránh nhé"""

CSS = """
:root {
  --storm-navy: #0e2431;
  --rescue-red: #d94841;
  --safe-green: #1f8f5f;
  --signal-amber: #d7a33d;
  --mist: #edf3f5;
  --ink: #10202b;
}
.gradio-container {
  background:
    radial-gradient(circle at top right, rgba(217, 72, 65, 0.18), transparent 28%),
    radial-gradient(circle at top left, rgba(36, 91, 127, 0.24), transparent 32%),
    linear-gradient(180deg, #f3f7f8 0%, #e7eef1 100%);
}
.hero-card {
  background: linear-gradient(135deg, rgba(14, 36, 49, 0.96), rgba(20, 54, 74, 0.94));
  color: white;
  padding: 18px 20px;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.09);
  box-shadow: 0 18px 40px rgba(7, 22, 32, 0.16);
}
.hero-card h1 {
  margin: 0 0 8px 0;
  font-size: 2rem;
}
.hero-card p {
  margin: 0;
  line-height: 1.55;
  color: rgba(255, 255, 255, 0.88);
}
.metric-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.metric-chip {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  padding: 10px 12px;
}
.metric-chip strong {
  display: block;
  font-size: 1rem;
}
.metric-chip span {
  font-size: 0.88rem;
  color: rgba(255, 255, 255, 0.72);
}
.result-card {
  border-radius: 18px;
  padding: 18px;
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(16, 32, 43, 0.08);
  box-shadow: 0 14px 32px rgba(16, 32, 43, 0.08);
}
.badge-wrap {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 12px 16px;
  border-radius: 999px;
  font-weight: 700;
  letter-spacing: 0.02em;
  font-size: 1rem;
}
.badge-emergency {
  background: rgba(217, 72, 65, 0.12);
  color: var(--rescue-red);
  border: 1px solid rgba(217, 72, 65, 0.28);
}
.badge-safe {
  background: rgba(31, 143, 95, 0.12);
  color: var(--safe-green);
  border: 1px solid rgba(31, 143, 95, 0.28);
}
.warning-box {
  margin-top: 12px;
  padding: 12px 14px;
  border-radius: 14px;
  background: rgba(215, 163, 61, 0.12);
  border: 1px solid rgba(215, 163, 61, 0.25);
  color: #6b4f16;
}
.filter-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.summary-card {
  border-radius: 16px;
  padding: 14px 16px;
  background: rgba(255, 255, 255, 0.84);
  border: 1px solid rgba(16, 32, 43, 0.08);
  box-shadow: 0 10px 26px rgba(16, 32, 43, 0.06);
}
.summary-card strong {
  display: block;
  font-size: 1.4rem;
  color: var(--storm-navy);
}
.summary-card span {
  color: #456174;
  font-size: 0.92rem;
}
.table-card {
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255, 255, 255, 0.84);
  border: 1px solid rgba(16, 32, 43, 0.08);
  box-shadow: 0 14px 32px rgba(16, 32, 43, 0.08);
}
.table-card h3 {
  margin: 0 0 10px 0;
  color: var(--storm-navy);
}
.table-wrap {
  max-height: 580px;
  overflow: auto;
  border-radius: 14px;
  border: 1px solid rgba(16, 32, 43, 0.08);
}
.table-wrap table {
  width: 100%;
  border-collapse: collapse;
  background: white;
}
.table-wrap th,
.table-wrap td {
  text-align: left;
  vertical-align: top;
  padding: 10px 12px;
  border-bottom: 1px solid rgba(16, 32, 43, 0.08);
  font-size: 0.93rem;
}
.table-wrap th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #eff5f7;
}
.table-wrap td:last-child {
  min-width: 360px;
  line-height: 1.45;
}
.signal-row {
  background: linear-gradient(90deg, rgba(217, 72, 65, 0.14), rgba(255, 255, 255, 0));
}
.signal-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 4px 10px;
  background: rgba(217, 72, 65, 0.12);
  color: var(--rescue-red);
  font-weight: 700;
  font-size: 0.82rem;
}
.signal-pill.safe {
  background: rgba(31, 143, 95, 0.12);
  color: var(--safe-green);
}
.footer-note {
  color: #365364;
  font-size: 0.95rem;
}
"""


@dataclass
class PredictionConfig:
    model_source: str
    threshold: float
    max_length: int
    device: torch.device


class BatchTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    def __len__(self) -> int:
        return len(next(iter(self.encodings.values()))) if self.encodings else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}


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
                raise RuntimeError("PEFT adapter checkpoint detected but `peft` is not installed.")
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

    def _predict_probabilities(self, clean_texts: list[str], batch_size: int) -> list[float]:
        if not clean_texts:
            return []
        dataset = BatchTextDataset(clean_texts, self.tokenizer, self.config.max_length)
        collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8 if self.config.device.type == "cuda" else None,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        probabilities: list[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(self.config.device) for key, value in batch.items()}
                logits = self.model(**batch).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
                probabilities.extend(float(value) for value in probs)
        return probabilities

    def _build_prediction_result(self, text: str, prob_cau_cuu: float, threshold: float) -> dict[str, Any]:
        model_positive = prob_cau_cuu >= threshold
        heuristic_override = self._should_override_to_emergency(text=text, prob_cau_cuu=prob_cau_cuu)
        is_emergency = model_positive or heuristic_override
        label = "cau_cuu" if is_emergency else "khong_cau_cuu"
        confidence = prob_cau_cuu if is_emergency else 1.0 - prob_cau_cuu
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "is_emergency": is_emergency,
            "warning": self._build_warning(text, prob_cau_cuu, model_positive, heuristic_override),
            "model_probability": round(prob_cau_cuu, 4),
            "heuristic_override": heuristic_override,
        }

    def predict(self, text: str, threshold: float | None = None) -> dict[str, Any]:
        effective_threshold = float(self.config.threshold if threshold is None else threshold)
        clean_text = self.preprocess(text)
        if not clean_text:
            return {
                "label": "khong_cau_cuu",
                "confidence": 0.0,
                "is_emergency": False,
                "warning": "Input text is empty.",
                "model_probability": 0.0,
                "heuristic_override": False,
            }
        probabilities = self._predict_probabilities([clean_text], batch_size=1)
        probability = probabilities[0] if probabilities else 0.0
        return self._build_prediction_result(text=text, prob_cau_cuu=probability, threshold=effective_threshold)

    def predict_many(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE, threshold: float | None = None) -> list[dict[str, Any]]:
        effective_threshold = float(self.config.threshold if threshold is None else threshold)
        clean_texts = [self.preprocess(text) for text in texts]
        results: list[dict[str, Any] | None] = [None] * len(texts)
        valid_indices = [index for index, clean_text in enumerate(clean_texts) if clean_text]
        probabilities: list[float] = []
        if valid_indices:
            probabilities = self._predict_probabilities([clean_texts[index] for index in valid_indices], batch_size=batch_size)
        for index, probability in zip(valid_indices, probabilities):
            results[index] = self._build_prediction_result(text=texts[index], prob_cau_cuu=float(probability), threshold=effective_threshold)
        for index, clean_text in enumerate(clean_texts):
            if results[index] is not None:
                continue
            results[index] = {
                "label": "khong_cau_cuu",
                "confidence": 0.0,
                "is_emergency": False,
                "warning": "Input text is empty." if not clean_text else "Prediction unavailable.",
                "model_probability": 0.0,
                "heuristic_override": False,
            }
        return [result for result in results if result is not None]

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


def load_threshold() -> float:
    env_threshold = os.getenv("MODEL_THRESHOLD")
    if env_threshold:
        return float(env_threshold)
    if DEFAULT_CONFIG_JSON.exists():
        payload = json.loads(DEFAULT_CONFIG_JSON.read_text(encoding="utf-8"))
        if "optimal_threshold" in payload:
            return float(payload["optimal_threshold"])
    if DEFAULT_SUMMARY_JSON.exists():
        payload = json.loads(DEFAULT_SUMMARY_JSON.read_text(encoding="utf-8"))
        if "threshold" in payload:
            return float(payload["threshold"])
    return DEFAULT_THRESHOLD


def resolve_device() -> torch.device:
    if os.getenv("DEVICE") == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predictor() -> CauCuuPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        config = PredictionConfig(
            model_source=os.getenv("MODEL_SOURCE", DEFAULT_MODEL_SOURCE),
            threshold=load_threshold(),
            max_length=int(os.getenv("MAX_LENGTH", "256")),
            device=resolve_device(),
        )
        _PREDICTOR = CauCuuPredictor(config)
    return _PREDICTOR


def build_badge_html(result: dict[str, Any]) -> str:
    if result["is_emergency"]:
        return (
            '<div class="result-card">'
            '<div class="badge-wrap badge-emergency">🆘 CẦU CỨU</div>'
            f'<div style="margin-top:12px;font-size:1.05rem;color:#233642;"><strong>Confidence:</strong> {result["confidence"]:.4f}</div>'
            f'<div class="warning-box">{result["warning"]}</div>'
            '</div>'
        )
    return (
        '<div class="result-card">'
        '<div class="badge-wrap badge-safe">✅ KHÔNG CẦU CỨU</div>'
        f'<div style="margin-top:12px;font-size:1.05rem;color:#233642;"><strong>Confidence:</strong> {result["confidence"]:.4f}</div>'
        f'<div class="warning-box">{result["warning"]}</div>'
        '</div>'
    )


def predict_single(text: str) -> tuple[str, str, dict[str, Any]]:
    predictor = get_predictor()
    result = predictor.predict(text)
    details = {
        "model_source": predictor.config.model_source,
        "model_load_mode": predictor.model_load_mode,
        "threshold": predictor.config.threshold,
        **result,
    }
    summary = (
        f"Label: {result['label']}\n"
        f"Is emergency: {result['is_emergency']}\n"
        f"Confidence: {result['confidence']:.4f}\n"
        f"Model probability (cau_cuu): {result['model_probability']:.4f}\n"
        f"Heuristic override: {result['heuristic_override']}"
    )
    return build_badge_html(result), summary, details


def predict_batch(text_block: str) -> pd.DataFrame:
    predictor = get_predictor()
    lines = [line.strip() for line in (text_block or "").splitlines() if line.strip()]
    results = predictor.predict_many(lines, batch_size=DEFAULT_BATCH_SIZE)
    rows: list[dict[str, Any]] = []
    for index, (line, result) in enumerate(zip(lines, results), start=1):
        rows.append(
            {
                "index": index,
                "text": line,
                "label": result["label"],
                "confidence": result["confidence"],
                "is_emergency": result["is_emergency"],
                "warning": result["warning"],
                "model_probability": result["model_probability"],
                "heuristic_override": result["heuristic_override"],
            }
        )
    if not rows:
        rows.append(
            {
                "index": 0,
                "text": "",
                "label": "khong_cau_cuu",
                "confidence": 0.0,
                "is_emergency": False,
                "warning": "No input lines were provided.",
                "model_probability": 0.0,
                "heuristic_override": False,
            }
        )
    return pd.DataFrame(rows)


def filter_uploaded_json(json_file_path: str | None, threshold: float, sort_by: str) -> tuple[str, str, str | None]:
    try:
        return filter_uploaded_json_with_predictor(
            json_file_path=json_file_path,
            threshold=threshold,
            sort_by=sort_by,
            predictor=get_predictor(),
            batch_size=DEFAULT_BATCH_SIZE,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


with gr.Blocks(css=CSS, title="PhoBERT Cầu Cứu Demo") as demo:
    gr.HTML(
        """
        <div class="hero-card">
          <h1>PhoBERT Cầu Cứu Detector</h1>
          <p>Demo phân loại comment Facebook tiếng Việt để ưu tiên các tín hiệu cầu cứu trong thiên tai. Mô hình ưu tiên recall cao cho nhãn <strong>cau_cuu</strong>, sau đó được bọc thêm lớp safety override cho các câu SOS rất rõ.</p>
          <div class="metric-strip">
            <div class="metric-chip"><strong>Recall</strong><span>0.9091 trên test</span></div>
            <div class="metric-chip"><strong>F1 cau_cuu</strong><span>0.8054 trên test</span></div>
            <div class="metric-chip"><strong>Decision threshold</strong><span>0.4000</span></div>
          </div>
        </div>
        """
    )

    with gr.Tab("Single Comment"):
        with gr.Row():
            with gr.Column(scale=5):
                single_input = gr.Textbox(
                    label="Comment tiếng Việt",
                    lines=6,
                    placeholder="Ví dụ: Cứu với, nhà em đang ngập tới mái, còn người già bị kẹt...",
                )
                single_btn = gr.Button("Phân tích comment", variant="primary")
            with gr.Column(scale=4):
                single_badge = gr.HTML()
                single_summary = gr.Textbox(label="Tóm tắt", lines=5, interactive=False)
                single_details = gr.JSON(label="Chi tiết")

        gr.Examples(examples=SINGLE_EXAMPLES, inputs=single_input)
        single_btn.click(
            fn=predict_single,
            inputs=single_input,
            outputs=[single_badge, single_summary, single_details],
            api_name="predict_single",
        )
        single_input.submit(
            fn=predict_single,
            inputs=single_input,
            outputs=[single_badge, single_summary, single_details],
            api_name="submit_single",
        )

    with gr.Tab("Batch Comments"):
        batch_input = gr.Textbox(
            label="Nhiều comments, mỗi dòng một comment",
            lines=10,
            placeholder=BATCH_EXAMPLE,
        )
        batch_btn = gr.Button("Phân tích hàng loạt", variant="secondary")
        batch_table = gr.Dataframe(
            headers=["index", "text", "label", "confidence", "is_emergency", "warning", "model_probability", "heuristic_override"],
            datatype=["number", "str", "str", "number", "bool", "str", "number", "bool"],
            interactive=False,
            wrap=True,
            label="Kết quả batch",
        )
        batch_btn.click(
            fn=predict_batch,
            inputs=batch_input,
            outputs=batch_table,
            api_name="predict_batch",
        )

    with gr.Tab("Upload JSON File Filter"):
        with gr.Row():
            json_input = gr.File(label="Upload file JSON Facebook post", file_types=[".json"], type="filepath")
            threshold_slider = gr.Slider(
                minimum=0.3,
                maximum=0.7,
                value=load_threshold(),
                step=0.01,
                label="Threshold",
                info="Default là optimal threshold hiện tại.",
            )
            sort_by_radio = gr.Radio(
                choices=[("Confidence", "confidence"), ("Reaction", "reaction_count")],
                value="confidence",
                label="Sắp xếp bảng theo",
            )
        filter_btn = gr.Button("Lọc Comment Cầu Cứu", variant="primary")
        filter_summary_html = gr.HTML()
        filter_results_html = gr.HTML()
        filter_download = gr.File(label="Tải filtered_comments.json")

        filter_btn.click(
            fn=filter_uploaded_json,
            inputs=[json_input, threshold_slider, sort_by_radio],
            outputs=[filter_summary_html, filter_results_html, filter_download],
            api_name="filter_json",
        )

    gr.Markdown(
        """
        <div class="footer-note">
        Lưu ý: Đây là công cụ hỗ trợ ưu tiên bình luận cần cứu hộ. Với các trường hợp nghiêm trọng, vẫn cần xác minh thủ công thông tin vị trí, số điện thoại và trạng thái cứu hộ hiện tại.
        </div>
        """
    )


if __name__ == "__main__":
    demo.launch()



