from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    PEFT_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    PEFT_IMPORT_ERROR = exc


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


LOGGER = logging.getLogger("train_caucu")
ID2LABEL = {0: "khong_cau_cuu", 1: "cau_cuu"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}
SUPPORTED_UNSLOTH_MODEL_TYPES = {"llama", "mistral", "qwen2", "gemma", "phi", "phi3"}


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


ACCELERATE_AVAILABLE = package_available("accelerate")
BITSANDBYTES_AVAILABLE = package_available("bitsandbytes")
UNSLOTH_AVAILABLE = package_available("unsloth")


class EncodedCommentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        text_column = next(
            (column for column in ("model_input_text", "segmented_text", "normalized_text", "text") if column in frame.columns),
            None,
        )
        if text_column is None:
            raise ValueError("Expected one of model_input_text/segmented_text/normalized_text/text in split CSV.")

        texts = frame[text_column].fillna("").astype(str).tolist()
        self.ids = frame["id"].astype(str).tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


class WeightedClassificationTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        outputs = model(**model_inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT for Vietnamese emergency comment detection.")
    parser.add_argument("--base-model", default="vinai/phobert-base")
    parser.add_argument("--train-csv", type=Path, default=data_dir / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=data_dir / "val.csv")
    parser.add_argument("--class-weights-json", type=Path, default=data_dir / "class_weights.json")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "saved_model")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="phobert-caucu-v1")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_runtime_dependencies() -> None:
    if not ACCELERATE_AVAILABLE:
        raise RuntimeError("Missing dependency: accelerate. Install with `pip install accelerate`.")
    if PEFT_IMPORT_ERROR is not None:
        raise RuntimeError(f"Missing dependency: peft. Install with `pip install peft`. Original error: {PEFT_IMPORT_ERROR}")


def load_dataframe(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["label"] = frame["label"].astype(int)
    if "confidence" in frame.columns:
        frame["confidence"] = frame["confidence"].astype(float)
    return frame


def load_class_weights(path: Path) -> torch.Tensor:
    raw_weights = json.loads(path.read_text(encoding="utf-8"))
    ordered = [float(raw_weights[str(index)]) for index in sorted(ID2LABEL)]
    return torch.tensor(ordered, dtype=torch.float)


def detect_device_info() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    info: dict[str, Any] = {
        "cuda_available": cuda_available,
        "device": "cuda" if cuda_available else "cpu",
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_name": None,
        "cuda_memory_gb": None,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        info["cuda_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(properties.total_memory / 1024**3, 2)
    return info


def resolve_quantization(device_info: dict[str, Any]) -> tuple[BitsAndBytesConfig | None, str]:
    if not device_info["cuda_available"]:
        return None, "disabled_cpu_only"
    if not BITSANDBYTES_AVAILABLE:
        return None, "disabled_bitsandbytes_missing"
    if (device_info["cuda_memory_gb"] or 0.0) < 8.0:
        config = BitsAndBytesConfig(load_in_8bit=True)
        return config, "8bit"
    return None, "disabled_vram_gte_8gb"


def resolve_precision_flags(device_info: dict[str, Any]) -> tuple[bool, bool]:
    if not device_info["cuda_available"]:
        return False, False
    supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return supports_bf16, not supports_bf16


def infer_lora_target_modules(model: torch.nn.Module) -> list[str]:
    discovered: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in {"query", "key", "value"}:
                discovered.add(suffix)
    return sorted(discovered) or ["query", "value"]


def infer_modules_to_save(model: torch.nn.Module) -> list[str] | None:
    modules: list[str] = []
    for candidate in ("classifier", "score", "pre_classifier"):
        if hasattr(model, candidate):
            modules.append(candidate)
    return modules or None


def resolve_unsloth_info(base_model: str, model_type: str, device_info: dict[str, Any]) -> dict[str, Any]:
    if not device_info["cuda_available"]:
        return {"enabled": False, "reason": "CUDA unavailable"}
    if not UNSLOTH_AVAILABLE:
        return {"enabled": False, "reason": "unsloth package not installed"}
    if model_type not in SUPPORTED_UNSLOTH_MODEL_TYPES or "phobert" in base_model.lower():
        return {
            "enabled": False,
            "reason": f"Unsloth optimizes decoder-only families; model_type={model_type} for PhoBERT sequence classification is skipped.",
        }
    return {
        "enabled": False,
        "reason": "Unsloth branch intentionally disabled because this step uses AutoModelForSequenceClassification.",
    }


def build_model_and_tokenizer(args: argparse.Namespace) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    device_info = detect_device_info()
    quantization_config, quantization_mode = resolve_quantization(device_info)

    config = AutoConfig.from_pretrained(args.base_model)
    config.num_labels = len(ID2LABEL)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    config.problem_type = "single_label_classification"

    unsloth_info = resolve_unsloth_info(args.base_model, getattr(config, "model_type", "unknown"), device_info)
    LOGGER.info("Unsloth status: %s", unsloth_info["reason"])
    LOGGER.info("Quantization mode: %s", quantization_mode)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token

    model_kwargs: dict[str, Any] = {"config": config}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(device_info["cuda_available"]),
        )
    elif device_info["cuda_available"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = infer_lora_target_modules(model)
    modules_to_save = infer_modules_to_save(model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    runtime_info = {
        **device_info,
        "quantization_mode": quantization_mode,
        "unsloth": unsloth_info,
        "lora_target_modules": target_modules,
        "modules_to_save": modules_to_save or [],
        "base_model": args.base_model,
    }
    return model, tokenizer, runtime_info


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=-1)
    labels = eval_prediction.label_ids

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_cau_cuu": float(f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)),
        "recall_cau_cuu": float(recall_score(labels, preds, average="binary", pos_label=1, zero_division=0)),
    }


def save_runtime_metadata(output_dir: Path, args: argparse.Namespace, runtime_info: dict[str, Any], class_weights: torch.Tensor) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "base_model": args.base_model,
            "num_labels": len(ID2LABEL),
            "id2label": ID2LABEL,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "run_name": args.run_name,
        },
        "runtime": runtime_info,
        "class_weights": {str(index): float(weight) for index, weight in enumerate(class_weights.tolist())},
        "package_availability": {
            "accelerate": ACCELERATE_AVAILABLE,
            "peft": PEFT_IMPORT_ERROR is None,
            "bitsandbytes": BITSANDBYTES_AVAILABLE,
            "unsloth": UNSLOTH_AVAILABLE,
        },
    }
    (output_dir / "training_setup.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    configure_logging()
    args = parse_args()
    ensure_runtime_dependencies()
    set_seed(args.seed)

    train_df = load_dataframe(args.train_csv)
    val_df = load_dataframe(args.val_csv)
    class_weights = load_class_weights(args.class_weights_json)

    model, tokenizer, runtime_info = build_model_and_tokenizer(args)
    save_runtime_metadata(args.output_dir, args, runtime_info, class_weights)

    train_dataset = EncodedCommentDataset(train_df, tokenizer, args.max_length)
    val_dataset = EncodedCommentDataset(val_df, tokenizer, args.max_length)

    bf16, fp16 = resolve_precision_flags(runtime_info)
    optim_name = "paged_adamw_8bit" if runtime_info["quantization_mode"] == "8bit" else "adamw_torch"

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim=optim_name,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(runtime_info["cuda_available"]),
        use_cpu=not runtime_info["cuda_available"],
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_recall_cau_cuu",
        greater_is_better=True,
        report_to=args.report_to,
        run_name=args.run_name,
        logging_dir=str(args.output_dir / "logs"),
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_pin_memory=bool(runtime_info["cuda_available"]),
        seed=args.seed,
    )

    trainer = WeightedClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if runtime_info["cuda_available"] else None,
        ),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    LOGGER.info("Starting training with %s train rows and %s val rows", len(train_dataset), len(val_dataset))
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()

    train_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    summary = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Best checkpoint saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
