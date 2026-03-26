from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


EXCLUDED_FILE_NAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
}
EXCLUDED_FILE_PREFIXES = (
    "rng_state",
    "events.out.tfevents",
)
EXCLUDED_DIR_PREFIXES = (
    "checkpoint-",
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results_final"

    parser = argparse.ArgumentParser(description="Prepare a clean Hugging Face model release folder and model card.")
    parser.add_argument("--source-model-dir", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, default=results_dir / "evaluation_summary.json")
    parser.add_argument("--export-dir", type=Path, default=project_dir / "hf_model_release")
    parser.add_argument("--repo-id", default="dat201204/phobert-vi-caucu-classifier")
    parser.add_argument("--model-name", default="PhoBERT Vietnamese Cau Cuu Classifier")
    parser.add_argument("--base-model", default="vinai/phobert-base")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--public", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def should_skip(path: Path) -> bool:
    if path.is_dir():
        return any(path.name.startswith(prefix) for prefix in EXCLUDED_DIR_PREFIXES)
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    return any(path.name.startswith(prefix) for prefix in EXCLUDED_FILE_PREFIXES)


def copy_release_files(source_dir: Path, export_dir: Path) -> list[str]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source model directory not found: {source_dir}")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for item in source_dir.iterdir():
        if should_skip(item):
            continue
        target = export_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied.append(item.name)
    return sorted(copied)


def build_model_card(args: argparse.Namespace, summary: dict[str, Any]) -> str:
    threshold = float(summary.get("threshold", 0.5))
    threshold_source = summary.get("threshold_source", "unknown")
    target_recall = float(summary.get("target_recall", 0.85))

    validation_metrics = summary.get("validation_metrics", {})
    test_metrics = summary.get("test_metrics", {})
    confusion = summary.get("confusion_matrix", [[0, 0], [0, 0]])

    return f'''---
language:
- {args.language}
license: {args.license}
library_name: transformers
pipeline_tag: text-classification
tags:
- vietnamese
- phobert
- disaster-response
- emergency-detection
- text-classification
- peft
- lora
base_model: {args.base_model}
---

# {args.model_name}

PhoBERT-based Vietnamese Facebook comment classifier for detecting **"cầu cứu"** comments during natural-disaster situations.

## Labels

- `0`: `khong_cau_cuu`
- `1`: `cau_cuu`

## Intended use

This model is designed to prioritize **high recall** for emergency rescue requests in Vietnamese social-media comments, especially when comments may contain distress language, location hints, phone numbers, or SOS markers.

## Training setup

- Base model: `{args.base_model}`
- Fine-tuning method: LoRA / PEFT
- Evaluation checkpoint source: `{summary.get("model_dir", str(args.source_model_dir))}`
- Decision threshold for deployment: `{threshold:.4f}`
- Threshold selection policy: `{threshold_source}` with validation target recall `{target_recall:.2f}`

## Validation metrics at selected threshold

- Accuracy: `{float(validation_metrics.get("accuracy", 0.0)):.4f}`
- F1 macro: `{float(validation_metrics.get("f1_macro", 0.0)):.4f}`
- F1 (`cau_cuu`): `{float(validation_metrics.get("f1_cau_cuu", 0.0)):.4f}`
- Recall (`cau_cuu`): `{float(validation_metrics.get("recall_cau_cuu", 0.0)):.4f}`
- Precision (`cau_cuu`): `{float(validation_metrics.get("precision_cau_cuu", 0.0)):.4f}`

## Test metrics

- Accuracy: `{float(test_metrics.get("accuracy", 0.0)):.4f}`
- F1 macro: `{float(test_metrics.get("f1_macro", 0.0)):.4f}`
- F1 (`cau_cuu`): `{float(test_metrics.get("f1_cau_cuu", 0.0)):.4f}`
- Recall (`cau_cuu`): `{float(test_metrics.get("recall_cau_cuu", 0.0)):.4f}`
- Precision (`cau_cuu`): `{float(test_metrics.get("precision_cau_cuu", 0.0)):.4f}`

## Confusion matrix on test set

```text
{confusion[0][0]} {confusion[0][1]}
{confusion[1][0]} {confusion[1][1]}
```

## Recommended inference rule

Convert logits to probabilities and classify as `cau_cuu` when:

```python
prob_cau_cuu >= {threshold:.4f}
```

This threshold was chosen on the validation set to preserve strong recall while improving `F1(cau_cuu)` and overall accuracy.

## Example loading code

```python
import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

repo_id = "{args.repo_id}"
threshold = {threshold:.4f}

tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=False)
model = AutoPeftModelForSequenceClassification.from_pretrained(repo_id)
model.eval()

text = "Cuu voi, nha em dang ngap va co nguoi gia bi ket"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**inputs).logits
    prob_cau_cuu = torch.softmax(logits, dim=-1)[0, 1].item()

label = "cau_cuu" if prob_cau_cuu >= threshold else "khong_cau_cuu"
print({{"label": label, "prob_cau_cuu": prob_cau_cuu}})
```

## Limitations

- The dataset was weakly supervised in the first labeling stage and may contain residual noise.
- The model is optimized for disaster-response triage, not for general sentiment or topic classification.
- Human verification is still recommended for high-stakes rescue coordination.
'''


def write_model_card(export_dir: Path, content: str) -> Path:
    readme_path = export_dir / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    return readme_path


def main() -> None:
    args = parse_args()
    summary = load_json(args.evaluation_summary)
    copied_files = copy_release_files(args.source_model_dir, args.export_dir)
    readme_path = write_model_card(args.export_dir, build_model_card(args, summary))

    create_command = f"hf repos create {args.repo_id} --type model --exist-ok"
    if args.public:
        create_command += " --public"

    print(f"Prepared release folder: {args.export_dir}")
    print(f"Copied files ({len(copied_files)}): {copied_files}")
    print(f"Model card written: {readme_path}")
    print()
    print("Run these commands on Colab to push to Hugging Face Hub:")
    print(create_command)
    print(f"hf upload-large-folder {args.repo_id} {args.export_dir} --type model")


if __name__ == "__main__":
    main()
