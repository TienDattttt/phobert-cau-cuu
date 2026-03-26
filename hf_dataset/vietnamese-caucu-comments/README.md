---
language:
- vi
license: other
pretty_name: Vietnamese Cau Cuu Facebook Comments
tags:
- vietnamese
- disaster-response
- emergency-detection
- facebook-comments
- text-classification
task_categories:
- text-classification
task_ids:
- binary-classification
size_categories:
- 1K<n<10K
annotations_creators:
- machine-generated
source_datasets:
- original
---

# Vietnamese Cau Cuu Facebook Comments

## Dataset Summary

This dataset contains Vietnamese Facebook comments collected from a natural-disaster discussion thread and auto-labeled for binary emergency detection.

The target task is to detect whether a comment is a real-time rescue request (`cau_cuu`) versus a non-emergency comment (`khong_phai_cau_cuu`).

This release is intended as a bootstrap dataset for triage modeling and should be treated as a weakly supervised resource. Human review is strongly recommended before production use.

## Task Definition

- `0`: `khong_phai_cau_cuu`
  Non-emergency content such as sympathy, reposts, hotline aggregation, updates that the family is already safe, or unrelated discussion.
- `1`: `cau_cuu`
  Active rescue requests where people are trapped, in immediate danger, isolated, or explicitly requesting emergency evacuation/support.

Priority metric for downstream models: recall on label `1`.

## Data Source

- Source type: Vietnamese Facebook comments from a disaster-related post/thread.
- Data was flattened from both top-level comments and nested replies.
- Original extraction and weak labeling were produced locally for research and experimentation.

## Data Fields

- `id`: Stable synthetic identifier derived from comment tree position.
- `text`: Raw Vietnamese comment text.
- `label`: Binary weak label (`0` or `1`).
- `confidence`: Heuristic confidence score from the auto-labeling pipeline.

## Class Distribution

Current version statistics:

- Total rows: `1492`
- Label `0`: `1050`
- Label `1`: `442`

## Labeling Method

Labels were assigned with a rule-based weak supervision pipeline using:

- urgent rescue keywords such as `cứu`, `mắc kẹt`, `ngập tới mái`, `khẩn cấp`, `SOS`
- structural signals such as phone numbers, map links, GPS-like coordinates, and location mentions
- negative filters for resolved cases (`đã được cứu`, `đã an toàn`) and hotline/broadcast style comments

Because labels are weakly supervised, false positives and false negatives remain possible.

## Recommended Use

- Training or bootstrapping a Vietnamese emergency comment classifier
- Error analysis and heuristic refinement
- Human-in-the-loop triage experiments

## Limitations and Ethics

- Comments may contain sensitive situational information such as phone numbers and addresses.
- Labels are machine-generated and not fully human-verified.
- Do not use this dataset for surveillance or any harmful downstream purpose.
- Review privacy, legal, and platform-policy constraints before redistribution or deployment.

## Citation

If you use this dataset, please cite the project/repository that publishes this dataset card and describe it as a weakly supervised Vietnamese rescue-request classification dataset.
