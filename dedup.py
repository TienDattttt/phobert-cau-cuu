from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class DedupSummary:
    before_count: int
    after_count: int
    duplicates_removed: int
    exact_duplicates_removed: int
    near_duplicates_removed: int
    output_file: str


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"

    parser = argparse.ArgumentParser(description="Deduplicate filtered Vietnamese SOS comments.")
    parser.add_argument("--input-file", type=Path, default=results_dir / "filtered_comments.json")
    parser.add_argument("--output-file", type=Path, default=results_dir / "filtered_deduped_comments.json")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = " ".join(normalized.split())
    return normalized.strip()


def load_comments(input_file: Path) -> list[dict[str, Any]]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    payload = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Expected input JSON to be a list of filtered comments.")
    return [item for item in payload if isinstance(item, dict)]


def parse_timestamp(timestamp: str) -> tuple[int, str]:
    raw_value = str(timestamp or "").strip()
    if not raw_value:
        return (1, "")

    candidates = [raw_value]
    if raw_value.endswith("Z"):
        candidates.append(raw_value[:-1] + "+00:00")

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            return (0, parsed.isoformat())
        except ValueError:
            continue
    return (1, raw_value)


def reaction_count(comment: dict[str, Any]) -> int:
    try:
        return int(str(comment.get("reaction_count", 0)).strip())
    except (TypeError, ValueError):
        return 0


def confidence_score(comment: dict[str, Any]) -> float:
    try:
        return float(comment.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def keep_preferred_comment(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_reactions = reaction_count(left)
    right_reactions = reaction_count(right)
    if left_reactions != right_reactions:
        return left if left_reactions > right_reactions else right

    left_timestamp = parse_timestamp(left.get("timestamp", ""))
    right_timestamp = parse_timestamp(right.get("timestamp", ""))
    if left_timestamp != right_timestamp:
        return left if left_timestamp < right_timestamp else right

    left_confidence = confidence_score(left)
    right_confidence = confidence_score(right)
    if left_confidence != right_confidence:
        return left if left_confidence >= right_confidence else right

    left_id = str(left.get("id", ""))
    right_id = str(right.get("id", ""))
    return left if left_id <= right_id else right


def exact_deduplicate(comments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    deduped: dict[str, dict[str, Any]] = {}
    removed = 0

    for comment in comments:
        key = normalize_text(str(comment.get("text", "")))
        if not key:
            key = f"__empty__::{comment.get('id', '')}"

        existing = deduped.get(key)
        if existing is None:
            deduped[key] = comment
            continue

        deduped[key] = keep_preferred_comment(existing, comment)
        removed += 1

    ordered = sorted(
        deduped.values(),
        key=lambda item: (-confidence_score(item), -reaction_count(item), parse_timestamp(item.get("timestamp", "")), str(item.get("id", ""))),
    )
    return ordered, removed


def similarity_score(left_text: str, right_text: str) -> float:
    return SequenceMatcher(None, normalize_text(left_text), normalize_text(right_text)).ratio()


def near_deduplicate(comments: list[dict[str, Any]], similarity_threshold: float) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0

    for candidate in comments:
        candidate_text = str(candidate.get("text", ""))
        duplicate_index: int | None = None
        preferred_comment = candidate

        for index, existing in enumerate(kept):
            score = similarity_score(candidate_text, str(existing.get("text", "")))
            if score <= similarity_threshold:
                continue
            duplicate_index = index
            preferred_comment = keep_preferred_comment(existing, candidate)
            removed += 1
            break

        if duplicate_index is None:
            kept.append(candidate)
        else:
            kept[duplicate_index] = preferred_comment

    kept.sort(
        key=lambda item: (-confidence_score(item), -reaction_count(item), parse_timestamp(item.get("timestamp", "")), str(item.get("id", ""))),
    )
    return kept, removed


def save_comments(comments: list[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(comments, ensure_ascii=False, indent=2), encoding="utf-8")


def deduplicate(
    input_file: str,
    output_file: str,
    similarity_threshold: float = 0.85,
) -> dict[str, Any]:
    input_path = Path(input_file)
    output_path = Path(output_file)
    comments = load_comments(input_path)
    before_count = len(comments)

    after_exact, exact_removed = exact_deduplicate(comments)
    after_near, near_removed = near_deduplicate(after_exact, similarity_threshold)
    save_comments(after_near, output_path)

    summary = DedupSummary(
        before_count=before_count,
        after_count=len(after_near),
        duplicates_removed=before_count - len(after_near),
        exact_duplicates_removed=exact_removed,
        near_duplicates_removed=near_removed,
        output_file=str(output_path),
    )
    print(
        f"Trước: {summary.before_count} comments -> Sau: {summary.after_count} comments "
        f"(loại {summary.duplicates_removed} duplicates)"
    )
    return {
        "before_count": summary.before_count,
        "after_count": summary.after_count,
        "duplicates_removed": summary.duplicates_removed,
        "exact_duplicates_removed": summary.exact_duplicates_removed,
        "near_duplicates_removed": summary.near_duplicates_removed,
        "output_file": summary.output_file,
        "similarity_threshold": similarity_threshold,
    }


def main() -> None:
    args = parse_args()
    summary = deduplicate(
        input_file=str(args.input_file),
        output_file=str(args.output_file),
        similarity_threshold=args.similarity_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
