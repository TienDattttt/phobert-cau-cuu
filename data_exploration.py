from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


SOS_KEYWORDS = (
    "cứu",
    "cần cứu",
    "xin cứu",
    "kẹt",
    "mắc kẹt",
    "bị kẹt",
    "nước dâng",
    "ngập tới",
    "nước lên",
    "không thoát",
    "ko thoát",
    "thoát không được",
    "cần thuyền",
    "cần ca nô",
    "cần xuồng",
    "sos",
    "🆘",
    "khẩn cấp",
    "cứu hộ gấp",
    "ai cứu",
    "cần giúp đỡ gấp",
    "đang chìm",
    "trôi",
    "lũ cuốn",
    "sạt lở nhà",
)
EMERGENCY_TERMS = (
    "bị kẹt",
    "mắc kẹt",
    "cô lập",
    "lên mái",
    "lên nóc",
    "tới mái",
    "tới nóc",
    "lút mái",
    "ngập hết tầng",
    "nước lên",
    "nước dâng",
    "nước xiết",
    "mất liên lạc",
    "không trụ",
    "ko trụ",
    "không thoát",
    "ko thoát",
    "không ra được",
    "gọi cứu hộ chưa được",
    "đã gọi cứu hộ nhưng không được",
    "nước gần tới đầu",
    "sắp không chịu được",
    "sắp không trụ",
)
PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:\d[ .-]?){8,10}")
GPS_PATTERN = re.compile(r"[-+]?\d{1,3}[.,]\d+")
MAP_MARKERS = ("maps.app.goo.gl", "google.com/maps", "goo.gl/maps", "📍", "tọa độ")
NEGATIVE_SAMPLE_MARKERS = (
    "cầu mong",
    "cầu nguyện",
    "bình an",
    "đã được cứu",
    "đã an toàn",
    "đội sos",
    "cập nhật sđt",
    "đơn vị trực chiến",
    "tổng đài quốc gia",
    "lưu lại và chia sẻ",
    "mọi người lưu lại",
    "thông báo",
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    default_input = project_dir.parent / "10239674864474861.json"
    default_output = project_dir / "data" / "raw_comments.csv"

    parser = argparse.ArgumentParser(
        description="Extract Facebook comments and nested replies into a flat CSV."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=default_input,
        help="Path to the raw Facebook post JSON export.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output,
        help="Destination CSV path for flattened comments.",
    )
    return parser.parse_args()


def make_comment_id(post_id: str, source: str, indices: Iterable[int]) -> str:
    suffix = "-".join(f"{index:04d}" for index in indices)
    return f"{post_id}-{source}-{suffix}"


def flatten_comments(comments: list[dict], post_id: str) -> list[dict]:
    rows: list[dict] = []

    def visit(items: list[dict], lineage: list[int], source: str) -> None:
        for offset, item in enumerate(items, start=1):
            current_lineage = [*lineage, offset]
            text = item.get("text")
            author = item.get("author")
            timestamp = item.get("timestamp")
            reaction_count = item.get("reaction_count")

            try:
                parsed_reaction_count = int(str(reaction_count).strip())
            except (TypeError, ValueError):
                parsed_reaction_count = 0

            row_source = source if not lineage else "reply"
            rows.append(
                {
                    "id": make_comment_id(post_id, row_source, current_lineage),
                    "text": text if text is not None else "",
                    "source": row_source,
                    "author": author if author is not None else "",
                    "timestamp": timestamp if timestamp is not None else "",
                    "reaction_count": parsed_reaction_count,
                }
            )

            replies = item.get("replies") or []
            if replies:
                visit(replies, current_lineage, "reply")

    visit(comments, [], "top")
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "text", "source", "author", "timestamp", "reaction_count"]

    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_emergency_like(text: str) -> bool:
    normalized = text.casefold()
    if any(marker in normalized for marker in NEGATIVE_SAMPLE_MARKERS):
        return False

    has_state_marker = any(marker in normalized for marker in EMERGENCY_TERMS)
    has_sos_keyword = any(keyword in normalized for keyword in SOS_KEYWORDS)
    has_phone_or_location = bool(PHONE_PATTERN.search(text)) or bool(GPS_PATTERN.search(text)) or any(
        marker in normalized for marker in MAP_MARKERS
    )
    return has_state_marker and (has_sos_keyword or has_phone_or_location)


def score_sos_candidate(text: str) -> int:
    normalized = text.casefold()
    score = sum(1 for keyword in SOS_KEYWORDS if keyword in normalized)
    score += 2 * sum(1 for term in EMERGENCY_TERMS if term in normalized)

    if PHONE_PATTERN.search(text):
        score += 2
    if GPS_PATTERN.search(text) or any(marker in normalized for marker in MAP_MARKERS):
        score += 2
    if len(text) >= 120:
        score += 1

    return score


def print_summary(rows: list[dict]) -> None:
    total_comments = len(rows)
    non_empty_rows = [row for row in rows if row["text"].strip()]
    avg_length = (
        sum(len(row["text"]) for row in non_empty_rows) / len(non_empty_rows)
        if non_empty_rows
        else 0.0
    )
    null_count = total_comments - len(non_empty_rows)

    scored_rows = [row for row in rows if row["text"].strip() and is_emergency_like(row["text"])]
    sos_samples = sorted(
        scored_rows,
        key=lambda row: (score_sos_candidate(row["text"]), len(row["text"])),
        reverse=True,
    )[:5]

    print(f"Total comments: {total_comments}")
    print(f"Average text length (chars): {avg_length:.2f}")
    print(f"Null or empty text count: {null_count}")
    print("Sample SOS comments:")
    if not sos_samples:
        print("- No SOS-like comments found")
        return

    for index, row in enumerate(sos_samples, start=1):
        preview = " ".join(row["text"].split())
        print(f"{index}. {preview[:240]}")


def main() -> None:
    args = parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    post_id = str(payload.get("post_id", "unknown_post"))
    comments = payload.get("comments") or []

    rows = flatten_comments(comments, post_id)
    write_csv(rows, args.output_csv)
    print_summary(rows)


if __name__ == "__main__":
    main()
