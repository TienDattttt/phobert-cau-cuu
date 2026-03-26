from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


POSITIVE_KEYWORDS = {
    "cứu": 1.0,
    "cần cứu": 2.0,
    "xin cứu": 2.5,
    "cầu cứu": 3.0,
    "kẹt": 1.5,
    "mắc kẹt": 3.0,
    "bị kẹt": 3.0,
    "nước dâng": 2.0,
    "ngập tới": 2.0,
    "nước lên": 2.0,
    "không thoát": 3.0,
    "thoát không được": 3.0,
    "cần thuyền": 3.0,
    "cần ca nô": 3.0,
    "cần xuồng": 3.0,
    "sos": 2.5,
    "🆘": 3.0,
    "khẩn cấp": 3.0,
    "cứu hộ gấp": 3.0,
    "không trụ được": 3.5,
    "ai cứu": 2.0,
    "cần giúp đỡ gấp": 3.0,
    "đang chìm": 3.5,
    "trôi": 1.5,
    "lũ cuốn": 3.5,
    "sạt lở nhà": 3.5,
}

URGENT_STATE_MARKERS = {
    "ngập tới mái": 3.0,
    "ngập lên tới mái": 3.0,
    "tới mái": 2.0,
    "tới nóc": 2.0,
    "lút mái": 2.5,
    "lên mái": 2.5,
    "lên nóc": 2.5,
    "nóc nhà": 2.0,
    "cô lập": 2.5,
    "cô lập hoàn toàn": 3.0,
    "không ra được": 2.5,
    "ko ra được": 2.5,
    "không trụ": 2.5,
    "ko trụ": 2.5,
    "không thoát": 3.0,
    "ko thoát": 3.0,
    "mất liên lạc": 2.0,
    "không liên lạc được": 2.0,
    "giao thông chia cắt": 2.0,
    "chia cắt": 1.5,
    "gọi cứu hộ chưa được": 3.0,
    "đã gọi cứu hộ nhưng không được": 3.0,
    "nước lên rất nhanh": 2.5,
    "nước gần tới đầu": 3.0,
    "sắp không chịu được": 3.0,
    "bà bầu": 1.0,
    "trẻ em": 1.0,
    "em bé": 1.0,
    "người già": 1.0,
}

NEGATIVE_KEYWORDS = {
    "cầu nguyện": 2.0,
    "cầu mong": 2.0,
    "thương quá": 1.5,
    "chia sẻ": 1.0,
    "bình an": 2.0,
    "tin bão": 2.0,
    "dự báo": 2.0,
    "cảm ơn": 1.0,
    "đã được cứu": 6.0,
    "đã an toàn": 6.0,
    "đã sơ tán": 6.0,
    "đã ổn": 5.0,
    "thông báo": 2.5,
}

BROADCAST_MARKERS = {
    "tổng hợp số cứu hộ": 6.0,
    "số liên hệ cứu hộ": 6.0,
    "cập nhật sđt": 5.0,
    "đơn vị trực chiến": 5.0,
    "tổng đài quốc gia": 5.0,
    "copy gửi lại": 5.0,
    "khi gọi cứu hộ": 5.0,
    "chia theo khu vực": 5.0,
    "đầu mối": 4.0,
    "mọi người lưu lại": 5.0,
    "ưu tiên gọi": 4.0,
    "gọi đầu tiên": 4.0,
}

RESPONDER_MARKERS = (
    "mọi người nhắn tin cầu cứu nhiều quá",
    "tôi đọc không hết",
    "tôi mở status này",
    "mong bà con bình tĩnh chờ đợi",
    "chuyển được một ít thông tin",
)

DISASTER_KEYWORDS = (
    "lụt",
    "lũ",
    "ngập",
    "nước",
    "cô lập",
    "mưa lớn",
    "sạt lở",
    "chìm",
)

PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:\d[\s.-]?){8,10}")
GPS_PATTERN = re.compile(r"[-+]?\d{1,3}(?:[.,]\d+)")
PLUS_CODE_PATTERN = re.compile(r"\b[23456789CFGHJMPQRVWX]{4,8}\+[23456789CFGHJMPQRVWX]{2,7}\b")
MAP_MARKERS = ("maps.app.goo.gl", "google.com/maps", "goo.gl/maps", "tọa độ", "📍")
TRIVIAL_REPLY_PATTERN = re.compile(r"^[\W_]+$", re.UNICODE)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    parser = argparse.ArgumentParser(description="Auto-label Vietnamese rescue comments.")
    parser.add_argument("--input-csv", type=Path, default=data_dir / "raw_comments.csv")
    parser.add_argument("--output-csv", type=Path, default=data_dir / "labeled_comments.csv")
    parser.add_argument("--review-csv", type=Path, default=data_dir / "review_needed.csv")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", normalized).strip()


def score_matches(text: str, weighted_keywords: dict[str, float]) -> float:
    lowered = text.casefold()
    return sum(weight for keyword, weight in weighted_keywords.items() if keyword in lowered)


def contains_phone(text: str) -> bool:
    return bool(PHONE_PATTERN.search(text))


def contains_location(text: str) -> bool:
    lowered = text.casefold()
    return bool(GPS_PATTERN.search(text) or PLUS_CODE_PATTERN.search(text) or any(marker in lowered for marker in MAP_MARKERS))


def contains_disaster_keyword(text: str) -> bool:
    lowered = text.casefold()
    return any(keyword in lowered for keyword in DISASTER_KEYWORDS)


def is_trivial_reply(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True
    if len(compact) <= 3:
        return True
    return bool(TRIVIAL_REPLY_PATTERN.fullmatch(compact))


def classify_comment(text: str) -> tuple[int, float]:
    normalized = normalize_text(text)
    lowered = normalized.casefold()

    if not normalized or is_trivial_reply(normalized):
        return 0, 0.92

    positive_score = score_matches(normalized, POSITIVE_KEYWORDS)
    urgent_score = score_matches(normalized, URGENT_STATE_MARKERS)
    negative_score = score_matches(normalized, NEGATIVE_KEYWORDS)
    broadcast_score = score_matches(normalized, BROADCAST_MARKERS)
    negative_score += broadcast_score
    positive_score += urgent_score

    has_phone = contains_phone(normalized)
    has_location = contains_location(normalized)
    has_disaster = contains_disaster_keyword(normalized)
    has_resolution = any(marker in lowered for marker in ("đã được cứu", "đã an toàn", "đã sơ tán", "đã ổn"))
    has_broadcast = broadcast_score > 0
    has_responder_context = any(marker in lowered for marker in RESPONDER_MARKERS)

    if has_phone and has_disaster:
        positive_score += 2.0
    if has_location and (urgent_score > 0 or positive_score >= 2.5):
        positive_score += 2.5
    if any(token in lowered for token in ("trẻ nhỏ", "em bé", "người già", "bà bầu", "tai biến", "sơ sinh")) and has_disaster:
        positive_score += 1.5

    if has_resolution:
        return 0, 0.95
    if has_broadcast and urgent_score < 4.0:
        return 0, 0.92
    if has_responder_context:
        return 0, 0.9

    margin = positive_score - negative_score

    if positive_score >= 10.0 and (has_phone or has_location or urgent_score >= 4.0):
        return 1, 0.93
    if positive_score >= 7.0 and margin >= 1.0 and (has_phone or has_location or urgent_score >= 2.5):
        return 1, 0.86
    if positive_score >= 5.0 and has_phone and has_disaster:
        return 1, 0.8
    if positive_score >= 4.5 and (urgent_score >= 2.5 or (has_disaster and urgent_score >= 2.0)) and (has_phone or has_location or has_disaster):
        return 1, 0.78
    if negative_score >= positive_score + 2.0:
        return 0, 0.86
    if negative_score > positive_score:
        return 0, 0.8
    if positive_score > negative_score and (has_phone or has_location or urgent_score > 0):
        return 1, 0.76
    return 0, 0.6


def load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def save_rows(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "text", "label", "confidence"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int | float]]) -> None:
    label_zero = [row for row in rows if int(row["label"]) == 0]
    label_one = [row for row in rows if int(row["label"]) == 1]
    print("Class distribution:")
    print(f"label=0 -> {len(label_zero)}")
    print(f"label=1 -> {len(label_one)}")

    print("Sample label=0:")
    for index, row in enumerate(label_zero[:10], start=1):
        preview = " ".join(str(row["text"]).split())
        print(f"{index}. {preview[:200]}")

    print("Sample label=1:")
    for index, row in enumerate(label_one[:10], start=1):
        preview = " ".join(str(row["text"]).split())
        print(f"{index}. {preview[:200]}")


def main() -> None:
    args = parse_args()
    source_rows = load_rows(args.input_csv)
    labeled_rows: list[dict[str, str | int | float]] = []
    review_rows: list[dict[str, str | int | float]] = []

    for row in source_rows:
        label, confidence = classify_comment(row.get("text", ""))
        labeled_row = {
            "id": row.get("id", ""),
            "text": row.get("text", ""),
            "label": label,
            "confidence": confidence,
        }
        labeled_rows.append(labeled_row)
        if confidence < 0.75:
            review_rows.append(labeled_row)

    save_rows(labeled_rows, args.output_csv)
    save_rows(review_rows, args.review_csv)
    print_summary(labeled_rows)


if __name__ == "__main__":
    main()
