from __future__ import annotations

import json
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:\d[ .-]?){8,10}\d")
GPS_PATTERN = re.compile(r"[-+]?\d{1,3}[.,]\d+")
ADDRESS_HINT_PATTERN = re.compile(
    r"\b(thon|to|doi|xom|ap|xa|phuong|quan|huyen|duong|hem|ngo|khu pho|kp|so nha|thon\s+\d+|to\s+\d+)\b",
    re.IGNORECASE,
)
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def escape_html(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\n", "<br>")
    )


def make_comment_id(post_id: str, source: str, indices: Iterable[int]) -> str:
    suffix = "-".join(f"{index:04d}" for index in indices)
    return f"{post_id}-{source}-{suffix}"


def parse_reaction_count(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def normalize_whitespace(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = ZERO_WIDTH_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_for_rules(text: str) -> str:
    normalized = normalize_whitespace(text).lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return WHITESPACE_PATTERN.sub(" ", ascii_text).strip()


def is_effectively_empty(text: str) -> bool:
    if not text or not text.strip():
        return True
    without_urls = URL_PATTERN.sub(" ", text)
    without_symbols = re.sub(r"[\W_]+", "", without_urls, flags=re.UNICODE)
    return not without_symbols.strip()


def detect_signal_tags(text: str) -> tuple[list[str], bool, bool, bool]:
    normalized = normalize_for_rules(text)
    has_phone = bool(PHONE_PATTERN.search(text))
    has_gps = bool(GPS_PATTERN.search(text))
    has_address = bool(ADDRESS_HINT_PATTERN.search(normalized))
    tags = []
    if has_phone:
        tags.append("phone")
    if has_gps:
        tags.append("gps")
    if has_address:
        tags.append("address")
    return tags or ["-"], has_phone, has_gps, has_address


def load_json_payload(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(json_path.read_text(encoding=encoding))
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Unable to parse JSON file: {json_path}") from last_error


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
            rows.append(
                {
                    "id": make_comment_id(post_id, source, current_lineage),
                    "text": "" if text is None else str(text),
                    "author": author,
                    "timestamp": str(item.get("timestamp") or ""),
                    "reaction_count": parse_reaction_count(item.get("reaction_count")),
                    "source": source,
                    "parent_author": parent_author,
                    "post_id": post_id,
                }
            )
            replies = item.get("replies") or []
            if replies:
                visit(replies, current_lineage, author or None, "reply")

    visit(comments, [], None, "top")
    return rows


def render_filter_summary(summary: dict[str, Any]) -> str:
    return f"""
    <div class="filter-summary">
      <div class="summary-card"><strong>{summary['total_comments']}</strong><span>Tổng comments đầu vào</span></div>
      <div class="summary-card"><strong>{summary['kept_comments']}</strong><span>Comments giữ lại</span></div>
      <div class="summary-card"><strong>{summary['filter_rate']:.2%}</strong><span>Tỷ lệ bị lọc ra</span></div>
      <div class="summary-card"><strong>{summary['avg_confidence']:.4f}</strong><span>Confidence trung bình</span></div>
      <div class="summary-card"><strong>{summary['phone_or_gps_count']}</strong><span>Có phone hoặc GPS</span></div>
      <div class="summary-card"><strong>{summary['threshold']:.2f}</strong><span>Threshold đang dùng</span></div>
    </div>
    """


def render_filter_table(rows: list[dict[str, Any]], sort_by: str) -> str:
    if not rows:
        return "<div class='table-card'><h3>Kết quả lọc</h3><p>Không có comment cầu cứu nào vượt ngưỡng hiện tại.</p></div>"

    headers = ["Signal", "Confidence", "Reaction", "Author", "Timestamp", "Source", "Parent Author", "Text"]
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body_rows: list[str] = []
    for row in rows:
        has_signal = row["has_phone"] or row["has_gps"]
        signal_badge = (
            f"<span class='signal-pill'>🚨 {escape_html(row['signal_tags'])}</span>"
            if has_signal
            else f"<span class='signal-pill safe'>{escape_html(row['signal_tags'])}</span>"
        )
        row_class = "signal-row" if has_signal else ""
        body_rows.append(
            f"<tr class='{row_class}'>"
            f"<td>{signal_badge}</td>"
            f"<td>{row['confidence']:.4f}</td>"
            f"<td>{row['reaction_count']}</td>"
            f"<td>{escape_html(row['author'])}</td>"
            f"<td>{escape_html(row['timestamp'])}</td>"
            f"<td>{escape_html(row['source'])}</td>"
            f"<td>{escape_html(row['parent_author'] or '-')}</td>"
            f"<td>{escape_html(row['text'])}</td>"
            "</tr>"
        )

    sorted_label = "confidence" if sort_by == "confidence" else "reaction"
    return (
        "<div class='table-card'>"
        f"<h3>Kết quả lọc, sắp xếp theo {escape_html(sorted_label)}</h3>"
        f"<div class='table-wrap'><table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
        "</div>"
    )


def write_filtered_json(rows: list[dict[str, Any]]) -> str:
    temp_dir = Path(tempfile.mkdtemp(prefix="caucu_filter_"))
    output_path = temp_dir / "filtered_comments.json"
    payload = [
        {
            "id": row["id"],
            "text": row["text"],
            "author": row["author"],
            "timestamp": row["timestamp"],
            "reaction_count": row["reaction_count"],
            "confidence": row["confidence"],
            "source": row["source"],
            "parent_author": row["parent_author"],
            "post_id": row["post_id"],
        }
        for row in rows
    ]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def filter_uploaded_json_with_predictor(
    json_file_path: str | None,
    threshold: float,
    sort_by: str,
    predictor: Any,
    batch_size: int = 32,
) -> tuple[str, str, str | None]:
    if not json_file_path:
        raise ValueError("Hãy upload file JSON Facebook post trước khi chạy filter.")

    payload = load_json_payload(json_file_path)
    flattened = flatten_comments(payload)
    total_comments = len(flattened)
    candidate_rows = [row for row in flattened if not is_effectively_empty(row["text"])]
    predictions = predictor.predict_many([row["text"] for row in candidate_rows], batch_size=batch_size, threshold=threshold)

    filtered_rows: list[dict[str, Any]] = []
    for row, result in zip(candidate_rows, predictions):
        if not result["is_emergency"]:
            continue
        signal_tags, has_phone, has_gps, has_address = detect_signal_tags(row["text"])
        filtered_rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                "author": row["author"],
                "timestamp": row["timestamp"],
                "reaction_count": row["reaction_count"],
                "confidence": float(result["model_probability"]),
                "source": row["source"],
                "parent_author": row["parent_author"],
                "post_id": row["post_id"],
                "signal_tags": ", ".join(signal_tags),
                "has_phone": has_phone,
                "has_gps": has_gps,
                "has_address": has_address,
            }
        )

    if sort_by == "reaction_count":
        filtered_rows.sort(key=lambda item: (-int(item["reaction_count"]), -float(item["confidence"]), str(item["timestamp"])))
    else:
        filtered_rows.sort(key=lambda item: (-float(item["confidence"]), -int(item["reaction_count"]), str(item["timestamp"])))

    summary = {
        "total_comments": total_comments,
        "kept_comments": len(filtered_rows),
        "filter_rate": (total_comments - len(filtered_rows)) / total_comments if total_comments else 0.0,
        "avg_confidence": sum(float(row["confidence"]) for row in filtered_rows) / len(filtered_rows) if filtered_rows else 0.0,
        "phone_or_gps_count": sum(1 for row in filtered_rows if row["has_phone"] or row["has_gps"]),
        "threshold": float(threshold),
    }
    download_path = write_filtered_json(filtered_rows)
    return render_filter_summary(summary), render_filter_table(filtered_rows, sort_by), download_path
