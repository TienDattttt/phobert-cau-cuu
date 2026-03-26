from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import unicodedata
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PHONE_PATTERN = re.compile(r"(?:\+?84|0)(?:\d[ .-]?){8,10}\d")
GPS_PATTERN = re.compile(r"[-+]?\d{1,3}[.,]\d+")
ADDRESS_HINT_PATTERN = re.compile(
    r"\b(thon|to|doi|xom|ap|xa|phuong|quan|huyen|duong|hem|ngo|khu pho|kp|so nha|thon\s+\d+|to\s+\d+)\b",
    re.IGNORECASE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")
BASE_COLUMNS = ["id", "text", "author", "timestamp", "reaction_count", "confidence", "source", "parent_author", "post_id"]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"
    parser = argparse.ArgumentParser(description="Generate an HTML quality report for filtered SOS comments.")
    parser.add_argument("--input-file", type=Path, default=results_dir / "filtered_deduped_comments.json")
    parser.add_argument("--output-file", type=Path, default=results_dir / "filter_quality_report.html")
    parser.add_argument("--raw-json-file", type=Path, default=project_dir.parent / "10239674864474861.json")
    parser.add_argument("--filtered-file", type=Path, default=results_dir / "filtered_comments.json")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_for_rules(text: str) -> str:
    normalized = normalize_text(text).lower()
    normalized = unicodedata.normalize("NFKD", normalized)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return WHITESPACE_PATTERN.sub(" ", ascii_text).strip()


def get_series(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def load_comment_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in payload if isinstance(item, dict)]


def flatten_raw_comments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    comments = payload.get("comments") or []
    rows: list[dict[str, Any]] = []

    def visit(items: list[dict[str, Any]]) -> None:
        for item in items:
            rows.append(item)
            replies = item.get("replies") or []
            if replies:
                visit(replies)

    visit(comments)
    return rows


def infer_raw_total(raw_json_file: Path) -> int:
    if not raw_json_file.exists():
        return 0
    payload = json.loads(raw_json_file.read_text(encoding="utf-8"))
    return len(flatten_raw_comments(payload))


def parse_timestamp(value: Any) -> datetime | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    candidates = [raw_value]
    if raw_value.endswith("Z"):
        candidates.append(raw_value[:-1] + "+00:00")
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def safe_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_frame(comments: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(comments)
    if frame.empty:
        frame = pd.DataFrame(columns=BASE_COLUMNS)

    frame["id"] = get_series(frame, "id", "").fillna("").astype(str)
    frame["text"] = get_series(frame, "text", "").fillna("").astype(str).map(normalize_text)
    frame["author"] = get_series(frame, "author", "").fillna("").astype(str)
    frame["timestamp"] = get_series(frame, "timestamp", "").fillna("").astype(str)
    frame["reaction_count"] = get_series(frame, "reaction_count", 0).map(safe_int)
    frame["confidence"] = get_series(frame, "confidence", 0.0).map(safe_float)
    frame["source"] = get_series(frame, "source", "").fillna("").astype(str)
    frame["parent_author"] = get_series(frame, "parent_author", None).where(lambda series: pd.notna(series), None)
    frame["post_id"] = get_series(frame, "post_id", "").fillna("").astype(str)
    frame["normalized_rules_text"] = frame["text"].map(normalize_for_rules)
    frame["parsed_timestamp"] = frame["timestamp"].map(parse_timestamp)
    frame["hour_bucket"] = frame["parsed_timestamp"].map(lambda value: value.strftime("%Y-%m-%d %H:00") if value else "Unknown")
    frame["has_phone"] = frame["text"].map(lambda text: bool(PHONE_PATTERN.search(text)))
    frame["has_gps"] = frame["text"].map(lambda text: bool(GPS_PATTERN.search(text)))
    frame["has_address"] = frame["normalized_rules_text"].map(lambda text: bool(ADDRESS_HINT_PATTERN.search(text)))
    frame["signal_tags"] = frame.apply(
        lambda row: ", ".join(
            tag
            for tag, is_present in (
                ("phone", row["has_phone"]),
                ("gps", row["has_gps"]),
                ("address", row["has_address"]),
            )
            if is_present
        ) or "-",
        axis=1,
    )
    return frame


def figure_to_base64() -> str:
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_confidence_histogram(frame: pd.DataFrame) -> str:
    plt.figure(figsize=(7.2, 4.4))
    if not frame.empty:
        plt.hist(frame["confidence"], bins=12, color="#d94841", edgecolor="#fdf7f2")
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Comment Count")
    plt.grid(alpha=0.2)
    return figure_to_base64()


def build_timeline_chart(frame: pd.DataFrame) -> str:
    timeline = frame.groupby("hour_bucket").size().sort_index() if not frame.empty else pd.Series(dtype=int)
    plt.figure(figsize=(9.2, 4.4))
    if not timeline.empty:
        plt.plot(range(len(timeline)), timeline.values, marker="o", linewidth=2.0, color="#2563eb")
        plt.xticks(range(len(timeline)), timeline.index, rotation=45, ha="right")
    plt.title("Comments Over Time")
    plt.xlabel("Hour")
    plt.ylabel("Comment Count")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    return figure_to_base64()


def format_comment_cell(text: str) -> str:
    escaped = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return escaped.replace("\n", "<br>")


def render_table(frame: pd.DataFrame, columns: list[str], title: str, limit: int = 10, highlight_signals: bool = False) -> str:
    if frame.empty:
        return f"<section class='card'><h2>{title}</h2><p>Không có dữ liệu.</p></section>"

    rows_html: list[str] = []
    subset = frame.head(limit)
    for _, row in subset.iterrows():
        row_has_signal = bool(row.get("has_phone") or row.get("has_gps") or row.get("has_address"))
        row_class = "signal-row" if highlight_signals and row_has_signal else ""
        cells = "".join(f"<td>{format_comment_cell(row.get(column, ''))}</td>" for column in columns)
        rows_html.append(f"<tr class='{row_class}'>{cells}</tr>")

    headers = "".join(f"<th>{column}</th>" for column in columns)
    body = "".join(rows_html)
    return (
        f"<section class='card'><h2>{title}</h2>"
        f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>"
        f"</section>"
    )


def render_summary_cards(raw_total: int, filtered_total: int, deduped_total: int, frame: pd.DataFrame) -> str:
    cards = [
        ("Tổng comments đầu vào", raw_total),
        ("Sau filter", filtered_total),
        ("Sau dedup", deduped_total),
        ("Có số điện thoại", int(frame["has_phone"].sum()) if not frame.empty else 0),
        ("Có GPS", int(frame["has_gps"].sum()) if not frame.empty else 0),
        ("Có địa chỉ", int(frame["has_address"].sum()) if not frame.empty else 0),
    ]
    card_html = "".join(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>"
        for label, value in cards
    )
    return f"<section class='metric-grid'>{card_html}</section>"


def generate_report(
    input_file: str,
    output_file: str,
    raw_json_file: str | None = None,
    filtered_file: str | None = None,
) -> dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    input_path = Path(input_file)
    output_path = Path(output_file)
    raw_json_path = Path(raw_json_file) if raw_json_file is not None else project_dir.parent / "10239674864474861.json"
    filtered_path = Path(filtered_file) if filtered_file is not None else project_dir / "results" / "filtered_comments.json"

    comments = load_comment_list(input_path)
    frame = build_frame(comments)
    raw_total = infer_raw_total(raw_json_path)
    filtered_total = len(load_comment_list(filtered_path)) if filtered_path.exists() else len(frame)
    deduped_total = len(frame)

    confidence_desc = frame.sort_values(by=["confidence", "reaction_count"], ascending=[False, False]) if not frame.empty else frame
    reactions_desc = frame.sort_values(by=["reaction_count", "confidence"], ascending=[False, False]) if not frame.empty else frame
    signal_frame = frame[(frame["has_phone"]) | (frame["has_gps"]) | (frame["has_address"])] if not frame.empty else frame

    confidence_hist = build_confidence_histogram(frame)
    timeline_chart = build_timeline_chart(frame)

    summary_cards_html = render_summary_cards(raw_total, filtered_total, deduped_total, frame)
    top_confidence_html = render_table(
        confidence_desc,
        ["confidence", "reaction_count", "author", "timestamp", "source", "signal_tags", "text"],
        "Top 10 Comments Có Confidence Cao Nhất",
        limit=10,
        highlight_signals=True,
    )
    top_reactions_html = render_table(
        reactions_desc,
        ["reaction_count", "confidence", "author", "timestamp", "source", "signal_tags", "text"],
        "Top 10 Comments Có Reaction Cao Nhất",
        limit=10,
        highlight_signals=True,
    )
    signals_html = render_table(
        signal_frame.sort_values(by=["has_phone", "has_gps", "has_address", "confidence", "reaction_count"], ascending=[False, False, False, False, False]),
        ["signal_tags", "confidence", "reaction_count", "author", "timestamp", "text"],
        "Comments Chứa Phone / GPS / Địa Chỉ",
        limit=50,
        highlight_signals=True,
    )

    duplicates_removed = max(filtered_total - deduped_total, 0)
    filter_rate = (raw_total - filtered_total) / raw_total if raw_total else 0.0
    dedup_rate = duplicates_removed / filtered_total if filtered_total else 0.0
    avg_confidence = frame["confidence"].mean() if not frame.empty else 0.0

    html = f"""
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>Filter Quality Report</title>
  <style>
    :root {{
      --bg: #f7f1e8;
      --card: #fffaf3;
      --ink: #1f2937;
      --muted: #6b7280;
      --accent: #d94841;
      --accent-soft: #fee2e2;
      --line: #eadfce;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: linear-gradient(180deg, #f3ebdf 0%, #fbf7f1 100%); color: var(--ink); }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 28px; }}
    .hero {{ background: linear-gradient(135deg, #fff8ef 0%, #fff1e6 60%, #fde2e4 100%); border: 1px solid var(--line); border-radius: 24px; padding: 28px; box-shadow: 0 14px 40px rgba(84, 45, 10, 0.08); }}
    h1 {{ margin: 0 0 10px; font-size: 34px; }}
    .subtitle {{ color: var(--muted); font-size: 16px; margin: 0; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 22px 0; }}
    .metric-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05); }}
    .metric-label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 10px; }}
    .metric-value {{ font-size: 30px; font-weight: 700; }}
    .summary-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-top: 18px; }}
    .summary-item {{ background: rgba(255,255,255,0.68); border: 1px solid var(--line); border-radius: 16px; padding: 16px; }}
    .summary-item strong {{ display: block; font-size: 18px; margin-bottom: 6px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 24px 0; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding: 20px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05); margin-top: 18px; }}
    .card h2 {{ margin: 0 0 14px; font-size: 22px; }}
    .chart {{ width: 100%; border-radius: 14px; border: 1px solid var(--line); background: white; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; vertical-align: top; border-bottom: 1px solid var(--line); padding: 12px 10px; font-size: 14px; }}
    th {{ position: sticky; top: 0; background: #fff7ec; z-index: 1; }}
    td:last-child {{ min-width: 420px; line-height: 1.45; }}
    .signal-row {{ background: linear-gradient(90deg, var(--accent-soft), rgba(255,255,255,0)); }}
    .note {{ color: var(--muted); font-size: 14px; }}
    @media (max-width: 960px) {{ .grid {{ grid-template-columns: 1fr; }} .container {{ padding: 16px; }} td:last-child {{ min-width: 260px; }} }}
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>Filter Quality Report</h1>
      <p class="subtitle">Đánh giá chất lượng pipeline lọc comment cầu cứu sau bước filter và dedup, sẵn sàng cho Info Extraction.</p>
      {summary_cards_html}
      <div class="summary-strip">
        <div class="summary-item"><strong>Tỷ lệ lọc</strong>{filter_rate:.2%} comments đã bị loại khỏi tập đầu vào.</div>
        <div class="summary-item"><strong>Giảm do dedup</strong>{duplicates_removed} comments trùng hoặc gần trùng, tương đương {dedup_rate:.2%} so với output sau filter.</div>
        <div class="summary-item"><strong>Confidence trung bình</strong>{avg_confidence:.4f}</div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Distribution Confidence Scores</h2>
        <img class="chart" src="data:image/png;base64,{confidence_hist}" alt="Confidence histogram">
      </div>
      <div class="card">
        <h2>Comments Theo Timeline</h2>
        <img class="chart" src="data:image/png;base64,{timeline_chart}" alt="Timeline chart">
      </div>
    </section>

    {top_confidence_html}
    {top_reactions_html}
    {signals_html}

    <section class="card">
      <h2>Ghi Chú</h2>
      <p class="note">Comments được highlight màu đỏ nhạt khi chứa số điện thoại, GPS hoặc địa chỉ, vì đây là tín hiệu quan trọng cho bước Info Extraction tiếp theo.</p>
      <p class="note">Input file: {format_comment_cell(str(input_path))}</p>
      <p class="note">Raw JSON file: {format_comment_cell(str(raw_json_path)) if raw_json_path.exists() else 'Không tìm thấy'}</p>
      <p class="note">Filtered file: {format_comment_cell(str(filtered_path)) if filtered_path.exists() else 'Không tìm thấy'}</p>
    </section>
  </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    location_signal_count = int(((frame["has_gps"]) | (frame["has_address"])).sum()) if not frame.empty else 0
    return {
        "raw_total_comments": raw_total,
        "filtered_comments": filtered_total,
        "deduped_comments": deduped_total,
        "duplicates_removed": duplicates_removed,
        "phone_count": int(frame["has_phone"].sum()) if not frame.empty else 0,
        "gps_count": int(frame["has_gps"].sum()) if not frame.empty else 0,
        "address_count": int(frame["has_address"].sum()) if not frame.empty else 0,
        "location_signal_count": location_signal_count,
        "output_file": str(output_path),
    }


def main() -> None:
    args = parse_args()
    summary = generate_report(
        input_file=str(args.input_file),
        output_file=str(args.output_file),
        raw_json_file=str(args.raw_json_file),
        filtered_file=str(args.filtered_file),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

