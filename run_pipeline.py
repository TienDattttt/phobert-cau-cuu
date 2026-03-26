from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dedup import deduplicate
from filter_pipeline import filter_cau_cuu_comments
from quality_report import generate_report


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    results_dir = project_dir / "results"

    parser = argparse.ArgumentParser(description="Run the full SOS filtering pipeline end-to-end.")
    parser.add_argument("--json-file", type=Path, default=project_dir.parent / "10239674864474861.json")
    parser.add_argument("--model-dir", type=Path, default=project_dir / "saved_model")
    parser.add_argument("--config-json", type=Path, default=project_dir / "config.json")
    parser.add_argument("--filtered-output", type=Path, default=results_dir / "filtered_comments.json")
    parser.add_argument("--deduped-output", type=Path, default=results_dir / "filtered_deduped_comments.json")
    parser.add_argument("--report-output", type=Path, default=results_dir / "filter_quality_report.html")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    filter_summary = filter_cau_cuu_comments(
        json_file=str(args.json_file),
        output_file=str(args.filtered_output),
        threshold=args.threshold,
        min_confidence=args.min_confidence,
        batch_size=args.batch_size,
        model_dir=str(args.model_dir),
        config_json=str(args.config_json),
        max_length=args.max_length,
        device=args.device,
    )
    print(f"Filtered: {filter_summary['cau_cuu_count']}/{filter_summary['total_comments']} comments giữ lại")

    dedup_summary = deduplicate(
        input_file=str(args.filtered_output),
        output_file=str(args.deduped_output),
        similarity_threshold=args.similarity_threshold,
    )

    report_summary = generate_report(
        input_file=str(args.deduped_output),
        output_file=str(args.report_output),
        raw_json_file=str(args.json_file),
        filtered_file=str(args.filtered_output),
    )

    final_payload = {
        "filter_summary": filter_summary,
        "dedup_summary": dedup_summary,
        "report_summary": report_summary,
    }
    print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    print("===== KẾT QUẢ PIPELINE =====")
    print(f"Tổng comments đầu vào:    {filter_summary['total_comments']}")
    print(
        f"Sau filter (cầu cứu):     {filter_summary['cau_cuu_count']} "
        f"({(filter_summary['cau_cuu_count'] / filter_summary['total_comments'] * 100) if filter_summary['total_comments'] else 0:.2f}%)"
    )
    print(f"Sau dedup:                {dedup_summary['after_count']}")
    print(f"Có số điện thoại:         {report_summary['phone_count']}")
    print(f"Có GPS/địa chỉ:           {report_summary['location_signal_count']}")
    print("Sẵn sàng cho Info Extraction ✅")


if __name__ == "__main__":
    main()

