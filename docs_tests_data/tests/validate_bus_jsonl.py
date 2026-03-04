from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def _strict_enabled(default: bool = True) -> bool:
    raw = str(os.getenv("M3_BUS_LOG_VALIDATE_STRICT", "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _record_has_required_keys(rec: Dict[str, Any]) -> bool:
    has_ts = "ts" in rec
    has_kind = "kind" in rec and str(rec.get("kind", "")).strip() != ""
    has_event = str(rec.get("event", "")).strip() != ""
    has_source_key = (
        str(rec.get("source", "")).strip() != ""
        and str(rec.get("key", "")).strip() != ""
    )
    has_schema = str(rec.get("schema_version", "")).strip() != ""
    has_writer = str(rec.get("writer_id", "")).strip() != ""
    has_record_id = str(rec.get("record_id", "")).strip() != ""
    return bool(has_ts and has_kind and (has_event or has_source_key) and has_schema and has_writer and has_record_id)


def validate_bus_jsonl(
    path: str | Path,
    max_lines: int = 50_000,
    strict: bool | None = None,
) -> Dict[str, Any]:
    target = Path(path)
    if strict is None:
        strict = _strict_enabled(default=True)
    if not target.exists():
        return {
            "path": str(target),
            "total_lines": 0,
            "parsed_lines": 0,
            "parse_errors": 0,
            "schema_errors": 0,
            "parse_error_rate": 0.0,
            "strict": bool(strict),
            "exists": False,
        }

    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]

    parsed = 0
    parse_errors = 0
    schema_errors = 0
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            parse_errors += 1
            continue
        if not isinstance(rec, dict):
            parse_errors += 1
            continue
        parsed += 1
        if strict and not _record_has_required_keys(rec):
            schema_errors += 1

    total = int(len(lines))
    parse_error_rate = float(parse_errors / max(1, total))
    return {
        "path": str(target),
        "total_lines": total,
        "parsed_lines": int(parsed),
        "parse_errors": int(parse_errors),
        "schema_errors": int(schema_errors),
        "parse_error_rate": parse_error_rate,
        "strict": bool(strict),
        "exists": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate bus.jsonl parse integrity.")
    parser.add_argument("path", help="Path to bus.jsonl")
    parser.add_argument("--max-lines", type=int, default=50_000)
    parser.add_argument("--threshold", type=float, default=0.0001, help="Allowed parse_error_rate")
    parser.add_argument("--strict", action="store_true", default=None)
    args = parser.parse_args()

    result = validate_bus_jsonl(
        path=args.path,
        max_lines=int(args.max_lines),
        strict=args.strict,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    parse_error_rate = float(result.get("parse_error_rate", 0.0))
    if parse_error_rate > float(args.threshold):
        return 2
    if bool(result.get("strict")) and int(result.get("schema_errors", 0)) > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
