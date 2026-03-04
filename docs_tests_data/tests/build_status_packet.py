#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_MD = "STATUS_PACKET.md"
OUTPUT_SVG = "VISUAL_MAP.svg"
CANVAS_W = 1600
CANVAS_H = 1100

LEGACY_OUTPUTS = (
    "STATUS.md",
    "ARCHITECTURE_STATE.md",
    "RUNBOOK.md",
    "BACKLOG.md",
    "VISUAL_MAP.md",
    "CLASS_CATALOG.md",
    "CHANGE_REPORT.md",
    "CLASS_INDEX.json",
    "CHANGESET.json",
    "VISUAL_MAP.html",
    "visual_module_graph.svg",
    "visual_class_density.svg",
    "visual_change_heatmap.svg",
)

KEY_FILES = [
    "m3/m3_core.py",
    "llm_adapter/llm_core.py",
    "llm_adapter/m3_control_bridge.py",
    "llm_adapter/m3_plastic_policy.py",
    "m3/torch_policy.py",
    "m3/reward.py",
    "llm_adapter/memory.py",
    "m3/features.py",
    "llm_adapter/tokenization.py",
    "llm_adapter/remote.py",
    "llm_adapter/layers.py",
    "run_llm_adapter.py",
]

ENV_PATTERNS = [
    re.compile(r"os\.environ\.get\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*(?:,\s*([^)]+))?\)"),
    re.compile(r"os\.getenv\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*(?:,\s*([^)]+))?\)"),
    re.compile(r"os\.environ\s*\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]"),
]

CJK_RE = re.compile(r"[\u1100-\u11ff\u2e80-\u9fff\ua960-\ua97f\uac00-\ud7af]")


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int


def run(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_lines(cmd: list[str], cwd: Path) -> list[str]:
    out = run(cmd, cwd)
    if out == "UNKNOWN" or not out:
        return []
    return [line for line in out.splitlines() if line.strip()]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def safe_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def short_sha(value: str | None, size: int = 12) -> str:
    if not value:
        return "UNKNOWN"
    v = str(value).strip()
    return v[:size] if v else "UNKNOWN"


def latest_jsonl_tail(artifacts_dir: Path, max_lines: int) -> list[str]:
    candidates = sorted(artifacts_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return []
    lines = candidates[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _jsonl_records(path: Path, max_lines: int = 0) -> tuple[list[dict[str, Any]], int, int]:
    if not path.exists():
        return [], 0, 0
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]
    total = 0
    parse_errors = 0
    rows: list[dict[str, Any]] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except Exception:
            parse_errors += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            parse_errors += 1
    return rows, total, parse_errors


def _norm_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def collect_runtime_kpis(out_dir: Path) -> dict[str, float]:
    bus_rows, bus_total, bus_parse_errors = _jsonl_records(out_dir / "bus.jsonl", max_lines=50_000)
    _ = bus_rows
    bus_error_rate = float(bus_parse_errors / max(1, bus_total))

    log_path = out_dir / "llm_adapter.log"
    hf_failures = 0
    hf_success = 0
    if log_path.exists():
        for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = str(raw or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                lo = line.lower()
                if "hfbackend" in lo and "generation failed" in lo:
                    hf_failures += 1
                continue
            kind = str(obj.get("kind", "")).strip().lower()
            reason_code = str(obj.get("reason_code", "")).strip().lower()
            if kind == "hf_runtime_failure":
                hf_failures += 1
            if reason_code == "hf_generate_ok":
                hf_success += 1
    hf_den = max(1, hf_failures + hf_success)
    hf_failure_rate = float(hf_failures / hf_den)

    meaning_rows, meaning_total, _meaning_parse_errors = _jsonl_records(out_dir / "meaning_state.jsonl")
    unknown_abstain_hits = 0
    for rec in meaning_rows:
        intent = str(rec.get("intent", "")).strip().lower()
        answer_type = str(rec.get("answer_type", "")).strip().lower()
        if intent == "unknown" or answer_type == "abstain":
            unknown_abstain_hits += 1
    unknown_abstain_ratio = float(unknown_abstain_hits / max(1, meaning_total))

    chat_rows, _chat_total, _chat_parse_errors = _jsonl_records(out_dir / "chat_history.jsonl")
    assistant_texts = [
        _norm_text(rec.get("text") or rec.get("content"))
        for rec in chat_rows
        if str(rec.get("role", "")).strip().lower() in {"assistant", "m3", "bot"}
    ]
    assistant_texts = [txt for txt in assistant_texts if txt]
    unique_assistant = len(set(assistant_texts))
    repeats = max(0, len(assistant_texts) - unique_assistant)
    response_repeat_ratio = float(repeats / max(1, len(assistant_texts)))

    return {
        "hf_generation_failure_rate": hf_failure_rate,
        "bus_jsonl_parse_error_rate": bus_error_rate,
        "unknown_abstain_ratio": unknown_abstain_ratio,
        "response_repeat_ratio": response_repeat_ratio,
    }


def load_source_revision(out_dir: Path, repo_root: Path) -> dict[str, Any]:
    candidates = [
        out_dir / "source_revision.json",
        repo_root / "docs_tests_data" / "source_revision.json",
    ]
    for path in candidates:
        obj = safe_json(path)
        if isinstance(obj, dict):
            return obj
    return {}


def discover_entry_points(repo_root: Path) -> list[str]:
    entries: set[str] = set()
    for py in sorted(repo_root.glob("*.py")):
        entries.add(py.name)
    for rel in KEY_FILES:
        path = repo_root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "if __name__ == \"__main__\"" in text or "if __name__ == '__main__'" in text:
            entries.add(rel)
    return sorted(entries)


def discover_env_toggles(repo_root: Path) -> list[tuple[str, str, str]]:
    found: dict[tuple[str, str], tuple[str, str, str]] = {}
    for rel in KEY_FILES:
        path = repo_root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in ENV_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1)
                default_group = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                default = (default_group or "None").strip()
                found[(name, rel)] = (name, default, rel)
    return sorted(found.values(), key=lambda item: (item[0], item[2]))


def module_overview(repo_root: Path) -> tuple[list[str], list[tuple[str, str]]]:
    modules: list[str] = []
    edges: set[tuple[str, str]] = set()
    for rel in KEY_FILES:
        path = repo_root / rel
        if not path.exists():
            continue
        modules.append(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        imports = set(re.findall(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", text, flags=re.M))
        for imp in sorted(imports):
            if imp.startswith("m3") or imp.startswith("llm_adapter"):
                edges.add((rel, imp))
    return modules, sorted(edges)


def parse_classes_and_functions(py_path: Path) -> dict[str, int]:
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {"classes": 0, "functions": 0, "methods": 0}
    classes = 0
    functions = 0
    methods = 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes += 1
            methods += sum(1 for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
    return {"classes": classes, "functions": functions, "methods": methods}


def collect_structure_counts(repo_root: Path) -> dict[str, int]:
    totals = {"files": 0, "classes": 0, "functions": 0, "methods": 0}
    for rel in KEY_FILES:
        path = repo_root / rel
        if not path.exists():
            continue
        counts = parse_classes_and_functions(path)
        totals["files"] += 1
        totals["classes"] += counts["classes"]
        totals["functions"] += counts["functions"]
        totals["methods"] += counts["methods"]
    return totals


def resolve_base_head(repo_root: Path, base_sha: str, head_sha: str) -> tuple[str | None, str | None]:
    head = head_sha.strip() or run(["git", "rev-parse", "HEAD"], repo_root)
    if head == "UNKNOWN":
        return None, None
    base = base_sha.strip() or run(["git", "rev-parse", f"{head}~1"], repo_root)
    if base == "UNKNOWN":
        base = None
    return base, head


def collect_changes(repo_root: Path, base: str | None, head: str | None) -> list[dict[str, Any]]:
    if not base or not head:
        return []
    rows: dict[str, dict[str, Any]] = {}
    for line in run_lines(["git", "diff", "--name-status", f"{base}..{head}"], repo_root):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status_raw = parts[0].strip()
        status = status_raw[:1] if status_raw else "M"
        path = parts[-1].strip().replace("\\", "/")
        rows[path] = {
            "status": status,
            "status_raw": status_raw,
            "path": path,
            "added": 0,
            "deleted": 0,
            "churn": 0,
        }
    for line in run_lines(["git", "diff", "--numstat", f"{base}..{head}"], repo_root):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        path = parts[-1].replace("\\", "/")
        added = int(parts[0]) if parts[0].isdigit() else 0
        deleted = int(parts[1]) if parts[1].isdigit() else 0
        row = rows.get(path) or {"status": "M", "status_raw": "M", "path": path, "added": 0, "deleted": 0, "churn": 0}
        row["added"] = added
        row["deleted"] = deleted
        row["churn"] = added + deleted
        rows[path] = row
    return sorted(rows.values(), key=lambda item: (-int(item["churn"]), str(item["path"])))


def gather_todos(repo_root: Path, limit: int = 40) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    suffixes = {".py", ".md", ".txt", ".json", ".yml", ".yaml"}
    skip = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", "node_modules"}
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in skip for part in path.parts):
            continue
        if "docs_tests_data" in path.parts and "tests" not in path.parts:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        rel = rel_path(path, repo_root)
        if rel in {"tools/build_status_packet.py", "docs_tests_data/tests/build_status_packet.py"}:
            continue
        for idx, line in enumerate(lines, start=1):
            if "TODO" in line or "FIXME" in line:
                hits.append((rel, idx, normalize_text(line)[:180]))
                if len(hits) >= limit:
                    return hits
    return hits


def cleanup_legacy_outputs(out_dir: Path) -> None:
    for name in LEGACY_OUTPUTS:
        target = out_dir / name
        if target.exists() and target.is_file():
            target.unlink()


def compact_object(value: Any, depth: int = 0, max_depth: int = 2, max_items: int = 12) -> Any:
    if depth >= max_depth:
        if isinstance(value, (dict, list, tuple)):
            return "<trimmed>"
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, key in enumerate(sorted(value.keys(), key=lambda x: str(x))):
            if idx >= max_items:
                out["..."] = f"+{len(value) - max_items} more"
                break
            out[str(key)] = compact_object(value[key], depth + 1, max_depth=max_depth, max_items=max_items)
        return out
    if isinstance(value, (list, tuple)):
        out_list: list[Any] = []
        for idx, item in enumerate(value):
            if idx >= max_items:
                out_list.append(f"... +{len(value) - max_items} more")
                break
            out_list.append(compact_object(item, depth + 1, max_depth=max_depth, max_items=max_items))
        return out_list
    return value


def scalar_to_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}" if abs(value) < 1000 else f"{value:,.2f}"
    return str(value)


def collect_metric_rows(metrics: Any, limit: int = 12) -> list[tuple[str, str]]:
    if not isinstance(metrics, dict):
        return [("metrics", "UNKNOWN")]
    preferred = (
        "hf_generation_failure_rate",
        "bus_jsonl_parse_error_rate",
        "unknown_abstain_ratio",
        "response_repeat_ratio",
        "status",
        "loss",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "latency_ms",
        "throughput",
        "step",
        "epoch",
    )
    rows: list[tuple[str, str]] = []
    used: set[str] = set()
    for key in preferred:
        if key in metrics:
            rows.append((key, scalar_to_text(metrics[key])))
            used.add(key)
        if len(rows) >= limit:
            return rows
    for key in sorted(metrics.keys(), key=lambda x: str(x)):
        if key in used:
            continue
        value = metrics[key]
        if isinstance(value, (str, int, float, bool)):
            rows.append((str(key), scalar_to_text(value)))
        if len(rows) >= limit:
            break
    return rows or [("metrics", "EMPTY_OBJECT")]


def infer_status(run_summary: Any) -> str:
    if not isinstance(run_summary, dict):
        return "UNKNOWN"
    raw = str(run_summary.get("status", "")).strip().lower()
    if raw in {"success", "passed", "pass", "ok", "true"}:
        return "PASS"
    if raw in {"failed", "failure", "fail", "false", "error"}:
        return "FAIL"
    return "UNKNOWN"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_status_packet_md(
    ts: str,
    branch: str,
    sha: str,
    base: str | None,
    head: str | None,
    fallback_source: str,
    status_text: str,
    metric_rows: list[tuple[str, str]],
    run_summary: Any,
    jsonl_tail: list[str],
    modules: list[str],
    deps: list[tuple[str, str]],
    entry_points: list[str],
    env_toggles: list[tuple[str, str, str]],
    changes: list[dict[str, Any]],
    todos: list[tuple[str, int, str]],
    structure_counts: dict[str, int],
    out_rel: str,
    artifacts_rel: str,
) -> str:
    lines: list[str] = []
    lines.append("# STATUS_PACKET")
    lines.append("")
    lines.append("![Status Packet Dashboard](VISUAL_MAP.svg)")
    lines.append("")
    lines.append("## Snapshot")
    lines.append("")
    snapshot_rows = [
        ["Generated At (UTC)", f"`{ts}`"],
        ["Status", f"**{status_text}**"],
        ["Branch", f"`{branch}`"],
        ["Commit", f"`{short_sha(sha)}`"],
        ["Base SHA", f"`{short_sha(base)}`"],
        ["Head SHA", f"`{short_sha(head)}`"],
        ["Revision Source", f"`{fallback_source}`"],
    ]
    lines.append(markdown_table(["Field", "Value"], snapshot_rows))
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append(markdown_table(["Metric", "Value"], [[f"`{k}`", f"`{v}`"] for k, v in metric_rows]))
    lines.append("")
    lines.append("## Latest Run Summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(compact_object(run_summary), indent=2, ensure_ascii=False, sort_keys=True))
    lines.append("```")
    if jsonl_tail:
        lines.append("")
        lines.append(f"Latest JSONL tail from `{artifacts_rel}`:")
        lines.append("")
        lines.append("```jsonl")
        lines.extend(jsonl_tail[:40])
        lines.append("```")
    lines.append("")
    lines.append("## Architecture Highlights")
    lines.append("")
    lines.append(
        markdown_table(
            ["Metric", "Value"],
            [
                ["Tracked modules", f"`{len(modules)}`"],
                ["Dependency edges", f"`{len(deps)}`"],
                ["Entry points", f"`{len(entry_points)}`"],
                ["Tracked classes", f"`{structure_counts.get('classes', 0)}`"],
                ["Top-level functions", f"`{structure_counts.get('functions', 0)}`"],
                ["Class methods", f"`{structure_counts.get('methods', 0)}`"],
                ["Changed files", f"`{len(changes)}`"],
            ],
        )
    )
    lines.append("")
    lines.append("Top dependency edges:")
    if deps:
        for src, dst in deps[:8]:
            lines.append(f"- `{src} -> {dst}`")
        if len(deps) > 8:
            lines.append(f"- `+{len(deps) - 8} more`")
    else:
        lines.append("- `UNKNOWN`")
    lines.append("")
    lines.append("Top env toggles:")
    if env_toggles:
        env_rows = [[f"`{n}`", f"`{normalize_text(d)}`", f"`{r}`"] for n, d, r in env_toggles[:10]]
        lines.append(markdown_table(["Variable", "Default", "File"], env_rows))
        if len(env_toggles) > 10:
            lines.append(f"\n`+{len(env_toggles) - 10} more toggles`")
    else:
        lines.append("- `UNKNOWN`")
    lines.append("")
    lines.append("## Top Changed Files")
    lines.append("")
    if changes:
        change_rows = []
        for item in changes[:20]:
            change_rows.append(
                [
                    f"`{item.get('status', 'M')}`",
                    f"`{normalize_text(item.get('path', ''))}`",
                    str(int(item.get("added", 0))),
                    str(int(item.get("deleted", 0))),
                    str(int(item.get("churn", 0))),
                ]
            )
        lines.append(markdown_table(["Status", "File", "+", "-", "Churn"], change_rows))
        if len(changes) > 20:
            lines.append(f"\n`+{len(changes) - 20} more files`")
    else:
        lines.append("- No diff data available for the current base/head range.")
    lines.append("")
    lines.append("## Risks & TODO Summary")
    lines.append("")
    if todos:
        for rel, line_no, text in todos[:12]:
            lines.append(f"- `{rel}:{line_no}` {text}")
        if len(todos) > 12:
            lines.append(f"- `+{len(todos) - 12} more`")
    else:
        lines.append("- No TODO/FIXME markers found in scanned source files.")
    lines.append("")
    lines.append("## Repro Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("python docs_tests_data/tests/build_status_packet.py \\")
    lines.append("  --repo-root . \\")
    lines.append(f"  --out-dir {out_rel} \\")
    lines.append(f"  --artifacts-dir {artifacts_rel}")
    lines.append("```")
    lines.append("")
    lines.append("## Output Contract")
    lines.append("")
    lines.append("- `docs_tests_data/STATUS_PACKET.md`")
    lines.append("- `docs_tests_data/VISUAL_MAP.svg`")
    lines.append("")
    lines.append("Legacy packet outputs are cleaned automatically by this generator.")
    return "\n".join(lines)


def measure_text_px(text: str, font_size: int) -> int:
    width = 0.0
    for ch in text:
        if ch in " il.,:;|'`":
            width += font_size * 0.33
        elif CJK_RE.match(ch):
            width += font_size * 1.0
        elif ord(ch) > 127:
            width += font_size * 0.9
        elif ch.isupper():
            width += font_size * 0.62
        else:
            width += font_size * 0.56
    return int(width) + 1


def fit_text_single_line(text: Any, max_px: int, font_size: int) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    if measure_text_px(normalized, font_size) <= max_px:
        return normalized
    ellipsis = "..."
    if measure_text_px(ellipsis, font_size) > max_px:
        return ellipsis
    low = 0
    high = len(normalized)
    while low < high:
        mid = (low + high + 1) // 2
        candidate = normalized[:mid].rstrip() + ellipsis
        if measure_text_px(candidate, font_size) <= max_px:
            low = mid
        else:
            high = mid - 1
    return normalized[:low].rstrip() + ellipsis


def wrap_text_lines(text: Any, max_px: int, font_size: int, max_lines: int) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return [""]
    tokens = re.findall(r"\S+\s*", normalized) or list(normalized)
    lines: list[str] = []
    current = ""
    for token in tokens:
        trial = current + token
        if not current or measure_text_px(trial, font_size) <= max_px:
            current = trial
            continue
        lines.append(current.rstrip())
        current = token.lstrip()
    if current.strip():
        lines.append(current.rstrip())
    if len(lines) <= max_lines:
        return [fit_text_single_line(line, max_px, font_size) for line in lines]
    trimmed = lines[: max_lines - 1]
    overflow = " ".join(lines[max_lines - 1 :])
    trimmed.append(fit_text_single_line(overflow, max_px, font_size))
    if not trimmed[-1].endswith("..."):
        trimmed[-1] = fit_text_single_line(trimmed[-1] + "...", max_px, font_size)
    return trimmed


def xml_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def assert_cards_valid(cards: list[Rect]) -> None:
    for card in cards:
        if card.x < 0 or card.y < 0 or card.x + card.w > CANVAS_W or card.y + card.h > CANVAS_H:
            raise ValueError(f"Card out of bounds: {card}")
    for i in range(len(cards)):
        a = cards[i]
        for b in cards[i + 1 :]:
            overlaps = not (
                a.x + a.w <= b.x
                or b.x + b.w <= a.x
                or a.y + a.h <= b.y
                or b.y + b.h <= a.y
            )
            if overlaps:
                raise ValueError(f"Card overlap detected: {a} <-> {b}")


def line_capacity(card: Rect, top: int = 68, bottom: int = 20, line_h: int = 20) -> int:
    usable = max(0, card.h - top - bottom)
    return max(1, usable // line_h)


def render_card_title(svg: list[str], card: Rect, title: str, subtitle: str | None = None) -> int:
    y = card.y + 34
    svg.append(f'<text x="{card.x + 20}" y="{y}" class="card-title">{xml_escape(fit_text_single_line(title, card.w - 40, 20))}</text>')
    if subtitle:
        svg.append(f'<text x="{card.x + 20}" y="{y + 20}" class="muted">{xml_escape(fit_text_single_line(subtitle, card.w - 40, 13))}</text>')
    return card.y + 70


def write_visual_dashboard_svg(
    path: Path,
    ts: str,
    sha: str,
    branch: str,
    base: str | None,
    head: str | None,
    status_text: str,
    modules: list[str],
    deps: list[tuple[str, str]],
    entry_points: list[str],
    env_toggles: list[tuple[str, str, str]],
    changes: list[dict[str, Any]],
    todos: list[tuple[str, int, str]],
) -> None:
    margin = 24
    gap = 18
    content_w = CANVAS_W - margin * 2

    header_card = Rect(margin, margin, content_w, 118)
    kpi_card = Rect(margin, header_card.y + header_card.h + gap, content_w, 134)

    col_w = (content_w - gap) // 2
    section_y = kpi_card.y + kpi_card.h + gap
    left_h = max(220, 72 + (7 + min(6, len(env_toggles))) * 19)
    right_h = max(220, 72 + (2 + min(10, len(changes))) * 19)
    section_h = max(left_h, right_h)
    min_table_h = 260
    if section_y + section_h + gap + min_table_h + margin > CANVAS_H:
        section_h = max(200, CANVAS_H - (section_y + gap + min_table_h + margin))

    left_card = Rect(margin, section_y, col_w, section_h)
    right_card = Rect(margin + col_w + gap, section_y, content_w - col_w - gap, section_h)
    table_card = Rect(margin, section_y + section_h + gap, content_w, CANVAS_H - (section_y + section_h + gap) - margin)

    cards = [header_card, kpi_card, left_card, right_card, table_card]
    assert_cards_valid(cards)
    top_changes = sorted(changes, key=lambda item: int(item.get("churn", 0)), reverse=True)

    svg: list[str] = []
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}" role="img" aria-labelledby="title desc">')
    svg.append('<title id="title">Status Packet Dashboard</title>')
    svg.append('<desc id="desc">Consolidated architecture and change summary for the latest status packet.</desc>')
    svg.append("<style>")
    svg.append(".bg { fill: #eef3f9; } .card { fill: #ffffff; stroke: #cdd8e6; stroke-width: 1; } .header-card { fill: #f8fbff; stroke: #bdd4ee; stroke-width: 1.2; }")
    svg.append(".table-head { fill: #f1f6fd; } .title { font-size: 34px; font-weight: 700; fill: #0f172a; } .subtitle { font-size: 14px; fill: #334155; }")
    svg.append(".card-title { font-size: 20px; font-weight: 700; fill: #0f172a; } .kpi-label { font-size: 12px; fill: #475569; } .kpi-value { font-size: 26px; font-weight: 700; fill: #0b3b76; }")
    svg.append(".body { font-size: 14px; fill: #0f172a; } .muted { font-size: 13px; fill: #475569; } .table-title { font-size: 19px; font-weight: 700; fill: #0f172a; }")
    svg.append(".table-head-text { font-size: 12px; font-weight: 700; fill: #334155; } .table-row { font-size: 12px; fill: #0f172a; } .row-line { stroke: #e2e8f0; stroke-width: 1; }")
    svg.append(".chip { fill: #e5eef9; stroke: #c6d7ed; stroke-width: 1; } .kpi-box { fill: #f8fbff; stroke: #d2deef; stroke-width: 1; }")
    svg.append("</style>")
    svg.append("<g font-family=\"Inter, Arial, 'Noto Sans KR', sans-serif\">")
    svg.append(f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" class="bg"/>')

    for card in cards:
        klass = "header-card" if card == header_card else "card"
        svg.append(f'<rect x="{card.x}" y="{card.y}" width="{card.w}" height="{card.h}" rx="14" ry="14" class="{klass}"/>')

    svg.append(f'<text x="{header_card.x + 20}" y="{header_card.y + 44}" class="title">{xml_escape(fit_text_single_line("Status Packet Dashboard", header_card.w - 40, 34))}</text>')
    sub1 = f"Generated (UTC): {ts} | Status: {status_text}"
    sub2 = f"Branch: {branch} | Commit: {short_sha(sha)} | Base: {short_sha(base)} | Head: {short_sha(head)}"
    svg.append(f'<text x="{header_card.x + 20}" y="{header_card.y + 72}" class="subtitle">{xml_escape(fit_text_single_line(sub1, header_card.w - 40, 14))}</text>')
    svg.append(f'<text x="{header_card.x + 20}" y="{header_card.y + 92}" class="subtitle">{xml_escape(fit_text_single_line(sub2, header_card.w - 40, 14))}</text>')

    kpis = [("Status", status_text), ("Modules", str(len(modules))), ("Edges", str(len(deps))), ("Entry points", str(len(entry_points))), ("Changed files", str(len(changes))), ("TODO/FIXME", str(len(todos)))]
    box_gap = 14
    box_w = (kpi_card.w - box_gap * (len(kpis) + 1)) // len(kpis)
    box_y = kpi_card.y + 20
    for i, (label, value) in enumerate(kpis):
        x = kpi_card.x + box_gap + i * (box_w + box_gap)
        svg.append(f'<rect x="{x}" y="{box_y}" width="{box_w}" height="{kpi_card.h - 40}" rx="10" ry="10" class="kpi-box"/>')
        svg.append(f'<text x="{x + 14}" y="{box_y + 30}" class="kpi-label">{xml_escape(fit_text_single_line(label, box_w - 28, 12))}</text>')
        svg.append(f'<text x="{x + 14}" y="{box_y + 68}" class="kpi-value">{xml_escape(fit_text_single_line(value, box_w - 28, 26))}</text>')

    left_cursor = render_card_title(svg, left_card, "Architecture Highlights", "Modules, edges, entry points, env toggles")
    left_lines = [f"Tracked modules: {len(modules)}", f"Internal dependency edges: {len(deps)}", f"Entrypoints: {len(entry_points)}", f"Top edge: {deps[0][0]} -> {deps[0][1]}" if deps else "Top edge: UNKNOWN", "Top env toggles:"]
    for name, default, rel in env_toggles[:6]:
        left_lines.append(f"- {name} (default={normalize_text(default)}) @ {rel}")
    cap_left = line_capacity(left_card)
    if len(left_lines) > cap_left:
        left_lines = left_lines[: cap_left - 1] + [f"+{len(left_lines) - cap_left + 1} more"]
    for line in left_lines:
        wrapped = wrap_text_lines(line, left_card.w - 40, 14, 1)[0]
        svg.append(f'<text x="{left_card.x + 20}" y="{left_cursor}" class="body">{xml_escape(wrapped)}</text>')
        left_cursor += 20

    right_cursor = render_card_title(svg, right_card, "Top Changes", "Highest churn files in current diff range")
    right_lines = [f"{idx}. {normalize_text(row.get('path', ''))} (churn={int(row.get('churn', 0))})" for idx, row in enumerate(top_changes[:10], start=1)] or ["No diff data available."]
    cap_right = line_capacity(right_card)
    if len(right_lines) > cap_right:
        right_lines = right_lines[: cap_right - 1] + [f"+{len(right_lines) - cap_right + 1} more"]
    for line in right_lines:
        wrapped = wrap_text_lines(line, right_card.w - 40, 14, 1)[0]
        svg.append(f'<text x="{right_card.x + 20}" y="{right_cursor}" class="body">{xml_escape(wrapped)}</text>')
        right_cursor += 20

    svg.append(f'<text x="{table_card.x + 20}" y="{table_card.y + 34}" class="table-title">{xml_escape(fit_text_single_line("Changed Files", table_card.w - 40, 19))}</text>')
    head_y = table_card.y + 48
    svg.append(f'<rect x="{table_card.x + 12}" y="{head_y}" width="{table_card.w - 24}" height="28" rx="8" ry="8" class="table-head"/>')
    col_status, col_file = table_card.x + 24, table_card.x + 86
    col_add, col_del, col_churn = table_card.x + table_card.w - 256, table_card.x + table_card.w - 174, table_card.x + table_card.w - 90
    file_w = col_add - col_file - 12
    svg.append(f'<text x="{col_status}" y="{head_y + 19}" class="table-head-text">ST</text>')
    svg.append(f'<text x="{col_file}" y="{head_y + 19}" class="table-head-text">FILE</text>')
    svg.append(f'<text x="{col_add}" y="{head_y + 19}" class="table-head-text">+</text>')
    svg.append(f'<text x="{col_del}" y="{head_y + 19}" class="table-head-text">-</text>')
    svg.append(f'<text x="{col_churn}" y="{head_y + 19}" class="table-head-text">CHURN</text>')

    row_h = 22
    row_y0 = head_y + 44
    max_rows = max(1, (table_card.h - 92) // row_h)
    rows = top_changes[:max_rows]
    if top_changes and len(top_changes) > max_rows:
        rows = top_changes[: max_rows - 1]

    if not rows:
        svg.append(f'<text x="{table_card.x + 20}" y="{row_y0 + 8}" class="muted">{xml_escape("No changed files found for current base/head range.")}</text>')
    else:
        for idx, row in enumerate(rows):
            y = row_y0 + idx * row_h
            svg.append(f'<text x="{col_status}" y="{y}" class="table-row">{xml_escape(fit_text_single_line(row.get("status", "M"), 30, 12))}</text>')
            svg.append(f'<text x="{col_file}" y="{y}" class="table-row">{xml_escape(fit_text_single_line(row.get("path", ""), file_w, 12))}</text>')
            svg.append(f'<text x="{col_add}" y="{y}" class="table-row">{int(row.get("added", 0))}</text>')
            svg.append(f'<text x="{col_del}" y="{y}" class="table-row">{int(row.get("deleted", 0))}</text>')
            svg.append(f'<text x="{col_churn}" y="{y}" class="table-row">{int(row.get("churn", 0))}</text>')
            svg.append(f'<line x1="{table_card.x + 16}" y1="{y + 6}" x2="{table_card.x + table_card.w - 16}" y2="{y + 6}" class="row-line"/>')
        hidden = len(top_changes) - len(rows)
        if hidden > 0:
            mx, my = table_card.x + 16, row_y0 + len(rows) * row_h
            svg.append(f'<rect x="{mx}" y="{my - 14}" width="150" height="20" rx="9" ry="9" class="chip"/>')
            svg.append(f'<text x="{mx + 10}" y="{my}" class="muted">+{hidden} more files</text>')

    svg.append("</g>")
    svg.append("</svg>")
    xml_text = "\n".join(svg)
    ET.fromstring(xml_text)
    write_text(path, xml_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified status packet outputs.")
    parser.add_argument("--repo-root", default=str(_DEFAULT_REPO_ROOT))
    parser.add_argument("--out-dir", default="docs_tests_data")
    parser.add_argument("--artifacts-dir", default="docs_tests_data/artifacts/latest_run")
    parser.add_argument("--max-log-lines", type=int, default=40)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", default="")
    return parser.parse_args()


def build_docs(repo_root: Path, out_dir: Path, artifacts_dir: Path, max_log_lines: int, base_sha: str, head_sha: str) -> None:
    ts = now_iso()
    sha = run(["git", "rev-parse", "HEAD"], repo_root)
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    base, head = resolve_base_head(repo_root, base_sha, head_sha)
    fallback_source = "git"
    source_revision = load_source_revision(out_dir, repo_root)
    if branch == "UNKNOWN":
        branch = str(source_revision.get("branch", "") or "UNKNOWN")
        fallback_source = "source_revision.json"
    if sha == "UNKNOWN":
        sha = str(source_revision.get("commit", "") or "UNKNOWN")
        fallback_source = "source_revision.json"
    if (base in {"", "UNKNOWN", None}) and source_revision.get("base"):
        base = str(source_revision.get("base"))
        fallback_source = "source_revision.json"
    if (head in {"", "UNKNOWN", None}) and source_revision.get("head"):
        head = str(source_revision.get("head"))
        fallback_source = "source_revision.json"

    metrics = safe_json(artifacts_dir / "metrics.json") if artifacts_dir.exists() else None
    run_summary = safe_json(artifacts_dir / "run_summary.json") if artifacts_dir.exists() else None
    jsonl_tail = latest_jsonl_tail(artifacts_dir, max_log_lines) if artifacts_dir.exists() else []
    runtime_kpis = collect_runtime_kpis(out_dir)
    replay_metrics = safe_json(out_dir / "replay_metrics.json")

    modules, deps = module_overview(repo_root)
    entry_points = discover_entry_points(repo_root)
    env_toggles = discover_env_toggles(repo_root)
    changes = collect_changes(repo_root, base, head)
    todos = gather_todos(repo_root, limit=40)
    structure_counts = collect_structure_counts(repo_root)
    metric_payload: dict[str, Any] = {}
    if isinstance(metrics, dict):
        metric_payload.update(metrics)
    metric_payload.update(runtime_kpis)
    if isinstance(replay_metrics, dict):
        metric_payload.update({k: v for k, v in replay_metrics.items() if isinstance(v, (int, float))})

    md = build_status_packet_md(
        ts=ts,
        branch=branch,
        sha=sha,
        base=base,
        head=head,
        fallback_source=fallback_source,
        status_text=infer_status(run_summary),
        metric_rows=collect_metric_rows(metric_payload, limit=12),
        run_summary=run_summary,
        jsonl_tail=jsonl_tail,
        modules=modules,
        deps=deps,
        entry_points=entry_points,
        env_toggles=env_toggles,
        changes=changes,
        todos=todos,
        structure_counts=structure_counts,
        out_rel=rel_path(out_dir, repo_root),
        artifacts_rel=rel_path(artifacts_dir, repo_root),
    )
    write_text(out_dir / OUTPUT_MD, md)
    write_visual_dashboard_svg(
        path=out_dir / OUTPUT_SVG,
        ts=ts,
        sha=sha,
        branch=branch,
        base=base,
        head=head,
        status_text=infer_status(run_summary),
        modules=modules,
        deps=deps,
        entry_points=entry_points,
        env_toggles=env_toggles,
        changes=changes,
        todos=todos,
    )
    cleanup_legacy_outputs(out_dir)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    build_docs(repo_root, out_dir, artifacts_dir, int(args.max_log_lines), str(args.base_sha or ""), str(args.head_sha or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
