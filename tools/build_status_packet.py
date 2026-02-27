#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import subprocess
from pathlib import Path
from typing import Any

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
    "m3/visualization.py",
    "run_llm_adapter.py",
]

ENV_PATTERNS = [
    re.compile(r"os\\.environ\\.get\\(\\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\\s*(?:,\\s*([^\\)]+))?\\)"),
    re.compile(r"os\\.getenv\\(\\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\\s*(?:,\\s*([^\\)]+))?\\)"),
    re.compile(r"os\\.environ\\s*\\[\\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\\s*\\]"),
]


def run(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "UNKNOWN"


def run_lines(cmd: list[str], cwd: Path) -> list[str]:
    out = run(cmd, cwd)
    if out == "UNKNOWN" or not out:
        return []
    return [ln for ln in out.splitlines() if ln.strip()]


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    write(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def safe_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_json_block(data: Any) -> str:
    if data is None:
        return "UNKNOWN"
    return "```json\n" + json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n```"


def _normalize_markdown_spacing(markdown: str) -> str:
    """Normalize markdown spacing to satisfy common markdownlint rules.

    - Ensure headings are surrounded by at least one blank line.
    - Ensure fenced code blocks are separated from surrounding text.
    - Avoid collapsing non-code blank-lines into a hard constraint (keep single blank).
    """

    lines = markdown.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    in_fence = False

    def is_heading(line: str) -> bool:
        return bool(re.match(r"^\s{0,3}#{1,6}\s+.+", line))

    def is_fence(line: str) -> bool:
        return line.startswith("```")

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if in_fence:
            if is_fence(line):
                out.append(line)
                in_fence = False
                if i + 1 < len(lines) and lines[i + 1].strip():
                    out.append("")
                i += 1
                continue
            out.append(line)
            i += 1
            continue

        if is_fence(line):
            if out and out[-1] != "":
                out.append("")
            out.append(line)
            in_fence = True
            i += 1
            continue

        if is_heading(line):
            if out and out[-1] != "":
                out.append("")
            out.append(line)
            if i + 1 < len(lines) and lines[i + 1].strip():
                out.append("")
            i += 1
            continue

        if line == "":
            if not out or out[-1] != "":
                out.append("")
            i += 1
            continue

        out.append(line)
        i += 1

    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()

    return "\n".join(out) + "\n"


def file_header(ts: str, sha: str, branch: str) -> str:
    return (
        f"- Generated: {ts}\n"
        f"- Commit: `{sha}`\n"
        f"- Branch: `{branch}`\n"
        "- Repro seed/config: deterministic (no randomness used in generator)\n"
    )


def latest_jsonl_tail(artifacts_dir: Path, max_lines: int) -> list[str]:
    candidates = sorted(artifacts_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return []
    lines = candidates[-1].read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def discover_entry_points(repo_root: Path) -> list[str]:
    entries: set[str] = set()
    for py in sorted(repo_root.glob("*.py")):
        entries.add(py.name)
    for rel in KEY_FILES:
        path = repo_root / rel
        if not path.exists():
            continue
        txt = path.read_text(encoding="utf-8", errors="replace")
        if "if __name__ == \"__main__\"" in txt or "if __name__ == '__main__'" in txt:
            entries.add(rel)
    return sorted(entries)


def discover_env_toggles(repo_root: Path) -> list[tuple[str, str, str]]:
    found: dict[tuple[str, str], tuple[str, str, str]] = {}
    for rel in KEY_FILES:
        path = repo_root / rel
        if not path.exists():
            continue
        txt = path.read_text(encoding="utf-8", errors="replace")
        for pattern in ENV_PATTERNS:
            for m in pattern.finditer(txt):
                name = m.group(1)
                default = (m.group(2) or "None").strip()
                found[(name, rel)] = (name, default, rel)
    return sorted(found.values(), key=lambda x: (x[0], x[2]))


def module_overview(repo_root: Path) -> tuple[list[str], list[tuple[str, str]]]:
    modules = []
    edges: set[tuple[str, str]] = set()
    for rel in KEY_FILES:
        p = repo_root / rel
        if not p.exists():
            continue
        modules.append(rel)
        txt = p.read_text(encoding="utf-8", errors="replace")
        imports = set(re.findall(r"^\\s*(?:from|import)\\s+([a-zA-Z0-9_\\.]+)", txt, flags=re.M))
        for imp in sorted(imports):
            if imp.startswith("m3") or imp.startswith("llm_adapter"):
                edges.add((rel, imp))
    return modules, sorted(edges)


def gather_todos(repo_root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts or "docs_tests_data" in path.parts or "tools" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".json"}:
            continue
        try:
            for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if "TODO" in line or "FIXME" in line:
                    rel = path.relative_to(repo_root)
                    hits.append(f"- `{rel}:{idx}` {line.strip()[:140]}")
        except Exception:
            continue
    return hits[:30]


def parse_classes_and_functions(py_path: Path) -> dict[str, Any]:
    try:
        text = py_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except Exception:
        return {"classes": [], "functions": []}
    classes: list[dict[str, Any]] = []
    funcs: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append({"name": node.name, "methods": methods, "lineno": int(getattr(node, "lineno", 0))})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(node.name)
    return {"classes": classes, "functions": funcs}


def collect_class_catalog(repo_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    summary = {"files": 0, "classes": 0, "functions": 0, "methods": 0}
    for rel in KEY_FILES:
        p = repo_root / rel
        if not p.exists() or p.suffix != ".py":
            continue
        info = parse_classes_and_functions(p)
        out[rel] = info
        summary["files"] += 1
        summary["classes"] += len(info["classes"])
        summary["functions"] += len(info["functions"])
        summary["methods"] += sum(len(c["methods"]) for c in info["classes"])
    return {"summary": summary, "files": out}


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
    for ln in run_lines(["git", "diff", "--name-status", f"{base}..{head}"], repo_root):
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        status_raw = parts[0].strip()
        status = status_raw[:1]
        path = parts[-1].strip().replace("\\", "/")
        rows[path] = {
            "status": status,
            "status_raw": status_raw,
            "path": path,
            "added": 0,
            "deleted": 0,
            "churn": 0,
        }
    for ln in run_lines(["git", "diff", "--numstat", f"{base}..{head}"], repo_root):
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        a, d, path = parts[0], parts[1], parts[-1].replace("\\", "/")
        added = int(a) if a.isdigit() else 0
        deleted = int(d) if d.isdigit() else 0
        row = rows.get(path) or {"status": "M", "status_raw": "M", "path": path, "added": 0, "deleted": 0, "churn": 0}
        row["added"] = added
        row["deleted"] = deleted
        row["churn"] = added + deleted
        rows[path] = row
    out = sorted(rows.values(), key=lambda x: (-int(x["churn"]), x["path"]))
    return out


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_module_graph_svg(path: Path, edges: list[tuple[str, str]]) -> None:
    # Deprecated: kept for backwards compatibility only; no-op placeholder.
    if not edges:
        return write(path, '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>')
    return write(path, '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>')


def write_change_heatmap_svg(path: Path, changes: list[dict[str, Any]]) -> None:
    # Deprecated.
    return write(path, '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>')


def write_class_overview_svg(path: Path, class_catalog: dict[str, Any]) -> None:
    # Deprecated.
    return write(path, '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>')


def build_visual_map_md(hdr: str) -> str:
    return (
        "# VISUAL_MAP\n\n"
        f"{hdr}\n\n"
        "![M3 Architecture Dashboard](VISUAL_MAP.svg)\n"
    )


def write_visual_dashboard_svg(
    path: Path,
    ts: str,
    sha: str,
    branch: str,
    base: str | None,
    head: str | None,
    modules: list[str],
    deps: list[tuple[str, str]],
    class_catalog: dict[str, Any],
    changes: list[dict[str, Any]],
) -> None:
    class_rows: list[tuple[str, int]] = []
    for rel, info in class_catalog.get("files", {}).items():
        cnt = len(info.get("classes", []))
        if cnt > 0:
            class_rows.append((rel, cnt))
    class_rows.sort(key=lambda x: (-x[1], x[0]))

    top_deps = sorted(deps)[:18]
    top_changes = sorted(changes, key=lambda c: int(c.get("churn", 0)), reverse=True)[:14]

    def norm(v: int, m: int) -> int:
        if m <= 0:
            return 0
        return max(6, int((v / m) * 620))

    max_classes = max((cnt for _, cnt in class_rows), default=1)
    max_churn = max((int(c.get("churn", 0)) for c in top_changes), default=1)
    dep_count = len(deps)
    change_count = len(changes)
    class_count = sum(cnt for _, cnt in class_rows)
    file_count = int(class_catalog.get("summary", {}).get("files", 0))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1460" height="980" viewBox="0 0 1460 980" role="img">',
        '<defs>',
        '  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '    <stop offset="0%" stop-color="#0b1020"/>',
        '    <stop offset="100%" stop-color="#111827"/>',
        "  </linearGradient>",
        '  <filter id="cardShadow" x="-20%" y="-20%" width="140%" height="140%">',
        '    <feDropShadow dx="0" dy="8" stdDeviation="10" flood-opacity="0.22"/>',
        "  </filter>",
        "</defs>",
        '<rect width="1460" height="980" fill="url(#bg)"/>',
        '<g font-family="Inter, Arial, sans-serif">',
        '<rect x="24" y="24" width="1412" height="72" rx="16" fill="#111827" fill-opacity="0.94" stroke="#334155" stroke-width="1" filter="url(#cardShadow)"/>',
        '<text x="48" y="66" fill="#f1f5f9" font-size="30" font-weight="700">M3 Architecture Visual Dashboard</text>',
        f'<text x="48" y="95" fill="#94a3b8" font-size="14">Generated: {ts}  |  Commit: {_svg_escape(sha[:12])}  |  Branch: {_svg_escape(branch)}</text>',
        f'<text x="48" y="113" fill="#94a3b8" font-size="12">Base: {_svg_escape(base or "UNKNOWN")}  |  Head: {_svg_escape(head or "UNKNOWN")}</text>',
        '<rect x="24" y="122" width="456" height="180" rx="14" fill="#17223a" fill-opacity="0.94" stroke="#334155" filter="url(#cardShadow)"/>',
        '<text x="44" y="148" fill="#cbd5e1" font-size="16" font-weight="600">Module graph</text>',
        f'<text x="44" y="170" fill="#9ca3af" font-size="12">{len(top_deps)} dependencies visualized</text>',
        f'<text x="44" y="188" fill="#9ca3af" font-size="12">Tracked modules: {len(modules)}</text>',
        '<text x="44" y="206" fill="#e2e8f0" font-size="12">Top edges:</text>',
    ]

    y = 228
    if top_deps:
        for src, dst in top_deps[:8]:
            lines.append(f'<text x="44" y="{y}" fill="#60a5fa" font-size="11">{_svg_escape(src)}</text>')
            lines.append(f'<text x="470" y="{y}" fill="#9ca3af" font-size="11">→</text>')
            lines.append(f'<text x="486" y="{y}" fill="#86efac" font-size="11">{_svg_escape(dst)}</text>')
            y += 18
    else:
        lines.append('<text x="44" y="228" fill="#9ca3af" font-size="12">No internal dependency edges found</text>')

    lines += [
        '<rect x="496" y="122" width="456" height="180" rx="14" fill="#17223a" fill-opacity="0.94" stroke="#334155" filter="url(#cardShadow)"/>',
        '<text x="516" y="148" fill="#cbd5e1" font-size="16" font-weight="600">Class distribution</text>',
        f'<text x="516" y="170" fill="#9ca3af" font-size="12">{len(class_rows)} files with classes</text>',
        '<text x="516" y="190" fill="#9ca3af" font-size="12">Top files shown (max 8)</text>',
    ]
    y = 214
    for rel, cnt in class_rows[:8]:
        w = norm(cnt, max_classes)
        lines.extend([
            f'<rect x="516" y="{y}" width="{w}" height="14" fill="#8b5cf6" opacity="0.92"/>',
            f'<text x="{516 + w + 10}" y="{y+11}" fill="#f8fafc" font-size="10" font-weight="600">{_svg_escape(rel)} ({cnt})</text>',
        ])
        y += 24

    lines += [
        '<rect x="968" y="122" width="468" height="180" rx="14" fill="#17223a" fill-opacity="0.94" stroke="#334155" filter="url(#cardShadow)"/>',
        '<text x="988" y="148" fill="#cbd5e1" font-size="16" font-weight="600">System health</text>',
        f'<text x="988" y="170" fill="#9ca3af" font-size="12">Changed files: {change_count}</text>',
        f'<text x="988" y="190" fill="#9ca3af" font-size="12">Classes: {class_count}, Files: {file_count}, Edges: {dep_count}</text>',
        '<text x="988" y="214" fill="#f8fafc" font-size="14">Change pressure (normalized)</text>',
    ]
    y = 236
    if top_changes:
        for idx, c in enumerate(top_changes[:6], start=1):
            churn = int(c.get("churn", 0))
            w = norm(churn, max_churn)
            lines.extend([
                f'<text x="988" y="{y+2}" fill="#9ca3af" font-size="10">{idx}.</text>',
                f'<rect x="1008" y="{y-7}" width="{w}" height="12" fill="#22c55e" opacity="0.9"/>',
                f'<text x="1008" y="{y+9}" fill="#e2e8f0" font-size="10">{_svg_escape(str(c.get("path", ""))[:48])}</text>',
                f'<text x="1450" y="{y+9}" fill="#cbd5e1" font-size="10" text-anchor="end">{churn}</text>',
            ])
            y += 24
    else:
        lines.append('<text x="988" y="236" fill="#9ca3af" font-size="12">No diff data available</text>')

    lines += [
        '<rect x="24" y="318" width="1412" height="620" rx="16" fill="#0f172a" fill-opacity="0.94" stroke="#334155" filter="url(#cardShadow)"/>',
        '<text x="44" y="350" fill="#cbd5e1" font-size="17" font-weight="600">Changed files (top by churn)</text>',
        '<text x="44" y="372" fill="#9ca3af" font-size="12">status  |  file  | + lines  | - lines  | churn</text>',
    ]
    y = 398
    if top_changes:
        for c in top_changes:
            lines.extend([
                f'<text x="44" y="{y}" fill="#94a3b8" font-size="11">{_svg_escape(str(c.get("status", "M"))):>2}</text>',
                f'<text x="86" y="{y}" fill="#bfdbfe" font-size="11" text-anchor="start">{_svg_escape(str(c.get("path", ""))[:115])}</text>',
                f'<text x="1160" y="{y}" fill="#cbd5e1" font-size="11" text-anchor="end">{int(c.get("added", 0)):>5}</text>',
                f'<text x="1248" y="{y}" fill="#f9fafb" font-size="11" text-anchor="end">{int(c.get("deleted", 0)):>5}</text>',
                f'<text x="1360" y="{y}" fill="#86efac" font-size="11" text-anchor="end">{int(c.get("churn", 0)):>6}</text>',
                f'<line x1="44" y1="{y+4}" x2="1432" y2="{y+4}" stroke="#1f2937" stroke-width="1"/>',
            ])
            y += 20
            if y > 900:
                break
    else:
        lines.append('<text x="44" y="398" fill="#9ca3af" font-size="12">No changed files in this base→head range.</text>')

    lines += [
        "</g>",
        "</svg>",
    ]
    write(path, "\n".join(lines))


def build_class_catalog_md(hdr: str, class_catalog: dict[str, Any]) -> str:
    s = class_catalog.get("summary", {})
    md = f"# CLASS_CATALOG\n\n{hdr}\n## Summary\n"
    md += f"- Files: `{s.get('files',0)}`\n- Classes: `{s.get('classes',0)}`\n- Functions: `{s.get('functions',0)}`\n- Methods: `{s.get('methods',0)}`\n"
    for rel, info in sorted(class_catalog.get("files", {}).items()):
        classes = info.get("classes", [])
        funcs = info.get("functions", [])
        if not classes and not funcs:
            continue
        md += "\n## " + rel + "\n"
        if classes:
            md += "\n### Classes\n\n"
            for c in classes:
                md += f"- `{c.get('name','')}` (L{c.get('lineno',0)})\n"
                methods = c.get("methods", [])
                if methods:
                    md += "  - methods:\n"
                    for m in methods[:24]:
                        md += f"    - `{m}`\n"
                    if len(methods) > 24:
                        md += "    - ...\n"
            md += "\n"
        if funcs:
            md += "\n### Top-level Functions\n\n"
            for fn in funcs[:40]:
                md += f"- `{fn}`\n"
        md += "\n"
    return md


def build_change_report_md(hdr: str, base: str | None, head: str | None, changes: list[dict[str, Any]]) -> str:
    md = f"# CHANGE_REPORT\n\n{hdr}\n## Diff Context\n"
    md += f"- Base SHA: `{base or 'UNKNOWN'}`\n- Head SHA: `{head or 'UNKNOWN'}`\n- Changed files: `{len(changes)}`\n"
    if not changes:
        md += "\n- No git diff data available.\n"
        return md
    md += "\n## Changed Files\n| Status | File | + | - | Churn |\n|---|---|---:|---:|---:|\n"
    for c in changes[:80]:
        md += f"| `{c.get('status','M')}` | `{c.get('path','')}` | {int(c.get('added',0))} | {int(c.get('deleted',0))} | {int(c.get('churn',0))} |\n"
    return md


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build status packet markdown docs")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--out-dir", default="docs_tests_data")
    p.add_argument("--artifacts-dir", default="docs_tests_data/artifacts/latest_run")
    p.add_argument("--max-log-lines", type=int, default=200)
    p.add_argument("--base-sha", default="")
    p.add_argument("--head-sha", default="")
    return p.parse_args()


def build_docs(repo_root: Path, out_dir: Path, artifacts_dir: Path, max_log_lines: int, base_sha: str, head_sha: str) -> None:
    ts = now_iso()
    sha = run(["git", "rev-parse", "HEAD"], repo_root)
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    hdr = file_header(ts, sha, branch)
    out_rel = rel_path(out_dir, repo_root)
    artifacts_rel = rel_path(artifacts_dir, repo_root)

    base, head = resolve_base_head(repo_root, base_sha, head_sha)
    changes = collect_changes(repo_root, base, head)

    metrics = safe_json(artifacts_dir / "metrics.json") if artifacts_dir.exists() else None
    run_summary = safe_json(artifacts_dir / "run_summary.json") if artifacts_dir.exists() else None
    jsonl_tail = latest_jsonl_tail(artifacts_dir, max_log_lines) if artifacts_dir.exists() else []

    status_text = "UNKNOWN"
    if isinstance(run_summary, dict):
        if run_summary.get("status") in {"success", "passed", True}:
            status_text = "PASS"
        elif run_summary.get("status") in {"failed", "failure", False}:
            status_text = "FAIL"

    status_md = (
        "# STATUS\n\n"
        f"{hdr}\n\n"
        "## As-of\n\n"
        f"- As-of: `{ts}`\n"
        f"- Commit SHA: `{sha}`\n\n"
        "## Key metrics summary\n\n"
        f"{format_json_block(metrics)}\n\n"
        "## Latest experiment results\n\n"
        f"{format_json_block(run_summary)}\n\n"
        "## Diff Snapshot\n\n"
        f"- Base SHA: `{base or 'UNKNOWN'}`\n"
        f"- Head SHA: `{head or 'UNKNOWN'}`\n"
        f"- Changed files: `{len(changes)}`\n\n"
        "## Success/failure conditions\n\n"
        f"- Overall status: **{status_text}**\n"
        "- Verification items:\n"
        "  - Metrics file present\n"
        f"    - `{artifacts_rel}/metrics.json`\n"
        "  - Run summary present\n"
        f"    - `{artifacts_rel}/run_summary.json`\n"
        "  - Recent log lines parseable\n"
        f"    - `{artifacts_rel}/*.jsonl`\n\n"
        f"## Latest JSONL tail ({max_log_lines} lines max)\n\n"
    )
    if jsonl_tail:
        status_md += "```jsonl\n" + "\n".join(jsonl_tail) + "\n```\n"
    else:
        status_md += "- UNKNOWN (no jsonl artifacts found)\n"

    modules, deps = module_overview(repo_root)
    entries = discover_entry_points(repo_root)
    envs = discover_env_toggles(repo_root)
    arch_md = f"# ARCHITECTURE_STATE\n\n{hdr}\n## Module graph / dependency overview\n\n### Key modules\n" + "\n".join(f"- `{m}`" for m in modules)
    arch_md += "\n\n### Internal dependency edges\n" + ("\n".join(f"- `{a}` -> `{b}`" for a, b in deps) if deps else "- UNKNOWN")
    arch_md += "\n\n## Entry points\n" + ("\n".join(f"- `{e}`" for e in entries) if entries else "- UNKNOWN")
    arch_md += "\n\n## Env toggles used in run paths\n"
    if envs:
        arch_md += "| Variable | Default | File |\n|---|---|---|\n"
        for name, default, rel in envs:
            arch_md += f"| `{name}` | `{default}` | `{rel}` |\n"
    else:
        arch_md += "- UNKNOWN\n"
    arch_md += "\n## Extended Docs\n- `docs_tests_data/VISUAL_MAP.md`\n- `docs_tests_data/VISUAL_MAP.svg`\n- `docs_tests_data/CLASS_CATALOG.md`\n- `docs_tests_data/CHANGE_REPORT.md`\n"

    req_exists = (repo_root / "requirements.txt").exists()
    install_line = "python -m pip install -r requirements.txt" if req_exists else "python -m pip install <dependencies>  # requirements.txt not found"
    runbook_md = (
        "# RUNBOOK\n\n"
        f"{hdr}\n\n"
        "## Repro commands (local + CI)\n\n"
        "```bash\n"
        f"{install_line}\n"
        "python tools/build_status_packet.py \\\n"
        f"  --repo-root . \\\n  --out-dir {out_rel} \\\n"
        f"  --artifacts-dir {artifacts_rel}\n"
        "python run_llm_adapter.py\n"
        "```\n\n"
        "## Verification checklist\n\n"
        f"- [ ] `{out_rel}/STATUS.md` generated.\n"
        f"- [ ] `{out_rel}/VISUAL_MAP.md` generated.\n"
        f"- [ ] `{out_rel}/VISUAL_MAP.svg` generated.\n"
        f"- [ ] `{out_rel}/CLASS_CATALOG.md` generated.\n"
        f"- [ ] `{out_rel}/CHANGE_REPORT.md` generated.\n"
    )

    todos = gather_todos(repo_root)
    backlog_md = f"""# BACKLOG\n\n{hdr}\n## Prioritized next actions\n\n### P0 (critical)\n\n- Keep `{out_rel}/CHANGE_REPORT.md` synced with PR/commit delta.\n- Keep dashboard (`VISUAL_MAP.svg`) current with architecture changes.\n\n## Open risks / tech debt\n"""
    backlog_md += ("\n".join(todos) if todos else "- No TODO/FIXME markers found.") + "\n"

    class_catalog = collect_class_catalog(repo_root)
    write_visual_dashboard_svg(
        out_dir / "VISUAL_MAP.svg",
        ts,
        sha,
        branch,
        base,
        head,
        modules,
        deps,
        class_catalog,
        changes,
    )
    visual_md = build_visual_map_md(hdr)
    class_md = build_class_catalog_md(hdr, class_catalog)
    change_md = build_change_report_md(hdr, base, head, changes)

    write(out_dir / "STATUS.md", _normalize_markdown_spacing(status_md))
    write(out_dir / "ARCHITECTURE_STATE.md", _normalize_markdown_spacing(arch_md))
    write(out_dir / "RUNBOOK.md", _normalize_markdown_spacing(runbook_md))
    write(out_dir / "BACKLOG.md", _normalize_markdown_spacing(backlog_md))
    write(out_dir / "VISUAL_MAP.md", visual_md)
    for legacy in (
        "VISUAL_MAP.html",
        "visual_module_graph.svg",
        "visual_class_density.svg",
        "visual_change_heatmap.svg",
    ):
        legacy_path = out_dir / legacy
        if legacy_path.exists():
            legacy_path.unlink()
    write(out_dir / "CLASS_CATALOG.md", _normalize_markdown_spacing(class_md))
    write(out_dir / "CHANGE_REPORT.md", _normalize_markdown_spacing(change_md))
    write_json(out_dir / "CLASS_INDEX.json", class_catalog)
    write_json(out_dir / "CHANGESET.json", {"base_sha": base, "head_sha": head, "changes": changes})


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    build_docs(repo_root, out_dir, artifacts_dir, int(args.max_log_lines), str(args.base_sha or ""), str(args.head_sha or ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

