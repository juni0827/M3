#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

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


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def file_header(ts: str, sha: str, branch: str) -> str:
    return (
        f"- Generated: {ts}\n"
        f"- Commit: `{sha}`\n"
        f"- Branch: `{branch}`\n"
        "- Repro seed/config: deterministic (no randomness used in generator)\n"
    )


def safe_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
                key = (name, rel)
                found[key] = (name, default, rel)
    return sorted(found.values(), key=lambda x: (x[0], x[2]))


def module_overview(repo_root: Path) -> tuple[list[str], list[str]]:
    modules = []
    dependencies = []
    for rel in KEY_FILES:
        p = repo_root / rel
        if not p.exists():
            continue
        modules.append(rel)
        txt = p.read_text(encoding="utf-8", errors="replace")
        imports = set(re.findall(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", txt, flags=re.M))
        for imp in sorted(imports):
            if imp.startswith("m3") or imp.startswith("llm_adapter"):
                dependencies.append(f"- `{rel}` -> `{imp}`")
    return modules, sorted(set(dependencies))


def gather_todos(repo_root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if "docs" in path.parts or "tools" in path.parts:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build status packet markdown docs")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--out-dir", default="docs")
    p.add_argument("--artifacts-dir", default="artifacts/latest_run")
    p.add_argument("--max-log-lines", type=int, default=200)
    return p.parse_args()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def format_json_block(data) -> str:
    if data is None:
        return "UNKNOWN"
    return "```json\n" + json.dumps(data, indent=2, sort_keys=True) + "\n```"


def recent_changes(repo_root: Path) -> str:
    log = run(["git", "log", "--oneline", "-10"], repo_root)
    if log == "UNKNOWN" or not log:
        return "- UNKNOWN"
    return "\n".join(f"- {line}" for line in log.splitlines())


def build_docs(repo_root: Path, out_dir: Path, artifacts_dir: Path, max_log_lines: int) -> None:
    ts = now_iso()
    sha = run(["git", "rev-parse", "HEAD"], repo_root)
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    hdr = file_header(ts, sha, branch)

    metrics = safe_json(artifacts_dir / "metrics.json") if artifacts_dir.exists() else None
    run_summary = safe_json(artifacts_dir / "run_summary.json") if artifacts_dir.exists() else None
    jsonl_tail = latest_jsonl_tail(artifacts_dir, max_log_lines) if artifacts_dir.exists() else []

    status_text = "UNKNOWN"
    if isinstance(run_summary, dict):
        if run_summary.get("status") in {"success", "passed", True}:
            status_text = "PASS"
        elif run_summary.get("status") in {"failed", "failure", False}:
            status_text = "FAIL"

    status_md = f"""# STATUS\n\n{hdr}\n## As-of\n- As-of: `{ts}`\n- Commit SHA: `{sha}`\n\n## Key metrics summary\n{format_json_block(metrics)}\n\n## Latest experiment results\n{format_json_block(run_summary)}\n\n## Success/failure conditions\n- Overall status: **{status_text}**\n- Verification items:\n  - Metrics file present (`artifacts/latest_run/metrics.json`)\n  - Run summary present (`artifacts/latest_run/run_summary.json`)\n  - Recent log lines parseable (`*.jsonl`)\n- Failure conditions:\n  - Missing core artifact files => status remains `UNKNOWN`\n  - `run_summary.status` explicitly failing => `FAIL`\n- Repro seed+config:\n  - Seed: `N/A (deterministic generator)`\n  - Config: `tools/build_status_packet.py --repo-root . --out-dir docs --artifacts-dir artifacts/latest_run --max-log-lines {max_log_lines}`\n\n## Recent changes summary\n{recent_changes(repo_root)}\n\n## Latest JSONL tail ({max_log_lines} lines max)\n"""
    if jsonl_tail:
        status_md += "```jsonl\n" + "\n".join(jsonl_tail) + "\n```\n"
    else:
        status_md += "- UNKNOWN (no jsonl artifacts found)\n"

    modules, deps = module_overview(repo_root)
    entries = discover_entry_points(repo_root)
    envs = discover_env_toggles(repo_root)
    arch_md = f"""# ARCHITECTURE_STATE\n\n{hdr}\n## Module graph / dependency overview\n### Key modules\n""" + "\n".join(f"- `{m}`" for m in modules) + "\n\n### Internal dependency edges\n" + ("\n".join(deps) if deps else "- UNKNOWN") + "\n\n## Entry points\n" + ("\n".join(f"- `{e}`" for e in entries) if entries else "- UNKNOWN") + "\n\n## Env toggles used in run paths\n"
    if envs:
        arch_md += "| Variable | Default | File |\n|---|---|---|\n"
        for name, default, rel in envs:
            arch_md += f"| `{name}` | `{default}` | `{rel}` |\n"
    else:
        arch_md += "- UNKNOWN\n"
    arch_md += "\n## Critical path (high-level)\n1. Start from `run_llm_adapter.py` or root scripts.\n2. Initialize M3 core (`m3/m3_core.py`) and adapter runtime (`llm_adapter/llm_core.py`).\n3. Route state/reward/features through policy modules and control bridge.\n4. Emit outputs and logs, optionally persisted as run artifacts.\n"

    req_exists = (repo_root / "requirements.txt").exists()
    install_line = "python -m pip install -r requirements.txt" if req_exists else "python -m pip install <dependencies>  # requirements.txt not found"
    runbook_md = f"""# RUNBOOK\n\n{hdr}\n## Repro commands (local + CI)\n```bash\n{install_line}\npython tools/build_status_packet.py --repo-root . --out-dir docs --artifacts-dir artifacts/latest_run\npython run_llm_adapter.py\n```\n\n## Seed/config conventions (determinism)\n- Status packet generator uses no randomness.\n- Repro seed: `N/A`\n- Repro config: CLI flags to `tools/build_status_packet.py` are the source of truth.\n\n## Env var setup example\n```bash\nexport M3_ENV=dev\nexport PYTHONUNBUFFERED=1\npython run_llm_adapter.py\n```\n\n## Verification checklist\n- [ ] `docs/STATUS.md` generated with current commit SHA.\n- [ ] `docs/ARCHITECTURE_STATE.md` includes entry points and env toggles.\n- [ ] `docs/RUNBOOK.md` contains deterministic repro steps.\n- [ ] `docs/BACKLOG.md` contains P0/P1/P2 and investor demo checklist.\n\n## Failure conditions\n- Missing artifact directory is allowed, but must render `UNKNOWN` metrics.\n- Git metadata unavailable => SHA/branch become `UNKNOWN`.\n- Script exits non-zero on file write errors.\n\n## Debug routines (common issues)\n1. Validate git context: `git rev-parse HEAD && git rev-parse --abbrev-ref HEAD`.\n2. Confirm artifact paths: `ls -la artifacts/latest_run`.\n3. Re-run with bounded log tail: `python tools/build_status_packet.py --max-log-lines 50`.\n4. Inspect markdown diff: `git diff -- docs/`.\n"""

    todos = gather_todos(repo_root)
    ref_lines = []
    for md in sorted(repo_root.rglob("*.md")):
        if ".git" in md.parts or "docs" in md.parts:
            continue
        txt = md.read_text(encoding="utf-8", errors="replace")
        for ln in txt.splitlines():
            if re.search(r"#\d+", ln):
                ref_lines.append(f"- `{md.relative_to(repo_root)}`: {ln.strip()[:120]}")
    backlog_md = f"""# BACKLOG\n\n{hdr}\n## Prioritized next actions\n### P0 (critical)\n- Keep `docs/STATUS.md` synced with latest commit and artifacts.\n- Validate PASS/FAIL/UNKNOWN mapping from `run_summary.status`.\n- Ensure CI auto-commit path only runs on `main` and schedule events.\n\n### P1 (important)\n- Expand architecture dependency extraction for richer module graphing.\n- Add additional run artifact parsers when new outputs are introduced.\n- Tighten verification checklist with project-specific acceptance gates.\n\n### P2 (nice-to-have)\n- Improve trend summaries across multiple historical runs.\n- Add richer release/demo readiness indicators from artifact history.\n\n## Open risks / tech debt\n"""
    backlog_md += ("\n".join(todos) if todos else "- No TODO/FIXME markers found.") + "\n\n"
    backlog_md += "## Issue references found in docs\n" + ("\n".join(ref_lines[:20]) if ref_lines else "- None found.") + "\n\n"
    backlog_md += """## Investor demo readiness checklist
- [ ] Demo scenario script is reproducible from clean checkout.
- [ ] Core runtime path (`run_llm_adapter.py`) runs without manual patching.
- [ ] Latest metrics and run summary are attached in `artifacts/latest_run/`.
- [ ] PASS/FAIL/UNKNOWN status is understandable by non-engineers.
- [ ] Architecture and runbook docs are current for the demo date.
"""

    write(out_dir / "STATUS.md", status_md)
    write(out_dir / "ARCHITECTURE_STATE.md", arch_md)
    write(out_dir / "RUNBOOK.md", runbook_md)
    write(out_dir / "BACKLOG.md", backlog_md)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    build_docs(repo_root, out_dir, artifacts_dir, args.max_log_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
