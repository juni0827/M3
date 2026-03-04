from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

_LOCK_MAP: Dict[str, threading.Lock] = {}
_LOCK_MAP_GUARD = threading.Lock()


def _path_lock(path: str) -> threading.Lock:
    key = os.path.abspath(str(path or ""))
    with _LOCK_MAP_GUARD:
        lock = _LOCK_MAP.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCK_MAP[key] = lock
        return lock


def _writer_id() -> str:
    return f"pid:{os.getpid()}-tid:{threading.get_ident()}"


def normalize_jsonl_record(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record or {})
    if "ts" not in rec:
        rec["ts"] = float(time.time())
    else:
        try:
            rec["ts"] = float(rec["ts"])
        except Exception:
            rec["ts"] = float(time.time())
    kind = str(rec.get("kind", "")).strip()
    if not kind:
        rec["kind"] = "log_event"
    has_event = str(rec.get("event", "")).strip() != ""
    has_source_key = str(rec.get("source", "")).strip() != "" and str(rec.get("key", "")).strip() != ""
    if not has_event and not has_source_key:
        rec["event"] = "record"
    rec.setdefault("schema_version", "jsonl_writer_v1")
    rec.setdefault("writer_id", _writer_id())
    rec.setdefault("record_id", uuid.uuid4().hex[:16])
    return rec


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> Dict[str, Any]:
    target = os.path.abspath(str(path))
    rec = normalize_jsonl_record(record)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    lock = _path_lock(target)
    with lock:
        with open(target, "a", encoding="utf-8") as f:
            f.write(line)
    return rec
