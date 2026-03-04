from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from m3.io.jsonl_writer import append_jsonl


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"shadow", "observe", "dryrun"}:
        return "shadow"
    if mode in {"enforce", "active", "apply"}:
        return "enforce"
    return "enforce"


def _call_optional(obj: Any, name: str, *args: Any, default: Any = None) -> Any:
    fn = getattr(obj, name, None)
    if callable(fn):
        try:
            return fn(*args)
        except Exception:
            return default
    return default


def _resolve_log_dir() -> str:
    base = str(
        os.getenv("LLM_ADAPTER_LOG_DIR")
        or Path(__file__).resolve().parents[1] / "docs_tests_data"
    ).strip()
    if not base:
        base = str(Path(__file__).resolve().parents[1] / "docs_tests_data")
    if not os.path.isabs(base):
        base = os.path.abspath(str(Path(__file__).resolve().parents[1] / base))
    return base


def resolve_control_decision_log_path() -> str:
    raw = str(os.getenv("M3_CONTROL_DECISION_LOG", "") or "").strip()
    if not raw:
        raw = os.path.join(_resolve_log_dir(), "control_decision.jsonl")
    elif not os.path.isabs(raw):
        raw = os.path.abspath(os.path.join(_resolve_log_dir(), raw))
    return os.path.abspath(raw)


@dataclass(frozen=True)
class M3ControlDecision:
    schema_version: str
    control_decision_id: str
    ts: float
    decision_mode: str
    selection_mode: str
    allow_state_context: bool
    allow_bridge: bool
    allow_decode_control: bool
    allow_memory_retrieval: bool
    quality_gate_on: bool
    fallback_mode: str
    repeat_block_policy: Dict[str, Any]
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "control_decision_id": str(self.control_decision_id),
            "ts": float(self.ts),
            "decision_mode": str(self.decision_mode),
            "selection_mode": str(self.selection_mode),
            "allow_state_context": bool(self.allow_state_context),
            "allow_bridge": bool(self.allow_bridge),
            "allow_decode_control": bool(self.allow_decode_control),
            "allow_memory_retrieval": bool(self.allow_memory_retrieval),
            "quality_gate_on": bool(self.quality_gate_on),
            "fallback_mode": str(self.fallback_mode),
            "repeat_block_policy": dict(self.repeat_block_policy),
            "reason_codes": list(self.reason_codes),
        }


def build_control_decision(policy: Any, *, prompt: str = "", source: str = "") -> M3ControlDecision:
    mode = _normalize_mode(str(os.getenv("M3_CONTROL_PLANE_MODE", "enforce")))
    selection_mode = str(
        _call_optional(policy, "_control_selection_mode", default=os.getenv("M3_CONTROL_SELECTION_MODE", "state"))
        or "state"
    ).strip().lower()
    allow_state_context = bool(_call_optional(policy, "_control_allows", "state_context", default=False))
    allow_bridge = bool(_call_optional(policy, "_control_allows", "bridge", default=False))
    allow_decode_control = bool(_call_optional(policy, "_control_allows", "decode_control", default=False))
    allow_memory_retrieval = bool(_call_optional(policy, "_control_allows", "memory_retrieval", default=False))
    quality_gate_allowed = bool(_call_optional(policy, "_control_allows", "quality_gate", default=False))
    quality_gate_on = bool(quality_gate_allowed and getattr(policy, "_quality_gate", None) is not None)

    reason_codes: List[str] = []
    fallback_mode = "normal"
    if bool(getattr(policy, "_hf_circuit_open", False)):
        fallback_mode = "circuit_breaker"
        reason_codes.append("hf_circuit_open")
    elif bool(_call_optional(policy, "_hf_runtime_cooldown_active", default=False)):
        fallback_mode = "cooldown_fallback"
        reason_codes.append("hf_runtime_cooldown")
    elif not bool(getattr(policy, "use_hf", False)):
        fallback_mode = "hf_unavailable"
        reason_codes.append("hf_unavailable_or_fallback")
    else:
        reason_codes.append("hf_generate_ok")
    if not quality_gate_on:
        reason_codes.append("quality_gate_off")
    repeat_block_policy = {
        "threshold": float(os.getenv("M3_REPEAT_BLOCK_THRESHOLD", "0.90")),
        "streak": int(os.getenv("M3_REPEAT_BLOCK_STREAK", "2")),
        "source": str(source or ""),
    }
    return M3ControlDecision(
        schema_version="m3_control_decision_v1",
        control_decision_id=uuid.uuid4().hex[:16],
        ts=float(time.time()),
        decision_mode=mode,
        selection_mode=selection_mode,
        allow_state_context=allow_state_context,
        allow_bridge=allow_bridge,
        allow_decode_control=allow_decode_control,
        allow_memory_retrieval=allow_memory_retrieval,
        quality_gate_on=quality_gate_on,
        fallback_mode=fallback_mode,
        repeat_block_policy=repeat_block_policy,
        reason_codes=list(dict.fromkeys(reason_codes)),
    )


def log_control_decision(decision: M3ControlDecision, path: Optional[str] = None) -> Dict[str, Any]:
    target = path or resolve_control_decision_log_path()
    record = decision.to_dict()
    record.update(
        {
            "kind": "control_decision_shadow",
            "event": "control_decision",
        }
    )
    return append_jsonl(target, record)
