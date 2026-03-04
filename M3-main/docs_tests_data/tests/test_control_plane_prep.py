from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

from m3.attr_contract import attr_get_optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_adapter.control_plane import build_control_decision
from m3.io.jsonl_writer import append_jsonl
from m3.m3_core import M3ConsciousnessCore


class ControlPlanePrepTests(unittest.TestCase):
    def test_generate_research_plan_noop_fixed(self):
        core = M3ConsciousnessCore.__new__(M3ConsciousnessCore)
        core.llm_adapter = types.SimpleNamespace(
            generate=lambda prompt, max_len=200, source="research_plan": "plan: explore -> verify -> synthesize"
        )
        out = M3ConsciousnessCore._generate_research_plan(
            core,
            m3_state={"phi": 0.22, "activation": 0.51},
            needs=[{"type": "resource_constraint", "source": "energy", "intensity": 0.7}],
            question={"formulation": "How should energy be allocated?"},
        )
        self.assertIsNotNone(out)
        self.assertIn("plan:", str(out))

    def test_jsonl_writer_threadsafe_append(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "writer.jsonl"

            def _worker(offset: int):
                for i in range(400):
                    append_jsonl(
                        path,
                        {
                            "kind": "unit_test",
                            "event": "append",
                            "source": "test",
                            "key": f"k{offset + i}",
                        },
                    )

            ths = []
            for idx in range(8):
                th = threading.Thread(target=_worker, args=(idx * 1000,), daemon=True)
                th.start()
                ths.append(th)
            for th in ths:
                th.join(timeout=5.0)

            rows = []
            for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
                rows.append(json.loads(raw))
            self.assertEqual(len(rows), 8 * 400)
            for rec in rows[:50]:
                self.assertTrue(str(rec.get("kind", "")).strip())
                self.assertTrue(str(rec.get("schema_version", "")).strip())
                self.assertTrue(str(rec.get("writer_id", "")).strip())
                self.assertTrue(str(rec.get("record_id", "")).strip())

    def test_control_decision_shadow_enforce_consistency(self):
        class _StubPolicy:
            use_hf = True
            _hf_circuit_open = False
            _quality_gate = object()

            def _control_selection_mode(self):
                return "full"

            def _control_allows(self, feature: str) -> bool:
                return feature in {
                    "state_context",
                    "bridge",
                    "decode_control",
                    "memory_retrieval",
                    "quality_gate",
                }

            def _hf_runtime_cooldown_active(self):
                return False

        stub = _StubPolicy()
        old_mode = os.getenv("M3_CONTROL_PLANE_MODE")
        try:
            os.environ["M3_CONTROL_PLANE_MODE"] = "shadow"
            d_shadow = build_control_decision(stub, prompt="p", source="generate")
            os.environ["M3_CONTROL_PLANE_MODE"] = "enforce"
            d_enforce = build_control_decision(stub, prompt="p", source="generate")
        finally:
            if old_mode is None:
                os.environ.pop("M3_CONTROL_PLANE_MODE", None)
            else:
                os.environ["M3_CONTROL_PLANE_MODE"] = old_mode

        self.assertEqual(d_shadow.selection_mode, d_enforce.selection_mode)
        self.assertEqual(d_shadow.allow_bridge, d_enforce.allow_bridge)
        self.assertEqual(d_shadow.allow_decode_control, d_enforce.allow_decode_control)
        self.assertEqual(d_shadow.allow_memory_retrieval, d_enforce.allow_memory_retrieval)
        self.assertEqual(d_shadow.quality_gate_on, d_enforce.quality_gate_on)
        self.assertEqual(d_shadow.decision_mode, "shadow")
        self.assertEqual(d_enforce.decision_mode, "enforce")

    def test_replay_runtime_kpis_script_api(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            append_jsonl(out_dir / "bus.jsonl", {"kind": "feature_signal", "event": "feature_push", "source": "x", "key": "y"})
            append_jsonl(out_dir / "llm_adapter.log", {"kind": "hf_runtime_failure", "reason_code": "hf_runtime_failure"})
            append_jsonl(out_dir / "llm_adapter.log", {"kind": "diag", "reason_code": "hf_generate_ok"})
            fixture = PROJECT_ROOT / "tests" / "fixtures" / "replay_chat_history.jsonl"
            mod_path = PROJECT_ROOT / "tests" / "replay_runtime_kpis.py"
            spec = importlib.util.spec_from_file_location("replay_runtime_kpis", str(mod_path))
            module = importlib.util.module_from_spec(spec)
            assert spec is not None and spec.loader is not None
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            result = module.run_replay_kpis(out_dir=out_dir, fixture_path=fixture)
            self.assertIn("baseline", result)
            self.assertIn("replay", result)
            self.assertIn("unknown_abstain_ratio", result["replay"])
            self.assertIn("response_repeat_ratio", result["replay"])


if __name__ == "__main__":
    unittest.main()
