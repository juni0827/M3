from __future__ import annotations

import os
import sys
import types
from collections import deque
from pathlib import Path

import numpy as np
import pytest

TEST_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        parent
        for parent in TEST_FILE.parents
        if (parent / "llm_adapter").is_dir() and (parent / "m3").is_dir()
    ),
    TEST_FILE.parents[1],
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_adapter.llm_core import HFRuntimeFailure, HFBackend, TorchConversationalPolicy, _configure_external_library_logging
from m3.m3_core import FeatureBank, M3ConsciousnessCore, _validate_feature_bank_contract


def _make_action(reconnect_gain: float = 1.0) -> dict[str, float | str]:
    return {
        "exploration_factor": 0.0,
        "learning_rate": 0.0,
        "strategy": "normal",
        "sparsify_gain": 0.0,
        "prune_gain": 0.0,
        "reconnect_gain": float(reconnect_gain),
        "stability_gain": 0.0,
        "gate_adjust": 1.0,
        "kd_adjust": 1.0,
    }


def _make_core_for_execute_action(k: int = 16, cols: int = 5) -> M3ConsciousnessCore:
    core = M3ConsciousnessCore.__new__(M3ConsciousnessCore)
    core.K = int(k)
    core.n = 20
    core.groups = [np.arange(i * 5, (i + 1) * 5) for i in range(4)]
    core.h = np.zeros((core.n,), dtype=np.float32)
    core.U = np.zeros((core.K, int(cols)), dtype=np.float32)
    core.rng = np.random.default_rng(1234)
    core.P_obs_history = deque(maxlen=1000)
    core.base_vec = np.full((core.K,), 1.0 / max(1, core.K), dtype=np.float32)
    core.stability_window = deque(maxlen=50)
    core.gate_mid = 0.7
    core.kd_eff = 5.1
    core.l_ctrl = np.zeros((core.K,), dtype=np.float32)
    core.hi = 0
    core.lo = 1
    return core


def _make_fake_logging(has_many: bool, has_one: bool):
    calls = {"set": 0, "many": 0, "one": 0}

    def _set_verbosity_error():
        calls["set"] += 1

    module = types.SimpleNamespace(set_verbosity_error=_set_verbosity_error)
    if has_many:
        def _disable_many():
            calls["many"] += 1
        module.disable_progress_bars = _disable_many
    if has_one:
        def _disable_one():
            calls["one"] += 1
        module.disable_progress_bar = _disable_one
    return module, calls


def test_feature_bank_contract_passes_for_real_instance():
    fb = FeatureBank(max_dim=32, embed_dim=8)
    _validate_feature_bank_contract(fb, "test_feature_bank_contract_passes_for_real_instance")


def test_feature_bank_contract_raises_for_missing_methods():
    class _BrokenFB:
        def build(self):  # pragma: no cover
            return None

    with pytest.raises(AttributeError) as exc:
        _validate_feature_bank_contract(_BrokenFB(), "unit-test")
    msg = str(exc.value)
    assert "unit-test" in msg
    assert "missing callable method(s)" in msg
    assert "_register_default_specs" in msg
    assert "set_log_paths" in msg


def test_execute_action_reconnect_gain_does_not_overflow_columns():
    core = _make_core_for_execute_action(k=24, cols=5)
    action = _make_action(reconnect_gain=1.0)
    for _ in range(300):
        delta_hat, p_obs = M3ConsciousnessCore._execute_action(core, action)
        assert np.isfinite(float(delta_hat))
        assert p_obs.shape == (core.K,)


def test_execute_action_raises_when_u_and_w_shapes_mismatch():
    core = _make_core_for_execute_action(k=24, cols=6)
    action = _make_action(reconnect_gain=0.5)
    with pytest.raises(ValueError, match="U/W shape mismatch"):
        M3ConsciousnessCore._execute_action(core, action)


def test_hf_logging_compat_prefers_plural_api_when_available():
    tlog, t_calls = _make_fake_logging(has_many=False, has_one=True)
    hlog, h_calls = _make_fake_logging(has_many=True, has_one=True)
    _configure_external_library_logging(transformers_logging=tlog, hf_hub_logging=hlog)
    assert t_calls["set"] == 1
    assert t_calls["one"] == 1
    assert h_calls["set"] == 1
    assert h_calls["many"] == 1
    assert h_calls["one"] == 0


def test_hf_logging_compat_uses_singular_when_plural_missing():
    tlog, t_calls = _make_fake_logging(has_many=False, has_one=False)
    hlog, h_calls = _make_fake_logging(has_many=False, has_one=True)
    _configure_external_library_logging(transformers_logging=tlog, hf_hub_logging=hlog)
    assert t_calls["set"] == 1
    assert h_calls["set"] == 1
    assert h_calls["one"] == 1


def test_hf_logging_compat_skips_when_no_disable_api_exists():
    tlog, t_calls = _make_fake_logging(has_many=False, has_one=False)
    hlog, h_calls = _make_fake_logging(has_many=False, has_one=False)
    _configure_external_library_logging(transformers_logging=tlog, hf_hub_logging=hlog)
    assert t_calls["set"] == 1
    assert t_calls["many"] == 0 and t_calls["one"] == 0
    assert h_calls["set"] == 1
    assert h_calls["many"] == 0 and h_calls["one"] == 0


def test_hf_backend_auto_control_mode_initializes_health_window():
    backend = HFBackend()
    assert hasattr(backend, "_control_health_window")
    assert hasattr(backend, "_auto_mode_fail_streak")
    stats = backend._compute_recent_control_stats()
    assert stats["count"] == 0.0
    assert stats["fail_ratio"] == 0.0
    assert stats["consecutive_failures"] == 0.0


def test_hf_backend_bridge_safe_handles_legacy_instance_without_control_window(monkeypatch):
    legacy = HFBackend.__new__(HFBackend)
    monkeypatch.setenv("M3_CONTROL_SELECTION_MODE", "auto")
    assert HFBackend._bridge_enabled_safe(legacy) is False
    assert hasattr(legacy, "_control_health_window")
    assert hasattr(legacy, "_auto_mode_fail_streak")


def test_hf_runtime_failure_window_triggers_cooldown(monkeypatch):
    policy = TorchConversationalPolicy.__new__(TorchConversationalPolicy)
    policy._hf_failure_events = deque(maxlen=256)
    policy._hf_runtime_cooldown_until = 0.0
    monkeypatch.setenv("M3_HF_FAILURE_WINDOW_SEC", "30")
    monkeypatch.setenv("M3_HF_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("M3_HF_FAILURE_COOLDOWN_SEC", "20")
    first = TorchConversationalPolicy._register_hf_runtime_failure(
        policy,
        reason_code="hf_runtime_failure",
        phase="decode_forward",
        model_output_shape="1x32000",
        has_logits=True,
    )
    second = TorchConversationalPolicy._register_hf_runtime_failure(
        policy,
        reason_code="hf_runtime_failure",
        phase="decode_forward",
        model_output_shape="1x32000",
        has_logits=True,
    )
    assert int(first["failure_window_count"]) == 1
    assert int(second["failure_window_count"]) >= 2
    assert float(second["cooldown_until"]) > 0.0


def test_hf_failure_details_extract_from_runtime_exception():
    policy = TorchConversationalPolicy.__new__(TorchConversationalPolicy)
    err = HFRuntimeFailure(
        reason_code="hf_runtime_failure",
        phase="decode_hidden",
        model_output_shape="1x2048",
        has_logits=False,
    )
    details = TorchConversationalPolicy._extract_hf_failure_details(policy, err, hf=None)
    assert details["reason_code"] == "hf_runtime_failure"
    assert details["phase"] == "decode_hidden"
    assert details["model_output_shape"] == "1x2048"
    assert details["has_logits"] is False


def test_stabilization_template_rotates_three_variants():
    policy = TorchConversationalPolicy.__new__(TorchConversationalPolicy)
    policy._safe_fallback_turn = 0
    first = TorchConversationalPolicy._next_stabilization_template(policy, "en")
    second = TorchConversationalPolicy._next_stabilization_template(policy, "en")
    third = TorchConversationalPolicy._next_stabilization_template(policy, "en")
    fourth = TorchConversationalPolicy._next_stabilization_template(policy, "en")
    assert first != second
    assert second != third
    assert fourth == first


def test_panels_build_without_extra_dependencies():
    fb = FeatureBank(max_dim=64, embed_dim=16)
    core = types.SimpleNamespace(
        phi_calculator=types.SimpleNamespace(phi_history=[0.1, 0.2]),
        world_state={"stability": 0.9, "delta_hat": 0.2},
        energy_ctrl=types.SimpleNamespace(activation_level=0.7, cognitive_energy=0.9, energy_capacity=1.0),
        unified_subject=types.SimpleNamespace(unity_score=0.8),
        qualia=types.SimpleNamespace(arousal=0.3, valence=0.4, entropy=0.5, engagement=0.6, frustration=0.1),
        bus=None,
    )
    panels = fb.panels(core)
    assert isinstance(panels, list)
    assert len(panels) > 0
    assert all(isinstance(p, np.ndarray) for p in panels)
