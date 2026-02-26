# ARCHITECTURE_STATE

- Generated: 2026-02-26T01:02:17+00:00
- Commit: `6b6cf0424f5a1e68d1327bd2f945ce722fd0e33d`
- Branch: `main`
- Repro seed/config: deterministic (no randomness used in generator)

## Module graph / dependency overview
### Key modules
- `m3/m3_core.py`
- `llm_adapter/llm_core.py`
- `llm_adapter/m3_control_bridge.py`
- `llm_adapter/m3_plastic_policy.py`
- `m3/torch_policy.py`
- `m3/reward.py`
- `llm_adapter/memory.py`
- `m3/features.py`
- `llm_adapter/tokenization.py`
- `llm_adapter/remote.py`
- `llm_adapter/layers.py`
- `m3/visualization.py`
- `run_llm_adapter.py`

### Internal dependency edges
- `llm_adapter/llm_core.py` -> `llm_adapter.config`
- `llm_adapter/llm_core.py` -> `llm_adapter.layers`
- `llm_adapter/llm_core.py` -> `llm_adapter.m3_control_bridge`
- `llm_adapter/llm_core.py` -> `llm_adapter.memory`
- `llm_adapter/llm_core.py` -> `llm_adapter.remote`
- `llm_adapter/llm_core.py` -> `llm_adapter.tokenization`
- `llm_adapter/llm_core.py` -> `llm_adapter`
- `llm_adapter/llm_core.py` -> `m3.device`
- `llm_adapter/m3_plastic_policy.py` -> `llm_adapter.config`
- `llm_adapter/m3_plastic_policy.py` -> `llm_adapter.layers`
- `llm_adapter/m3_plastic_policy.py` -> `llm_adapter.tokenization`
- `llm_adapter/m3_plastic_policy.py` -> `m3.device`
- `llm_adapter/memory.py` -> `llm_adapter.config`
- `llm_adapter/tokenization.py` -> `llm_adapter.config`
- `m3/features.py` -> `m3.visualization`
- `m3/m3_core.py` -> `llm_adapter.config`
- `m3/m3_core.py` -> `llm_adapter`
- `m3/m3_core.py` -> `m3.config`
- `m3/m3_core.py` -> `m3.features`
- `m3/m3_core.py` -> `m3.reward`
- `m3/m3_core.py` -> `m3.torch_policy`
- `m3/m3_core.py` -> `m3.visualization`
- `m3/torch_policy.py` -> `m3.device`
- `run_llm_adapter.py` -> `llm_adapter.llm_core`
- `run_llm_adapter.py` -> `llm_adapter`
- `run_llm_adapter.py` -> `m3.m3_core`

## Entry points
- `llm_adapter/llm_core.py`
- `m3/m3_core.py`
- `run_llm_adapter.py`

## Env toggles used in run paths
- UNKNOWN

## Critical path (high-level)
1. Start from `run_llm_adapter.py` or root scripts.
2. Initialize M3 core (`m3/m3_core.py`) and adapter runtime (`llm_adapter/llm_core.py`).
3. Route state/reward/features through policy modules and control bridge.
4. Emit outputs and logs, optionally persisted as run artifacts.
