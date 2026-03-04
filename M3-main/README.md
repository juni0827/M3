# M3

M3 is an experimental runtime that couples a consciousness-style core with an
LLM adapter.

This repository is operated with a stability-first policy:

- Keep runtime loops alive.
- Reduce fallback collapse.
- Preserve JSONL log integrity.
- Track improvements with replayable KPIs.

## Overview

The system has two main domains:

- `m3/`: core runtime, policy, feature flow, reward loop, and autonomy loops.
- `llm_adapter/`: backend integration, control bridge, and generation logic.

Runtime artifacts and regression tests are centralized in `docs_tests_data/`.

## Repository Layout

- `m3/m3_core.py`: main runtime core and autonomous loops.
- `m3/meaning_pipeline.py`: meaning interpretation and response planning.
- `m3/io/jsonl_writer.py`: shared JSONL writer with per-path lock.
- `llm_adapter/llm_core.py`: HF backend integration and generation path.
- `llm_adapter/control_plane.py`: unified control decision snapshot.
- `run_llm_adapter.py`: interactive runner entry point.
- `docs_tests_data/tests/`: validation and regression tests.
- `docs_tests_data/STATUS_PACKET.md`: generated architecture/KPI snapshot.
- `docs_tests_data/VISUAL_MAP.svg`: generated visual dashboard.

## Quick Start

Run commands from the repository root (`M3-main/`).

1. Create and activate a virtual environment.
2. Install dependencies used by this repo.
3. Start the runner.

```bash
python run_llm_adapter.py
```

## Runtime Data Contract

All generated runtime files are expected under `docs_tests_data/`.

- `llm_adapter.log`: adapter runtime log stream.
- `bus.jsonl`: consciousness bus stream.
- `meaning_state.jsonl`: meaning-state records.
- `response_plan.jsonl`: response-plan records.
- `control_decision.jsonl`: control-plane decisions.
- `baseline_metrics.json`: baseline KPI snapshot.
- `replay_metrics.json`: replay KPI snapshot.
- `STATUS_PACKET.md`: generated status packet.

## Control Plane

Control-plane modes and key variables:

- `M3_CONTROL_PLANE_MODE` (default `enforce`): `shadow` or `enforce`.
- `M3_CONTROL_DECISION_LOG`:
  default `docs_tests_data/control_decision.jsonl`.
- `M3_MEANING_DYNAMIC_UNCERTAINTY` (default `1`): dynamic uncertainty.
- `M3_PLAN_EVIDENCE_COUNT_MODE` (default `source_based`).
- `M3_BUS_LOG_VALIDATE_STRICT` (default `1`): strict bus schema checks.

## Validation Commands

Run focused regression checks:

```bash
python -m pytest -q -c docs_tests_data/pytest.ini \
  docs_tests_data/tests/test_fail_fast_featurebank_hf_logging.py \
  docs_tests_data/tests/test_m3_plan_features.py
```

Validate bus JSONL integrity:

```bash
python docs_tests_data/tests/validate_bus_jsonl.py \
  docs_tests_data/bus.jsonl \
  --max-lines 50000 \
  --threshold 0.0001
```

Replay runtime KPIs:

```bash
python docs_tests_data/tests/replay_runtime_kpis.py --out-dir docs_tests_data
```

Build status packet and visual map:

```bash
python docs_tests_data/tests/build_status_packet.py \
  --repo-root . \
  --out-dir docs_tests_data \
  --artifacts-dir docs_tests_data/artifacts/latest_run
```

## KPI Gates

Current operational gates:

- `bus_jsonl_parse_error_rate <= 0.01%`
- `HF generation failed` same-signature events `0 / 30 min`
- `unknown_abstain_ratio <= 40%`
- `response_repeat_ratio <= 35%`

## Troubleshooting

`ModuleNotFoundError: No module named 'llm_adapter'`

- Run commands from the repo root that contains `llm_adapter/` and `m3/`.
- Avoid launching tests from directories outside the repo root.

`STATUS_PACKET` fields show `UNKNOWN`

- If `.git` metadata is unavailable, provide
  `docs_tests_data/source_revision.json`.
- Re-run `build_status_packet.py` after updating runtime logs.

## Notes

- This README is intentionally concise and operational.
- Deep diagnostics come from generated artifacts in `docs_tests_data/`.
