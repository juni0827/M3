# RUNBOOK

- Generated: 2026-02-25T05:24:34+00:00
- Commit: `315dfa1efc04c6f90e4f844611fef56ccbcd7e4e`
- Branch: `work`
- Repro seed/config: deterministic (no randomness used in generator)

## Repro commands (local + CI)
```bash
python -m pip install <dependencies>  # requirements.txt not found
python tools/build_status_packet.py --repo-root . --out-dir docs --artifacts-dir artifacts/latest_run
python run_llm_adapter.py
```

## Seed/config conventions (determinism)
- Status packet generator uses no randomness.
- Repro seed: `N/A`
- Repro config: CLI flags to `tools/build_status_packet.py` are the source of truth.

## Env var setup example
```bash
export M3_ENV=dev
export PYTHONUNBUFFERED=1
python run_llm_adapter.py
```

## Verification checklist
- [ ] `docs/STATUS.md` generated with current commit SHA.
- [ ] `docs/ARCHITECTURE_STATE.md` includes entry points and env toggles.
- [ ] `docs/RUNBOOK.md` contains deterministic repro steps.
- [ ] `docs/BACKLOG.md` contains P0/P1/P2 and investor demo checklist.

## Failure conditions
- Missing artifact directory is allowed, but must render `UNKNOWN` metrics.
- Git metadata unavailable => SHA/branch become `UNKNOWN`.
- Script exits non-zero on file write errors.

## Debug routines (common issues)
1. Validate git context: `git rev-parse HEAD && git rev-parse --abbrev-ref HEAD`.
2. Confirm artifact paths: `ls -la artifacts/latest_run`.
3. Re-run with bounded log tail: `python tools/build_status_packet.py --max-log-lines 50`.
4. Inspect markdown diff: `git diff -- docs/`.
