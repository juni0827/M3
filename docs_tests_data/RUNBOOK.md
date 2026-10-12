# RUNBOOK

- Generated: 2026-10-12T02:58:09+00:00
- Commit: `d5399f84daff5943d500912d6d1a0b5cd973d484`
- Branch: `main`
- Repro seed/config: deterministic (no randomness used in generator)

## Repro commands (local + CI)

```bash
python -m pip install <dependencies>  # requirements.txt not found
python tools/build_status_packet.py \
  --repo-root . \
  --out-dir docs_tests_data \
  --artifacts-dir docs_tests_data/artifacts/latest_run
python run_llm_adapter.py
```

## Verification checklist

- [ ] `docs_tests_data/STATUS.md` generated.
- [ ] `docs_tests_data/VISUAL_MAP.md` generated.
- [ ] `docs_tests_data/VISUAL_MAP.svg` generated.
- [ ] `docs_tests_data/CLASS_CATALOG.md` generated.
- [ ] `docs_tests_data/CHANGE_REPORT.md` generated.
