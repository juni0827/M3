# RUNBOOK

- Generated: 2026-07-02T02:01:21+00:00
- Commit: `f66e8b3942a9029cca3240106deb1fce0c77a709`
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
