# RUNBOOK

- Generated: 2026-06-10T02:05:17+00:00
- Commit: `aeb7359deeea9ded25fc028152fe6878089eada5`
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
