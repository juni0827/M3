# RUNBOOK

- Generated: 2026-05-31T02:07:03+00:00
- Commit: `8c19753a00080dd75932a85b204dc169b674ab88`
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
