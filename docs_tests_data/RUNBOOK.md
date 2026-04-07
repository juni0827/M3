# RUNBOOK

- Generated: 2026-04-07T01:14:09+00:00
- Commit: `d494c4a060f1601a6bce2adfdd0098d4e2551183`
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
