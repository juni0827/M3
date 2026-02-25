# BACKLOG

- Generated: 2026-02-25T10:32:53+00:00
- Commit: `2c81f50965075bd7e3b771e5c5063809e79dacfa`
- Branch: `main`
- Repro seed/config: deterministic (no randomness used in generator)

## Prioritized next actions
### P0 (critical)
- Keep `docs/STATUS.md` synced with latest commit and artifacts.
- Validate PASS/FAIL/UNKNOWN mapping from `run_summary.status`.
- Ensure CI auto-commit path only runs on `main` and schedule events.

### P1 (important)
- Expand architecture dependency extraction for richer module graphing.
- Add additional run artifact parsers when new outputs are introduced.
- Tighten verification checklist with project-specific acceptance gates.

### P2 (nice-to-have)
- Improve trend summaries across multiple historical runs.
- Add richer release/demo readiness indicators from artifact history.

## Open risks / tech debt
- `m3/m3_core.py:17029` # TODO: Handle structured JSONL specifically if needed

## Issue references found in docs
- `docs_pr_plan.md`: ## PR #1 — 안전성 패치(타입/크래시/회귀 방지)
- `docs_pr_plan.md`: - adapter 입력 정규화 유틸만 revert (PR #1 단독 revert 가능)
- `docs_pr_plan.md`: ## PR #2 — 의미 분리(entropy 이원화)
- `docs_pr_plan.md`: ## PR #3 — 마이크로 업데이트 동역학 정합(energy coupling)
- `docs_pr_plan.md`: ## PR #4 — Φ 경로 일관성 및 스케일 정합
- `docs_pr_plan.md`: ## PR #5 — 책임 분리(IITPhiCalculator vs CES)
- `docs_pr_plan.md`: 1. PR #1 (안전성)
- `docs_pr_plan.md`: 2. PR #2 (entropy 의미 분리)
- `docs_pr_plan.md`: 3. PR #3 (에너지 결합)
- `docs_pr_plan.md`: 4. PR #4 (phi 경로/스케일)
- `docs_pr_plan.md`: 5. PR #5 (구조 리팩터)
- `docs_pr_plan.md`: > 앞 PR이 뒤 PR의 테스트 기반을 제공하도록 구성. 특히 #1~#3을 먼저 머지하면 #4~#5 리스크가 크게 줄어듦.

## Investor demo readiness checklist
- [ ] Demo scenario script is reproducible from clean checkout.
- [ ] Core runtime path (`run_llm_adapter.py`) runs without manual patching.
- [ ] Latest metrics and run summary are attached in `artifacts/latest_run/`.
- [ ] PASS/FAIL/UNKNOWN status is understandable by non-engineers.
- [ ] Architecture and runbook docs are current for the demo date.
