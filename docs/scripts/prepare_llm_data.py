#!/usr/bin/env python3
"""Prepare and analyze LLM training JSONL.
Produces statistics, deduped dataset, and train/val/test splits.

Usage: python scripts\prepare_llm_data.py
"""
import json
import random
import os
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / 'out_m3' / 'llm_training_data.jsonl'
OUT_DIR = ROOT / 'out_m3' / 'prepared'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
PER_TEMPLATE_CAP = int(os.environ.get('LLM_PREP_CAP', '2000'))  # per-prompt cap, None for unlimited
SEED = int(os.environ.get('LLM_PREP_SEED', '42'))
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Helper: load

def load_records(p):
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # try naive repair: remove trailing commas or stray characters
                try:
                    repaired = line.rstrip(',')
                    yield json.loads(repaired)
                except Exception:
                    continue


# 1) Raw scan
raw_total = 0
prompts = Counter()
labels = Counter()
anomalies = 0
raw_pairs = []
for rec in load_records(IN_PATH):
    raw_total += 1
    u = rec.get('user') if isinstance(rec, dict) else None
    a = rec.get('assistant') if isinstance(rec, dict) else None
    # Count anomalies: non-str or empty
    if not isinstance(u, str) or not isinstance(a, str) or not u.strip() or not a.strip():
        anomalies += 1
        continue
    u_s = u.strip()
    a_s = a.strip()
    prompts[u_s] += 1
    labels[a_s] += 1
    raw_pairs.append((u_s, a_s))

# Print raw stats
print('RAW_TOTAL:', raw_total)
print('VALID_PAIRS_AFTER_ANOMALY_FILTER:', len(raw_pairs))
print('ANOMALIES_REMOVED:', anomalies)
print('\nTop 20 labels:')
for k, v in labels.most_common(20):
    print(f'  {k!r}: {v}')
print('\nTop 20 duplicate prompts (prompt -> count):')
for k, v in prompts.most_common(20):
    print(f'  {k[:80]!r} -> {v}')

# 2) Deduplicate (prompt, assistant) pairs but keep count of duplicates per prompt
seen = set()
deduped = []
dup_counts = Counter()
for u, a in raw_pairs:
    key = (u, a)
    if key in seen:
        dup_counts[u] += 1
        continue
    seen.add(key)
    deduped.append({'user': u, 'assistant': a})

print('\nUNIQUE_PAIR_COUNT:', len(deduped))
print('PROMPT_DUPLICATE_COUNT:', sum(dup_counts.values()))
print('\nTop duplicates by prompt (pairs removed):')
for p, c in dup_counts.most_common(20):
    print(f'  {p[:80]!r} -> removed {c} duplicates')

# write deduped file
DEDUP_PATH = OUT_DIR / 'deduped.jsonl'
with open(DEDUP_PATH, 'w', encoding='utf-8') as f:
    for r in deduped:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# 3) Apply per-template cap
if PER_TEMPLATE_CAP is not None and PER_TEMPLATE_CAP > 0:
    by_prompt = defaultdict(list)
    for r in deduped:
        by_prompt[r['user']].append(r)
    capped = []
    for p, lst in by_prompt.items():
        if len(lst) > PER_TEMPLATE_CAP:
            capped.extend(lst[:PER_TEMPLATE_CAP])
        else:
            capped.extend(lst)
    print(f'APPLIED per-template cap = {PER_TEMPLATE_CAP}. Before: {len(deduped)}, After cap: {len(capped)}')
else:
    capped = deduped
    print('No per-template cap applied')

# 4) Shuffle and split
random.seed(SEED)
random.shuffle(capped)
N = len(capped)
train_n = int(N * TRAIN_RATIO)
val_n = int(N * VAL_RATIO)
train = capped[:train_n]
val = capped[train_n:train_n + val_n]
test = capped[train_n + val_n:]

for name, arr in (('train', train), ('val', val), ('test', test)):
    p = OUT_DIR / f'{name}.jsonl'
    with open(p, 'w', encoding='utf-8') as f:
        for r in arr:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

print('\nSPLIT COUNTS:')
print('  train:', len(train))
print('  val:  ', len(val))
print('  test: ', len(test))
print('\nOutput files written to', OUT_DIR)
print('DEDUPED file:', DEDUP_PATH)
print('Done.')
