#!/usr/bin/env python3
# prep_e2_subset.py - W1 first-trial subset prep (conversation-level 10%)
# Usage: python3 prep_e2_subset.py [corpus_dir (default ~/raven_w1_corpus_r1)]
# Samples ~10% of TRAIN conversations (not records - preserves conv-disjoint property)
# Output: train_subset10.jsonl + subset_manifest.json
import json, hashlib, os, sys

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/raven_w1_corpus_r1')
IN = os.path.join(D, 'train.jsonl')
OUT = os.path.join(D, 'train_subset10.jsonl')
SEED = 20261030
random_state = SEED

# deterministic seeded shuffle of conversation ids
import random
random.seed(SEED)
convs = {}
with open(IN) as f:
    for line in f:
        d = json.loads(line)
        convs.setdefault(d['conversation_id'], []).append(line)

ids = sorted(convs.keys())
random.shuffle(ids)
n_take = max(1, len(ids)//10)
take = set(ids[:n_take])

n = 0
with open(OUT, 'w') as f:
    for cid in ids:
        if cid in take:
            for line in convs[cid]:
                f.write(line)
                n += 1

h = hashlib.sha256(open(OUT,'rb').read()).hexdigest()
manifest = {
    'subset_id': 'RVN_RAVEN_W1_TRIAL_SUBSET_10PCT', 'seed': SEED,
    'source': IN, 'output': OUT,
    'conversations_total': len(ids), 'conversations_taken': n_take,
    'records': n, 'sha256': h,
}
with open(os.path.join(D,'subset_manifest.json'),'w') as f:
    json.dump(manifest, f, indent=2)
print(f'conversations: {n_take}/{len(ids)}')
print(f'records: {n}')
print(f'sha256: {h}')
