#!/usr/bin/env python3
# RVN_RAVEN_W1_CORPUS_R1 reproducible build script (R2: explicit paths, clear errors)
# Usage: python3 raven_w1_build_script.py /full/path/to/chatgpt.sqlite /full/path/to/output_dir
#   arg1: chatgpt.sqlite path (853MB, from Drive verified export /derived/)
#   arg2: output dir (default: ./raven_w1_corpus_r1)
# Intermediate files (general_pairs*.jsonl) are written INSIDE the output dir.
import sqlite3, json, hashlib, os, re, sys

def die(msg):
    print(f'ERROR: {msg}', file=sys.stderr)
    sys.exit(2)

if len(sys.argv) < 2:
    die('usage: python3 raven_w1_build_script.py /full/path/to/chatgpt.sqlite [output_dir]\n'
        '  arg1 must be the full path to chatgpt.sqlite (errno 2 = file not found at given path)')

DB = os.path.abspath(os.path.expanduser(sys.argv[1]))
OUT = os.path.abspath(os.path.expanduser(sys.argv[2])) if len(sys.argv) > 2 else os.path.abspath('raven_w1_corpus_r1')

if not os.path.isfile(DB):
    die(f'sqlite not found at: {DB}\n'
        '  Find it on your machine (verified export /derived/ folder) and pass the full path.')
try:
    db = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cur = db.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if 'conversations' not in tables or 'nodes' not in tables:
        die(f'wrong sqlite (missing conversations/nodes tables): {DB}')
except sqlite3.DatabaseError as e:
    die(f'cannot open sqlite: {e}')

os.makedirs(OUT, exist_ok=True)
GP = os.path.join(OUT, 'general_pairs.jsonl')
GP_ADMIT = os.path.join(OUT, 'general_pairs_admitted.jsonl')

def sha256(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()

print(f'[1/4] extracting pairs from {DB}')
out = open(GP, 'w')
n = 0
for (cid,) in cur.execute('SELECT conversation_id FROM conversations').fetchall():
    nodes = cur.execute('SELECT node_id, parent_id, role, content_type, text, is_current_path FROM nodes WHERE conversation_id=? ORDER BY create_time', (cid,)).fetchall()
    if not nodes:
        continue
    by_id = {x[0]: x for x in nodes}
    path_nodes = [x for x in nodes if x[5] == 1]
    tail = path_nodes[-1][0] if path_nodes else nodes[-1][0]
    chain, nid, seen = [], tail, set()
    while nid and nid in by_id and nid not in seen:
        seen.add(nid)
        chain.append(by_id[nid])
        nid = by_id[nid][1]
    chain.reverse()
    msgs = [(x[2], x[4]) for x in chain if x[2] and x[4]]
    i = 0
    pi = 0
    while i < len(msgs):
        if msgs[i][0] == 'user':
            j = i + 1
            at = []
            while j < len(msgs) and msgs[j][0] == 'assistant':
                at.append(msgs[j][1])
                j += 1
            if at:
                out.write(json.dumps({'conversation_id': cid, 'pair_index': pi, 'input': msgs[i][1], 'output': '\n\n'.join(at)}, ensure_ascii=False) + '\n')
                pi += 1
            i = j
        else:
            i += 1
    n += pi
out.close()
print(f'  pairs: {n}')

print('[2/4] admission filter (T3 exclusion per LEDGER_004)')
ADULT = re.compile(r'(?i)(satisfy a woman in bed|explicit content|sexual act|porn|nude|erotic)')
COPYRIGHT = re.compile(r'(?i)(lyrics of|verbatim excerpts|copyrighted songs)')
n_adm = 0
with open(GP) as f, open(GP_ADMIT, 'w') as o:
    for line in f:
        d = json.loads(line)
        t = d['input'] + '\n' + d['output']
        if ADULT.search(t) or COPYRIGHT.search(t):
            continue
        o.write(line)
        n_adm += 1
print(f'  admitted: {n_adm}')

print('[3/4] dedup + conversation-disjoint split')
SALT = 'RVN_RAVEN_W1_CORPUS_R1_CONV'
seen, uniq = set(), []
with open(GP_ADMIT) as f:
    for line in f:
        d = json.loads(line)
        h = sha256(d['input'] + '\x00' + d['output'])
        if h in seen:
            continue
        seen.add(h)
        uniq.append({'conversation_id': d['conversation_id'], 'pair_index': d['pair_index'], 'content_sha256': h,
                     'input_bytes': len(d['input'].encode()), 'output_bytes': len(d['output'].encode()),
                     'input': d['input'], 'output': d['output']})
ch = {}
train, val = [], []
for r in uniq:
    cid = r['conversation_id']
    if cid not in ch:
        ch[cid] = int(sha256(SALT + cid)[0], 16)
    (train if ch[cid] < 9 else val).append(r)

print('[4/4] writing corpus')
with open(os.path.join(OUT, 'train.jsonl'), 'w') as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with open(os.path.join(OUT, 'validation.jsonl'), 'w') as f:
    for r in val:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

tc = set(r['conversation_id'] for r in train)
vc = set(r['conversation_id'] for r in val)
manifest = {
    'corpus_id': 'RVN_RAVEN_W1_CORPUS_R1', 'created': '2026-10-30',
    'source': 'chatgpt-export-2026-09-20-verified (chatgpt.sqlite)',
    'admission': 'E1-B02 full approval per LEDGER_004; T3 22 excluded',
    'split': {'method': 'CONVERSATION-DISJOINT seeded sha256 conversation_id', 'salt': SALT,
              'train_records': len(train), 'validation_records': len(val),
              'train_conversations': len(tc), 'validation_conversations': len(vc), 'conversation_overlap': len(tc & vc)},
    'totals': {'unique_records': len(uniq)},
}
with open(os.path.join(OUT, 'admission_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
sums = [hashlib.sha256(open(os.path.join(OUT, fn), 'rb').read()).hexdigest() + '  ' + fn
        for fn in ['train.jsonl', 'validation.jsonl', 'admission_manifest.json']]
with open(os.path.join(OUT, 'SHA256SUMS'), 'w') as f:
    f.write('\n'.join(sums) + '\n')
print(f'train: {len(train)} | val: {len(val)} | overlap: {len(tc & vc)}')
for s in sums:
    print(s)
print(f'DONE -> {OUT}')
