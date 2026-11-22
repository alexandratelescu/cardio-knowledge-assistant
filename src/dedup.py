"""
Cleans and deduplicates the documents in data/docs.jsonl.
  0. cleaning: removes template lines repeated on many web pages (menus, banners,
     footers) and drops web pages with almost no text left
  1. exact duplicates: identical text (same hash)
  2. near duplicates: very similar text (MinHash + LSH, Jaccard >= 0.8)
Output: data/docs_dedup.jsonl (documents kept)
        data/duplicates.jsonl (documents removed + why)
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasketch import MinHash, MinHashLSH

DOCS = Path("data/docs.jsonl")
OUT = Path("data/docs_dedup.jsonl")
REMOVED = Path("data/duplicates.jsonl")

THRESHOLD = 0.8   # how similar two texts must be to count as duplicates
NUM_PERM = 128    # number of hash functions in each MinHash
K = 3             # shingle size: groups of 3 consecutive words
BOILERPLATE_SHARE = 0.1   # a line on 10%+ of the pages of one kind (and at least 4) counts as template
MIN_WORDS = 25            # web pages with fewer words after cleaning are dropped
NOTICE_MIN_WORDS = 8      # template lines this long are notices worth keeping once


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text, k=K):
    words = text.lower().split()
    if len(words) < k:                     # very short text: use single words
        return set(words)
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


def better(a, b):
    """Of two duplicates, keep the longer text; if equally long, the more recent one."""
    return max(a, b, key=lambda d: (len(d["text"]), d.get("date") or ""))


docs = [json.loads(line) for line in DOCS.read_text(encoding="utf-8").splitlines() if line.strip()]
docs_in = len(docs)
removed = []

# ---------- 0. remove page template + thin pages ----------
# each kind of page (blog, product, technology) has its own template, so look for it separately
web = [d for d in docs if d["id"].startswith("web:")]
for d in web:                                   # strip spaces, drop empty lines
    d["text"] = "\n".join(line.strip() for line in d["text"].splitlines() if line.strip())

boilerplate = set()
for source in {d["source"] for d in web}:
    group = [d for d in web if d["source"] == source]
    line_counts = Counter(line for d in group for line in set(d["text"].splitlines()))
    min_count = max(4, BOILERPLATE_SHARE * len(group))
    boilerplate |= {line for line, count in line_counts.items() if count >= min_count}

for d in web:
    d["text"] = "\n".join(line for line in d["text"].splitlines() if line not in boilerplate).strip()
    d["hash"] = sha(d["text"])

# long template lines are usually notices (e.g. medical disclaimers): keep them once, in one document
notices = sorted(line for line in boilerplate if len(line.split()) >= NOTICE_MIN_WORDS)
if notices:
    text = "\n".join(notices)
    docs.append({
        "id": "web:site-notices",
        "source": "site",
        "url": "https://kardia.com/",
        "title": "Kardia site-wide notices",
        "date": None,
        "section": None,
        "text": text,
        "hash": sha(text),
    })

kept_docs = []
for d in docs:
    # help center answers are often short on purpose, so only web pages are checked
    if d["id"].startswith("web:") and d["source"] != "site" and len(d["text"].split()) < MIN_WORDS:
        removed.append({**d, "duplicate_of": None, "reason": "thin"})
    else:
        kept_docs.append(d)
docs = kept_docs

# ---------- 1. exact duplicates ----------
by_hash = {}
for d in docs:
    h = d["hash"]
    if h not in by_hash:
        by_hash[h] = d
        continue
    keep = better(by_hash[h], d)
    drop = d if keep is by_hash[h] else by_hash[h]
    by_hash[h] = keep
    removed.append({**drop, "duplicate_of": keep["id"], "reason": "exact"})

unique = list(by_hash.values())
by_id = {d["id"]: d for d in unique}

# ---------- 2. near duplicates ----------
hashes = {d["id"]: minhash(d["text"]) for d in unique}

lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
for doc_id, m in hashes.items():
    lsh.insert(doc_id, m)

# union-find: puts documents that are similar (directly or through others) in one group
parent = {doc_id: doc_id for doc_id in by_id}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


pairs = []
for doc_id, m in hashes.items():
    for other in lsh.query(m):
        if other <= doc_id:                # skip itself and pairs already seen
            continue
        sim = m.jaccard(hashes[other])     # LSH finds candidates; this confirms them
        if sim >= THRESHOLD:
            pairs.append((doc_id, other, round(sim, 2)))
            parent[find(other)] = find(doc_id)

groups = {}
for doc_id in by_id:
    groups.setdefault(find(doc_id), []).append(by_id[doc_id])

kept = []
for members in groups.values():
    keep = members[0]
    for d in members[1:]:
        keep = better(keep, d)
    kept.append(keep)
    for d in members:
        if d is not keep:
            removed.append({**d, "duplicate_of": keep["id"], "reason": "near"})

# ---------- 3. save + report ----------
with OUT.open("w", encoding="utf-8") as f:
    for d in kept:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

with REMOVED.open("w", encoding="utf-8") as f:
    for d in removed:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

reasons = Counter(r["reason"] for r in removed)
print(f"documents in:          {docs_in}")
print(f"template lines found:  {len(boilerplate)}")
print(f"thin pages dropped:    {reasons['thin']}")
print(f"exact duplicates:      {reasons['exact']}")
print(f"near duplicates:       {reasons['near']}")
print(f"documents kept:        {len(kept)}")
print(f"notices kept once:     {len(notices)}")
for line in notices:
    print(f"  {line[:90]!r}")

print("\nthin pages dropped:")
for r in removed:
    if r["reason"] == "thin":
        print(f"  {r['url']}")

print("\ntemplate lines removed (check these by hand):")
for line in sorted(boilerplate)[:20]:
    print(f"  {line[:70]!r}")

print("\nmost similar pairs (check these by hand):")
for a, b, sim in sorted(pairs, key=lambda p: -p[2])[:10]:
    print(f"  {sim:.2f}  {by_id[a]['title'][:45]!r}  <->  {by_id[b]['title'][:45]!r}")