"""Run the data pipeline on downloaded Wikipedia data."""
import os
import sys
import time

os.chdir("/content/zenzei")

# Ensure directories exist
for d in ["data/cleaned", "data/deduped", "data/filtered"]:
    os.makedirs(d, exist_ok=True)

# Step 1: Convert raw text to JSONL format
print("=" * 50)
print("Step 1: Converting raw text to JSONL")
print("=" * 50)
input_txt = "data/raw/ja_wiki.txt"
input_jsonl = "data/raw/ja_wiki.jsonl"

if not os.path.exists(input_jsonl):
    import json
    count = 0
    with open(input_txt, "r") as fin, open(input_jsonl, "w") as fout:
        doc = []
        for line in fin:
            line = line.strip()
            if line:
                doc.append(line)
            else:
                if doc:
                    text = "\n".join(doc)
                    if len(text) > 50:
                        fout.write(json.dumps({"id": f"wiki_{count}", "text": text}, ensure_ascii=False) + "\n")
                        count += 1
                    doc = []
        if doc:
            text = "\n".join(doc)
            if len(text) > 50:
                fout.write(json.dumps({"id": f"wiki_{count}", "text": text}, ensure_ascii=False) + "\n")
                count += 1
    print(f"Created {count} documents in {input_jsonl}")
else:
    print(f"JSONL already exists: {input_jsonl}")

# Step 2: Clean
print()
print("=" * 50)
print("Step 2: Cleaning")
print("=" * 50)
t0 = time.time()
from zensei.data.clean import clean_file
stats = clean_file(input_jsonl, "data/cleaned/ja_wiki.jsonl", min_cjk_ratio=0.1, num_workers=4)
print(f"Done in {time.time()-t0:.1f}s — kept={stats['kept']}, filtered={stats['filtered']}")

# Step 3: Dedup
print()
print("=" * 50)
print("Step 3: Deduplication")
print("=" * 50)
t0 = time.time()
from zensei.data.dedup import dedup_file
stats = dedup_file("data/cleaned/ja_wiki.jsonl", "data/deduped/ja_wiki.jsonl", threshold=0.8, num_perm=128)
print(f"Done in {time.time()-t0:.1f}s — remaining={stats['remaining']}, removed={stats['duplicates_removed']}")

# Step 4: Filter
print()
print("=" * 50)
print("Step 4: Quality filtering")
print("=" * 50)
t0 = time.time()
from zensei.data.filter import filter_file
stats = filter_file("data/deduped/ja_wiki.jsonl", "data/filtered/ja_wiki.jsonl", min_len=100, max_len=500000, min_cjk_ratio=0.1)
print(f"Done in {time.time()-t0:.1f}s — kept={stats['kept']}, filtered={stats['filtered']}")

print()
print("=" * 50)
print("Data pipeline complete!")
print("=" * 50)

# Show file sizes
for path in ["data/raw/ja_wiki.jsonl", "data/cleaned/ja_wiki.jsonl", "data/deduped/ja_wiki.jsonl", "data/filtered/ja_wiki.jsonl"]:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        lines = sum(1 for _ in open(path))
        print(f"  {path}: {size_mb:.1f} MB, {lines} docs")
