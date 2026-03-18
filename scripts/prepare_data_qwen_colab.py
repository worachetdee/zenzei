"""Re-tokenize filtered data using Qwen2.5-7B's own tokenizer."""
import json
import os
import time

import numpy as np

os.chdir("/content/zenzei")

INPUT_PATH = "data/filtered/ja_wiki.jsonl"
OUTPUT_PREFIX = "data/processed/ja_wiki_qwen"
MODEL_NAME = "Qwen/Qwen2.5-7B"
MAX_SEQ_LEN = 512

os.makedirs("data/processed", exist_ok=True)

# Load Qwen tokenizer
print("Loading Qwen2.5-7B tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print(f"  Vocab size: {tokenizer.vocab_size}")

# Read and tokenize
print(f"Tokenizing {INPUT_PATH}...")
t0 = time.time()

all_token_ids = []
doc_count = 0

with open(INPUT_PATH, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            text = record.get("text", "")
            if not text:
                continue
        except json.JSONDecodeError:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        if ids:
            all_token_ids.append(ids)
            doc_count += 1

        if doc_count % 100000 == 0 and doc_count > 0:
            elapsed = time.time() - t0
            print(f"  Tokenized {doc_count:,} docs ({doc_count/elapsed:,.0f} docs/s)")

tokenize_time = time.time() - t0
total_tokens = sum(len(ids) for ids in all_token_ids)
print(f"  Done: {doc_count:,} docs, {total_tokens:,} tokens in {tokenize_time:.1f}s")

# Pack into binary (uint32 since Qwen vocab > 65535)
print("Packing into binary...")
t1 = time.time()

bin_path = OUTPUT_PREFIX + ".bin"
idx_path = OUTPUT_PREFIX + ".idx"

boundaries = [0]
for ids in all_token_ids:
    boundaries.append(boundaries[-1] + len(ids))

mmap = np.memmap(bin_path, dtype=np.uint32, mode="w+", shape=(total_tokens,))
offset = 0
for ids in all_token_ids:
    length = len(ids)
    mmap[offset:offset + length] = np.array(ids, dtype=np.uint32)
    offset += length
mmap.flush()
del mmap

index = {
    "num_documents": doc_count,
    "total_tokens": total_tokens,
    "dtype": "uint32",
    "document_boundaries": boundaries,
}
with open(idx_path, "w") as f:
    json.dump(index, f)

size_mb = os.path.getsize(bin_path) / (1024 * 1024)

print()
print("=" * 50)
print("Data preparation complete!")
print("=" * 50)
print(f"  Documents:  {doc_count:,}")
print(f"  Tokens:     {total_tokens:,}")
print(f"  Binary:     {bin_path} ({size_mb:.1f} MB)")
print(f"  Tokenizer:  {MODEL_NAME}")
print(f"  Time:       {tokenize_time + (time.time()-t1):.1f}s")
