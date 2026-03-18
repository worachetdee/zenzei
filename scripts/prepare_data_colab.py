"""Tokenize filtered data into binary format using SentencePiece tokenizer."""
import json
import os
import time

import numpy as np
import sentencepiece as spm

os.chdir("/content/zenzei")

# Config
INPUT_PATH = "data/filtered/ja_wiki.jsonl"
OUTPUT_PREFIX = "data/processed/ja_wiki"
SP_MODEL = "data/tokenizer/zensei_ja_sp.model"
MAX_SEQ_LEN = 4096

os.makedirs("data/processed", exist_ok=True)

# Load tokenizer
print("Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL)
print(f"  Vocab size: {sp.get_piece_size()}")

# Read and tokenize documents
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

        ids = sp.encode(text, out_type=int)
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        if ids:
            all_token_ids.append(ids)
            doc_count += 1

        if doc_count % 100000 == 0 and doc_count > 0:
            elapsed = time.time() - t0
            print(f"  Tokenized {doc_count:,} docs ({doc_count/elapsed:,.0f} docs/s)")

tokenize_time = time.time() - t0
print(f"  Done: {doc_count:,} docs in {tokenize_time:.1f}s")

# Compute total tokens
total_tokens = sum(len(ids) for ids in all_token_ids)
print(f"  Total tokens: {total_tokens:,}")

# Pack into binary
print("Packing into binary...")
t1 = time.time()

bin_path = OUTPUT_PREFIX + ".bin"
idx_path = OUTPUT_PREFIX + ".idx"

# Document boundaries
boundaries = [0]
for ids in all_token_ids:
    boundaries.append(boundaries[-1] + len(ids))

# Write binary (uint16 memmap)
mmap = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
offset = 0
for ids in all_token_ids:
    length = len(ids)
    mmap[offset:offset + length] = np.array(ids, dtype=np.uint16)
    offset += length
mmap.flush()
del mmap

# Write index
index = {
    "num_documents": doc_count,
    "total_tokens": total_tokens,
    "dtype": "uint16",
    "document_boundaries": boundaries,
}
with open(idx_path, "w") as f:
    json.dump(index, f)

pack_time = time.time() - t1
size_mb = os.path.getsize(bin_path) / (1024 * 1024)

print()
print("=" * 50)
print("Data preparation complete!")
print("=" * 50)
print(f"  Documents:  {doc_count:,}")
print(f"  Tokens:     {total_tokens:,}")
print(f"  Binary:     {bin_path} ({size_mb:.1f} MB)")
print(f"  Index:      {idx_path}")
print(f"  Time:       {tokenize_time + pack_time:.1f}s")
print(f"  Avg tokens/doc: {total_tokens/doc_count:.0f}")
