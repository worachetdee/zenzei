"""Download Japanese Wikipedia articles for tokenizer training."""
from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)

print("Downloading Japanese Wikipedia (50k articles)...")
ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train[:50000]")

with open("data/raw/ja_wiki.txt", "w") as f:
    for row in ds:
        f.write(row["text"] + "\n")

print(f"Saved {len(ds)} articles to data/raw/ja_wiki.txt")
