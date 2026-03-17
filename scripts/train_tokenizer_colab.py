"""Train the Japanese tokenizer on downloaded Wikipedia data."""
import subprocess
import sys

print("Training Japanese SentencePiece tokenizer on Wikipedia data...")
print("This takes ~10-30 minutes depending on the GPU/CPU.")

subprocess.run([
    sys.executable, "-m", "zensei.tokenizer.train_tokenizer",
    "--input_dir", "data/raw",
    "--model_prefix", "data/tokenizer/zensei_ja_sp",
    "--vocab_size", "32000",
], check=True)

# Quick sanity check
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/zensei_ja_sp.model")

text = "東京は日本の首都であり、世界最大の都市圏の一つです。"
tokens = sp.encode(text, out_type=str)
fertility = len(tokens) / len(text)

print()
print("=" * 50)
print(f"Vocab size: {sp.get_piece_size()}")
print(f"Sample: {text}")
print(f"Tokens: {tokens}")
print(f"Fertility: {fertility:.3f} tokens/char")
print("=" * 50)
print("Tokenizer training complete!")
