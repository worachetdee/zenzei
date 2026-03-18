"""Test the trained Zensei-7B model by generating Japanese text."""
import os
import torch

os.chdir("/content/zenzei")

CHECKPOINT_DIR = "checkpoints/zensei-7b-ja/final"
BASE_MODEL = "Qwen/Qwen2.5-7B"

print("=" * 60)
print("Zensei-7B Japanese Generation Test")
print("=" * 60)

# Load base model + LoRA
print("\nLoading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
model.eval()
print("Model loaded!")

# Test prompts
prompts = [
    "日本の首都は",
    "東京タワーは",
    "日本語の特徴は",
    "桜の花が",
    "人工知能の未来は",
]

print()
for prompt in prompts:
    print(f"Prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {text}")
    print("-" * 60)

# Compare with base model (no LoRA)
print("\n" + "=" * 60)
print("Comparison: Base Qwen2.5-7B (no Japanese training)")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_model.eval()

prompt = "日本の首都は"
print(f"\nPrompt: {prompt}")

inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)
with torch.no_grad():
    outputs = base_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
    )
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Base model: {text}")
print("-" * 60)
