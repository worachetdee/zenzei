"""Gradio Web UI for testing Zensei-7B Japanese generation.

Usage (Colab):
    !cd /content/zenzei && pip install gradio -q
    !cd /content/zenzei && python scripts/web_ui.py

Usage (local):
    python scripts/web_ui.py --model_path checkpoints/zensei-7b-ja-v2/final
"""
import argparse
import os
import torch


def load_model(model_path, base_model="Qwen/Qwen2.5-7B"):
    """Load base model + LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading base model: {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    print("Model loaded!")
    return model, tokenizer


def generate_text(
    prompt,
    model,
    tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def build_ui(model, tokenizer):
    """Build Gradio interface."""
    import gradio as gr

    def respond(prompt, max_tokens, temperature, top_p, rep_penalty):
        if not prompt.strip():
            return "プロンプトを入力してください。"
        return generate_text(
            prompt, model, tokenizer,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
        )

    examples = [
        ["日本の首都は"],
        ["東京タワーは"],
        ["日本語の特徴は"],
        ["桜の花が咲く季節に"],
        ["人工知能の未来は"],
        ["富士山は日本で"],
        ["日本料理の中で最も人気があるのは"],
        ["京都の歴史的な寺院として有名なのは"],
    ]

    with gr.Blocks(
        title="Zensei (禅精) — Japanese Language Model",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 禅精 Zensei — Japanese Language Model
            **Qwen2.5-7B + LoRA**, continued pretraining on Japanese Wikipedia.
            Type a Japanese prompt and see the model generate text.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt (プロンプト)",
                    placeholder="日本語のプロンプトを入力してください...",
                    lines=3,
                )
                output = gr.Textbox(
                    label="Generated Text (生成テキスト)",
                    lines=10,
                    interactive=False,
                )
                generate_btn = gr.Button("Generate (生成)", variant="primary")

            with gr.Column(scale=1):
                max_tokens = gr.Slider(
                    minimum=32, maximum=512, value=256, step=32,
                    label="Max Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                    label="Top-p",
                )
                rep_penalty = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.1, step=0.05,
                    label="Repetition Penalty",
                )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            label="Example Prompts (例文)",
        )

        generate_btn.click(
            respond,
            inputs=[prompt, max_tokens, temperature, top_p, rep_penalty],
            outputs=output,
        )
        prompt.submit(
            respond,
            inputs=[prompt, max_tokens, temperature, top_p, rep_penalty],
            outputs=output,
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Zensei Web UI")
    parser.add_argument(
        "--model_path",
        default="/content/drive/MyDrive/zensei_checkpoints/zensei-7b-ja-v2/final",
        help="Path to LoRA checkpoint",
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-7B",
        help="Base model name",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.base_model)
    demo = build_ui(model, tokenizer)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
