"""Evaluate tokenizer quality: fertility, UNK rate, and roundtrip accuracy.

Usage:
    python -m zensei.tokenizer.test_tokenizer \
        --base_tokenizer deepseek-ai/DeepSeek-V3 \
        --merged_tokenizer data/tokenizer/zensei_merged

    python -m zensei.tokenizer.test_tokenizer --config configs/tokenizer/tokenizer_train.yaml
"""

import logging
from typing import Optional

import fire
import yaml
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------- #
# Sample Japanese test texts (Wikipedia excerpts, news, literary)
# ------------------------------------------------------------------------------- #
SAMPLE_TEXTS: dict[str, str] = {
    "wikipedia": (
        "日本語（にほんご、にっぽんご）は、日本国内や、かつての日本領だった国、"
        "そして国外移民や移住者を含む日本人同士の間で使用されている言語。"
        "日本は法令によって公用語を規定していないが、法令その他の公用文は全て"
        "日本語で記述され、各種法令において日本語についての規定が存在する。"
    ),
    "news": (
        "政府は本日、新たな経済対策を発表した。物価高騰に苦しむ家計を支援する"
        "ため、総額5兆円規模の補正予算を編成する方針だ。岸田首相は記者会見で"
        "「国民生活を守るために全力を尽くす」と述べた。野党側は対策の規模が不十分"
        "だと批判している。"
    ),
    "literary": (
        "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
        "吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という"
        "人間中で一番獰悪な種族であったそうだ。"
    ),
    "technical": (
        "大規模言語モデルは、自然言語処理の分野において革新的な進歩をもたらした。"
        "トランスフォーマーアーキテクチャに基づくこれらのモデルは、数十億から数兆の"
        "パラメータを持ち、テキスト生成、翻訳、要約、質問応答など多様なタスクに"
        "対応できる。事前学習とファインチューニングの二段階訓練が一般的である。"
    ),
    "mixed_ja_en": (
        "Pythonは機械学習やDeep Learningの分野で広く使われているプログラミング言語です。"
        "TensorFlowやPyTorchなどのフレームワークを使って、GPUを活用した大規模な"
        "ニューラルネットワークの学習が可能です。最新のLLM技術では、Attentionメカニズムが"
        "重要な役割を果たしています。"
    ),
}


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_fertility(tokenizer: AutoTokenizer, text: str) -> float:
    """Compute fertility: average number of tokens per character.

    A lower fertility score indicates more efficient tokenization.
    Target for Japanese: < 1.5 tokens per character.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    num_chars = len(text)
    if num_chars == 0:
        return 0.0
    return len(tokens) / num_chars


def compute_unk_rate(tokenizer: AutoTokenizer, text: str) -> float:
    """Compute the percentage of UNK tokens in the tokenized output.

    Target: 0%.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 0:
        return 0.0

    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        return 0.0

    unk_count = sum(1 for tid in token_ids if tid == unk_id)
    return (unk_count / len(token_ids)) * 100.0


def check_roundtrip(tokenizer: AutoTokenizer, text: str) -> bool:
    """Check if encoding then decoding produces the original text."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    # Normalize whitespace for comparison (tokenizers may alter spacing)
    return decoded.strip() == text.strip()


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    """Format data as an aligned ASCII table."""
    all_rows = [headers] + rows
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*all_rows)]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def format_row(row: list[str]) -> str:
        cells = [f" {str(cell).ljust(w)} " for cell, w in zip(row, col_widths)]
        return "|" + "|".join(cells) + "|"

    lines = [sep, format_row(headers), sep]
    for row in rows:
        lines.append(format_row(row))
    lines.append(sep)
    return "\n".join(lines)


def evaluate(
    config: Optional[str] = None,
    base_tokenizer: Optional[str] = None,
    merged_tokenizer: Optional[str] = None,
) -> None:
    """Evaluate and compare base vs merged tokenizer on Japanese text.

    Args:
        config: Path to YAML configuration file.
        base_tokenizer: HuggingFace model name or path for the base tokenizer.
        merged_tokenizer: Path to the merged Zensei tokenizer directory.
    """
    # ------------------------------------------------------------------ #
    # Resolve configuration
    # ------------------------------------------------------------------ #
    cfg: dict = {}
    if config is not None:
        logger.info("Loading config from %s", config)
        cfg = load_config(config)

    base_tokenizer = base_tokenizer or cfg.get("base_tokenizer", "deepseek-ai/DeepSeek-V3")
    merged_tokenizer = merged_tokenizer or cfg.get("output_dir", "data/tokenizer/zensei_merged")

    # ------------------------------------------------------------------ #
    # Load tokenizers
    # ------------------------------------------------------------------ #
    logger.info("Loading base tokenizer: %s", base_tokenizer)
    base_tok = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)
    logger.info("Base tokenizer vocab size: %d", len(base_tok))

    logger.info("Loading merged tokenizer: %s", merged_tokenizer)
    merged_tok = AutoTokenizer.from_pretrained(merged_tokenizer, trust_remote_code=True)
    logger.info("Merged tokenizer vocab size: %d", len(merged_tok))

    # ------------------------------------------------------------------ #
    # Evaluate each sample text
    # ------------------------------------------------------------------ #
    detail_rows: list[list[str]] = []
    summary_base_fertility: list[float] = []
    summary_merged_fertility: list[float] = []
    summary_base_unk: list[float] = []
    summary_merged_unk: list[float] = []
    summary_base_rt: list[bool] = []
    summary_merged_rt: list[bool] = []

    for name, text in SAMPLE_TEXTS.items():
        # Base tokenizer metrics
        base_fert = compute_fertility(base_tok, text)
        base_unk = compute_unk_rate(base_tok, text)
        base_rt = check_roundtrip(base_tok, text)
        base_ntokens = len(base_tok.encode(text, add_special_tokens=False))

        # Merged tokenizer metrics
        merged_fert = compute_fertility(merged_tok, text)
        merged_unk = compute_unk_rate(merged_tok, text)
        merged_rt = check_roundtrip(merged_tok, text)
        merged_ntokens = len(merged_tok.encode(text, add_special_tokens=False))

        # Improvement
        fert_improvement = ((base_fert - merged_fert) / base_fert * 100) if base_fert > 0 else 0.0

        detail_rows.append([
            name,
            str(len(text)),
            f"{base_ntokens}",
            f"{merged_ntokens}",
            f"{base_fert:.3f}",
            f"{merged_fert:.3f}",
            f"{fert_improvement:+.1f}%",
            f"{base_unk:.2f}%",
            f"{merged_unk:.2f}%",
            "Yes" if base_rt else "NO",
            "Yes" if merged_rt else "NO",
        ])

        summary_base_fertility.append(base_fert)
        summary_merged_fertility.append(merged_fert)
        summary_base_unk.append(base_unk)
        summary_merged_unk.append(merged_unk)
        summary_base_rt.append(base_rt)
        summary_merged_rt.append(merged_rt)

    # ------------------------------------------------------------------ #
    # Print detailed results table
    # ------------------------------------------------------------------ #
    headers = [
        "Text",
        "Chars",
        "Tok(base)",
        "Tok(merged)",
        "Fert(base)",
        "Fert(merged)",
        "Fert Impr",
        "UNK%(base)",
        "UNK%(merged)",
        "RT(base)",
        "RT(merged)",
    ]

    print("\n" + "=" * 80)
    print("TOKENIZER EVALUATION RESULTS")
    print("=" * 80)
    print(f"Base tokenizer    : {base_tokenizer} (vocab={len(base_tok)})")
    print(f"Merged tokenizer  : {merged_tokenizer} (vocab={len(merged_tok)})")
    print()
    print(format_table(detail_rows, headers))

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    avg_base_fert = sum(summary_base_fertility) / len(summary_base_fertility)
    avg_merged_fert = sum(summary_merged_fertility) / len(summary_merged_fertility)
    avg_base_unk = sum(summary_base_unk) / len(summary_base_unk)
    avg_merged_unk = sum(summary_merged_unk) / len(summary_merged_unk)
    base_rt_pct = sum(summary_base_rt) / len(summary_base_rt) * 100
    merged_rt_pct = sum(summary_merged_rt) / len(summary_merged_rt) * 100

    fert_target = "PASS" if avg_merged_fert < 1.5 else "FAIL"
    unk_target = "PASS" if avg_merged_unk == 0.0 else "FAIL"
    rt_target = "PASS" if merged_rt_pct == 100.0 else "FAIL"

    summary_rows = [
        ["Avg Fertility", f"{avg_base_fert:.3f}", f"{avg_merged_fert:.3f}", "< 1.5", fert_target],
        ["Avg UNK Rate", f"{avg_base_unk:.2f}%", f"{avg_merged_unk:.2f}%", "0%", unk_target],
        ["Roundtrip Accuracy", f"{base_rt_pct:.0f}%", f"{merged_rt_pct:.0f}%", "100%", rt_target],
    ]
    summary_headers = ["Metric", "Base", "Merged", "Target", "Status"]

    print()
    print("SUMMARY")
    print("-" * 60)
    print(format_table(summary_rows, summary_headers))
    print()

    # ------------------------------------------------------------------ #
    # Final verdict
    # ------------------------------------------------------------------ #
    all_pass = all(s == "PASS" for s in [fert_target, unk_target, rt_target])
    if all_pass:
        logger.info("All quality targets met. Tokenizer is ready for use.")
    else:
        logger.warning(
            "Some quality targets were not met. Review the results above."
        )


if __name__ == "__main__":
    fire.Fire(evaluate)
