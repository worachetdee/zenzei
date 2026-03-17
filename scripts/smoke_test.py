#!/usr/bin/env python3
"""
Zensei Smoke Test — end-to-end validation of the full pipeline.

Tests all phases of the Zensei project using tiny configurations so that
the entire suite completes in under 60 seconds on a CPU.

Usage:
    python scripts/smoke_test.py --device cpu
    python scripts/smoke_test.py --device cuda --skip-tokenizer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Colorized output helpers
# ---------------------------------------------------------------------------

_COLOR_SUPPORTED: Optional[bool] = None


def _supports_color() -> bool:
    global _COLOR_SUPPORTED
    if _COLOR_SUPPORTED is not None:
        return _COLOR_SUPPORTED
    _COLOR_SUPPORTED = (
        hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("TERM") != "dumb"
    )
    return _COLOR_SUPPORTED


def _green(text: str) -> str:
    return f"\033[92m{text}\033[0m" if _supports_color() else text


def _red(text: str) -> str:
    return f"\033[91m{text}\033[0m" if _supports_color() else text


def _yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m" if _supports_color() else text


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _supports_color() else text


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    phase: str
    name: str
    passed: bool
    message: str
    elapsed: float  # seconds
    skipped: bool = False


RESULTS: list[TestResult] = []


def record(phase: str, name: str, passed: bool, message: str, elapsed: float, skipped: bool = False):
    RESULTS.append(TestResult(phase, name, passed, message, elapsed, skipped))
    status = _yellow("SKIP") if skipped else (_green("PASS") if passed else _red("FAIL"))
    print(f"  [{status}] {name} ({elapsed:.2f}s) — {message}")


# ---------------------------------------------------------------------------
# Synthetic Japanese text corpus
# ---------------------------------------------------------------------------

SAMPLE_JAPANESE_LINES = [
    "東京は日本の首都であり、世界最大の都市圏の一つです。",
    "桜の花が春に咲き、多くの人々が花見を楽しみます。",
    "富士山は日本で最も高い山で、標高は三千七百七十六メートルです。",
    "日本語には漢字、ひらがな、カタカナの三種類の文字があります。",
    "寿司は日本を代表する料理の一つで、世界中で人気があります。",
    "京都には多くの歴史的な寺院や神社があります。",
    "新幹線は日本の高速鉄道で、時速三百キロメートル以上で走ります。",
    "日本の四季は美しく、それぞれの季節に独自の風景があります。",
    "相撲は日本の国技であり、長い歴史を持つ伝統的なスポーツです。",
    "抹茶は日本の伝統的なお茶で、茶道の中心的な存在です。",
    "北海道は日本最北端の島で、美しい自然と豊かな食文化で知られています。",
    "大阪はたこ焼きやお好み焼きなどの粉物料理で有名です。",
    "日本のアニメーションは世界中で高い評価を受けています。",
    "温泉は日本の文化に深く根ざした入浴の習慣です。",
    "歌舞伎は江戸時代から続く日本の伝統的な演劇の形式です。",
    "日本の教育制度は六三三四制を採用しています。",
    "着物は日本の伝統的な衣装で、特別な行事の際に着用されます。",
    "俳句は五七五の音節で構成される日本の伝統的な詩の形式です。",
    "日本の人口は約一億二千万人で、高齢化が進んでいます。",
    "浮世絵は江戸時代に発展した日本の木版画の芸術形式です。",
    "コンピュータサイエンスは現代社会において重要な分野です。",
    "人工知能の研究は急速に進歩しており、様々な応用が期待されています。",
    "機械学習はデータから自動的にパターンを学習する技術です。",
    "深層学習はニューラルネットワークを用いた機械学習の手法です。",
    "自然言語処理は人間の言語をコンピュータで処理する技術です。",
    "トランスフォーマーモデルは注意機構を基盤とした神経回路網です。",
    "大規模言語モデルは膨大なテキストデータから学習されます。",
    "日本語の形態素解析は英語とは異なる課題があります。",
    "文字コードの標準化は多言語対応に不可欠な要素です。",
    "クラウドコンピューティングは現代のIT基盤として広く利用されています。",
]


def _make_jsonl_corpus(lines: list[str], output_path: str, duplicates: int = 0):
    """Write synthetic JSONL with optional duplicate entries."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(lines):
            record_obj = {"id": f"doc_{i}", "text": text}
            f.write(json.dumps(record_obj, ensure_ascii=False) + "\n")
        # Add duplicates of the first few lines
        for j in range(duplicates):
            record_obj = {"id": f"dup_{j}", "text": lines[j % len(lines)]}
            f.write(json.dumps(record_obj, ensure_ascii=False) + "\n")


# =========================================================================
# Phase 2: Model
# =========================================================================

def test_model(device: str):
    phase = "Phase 2: Model"
    print(f"\n{_bold(phase)}")

    # --- Load config ---
    t0 = time.time()
    try:
        from zensei.model.model import ModelArgs, Transformer

        config_path = Path(__file__).resolve().parent.parent / "configs" / "model" / "zensei_16B.json"
        args = ModelArgs.from_json(config_path)
        record(phase, "Load 16B config", True, f"dim={args.dim}, layers={args.n_layers}", time.time() - t0)
    except Exception as e:
        record(phase, "Load 16B config", False, str(e), time.time() - t0)
        return

    # --- Instantiate a TINY model for speed (override 16B config) ---
    t0 = time.time()
    try:
        tiny_args = ModelArgs(
            vocab_size=1000,
            dim=128,
            inter_dim=256,
            moe_inter_dim=64,
            n_layers=2,
            n_dense_layers=1,
            n_heads=4,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            route_scale=2.5,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            dtype="float32",
            max_seq_len=64,
            rope_theta=10000.0,
        )
        model = Transformer(tiny_args)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        record(phase, "Instantiate tiny model", True, f"{n_params:,} params on {device}", time.time() - t0)
    except Exception as e:
        record(phase, "Instantiate tiny model", False, str(e), time.time() - t0)
        return

    # --- Forward pass ---
    t0 = time.time()
    try:
        import torch
        batch, seq_len = 2, 16
        tokens = torch.randint(0, tiny_args.vocab_size, (batch, seq_len), device=device)
        logits, aux_loss = model(tokens)
        expected_shape = (batch, seq_len, tiny_args.vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        assert aux_loss.dim() == 0, "aux_loss should be scalar"
        record(phase, "Forward pass shape", True, f"logits shape={tuple(logits.shape)}", time.time() - t0)
    except Exception as e:
        record(phase, "Forward pass shape", False, str(e), time.time() - t0)

    # --- Vocab expansion (embed_resize) ---
    t0 = time.time()
    try:
        from zensei.training.embed_resize import resize_embedding_layer
        import torch

        old_vocab = tiny_args.vocab_size
        new_vocab = old_vocab + 128
        old_embed = model.tok_emb
        old_weight_snapshot = old_embed.weight.data[:old_vocab].clone()

        new_embed = resize_embedding_layer(old_embed, new_vocab, noise_std=0.01)

        assert new_embed.weight.shape[0] == new_vocab, (
            f"Expected vocab size {new_vocab}, got {new_embed.weight.shape[0]}"
        )
        # Verify old weights are preserved exactly
        diff = (new_embed.weight.data[:old_vocab] - old_weight_snapshot).abs().max().item()
        assert diff == 0.0, f"Old weights modified! max diff = {diff}"

        record(
            phase, "Vocab expansion (embed_resize)", True,
            f"{old_vocab} -> {new_vocab}, old weights preserved (diff={diff})",
            time.time() - t0,
        )
    except Exception as e:
        record(phase, "Vocab expansion (embed_resize)", False, str(e), time.time() - t0)


# =========================================================================
# Phase 3: Tokenizer
# =========================================================================

def test_tokenizer(device: str, skip: bool):
    phase = "Phase 3: Tokenizer"
    print(f"\n{_bold(phase)}")

    if skip:
        record(phase, "SentencePiece training", True, "Skipped (--skip-tokenizer)", 0.0, skipped=True)
        record(phase, "Encode/decode roundtrip", True, "Skipped (--skip-tokenizer)", 0.0, skipped=True)
        record(phase, "Fertility computation", True, "Skipped (--skip-tokenizer)", 0.0, skipped=True)
        return

    try:
        import sentencepiece as spm
    except ImportError:
        record(phase, "SentencePiece import", False, "sentencepiece not installed — use --skip-tokenizer", 0.0)
        return

    # --- Train a tiny SentencePiece model ---
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write training corpus
            corpus_path = os.path.join(tmpdir, "corpus.txt")
            with open(corpus_path, "w", encoding="utf-8") as f:
                # Write each sample line many times to give SP enough data
                for _ in range(10):
                    for line in SAMPLE_JAPANESE_LINES:
                        f.write(line + "\n")

            model_prefix = os.path.join(tmpdir, "zensei_tiny_sp")
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=1000,
                model_type="bpe",
                character_coverage=0.9995,
                byte_fallback=True,
                num_threads=1,
            )
            sp = spm.SentencePieceProcessor()
            sp.load(model_prefix + ".model")
            record(phase, "SentencePiece training", True, f"vocab_size={sp.get_piece_size()}", time.time() - t0)

            # --- Roundtrip test ---
            t0 = time.time()
            test_texts = SAMPLE_JAPANESE_LINES[:10]
            roundtrip_ok = 0
            for text in test_texts:
                ids = sp.encode(text, out_type=int)
                decoded = sp.decode(ids)
                if decoded == text:
                    roundtrip_ok += 1

            accuracy = roundtrip_ok / len(test_texts)
            passed = accuracy >= 0.8  # Allow some imperfection for tiny vocab
            record(
                phase, "Encode/decode roundtrip", passed,
                f"{roundtrip_ok}/{len(test_texts)} exact matches ({accuracy:.0%})",
                time.time() - t0,
            )

            # --- Fertility ---
            t0 = time.time()
            total_tokens = 0
            total_chars = 0
            for text in SAMPLE_JAPANESE_LINES:
                ids = sp.encode(text, out_type=int)
                total_tokens += len(ids)
                total_chars += len(text)

            fertility = total_tokens / total_chars if total_chars > 0 else float("inf")
            record(
                phase, "Fertility computation", True,
                f"fertility={fertility:.4f} (tokens/char)",
                time.time() - t0,
            )
    except Exception as e:
        record(phase, "SentencePiece training", False, str(e), time.time() - t0)


# =========================================================================
# Phase 4: Data Pipeline
# =========================================================================

def test_data_pipeline(device: str):
    phase = "Phase 4: Data Pipeline"
    print(f"\n{_bold(phase)}")

    # --- Test clean.py ---
    t0 = time.time()
    try:
        from zensei.data.clean import clean_text, process_line, cjk_ratio

        # Test clean_text on dirty input
        dirty = '<p>東京タワー</p> https://example.com は　　素晴らしい\x00場所です.'
        cleaned = clean_text(dirty)
        assert "https://" not in cleaned, "URL not removed"
        assert "<p>" not in cleaned, "HTML not removed"
        assert "\x00" not in cleaned, "Control char not removed"

        # Test CJK ratio
        ratio = cjk_ratio("東京は素晴らしい")
        assert ratio > 0.5, f"CJK ratio too low: {ratio}"

        # Test process_line
        line = json.dumps({"text": "東京は日本の首都であり、世界最大の都市圏の一つです。"}, ensure_ascii=False)
        result = process_line(line, min_cjk_ratio=0.3)
        assert result is not None, "Valid Japanese line should not be filtered"

        record(phase, "clean.py functions", True, "clean_text, cjk_ratio, process_line all OK", time.time() - t0)
    except Exception as e:
        record(phase, "clean.py functions", False, str(e), time.time() - t0)

    # --- Test clean_file end-to-end ---
    t0 = time.time()
    try:
        from zensei.data.clean import clean_file

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "raw.jsonl")
            output_path = os.path.join(tmpdir, "clean.jsonl")

            # Include some lines that should be filtered (English-only)
            lines = SAMPLE_JAPANESE_LINES + [
                "This is a purely English sentence with no Japanese at all.",
                "Another English line that should be filtered out by CJK ratio.",
            ]
            _make_jsonl_corpus(lines, input_path)
            stats = clean_file(input_path, output_path, min_cjk_ratio=0.3, num_workers=1)
            assert stats["kept"] > 0, "No lines kept"
            assert stats["filtered"] >= 2, "English lines should be filtered"

            record(
                phase, "clean_file end-to-end", True,
                f"kept={stats['kept']}, filtered={stats['filtered']}",
                time.time() - t0,
            )
    except Exception as e:
        record(phase, "clean_file end-to-end", False, str(e), time.time() - t0)

    # --- Test dedup.py ---
    t0 = time.time()
    try:
        from datasketch import MinHash, MinHashLSH
        from zensei.data.dedup import dedup_file

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "clean.jsonl")
            output_path = os.path.join(tmpdir, "dedup.jsonl")

            # Create corpus with 5 exact duplicates
            _make_jsonl_corpus(SAMPLE_JAPANESE_LINES[:10], input_path, duplicates=5)
            stats = dedup_file(input_path, output_path, threshold=0.8, num_perm=64)
            assert stats["duplicates_removed"] > 0, "Should have detected duplicates"
            assert stats["remaining"] <= stats["total"], "Remaining must be <= total"

            record(
                phase, "dedup_file end-to-end", True,
                f"total={stats['total']}, dupes_removed={stats['duplicates_removed']}, remaining={stats['remaining']}",
                time.time() - t0,
            )
    except ImportError:
        record(phase, "dedup_file end-to-end", False, "datasketch not installed", time.time() - t0)
    except Exception as e:
        record(phase, "dedup_file end-to-end", False, str(e), time.time() - t0)

    # --- Test filter.py ---
    t0 = time.time()
    try:
        from zensei.data.filter import filter_file, filter_document

        # Unit test filter_document
        reason = filter_document("短い", min_len=50)  # too short
        assert reason == "length", f"Expected 'length', got {reason}"

        long_jp = "東京は日本の首都です。" * 20
        reason = filter_document(long_jp, min_len=10)
        assert reason is None, f"Should pass but got reason={reason}"

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "dedup.jsonl")
            output_path = os.path.join(tmpdir, "filtered.jsonl")
            _make_jsonl_corpus(SAMPLE_JAPANESE_LINES, input_path)
            stats = filter_file(
                input_path, output_path,
                min_len=10, max_len=500_000,
                min_cjk_ratio=0.3,
            )
            assert stats["kept"] > 0, "No lines survived filtering"

            record(
                phase, "filter_file end-to-end", True,
                f"kept={stats['kept']}, filtered={stats['filtered']}",
                time.time() - t0,
            )
    except Exception as e:
        record(phase, "filter_file end-to-end", False, str(e), time.time() - t0)


# =========================================================================
# Phase 5: Training
# =========================================================================

def test_training(device: str):
    phase = "Phase 5: Training"
    print(f"\n{_bold(phase)}")

    import torch
    from zensei.model.model import ModelArgs, Transformer

    # --- Build tiny model ---
    t0 = time.time()
    try:
        tiny_args = ModelArgs(
            vocab_size=1000,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            n_layers=2,
            n_dense_layers=1,
            n_heads=4,
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            route_scale=2.5,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            dtype="float32",
            max_seq_len=64,
            rope_theta=10000.0,
        )
        model = Transformer(tiny_args).to(device)
        record(phase, "Build tiny training model", True, f"device={device}", time.time() - t0)
    except Exception as e:
        record(phase, "Build tiny training model", False, str(e), time.time() - t0)
        return

    # --- Run 10 training steps ---
    t0 = time.time()
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
        losses = []
        batch_size, seq_len = 8, 32
        n_steps = 20

        # Use a fixed repeated batch so the model can memorize it
        fixed_tokens = torch.randint(0, tiny_args.vocab_size, (batch_size, seq_len), device=device)

        for step in range(n_steps):
            inputs = fixed_tokens[:, :-1]
            targets = fixed_tokens[:, 1:]

            logits, aux_loss = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, tiny_args.vocab_size),
                targets.reshape(-1),
            )
            total_loss = loss + 0.01 * aux_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Check that loss decreased (first 3 vs last 3)
        first_loss = sum(losses[:3]) / 3
        last_loss = sum(losses[-3:]) / 3
        decreased = last_loss < first_loss
        record(
            phase, f"{n_steps} training steps (loss decreases)", decreased,
            f"first_avg={first_loss:.4f}, last_avg={last_loss:.4f}",
            time.time() - t0,
        )
    except Exception as e:
        record(phase, "10 training steps (loss decreases)", False, str(e), time.time() - t0)

    # --- Checkpoint save/load roundtrip ---
    t0 = time.time()
    try:
        from zensei.training.checkpoint import save_checkpoint, load_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            ckpt_dir = save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                step=10,
                path=tmpdir,
                use_deepspeed=False,
            )

            # Load into a fresh model
            model2 = Transformer(tiny_args).to(device)
            loaded_step = load_checkpoint(
                model=model2,
                optimizer=None,
                scheduler=None,
                path=ckpt_dir,
                use_deepspeed=False,
            )

            assert loaded_step == 10, f"Expected step 10, got {loaded_step}"

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                model.state_dict().items(), model2.state_dict().items()
            ):
                diff = (p1.cpu().float() - p2.cpu().float()).abs().max().item()
                assert diff < 1e-6, f"Weight mismatch in {n1}: max diff = {diff}"

            record(
                phase, "Checkpoint save/load roundtrip", True,
                f"step={loaded_step}, all weights match",
                time.time() - t0,
            )
    except Exception as e:
        record(phase, "Checkpoint save/load roundtrip", False, str(e), time.time() - t0)


# =========================================================================
# Phase 6: Eval
# =========================================================================

def test_eval(device: str):
    phase = "Phase 6: Eval"
    print(f"\n{_bold(phase)}")

    # --- Character-level F1 ---
    t0 = time.time()
    try:
        from zensei.eval.jaquad import char_f1_score, exact_match_score

        # Exact match
        f1 = char_f1_score("東京", "東京")
        assert abs(f1 - 1.0) < 1e-6, f"Exact match F1 should be 1.0, got {f1}"

        # Partial overlap
        f1 = char_f1_score("東京都", "東京")
        assert 0.0 < f1 < 1.0, f"Partial overlap F1 should be between 0 and 1, got {f1}"

        # No overlap
        f1 = char_f1_score("大阪", "東京")
        assert f1 == 0.0, f"No overlap F1 should be 0.0, got {f1}"

        # Empty strings
        f1 = char_f1_score("", "")
        assert f1 == 1.0, f"Both empty should be 1.0, got {f1}"

        # Exact match metric
        em = exact_match_score("東京", "東京")
        assert em == 1.0, f"EM should be 1.0, got {em}"
        em = exact_match_score("東京都", "東京")
        assert em == 0.0, f"EM should be 0.0, got {em}"

        record(phase, "Character-level F1 computation", True, "All known-input tests pass", time.time() - t0)
    except Exception as e:
        record(phase, "Character-level F1 computation", False, str(e), time.time() - t0)

    # --- Perplexity on dummy data ---
    t0 = time.time()
    try:
        import torch
        import torch.nn as nn
        from zensei.model.model import ModelArgs, Transformer

        # Build a tiny model for perplexity testing
        tiny_args = ModelArgs(
            vocab_size=200,
            dim=64,
            inter_dim=128,
            moe_inter_dim=32,
            n_layers=1,
            n_dense_layers=1,
            n_heads=2,
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            route_scale=2.5,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            dtype="float32",
            max_seq_len=64,
            rope_theta=10000.0,
        )
        model = Transformer(tiny_args).to(device).eval()

        # Compute perplexity manually on a random sequence
        tokens = torch.randint(0, tiny_args.vocab_size, (1, 32), device=device)
        with torch.no_grad():
            logits, _ = model(tokens)

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, tiny_args.vocab_size),
            shift_labels.view(-1),
        )
        ppl = math.exp(loss.item())
        # For a random model with vocab=200, perplexity should be around 200
        assert ppl > 1.0, f"Perplexity should be > 1.0, got {ppl}"
        assert ppl < 10000.0, f"Perplexity unreasonably high: {ppl}"

        record(
            phase, "Perplexity computation (dummy)", True,
            f"PPL={ppl:.2f} (expected ~{tiny_args.vocab_size} for random model)",
            time.time() - t0,
        )
    except Exception as e:
        record(phase, "Perplexity computation (dummy)", False, str(e), time.time() - t0)


# =========================================================================
# Phase 7: Serving
# =========================================================================

def test_serving(device: str):
    phase = "Phase 7: Serving"
    print(f"\n{_bold(phase)}")

    # --- Chat template ---
    t0 = time.time()
    try:
        from zensei.serving.chat_template import (
            apply_chat_template,
            parse_chat_messages,
            format_message,
            ROLE_TAGS,
            DEFAULT_SYSTEM_PROMPT,
        )

        messages = [
            {"role": "user", "content": "こんにちは！"},
        ]
        formatted = apply_chat_template(messages, add_generation_prompt=True)

        # Should contain system prompt (auto-added), user message, and assistant tag
        assert "<|system|>" in formatted, "Missing system tag"
        assert "<|user|>" in formatted, "Missing user tag"
        assert "<|assistant|>" in formatted, "Missing assistant tag"
        assert "こんにちは" in formatted, "Missing user content"

        # Roundtrip: format -> parse
        messages_with_system = [
            {"role": "system", "content": "テスト用システムプロンプト"},
            {"role": "user", "content": "質問です"},
            {"role": "assistant", "content": "回答です"},
        ]
        formatted = apply_chat_template(messages_with_system, add_generation_prompt=False)
        parsed = parse_chat_messages(formatted)
        assert len(parsed) == 3, f"Expected 3 messages, got {len(parsed)}"
        assert parsed[0]["role"] == "system"
        assert parsed[1]["role"] == "user"
        assert parsed[2]["role"] == "assistant"

        record(phase, "Chat template formatting", True, "format + parse roundtrip OK", time.time() - t0)
    except Exception as e:
        record(phase, "Chat template formatting", False, str(e), time.time() - t0)

    # --- Generate a few tokens ---
    t0 = time.time()
    try:
        import torch
        from zensei.model.model import ModelArgs, Transformer
        from zensei.serving.generate import _sample_token, _apply_repetition_penalty

        tiny_args = ModelArgs(
            vocab_size=200,
            dim=64,
            inter_dim=128,
            moe_inter_dim=32,
            n_layers=1,
            n_dense_layers=1,
            n_heads=2,
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            route_scale=2.5,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            dtype="float32",
            max_seq_len=64,
            rope_theta=10000.0,
        )
        model = Transformer(tiny_args).to(device).eval()

        # Test sampling helpers
        logits = torch.randn(200)
        token_id = _sample_token(logits, temperature=0.0)  # greedy
        assert 0 <= token_id < 200, f"Token ID out of range: {token_id}"

        # Test repetition penalty
        logits2 = torch.ones(200)
        penalized = _apply_repetition_penalty(logits2.clone(), [0, 1, 2], penalty=1.5)
        assert penalized[0] < logits2[0], "Repetition penalty should reduce logit for repeated token"

        # Test generation loop manually (no tokenizer needed)
        tokens = torch.randint(0, 200, (1, 8), device=device)
        generated_ids = []
        current_ids = tokens
        with torch.no_grad():
            for _ in range(5):
                out_logits, _ = model(current_ids)
                next_logits = out_logits[:, -1, :].squeeze(0)
                next_token = _sample_token(next_logits.cpu(), temperature=0.7, top_p=0.9, top_k=50)
                generated_ids.append(next_token)
                next_tensor = torch.tensor([[next_token]], device=device)
                current_ids = torch.cat([current_ids, next_tensor], dim=1)

        assert len(generated_ids) == 5, f"Expected 5 tokens, got {len(generated_ids)}"
        assert all(0 <= t < 200 for t in generated_ids), "Generated token out of range"

        record(
            phase, "Token generation (tiny model)", True,
            f"Generated 5 tokens: {generated_ids}",
            time.time() - t0,
        )
    except Exception as e:
        record(phase, "Token generation (tiny model)", False, str(e), time.time() - t0)


# =========================================================================
# Summary
# =========================================================================

def print_summary():
    print(f"\n{'=' * 70}")
    print(_bold("SMOKE TEST SUMMARY"))
    print(f"{'=' * 70}")

    # Table header
    print(f"{'Phase':<25} {'Test':<40} {'Status':<8} {'Time':>6}")
    print(f"{'-' * 25} {'-' * 40} {'-' * 8} {'-' * 6}")

    passed = 0
    failed = 0
    skipped = 0

    for r in RESULTS:
        if r.skipped:
            status = _yellow("SKIP")
            skipped += 1
        elif r.passed:
            status = _green("PASS")
            passed += 1
        else:
            status = _red("FAIL")
            failed += 1

        # Truncate long names for table alignment
        phase_short = r.phase.split(":")[0].strip() if ":" in r.phase else r.phase
        name_short = r.name[:38] + ".." if len(r.name) > 40 else r.name
        print(f"{phase_short:<25} {name_short:<40} {status:<8} {r.elapsed:>5.2f}s")

    print(f"{'-' * 25} {'-' * 40} {'-' * 8} {'-' * 6}")

    total = passed + failed + skipped
    total_time = sum(r.elapsed for r in RESULTS)

    summary = f"Total: {total} | {_green(f'PASS: {passed}')} | {_red(f'FAIL: {failed}')} | {_yellow(f'SKIP: {skipped}')} | Time: {total_time:.2f}s"
    print(summary)

    if failed > 0:
        print(f"\n{_red('SMOKE TEST FAILED')} — {failed} test(s) did not pass.")
        print("\nFailed tests:")
        for r in RESULTS:
            if not r.passed and not r.skipped:
                print(f"  - [{r.phase}] {r.name}: {r.message}")
        return 1
    else:
        print(f"\n{_green('ALL SMOKE TESTS PASSED')}")
        return 0


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Zensei end-to-end smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/smoke_test.py --device cpu
  python scripts/smoke_test.py --device cuda
  python scripts/smoke_test.py --device cpu --skip-tokenizer
        """,
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run tests on: cpu or cuda (default: cpu)",
    )
    parser.add_argument(
        "--skip-tokenizer", action="store_true",
        help="Skip tokenizer tests (if sentencepiece is not installed)",
    )
    args = parser.parse_args()

    # Validate device
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        print(_yellow("WARNING: --device cuda requested but CUDA not available. Falling back to CPU."))
        args.device = "cpu"

    print(_bold("=" * 70))
    print(_bold("ZENSEI SMOKE TEST"))
    print(_bold("=" * 70))
    print(f"Device: {args.device}")
    print(f"Skip tokenizer: {args.skip_tokenizer}")
    t_global = time.time()

    # Run all phases
    test_model(args.device)
    test_tokenizer(args.device, skip=args.skip_tokenizer)
    test_data_pipeline(args.device)
    test_training(args.device)
    test_eval(args.device)
    test_serving(args.device)

    total_time = time.time() - t_global
    print(f"\nTotal wall-clock time: {total_time:.2f}s")

    exit_code = print_summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
