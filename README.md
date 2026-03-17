# Zensei (禅精) — DeepSeek-V3 Fork Optimized for Japanese

Zensei is a continued pretraining fork of [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) that adapts the model for high-quality Japanese language understanding and generation. The project extends DeepSeek-V3's Mixture-of-Experts architecture with expanded Japanese vocabulary and trains on large-scale Japanese corpora.

## Architecture Highlights

| Property | Value |
|---|---|
| Base model | DeepSeek-V3 |
| Total parameters | 671B (MoE) |
| Activated parameters | 37B per token |
| Attention | Multi-head Latent Attention (MLA) |
| Expert routing | DeepSeekMoE with auxiliary-loss-free balancing |
| Vocabulary | 129K → 145K tokens (expanded with Japanese subwords) |
| Context length | 128K tokens (YaRN-extended) |

## Training Strategy

Zensei follows a two-stage continued pretraining approach:

1. **Stage 1 — Vocabulary Extension & Embedding Alignment**
   Expand the SentencePiece vocabulary with 16K Japanese subwords, initialize new embeddings via subword averaging, and align them with a short warmup on mixed Japanese/English data.

2. **Stage 2 — Full Continued Pretraining**
   Train on a large deduplicated Japanese corpus (Common Crawl, Wikipedia, books, code) with curriculum scheduling, progressing from short to long contexts.

## Quick Start

### Install

```bash
git clone https://github.com/your-org/zensei.git
cd zensei
make install
```

### Tokenizer Training

```bash
make train-tokenizer
```

### Data Preparation

```bash
make data-pipeline
```

### Training

```bash
make pretrain
```

### Evaluation

```bash
make eval
```

### Serving

```bash
make serve
```

## Project Structure

```
zensei/
├── zensei/                 # Main Python package
│   ├── model/              # Model architecture (MLA, MoE, DeepSeek blocks)
│   ├── tokenizer/          # Vocabulary expansion & tokenizer training
│   ├── data/               # Data pipeline, dedup, curriculum scheduling
│   ├── training/           # Training loops, DeepSpeed configs, FSDP
│   ├── eval/               # Benchmarks (JCommonsenseQA, JNLI, MARC-ja, …)
│   ├── serving/            # FastAPI inference server
│   └── cli/                # CLI entry points
├── configs/                # YAML training & model configs
├── scripts/                # Shell scripts for orchestration
├── tests/                  # Unit and integration tests
├── data/                   # Data directory (raw/ and processed/ are gitignored)
├── pyproject.toml          # Project metadata and dependencies
├── Makefile                # Common development commands
├── LICENSE-CODE            # MIT License (code)
└── LICENSE-MODEL           # DeepSeek Model License (weights)
```

## License

- **Code**: Released under the [MIT License](LICENSE-CODE).
- **Model weights**: Subject to the [DeepSeek Model License](LICENSE-MODEL). See the upstream license for usage terms and restrictions.
