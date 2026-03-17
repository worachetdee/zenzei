.PHONY: install lint format test smoke-test train-tokenizer data-pipeline pretrain eval serve clean

install:
	pip install -e ".[dev]"

lint:
	ruff check zensei/ tests/

format:
	ruff format zensei/ tests/

test:
	pytest tests/ -v

smoke-test:
	python scripts/smoke_test.py --device cpu

train-tokenizer:
	bash scripts/run_tokenizer_train.sh

data-pipeline:
	bash scripts/run_data_pipeline.sh

pretrain:
	bash scripts/run_pretrain.sh

eval:
	bash scripts/run_eval.sh

serve:
	bash scripts/run_server.sh

clean:
	rm -rf build dist *.egg-info
