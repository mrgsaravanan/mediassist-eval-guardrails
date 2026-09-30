PY ?= .venv/bin/python

.PHONY: install test adversarial eval eval-offline report judge-sanity live-test

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	$(PY) -m pytest -q

live-test:
	$(PY) -m pytest -q -m live

adversarial:
	$(PY) scripts/run_adversarial_tests.py

judge-sanity:
	$(PY) scripts/run_judge_sanity.py

eval:
	$(PY) scripts/run_eval.py --eval-set data/eval_set.jsonl --out reports/

eval-offline:
	$(PY) scripts/run_eval.py --eval-set data/eval_set.jsonl --out reports/ --offline

# Re-run the evaluation (exit code ignored) and promote the newest report to reports/sample_report.md
report:
	-$(PY) scripts/run_eval.py --eval-set data/eval_set.jsonl --out reports/
	cp "$$(ls -t reports/eval_report_*.md | head -1)" reports/sample_report.md
