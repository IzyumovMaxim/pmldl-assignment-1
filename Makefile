PYTHON ?= python3.12

.PHONY: setup download prepare train pipeline deploy test clean

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

download:
	.venv/bin/python -m code.datasets.download_data

prepare:
	.venv/bin/python -m code.datasets.prepare --config params.yaml

train:
	.venv/bin/python -m code.models.train --config params.yaml

pipeline:
	DVC_SITE_CACHE_DIR="$(CURDIR)/.dvc/site-cache" PATH="$(CURDIR)/.venv/bin:$$PATH" dvc repro

deploy:
	docker compose -f code/deployment/docker-compose.yml up -d --build

test:
	PYTHONPATH=. .venv/bin/python -m pytest -q

clean:
	docker compose -f code/deployment/docker-compose.yml down
