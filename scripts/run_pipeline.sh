#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

if [[ -x ".venv/bin/python" ]]; then
export PATH="$PROJECT_DIR/.venv/bin:$PATH"
fi
export DVC_SITE_CACHE_DIR="$PROJECT_DIR/.dvc/site-cache"

python -m code.datasets.download_data
dvc repro
docker compose -f code/deployment/docker-compose.yml up -d --build --remove-orphans
