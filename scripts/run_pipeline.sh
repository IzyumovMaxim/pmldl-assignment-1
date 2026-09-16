#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

# Scheduled jobs start with a minimal PATH on macOS and often cannot find
# Docker installed in /usr/local/bin or /opt/homebrew/bin.
export PATH="$PROJECT_DIR/.venv/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
export DVC_SITE_CACHE_DIR="$PROJECT_DIR/.dvc/site-cache"

python -m code.datasets.download_data
dvc repro
docker compose -f code/deployment/docker-compose.yml up -d --build --remove-orphans
