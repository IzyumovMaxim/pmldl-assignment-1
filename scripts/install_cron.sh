#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_LINE="*/5 * * * * cd $PROJECT_DIR && $PROJECT_DIR/scripts/run_pipeline.sh >> $PROJECT_DIR/logs/pipeline.log 2>&1"

(crontab -l 2>/dev/null | grep -Fv "$PROJECT_DIR/scripts/run_pipeline.sh" || true; echo "$CRON_LINE") | crontab -
echo "Installed: $CRON_LINE"

