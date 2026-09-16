#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_DIR/logs"

printf -v PROJECT_DIR_QUOTED '%q' "$PROJECT_DIR"
CRON_LINE="*/5 * * * * cd $PROJECT_DIR_QUOTED && $PROJECT_DIR_QUOTED/scripts/run_pipeline.sh >> $PROJECT_DIR_QUOTED/logs/pipeline.log 2>&1"

(crontab -l 2>/dev/null | grep -Fv "$PROJECT_DIR/scripts/run_pipeline.sh" || true; echo "$CRON_LINE") | crontab -
echo "Installed: $CRON_LINE"
