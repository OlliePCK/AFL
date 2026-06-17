#!/bin/bash
set -e

echo "[$(date)] AFL Model Container Starting..."

# Create log directory
mkdir -p /var/log/afl

# Configure local timezone for logs and cron if TZ is provided.
if [[ -n "${TZ:-}" && -f "/usr/share/zoneinfo/$TZ" ]]; then
  ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime
  echo "$TZ" > /etc/timezone
fi

# Pass environment variables to cron (cron runs in a clean env)
printenv | grep -E '^(ODDS_API_KEY|VISUAL_CROSSING_API_KEY|PATH|HOME|PYTHONPATH|TZ)=' > /etc/environment

# Install crontab
crontab /app/scripts/crontab
echo "[$(date)] Cron schedule installed"

# Start cron daemon in background
cron
echo "[$(date)] Cron daemon started"

# Run an initial catch-up refresh on startup so a midday restart doesn't leave
# completed results stale until the next scheduled sync window.
(
  cd /app &&
  python scripts/automate.py sync-results &&
  python scripts/automate.py predict-cached
) >> /var/log/afl/cron.log 2>&1 &

# Start Next.js dashboard (foreground — keeps container alive)
echo "[$(date)] Starting dashboard on port 3000..."
cd /app/dashboard && exec npm start
