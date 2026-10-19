#!/usr/bin/env bash
set -euo pipefail

app_dir="/opt/niftypilot"
db_file="$app_dir/data/niftypilot.db"
backup_dir="$app_dir/data/backups"
temporary="$backup_dir/.niftypilot-$(date +%F).db.tmp"
destination="$backup_dir/niftypilot-$(date +%F).db"

mkdir -p "$backup_dir"
sqlite3 "$db_file" ".backup '$temporary'"
check="$(sqlite3 "$temporary" 'PRAGMA quick_check')"
if [ "$check" != "ok" ]; then
  echo "Backup verification failed: $check" >&2
  exit 1
fi
mv "$temporary" "$destination"
if [ -f "$app_dir/models/regime.json" ] && [ -f "$app_dir/models/regime.metadata.json" ]; then
  cp "$app_dir/models/regime.json" "$backup_dir/regime-$(date +%F).json"
  cp "$app_dir/models/regime.metadata.json" "$backup_dir/regime-$(date +%F).metadata.json"
fi
find "$backup_dir" -type f -name 'niftypilot-*.db' -mtime +7 -delete
find "$backup_dir" -type f -name 'regime-*.json' -mtime +7 -delete
