#!/usr/bin/env bash
# Build and upload SENTINEL-15 to Namecheap shared hosting over SSH (port 21098).
# Usage: ./deploy.sh <cpanel-user> <server-host>     e.g. ./deploy.sh gnirxyzu server123.web-hosting.com
# One-time: authorise ~/.ssh/sentinel_deploy.pub in cPanel → SSH Access → Manage SSH Keys.
set -euo pipefail
USER_="${1:?cPanel username}"; HOST="${2:?server hostname}"
KEY=~/.ssh/sentinel_deploy
SSH=(ssh -i "$KEY" -p 21098 -o StrictHostKeyChecking=accept-new "$USER_@$HOST")
cd "$(dirname "$0")"

npm run verify          # JS model must still match Python
npm run build

# Site analyst config lives outside public_html so it can never be downloaded.
SECRETS=../.streamlit/secrets.toml
if [[ -f $SECRETS ]] && ! "${SSH[@]}" test -f sentinel-config.php; then
  KEYVAL=$(sed -n 's/^GEMINI_API_KEY *= *"\(.*\)"/\1/p' "$SECRETS")
  MODEL=$(sed -n 's/^LLM_MODEL *= *"\(.*\)"/\1/p' "$SECRETS")
  if [[ -n $KEYVAL ]]; then
    "${SSH[@]}" "umask 077; cat > sentinel-config.php" <<CFG
<?php return [
  'key' => '$KEYVAL',
  'model' => '${MODEL:-gemini-3.5-flash-lite}',
  'upstream' => 'https://generativelanguage.googleapis.com/v1beta/openai',
  'per_ip_hour' => 30,
  'per_day' => 600,
];
CFG
    echo "wrote ~/sentinel-config.php on the server (site analyst enabled)"
  fi
fi

# Upload; --delete only inside assets/ so cPanel's own files in public_html stay put.
rsync -az -e "ssh -i $KEY -p 21098" dist/ "$USER_@$HOST:public_html/"
rsync -az --delete -e "ssh -i $KEY -p 21098" dist/assets/ "$USER_@$HOST:public_html/assets/"
echo "deployed: https://gnirris.xyz (or http:// until the SSL certificate is issued)"
