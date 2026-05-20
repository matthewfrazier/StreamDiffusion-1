#!/usr/bin/env bash
set -euo pipefail

HTTPS_PORT=11555
INTERNAL_PORT=8384

echo "Configuring Tailscale serve: HTTPS :${HTTPS_PORT} -> http://127.0.0.1:${INTERNAL_PORT}"
sudo tailscale serve --bg --https "${HTTPS_PORT}" "http://127.0.0.1:${INTERNAL_PORT}"

echo ""
echo "Tailscale serve configured. Access the app at:"
tailscale status --self --json | python3 -c "
import json, sys
info = json.load(sys.stdin)
dns = info.get('Self', {}).get('DNSName', '').rstrip('.')
print(f'  https://{dns}:${HTTPS_PORT}')
" 2>/dev/null || echo "  https://$(hostname):${HTTPS_PORT}"
