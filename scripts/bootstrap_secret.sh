#!/usr/bin/env bash
#
# One-time bootstrap: create the Databricks secret scope and stash HF_TOKEN.
# DABs do not manage secrets, so this lives outside the bundle.
#
# Prerequisites:
#   - HF_TOKEN env var set (from https://huggingface.co/settings/tokens)
#   - PROFILE env var set to the databricks CLI profile for the target workspace
#     (run `databricks auth login --host <your-workspace>` to create one)
#   - `jq` installed
#
# Usage:
#   HF_TOKEN=hf_xxx PROFILE=my-workspace bash scripts/bootstrap_secret.sh
#
set -euo pipefail

: "${PROFILE:?Set PROFILE env var to your databricks CLI profile name}"
: "${HF_TOKEN:?Set HF_TOKEN env var (https://huggingface.co/settings/tokens) before running}"

SCOPE="${SCOPE:-loyalty-offers}"
KEY="${KEY:-HF_TOKEN}"

echo "[bootstrap] profile=$PROFILE scope=$SCOPE key=$KEY"

if databricks secrets list-scopes -p "$PROFILE" --output json \
    | jq -e --arg s "$SCOPE" '(.scopes // .)[]? | select(.name==$s)' >/dev/null 2>&1; then
  echo "[bootstrap] secret scope '$SCOPE' already exists"
else
  databricks secrets create-scope "$SCOPE" -p "$PROFILE"
  echo "[bootstrap] created secret scope '$SCOPE'"
fi

# `put-secret` reads the value from --string-value or stdin via --json.
databricks secrets put-secret "$SCOPE" "$KEY" --string-value "$HF_TOKEN" -p "$PROFILE"
echo "[bootstrap] wrote $SCOPE/$KEY (value length=${#HF_TOKEN})"

echo "[bootstrap] OK"
