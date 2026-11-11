#!/usr/bin/env bash
set -euo pipefail

BIN_PATH="${1:-./build/src/cryptflow2_crown}"
PORT="${CROWN_SMOKE_PORT:-42000}"
IP="${CROWN_SMOKE_IP:-127.0.0.1}"
THREADS="${CROWN_SMOKE_THREADS:-1}"
BITLENGTH="${CROWN_SMOKE_BITLENGTH:-41}"
SCALE_BITS="${CROWN_SMOKE_SCALE_BITS:-18}"

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "error: binary not found or not executable: ${BIN_PATH}" >&2
  exit 2
fi

cleanup() {
  if [[ -n "${P1_PID:-}" ]]; then
    kill "${P1_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${P2_PID:-}" ]]; then
    kill "${P2_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

LOG_DIR="$(mktemp -d)"
P1_LOG="${LOG_DIR}/party1.log"
P2_LOG="${LOG_DIR}/party2.log"

"${BIN_PATH}" "r=1" "ip=${IP}" "p=${PORT}" "nt=${THREADS}" "bl=${BITLENGTH}" "sf=${SCALE_BITS}" >"${P1_LOG}" 2>&1 &
P1_PID=$!
sleep 0.3
"${BIN_PATH}" "r=2" "ip=${IP}" "p=${PORT}" "nt=${THREADS}" "bl=${BITLENGTH}" "sf=${SCALE_BITS}" >"${P2_LOG}" 2>&1 &
P2_PID=$!

set +e
wait "${P1_PID}"
STATUS1=$?
wait "${P2_PID}"
STATUS2=$?
set -e

echo "===== party1 ====="
cat "${P1_LOG}"
echo "===== party2 ====="
cat "${P2_LOG}"

if [[ ${STATUS1} -ne 0 || ${STATUS2} -ne 0 ]]; then
  echo "smoke test failed: party1=${STATUS1}, party2=${STATUS2}" >&2
  exit 1
fi
