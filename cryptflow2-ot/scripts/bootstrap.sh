#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EZPC_DIR="${ROOT_DIR}/external/EzPC"
EZPC_COMMIT="${EZPC_COMMIT:-f24bf3e022dde3493cf0ffe3ebd92dd73936b734}"

if [[ ! -d "${EZPC_DIR}/.git" ]]; then
  mkdir -p "${ROOT_DIR}/external"
  git clone https://github.com/mpc-msri/EzPC.git "${EZPC_DIR}"
fi

git -C "${EZPC_DIR}" config core.autocrlf input
git -C "${EZPC_DIR}" fetch --depth 1 origin "${EZPC_COMMIT}"
git -C "${EZPC_DIR}" checkout "${EZPC_COMMIT}"
git -C "${EZPC_DIR}" apply "${ROOT_DIR}/patches/ezpc-sci-crown.patch"
cp "${ROOT_DIR}/patches/ot-phase-metrics.h" "${EZPC_DIR}/SCI/src/OT/ot-phase-metrics.h"
LEGACY_EZPC_PREFIX="/U""sers/[^[:space:]]+/EzPC"
sed -i -E "s|${LEGACY_EZPC_PREFIX}|${EZPC_DIR}|g" "${EZPC_DIR}/SCI/cmake/SCIConfig.cmake"

git -C "${EZPC_DIR}" submodule update --init --depth 1 SCI/extern/eigen SCI/extern/SEAL
git -C "${EZPC_DIR}/SCI/extern/SEAL" apply "${ROOT_DIR}/patches/seal-4.1-locks.patch"
