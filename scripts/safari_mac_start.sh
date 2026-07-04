#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="https://github.com/JABarios/safari/releases/download/model-v0.0.1/safari_lgbm_v0.zip"
MODEL_ZIP="safari_lgbm_v0.zip"
MODEL_DIR="models"
DATA_DIR="data"
OUTPUT_DIR="outputs"
MODEL_FILE="${MODEL_DIR}/safari_lgbm_v0.txt"

cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH."
  echo "Install Docker Desktop from: https://www.docker.com/products/docker-desktop/"
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed but not running."
  echo "Open Docker Desktop, wait until it is ready, then run this script again."
  exit 1
fi

mkdir -p "${DATA_DIR}" "${MODEL_DIR}" "${OUTPUT_DIR}"

if [ ! -f "${MODEL_FILE}" ]; then
  echo "Downloading SAFARI model..."
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "${MODEL_ZIP}" "${MODEL_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${MODEL_ZIP}" "${MODEL_URL}"
  else
    echo "Need curl or wget to download the model."
    exit 1
  fi

  echo "Unpacking model..."
  rm -rf safari_lgbm_v0
  unzip -q "${MODEL_ZIP}"
  cp safari_lgbm_v0/safari_lgbm_v0.txt "${MODEL_FILE}"
fi

edf_count="$(find "${DATA_DIR}" -type f \( -iname '*.edf' -o -iname '*.bdf' \) | wc -l | tr -d ' ')"
if [ "${edf_count}" = "0" ]; then
  echo
  echo "No EDF/BDF files found yet."
  echo "Put your recordings in:"
  echo "  $(pwd)/${DATA_DIR}"
  echo
fi

echo "Starting SAFARI..."
echo "Open this URL in your browser:"
echo "  http://127.0.0.1:8765"
echo
echo "Press Ctrl+C here to stop SAFARI."
docker compose up --build
