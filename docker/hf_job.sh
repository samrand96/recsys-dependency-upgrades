#!/bin/bash
# Entry point for a Hugging Face Job (protocol/second_host.md). Runs the harness in local mode.
#   /bundle  read-only: repo.bundle (git bundle of the commit), cases.txt (one case id per line)
#   /out     writable storage bucket: results are copied to /out/<JOB_NAME>/
#   SILENTRANK_KEEP_ARRAYS=1 also exports every run's raw arrays to /out/<JOB_NAME>/arrays/<case>.tar
# Image: ghcr.io/astral-sh/uv@sha256:<same digest as docker/Dockerfile.runner>
set -eu

BASE_DIGEST="${SILENTRANK_BASE_DIGEST:?set SILENTRANK_BASE_DIGEST to the runner base digest}"
JOB_NAME="${SILENTRANK_JOB_NAME:-job}"
OUT="/out/${JOB_NAME}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1

echo "== start $(date -u +%FT%TZ)"
apt-get update -qq
apt-get install --yes --no-install-recommends -qq g++ git ca-certificates curl > /dev/null

# Same environment as docker/Dockerfile.runner.
export UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SILENTRANK_RUNNER=local SILENTRANK_RUNNER_IMAGE="hfjob:${BASE_DIGEST}+g++"
# Deterministic cuBLAS for the GPU condition (protocol/pipelines_p7_gpu.md); harmless on CPU.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1 || echo "no GPU on this host" > "$OUT/nvidia-smi.txt"
lscpu > "$OUT/lscpu.txt" 2>&1 || true

git clone --quiet /bundle/repo.bundle /repo
cd /repo
git fetch --quiet --tags /bundle/repo.bundle || true
mkdir -p /venvs
echo "== commit $(git rev-parse HEAD)"

uv sync --locked --quiet
# The GroupLens download is sometimes cut off mid-file; resume it with curl. prepare_ml1m.py
# then verifies the publisher MD5 and the pinned SHA-256 before using the archive.
for attempt in 1 2 3 4 5 6; do
  curl -fsSL --retry 5 --retry-delay 5 -C - -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip && break
  echo "download attempt $attempt incomplete; resuming"; sleep 5
done
uv run --locked python docker/prepare_data_portable.py
git diff --quiet -- data/manifest.json || { echo "prepared data differ from the frozen artifact"; exit 1; }
uv run --locked silentrank init > /dev/null

CASES=$(grep -v '^#' /bundle/cases.txt | tr '\n' ' ')
echo "== prepare $(date -u +%T)"
uv run --locked silentrank env prepare-cases $CASES > "$OUT/prepare.json" || echo "prepare reported failures"

for case in $CASES; do
  echo "== $case $(date -u +%T)"
  uv run --locked silentrank run --case "$case" > "$OUT/run_$case.json" 2>&1 || echo "run $case exit=$?"
  uv run --locked silentrank verdict --case "$case" > /dev/null 2>&1 || true
  if [ "${SILENTRANK_KEEP_ARRAYS:-0}" = "1" ]; then
    # Raw arrays of this case, exported as soon as it finishes (protocol/cloud_reruns.md).
    uv run --locked python docker/export_arrays.py --case "$case" --out "$OUT/arrays" || echo "export $case exit=$?"
    mkdir -p "$OUT/results" && cp results/ledger.duckdb "$OUT/results/ledger.partial.duckdb" 2>/dev/null || true
  fi
done
uv run --locked silentrank report pilot > /dev/null 2>&1 || true

echo "== copy results $(date -u +%T)"
mkdir -p "$OUT/results/runs"
cp results/ledger.duckdb results/pilot_summary.md "$OUT/results/" 2>/dev/null || true
cp -r results/compare results/metrics "$OUT/results/" 2>/dev/null || true
for run in results/runs/*/; do
  name=$(basename "$run"); mkdir -p "$OUT/results/runs/$name"
  cp "$run"meta.json "$run"stage_hashes.json "$run"host.json "$run"worker_meta.json "$run"device.json "$OUT/results/runs/$name/" 2>/dev/null || true
done
echo "== done $(date -u +%FT%TZ)"
