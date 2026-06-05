#!/usr/bin/env bash
# Test suite for the fused MoE kernel. Requires a single node of 8× H100 (NVLink), the
# environment from pyproject.toml (`uv sync`), and MPI. From the repo root:
#     bash run_tests.sh
#
# Each test brings up MPI + NVSHMEM, imports `mega_kernel` (this repo's kernel) and `play`
# (the oracle), and checks vs play.py (indices) + torch (numerics). Hang-prone fused
# launches run under `timeout`.
set -u
cd "$(dirname "$0")"
ROOT="$PWD"
NVL="$ROOT/.venv/lib/python3.11/site-packages/nvidia/nvshmem/lib"
PY=".venv/bin/python"

# run <n_gpus> <test.py> [extra -x env...]
run() {
  local np="$1"; local f="$2"; shift 2
  echo ""; echo "──────────────────────── $f  (np=$np) ────────────────────────"
  timeout 700 mpirun -np "$np" \
    -x PYTHONPATH="$ROOT" -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
    -x LD_LIBRARY_PATH="$NVL:${LD_LIBRARY_PATH:-}" \
    "$@" "$PY" "$f" 2>&1 \
    | grep -E "PASS|FAIL|RESULT|SWEEP|inbox=|weights=|->|illegal|Error|Traceback" \
    || echo "  (no matching output — check the raw run)"
}

echo "### correctness ###"
run 2 tests/mega_d_test.py        # end-to-end fp16
run 2 tests/mega_bf16_test.py     # end-to-end bf16
run 2 tests/mega_sweep_test.py    # 15 randomized routings vs play.py
run 2 tests/mega_weight_test.py   # gating-weight transfer (HAS_WEIGHT)
run 8 tests/mega_scale_test.py    # the real architecture, 8 GPUs

echo ""; echo "### performance ###"
run 8 tests/mega_perf_test.py -x NTOK=4096                 # ~300 TF/s/GPU
run 8 tests/mega_perf_test.py -x NTOK=4096 -x GEMM_ONLY=1  # diagnostic: gemm-only ~560 TF/s

echo ""; echo "### regressions ###"
run 2 tests/mega_twocall_test.py  # host-driver idempotency across repeated run() calls (2 GPUs)
run 1 tests/mega_m0_test.py       # 0-token-expert GEMM groups (pass argv 0-3 for other cases)

echo ""; echo "### done ###"
