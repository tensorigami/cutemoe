# cutemoe

A single fused, persistent CUDA kernel — written in **CuteDSL** (`cutlass.cute`) — that does
**MoE token dispatch (all-to-all over NVSHMEM) + the up-projection grouped GEMM** in one launch,
with the communication hidden under the compute.

It's a faithful CuteDSL port of the fused expert-parallel mega-kernel from **UniEP** /
**triton-distributed** (Triton), built on NVIDIA's CUTLASS Hopper warp-specialized grouped-GEMM
example. On an 8×H100 node at the Qwen3-30B-A3B shape (hidden 2048, 128 experts, top-8, N=1536, bf16)
it reaches **~300 TFLOP/s/GPU**.

> **Why fuse dispatch and GEMM?** In an expert-parallel MoE layer, every GPU must *communicate*
> (scatter each token's copies to the GPUs owning its top-k experts over NVLink) and *compute* (the
> grouped GEMM over the gathered tokens). Done as two kernels, that's `time = T_comm + T_gemm`. Fusing
> them into one persistent kernel lets the comm hide under the compute, approaching `max(T_comm, T_gemm)`.

---

## Results

| | |
|---|---|
| **Throughput** | **~300 TFLOP/s/GPU** — bf16, 8×H100, real Qwen3-30B-A3B shape |
| **Correctness** | exact-integer vs an executable oracle (`play.py`) + tolerance vs torch, across fp16 + bf16, 15 randomized routings, and the full 8-GPU architecture |
| GEMM-path ceiling | ~560 TF/s (dispatch off) — the headroom a more aggressive comm/compute overlap would chase |

The two-granularity check is the methodology: **integer-exact vs `play.py`** for everything
index-shaped (which token-copy lands in which inbox row, which bells fire), and **tolerance vs torch**
for the GEMM numerics (`C == A_inbox @ B`).

---

## How it works

The kernel is one **persistent grid** whose CTAs pull work *tickets* from a single global atomic
counter (the reference's dynamic schedule):

```
task_id = atomic_add(counter, 1)
total   = num_dispatch + num_gemm_tiles
while task_id < total:
    if task_id < num_dispatch:  dispatch_tile(task_id)        # Crew A — comm
    else:                       gemm_tile(task_id - num_dispatch)  # Crew B — compute
    task_id = atomic_add(counter, 1)
```

Two "crews" share the grid, coupled **only** through symmetric-memory flags ("bells"):

- **Crew A — dispatch (comm).** Each dispatch tile strides over its share of the token-copy space and
  `put_warp`s each copy to the GPU that owns the routed expert, into a deterministic inbox slot. When
  this rank has delivered its last copy for an expert, it `fence`s and rings that expert's **bell**.
- **Crew B — grouped GEMM (compute).** The stock CUTLASS Hopper warp-specialized grouped GEMM
  (TMA + WGMMA), with one injected `signal_wait` at each group boundary: before loading expert *g*'s
  rows, wait until **every** sender has rung *g*'s bells, then matmul the dispatched rows by the
  expert weights.

`play.py` is the same pipeline written executably in pure Python (it's the oracle the kernel is
checked against): `build_inbox`/`build_dispatch_tables` produce the golden slots, `run_dispatch` is
Crew A, `run_gemm` is Crew B.

### The pieces that make it work (see `mega_kernel.py`)

- **Ticket → (group, tile) decode** (`get_work_for_ticket`). The stock scheduler walks an internal
  linear index; we decode an externally supplied ticket instead, and clamp the group-search z-index
  to 0 on out-of-range tickets so the search can't spin forever as the persistent loop drains.
- **One atomic, two warp groups agree** (`ticket_smem` + `ticket_bar`). The load warp's leader grabs
  the ticket and broadcasts it to the MMA warp group through shared memory + a named barrier. The
  barrier's participant count is `32 + num_mma_threads` (the load warp + the MMA warps) — *not* the
  whole CTA; the idle DMA warps must never reach it.
- **Whole-CTA dispatch.** Faithful to the reference's `num_warps`-wide dispatch tile, **every**
  participating warp issues puts — the load warp *and* every MMA warp, not just the load warp. Copies
  partition across (ticket, warp) by `ticket*n_disp_warps + warp_offset` (load = 0, MMA warp *k* =
  1+*k*), so each copy is issued exactly once. The runtime-atomic bell still fires exactly once per
  (expert, rank) — whoever issues that expert's last copy rings it — and `fence()` is PE-scoped, so it
  orders all of this PE's puts before the signal. **This is the single biggest perf lever** (see below).
- **bf16.** Hopper WGMMA does bf16 natively (same 16-bit atoms/layout as fp16). The nvshmem cute RMA
  has no bf16 entry in its dtype table, so the caller hands the dispatch an fp16 *view* of the bf16
  buffers — pure byte movement (the reference's `putmem` is byte-count-based); the GEMM reads them as
  bf16. See `tests/mega_bf16_test.py`.

### The perf story: 151 → ~300

The first end-to-end number was **151 TF/s/GPU**. A diagnostic — run the same kernel with dispatch
off (`GEMM_ONLY=1`: `num_dispatch=0`, pre-set bells) — measured the GEMM path alone at **~560 TF/s**.
So the GEMM and the ticket barrier were healthy; the whole gap was the **dispatch comm**, which was
under-parallelized: dispatch ran on a single warp per tile (~142 GB/s vs ~450 GB/s NVLink).

The reference uses the whole CTA. Restoring that — dispatch inlined across the load warp *and* all MMA
warps — cut the comm from ~0.99 ms to ~0.31 ms and took the end-to-end number **151 → ~300 TF/s/GPU**.

### Scope: one-stage vs two-stage

This implements the reference's **one-stage** path (`NUM_TAIL_SMS == 0`): dispatch and GEMM share the
ticket pool, dispatch sends one network put per *(token, expert)* copy. The reference's **two-stage**
path (`NUM_TAIL_SMS > 0`) adds a receiver-side *copy* stage: tokens are sent **once per destination
GPU** (deduped) into a staging buffer, and dedicated SMs then locally replicate them into each
expert's rows. That trades some SMs for **~34% less NVLink traffic** (a token's 8 copies hit ~5.25
distinct GPUs on average) plus a deeper send→copy→GEMM pipeline. It's the natural next step for more
throughput; not implemented here.

---

## Install

Single node of **8× H100** (NVLink), CUDA 12, Python 3.11. The exact stack is pinned in
`pyproject.toml` / `uv.lock`:

```bash
uv sync -p 3.11 --extra cute --extra torch --no-install-project
```

Key pins (these specific versions matter): `nvshmem4py-cu12 == 0.3.0` (has the device-side cute
bindings), `cuda-core 1.0.1`, `nvidia-cutlass-dsl == 4.4.2`, `nvidia-nvshmem-cu12 3.6.5`, `torch 2.8.0`,
`mpi4py`.

## Run the tests

```bash
bash run_tests.sh
```

Brings up MPI + NVSHMEM and runs each test, checking vs `play.py` (indices) and torch (numerics).

| test | GPUs | checks |
|------|------|--------|
| `mega_d_test.py` | 2 | end-to-end fp16: dispatch → bell → GEMM-on-the-dispatched-inbox |
| `mega_bf16_test.py` | 2 | same, bf16 |
| `mega_sweep_test.py` | 2 | 15 randomized routings (0-token experts, dropped tokens, max overlap, …) |
| `mega_weight_test.py` | 2 | the optional gating-weight transfer (`HAS_WEIGHT`) |
| `mega_scale_test.py` | 8 | the real architecture: 128 experts/16-per-GPU, top-8, K2048/N1536, bf16 |
| `mega_perf_test.py` | 8 | throughput (~300 TF/s; `GEMM_ONLY=1` for the ~560 diagnostic, `NTOK=` for size) |
| `mega_twocall_test.py`, `mega_m0_test.py` | 8 / 1 | regressions (driver idempotency; 0-token-expert GEMM groups) |

To run one test directly: `PYTHONPATH=. mpirun -np 2 -x PYTHONPATH … python tests/mega_d_test.py`
(see `run_tests.sh` for the full NVSHMEM env).

## Layout

```
mega_kernel.py      the fused kernel + its host driver run()
play.py             the executable oracle / spec the kernel is checked against
tests/              the suite (above)
run_tests.sh        runs the suite
pyproject.toml/uv.lock   the exact pinned environment
```

---

## References

This is a re-implementation; the algorithm and structure are not ours. Please cite the originals:

- **UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training** — the work this fused
  dispatch+GEMM mega-kernel comes from (it fuses the MoE-specific sub-graphs Dispatch+GroupGEMM and
  GroupGEMM+Combine, with a deterministic token-ordering scheme for numerical consistency under
  overlap). https://arxiv.org/abs/2604.19241
- **Triton-distributed** (ByteDance-Seed) — the Triton implementation we ported, specifically the
  kernel `mega_kernel_dispatch_token_moe_grouped_gemm` in `ep_all2all_fused.py`.
  https://github.com/ByteDance-Seed/Triton-distributed —
  paper: https://arxiv.org/abs/2504.19442
- **NVIDIA CUTLASS** — the Hopper warp-specialized grouped-GEMM example this kernel is built on (the
  GEMM machinery, TMA/WGMMA pipelines, and tile scheduler are upstream). https://github.com/NVIDIA/cutlass
  — see `LICENSE`.

## License

BSD-3-Clause. The GEMM machinery derives from NVIDIA's CUTLASS example (BSD-3-Clause; copyright
retained in `mega_kernel.py` and `LICENSE`); the fusion (dispatch, ticketing, bells, bf16 path) is
added under the same license.
