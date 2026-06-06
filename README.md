# cutemoe

**A single CuteDSL kernel that overlaps the two cross-GPU steps of a Mixture-of-Experts layer: routing each token to the GPUs that hold its experts, and the expert matmuls.**

~**300 TFLOP/s/GPU**, bf16, on 8×H100.

---

## What is this?

A [Mixture-of-Experts](https://en.wikipedia.org/wiki/Mixture_of_experts) (MoE) layer swaps one big feed-forward block for many small ones — the "experts" — and sends each token through only a few of them. Many more parameters, roughly the same compute per token.

To train a large MoE, you spread those experts across many GPUs.

That creates a problem: a token sitting on GPU 0 might be routed to experts that live on GPU 3 and GPU 5.

So every training step has two phases:

**1. Dispatch** *(networking)* — each GPU sends every token to the GPUs that hold its experts. A cross-GPU all-to-all over NVLink.

**2. Expert GEMM** *(compute)* — each GPU does a batched matmul over the tokens it just received: one matmul per expert, packed back to back — a *grouped GEMM*.

Do them as two separate kernels and you pay for both in series: `comm + compute`.

**This fuses them into one kernel.** The moment a token lands, the GPU can start its matmul — so the networking happens *underneath* the math instead of before it. The cost drops toward `max(comm, compute)`.

That overlap is the idea in the **UniEP** paper ([arXiv:2604.19241](https://arxiv.org/abs/2604.19241)).

This repo is a faithful [CuteDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html) port of UniEP's dispatch+GEMM kernel (originally written in Triton — see [References](#references)).

---

## Results

| metric | number |
|---|---|
| throughput | **~300 TFLOP/s/GPU** (bf16, 8×H100, 128 experts, top-8, hidden 2048, N 1536) |
| GEMM-path ceiling | ~560 TF/s (dispatch off — the headroom a deeper overlap would chase) |
| correctness | exact-integer vs an oracle + tolerance vs torch, across fp16/bf16, 15 routings, and the full 8-GPU shape |

Every run is checked two ways:

- **integer-exact vs `play.py`** — for the index logic (which token lands in which row, which "bells" fire).
- **tolerance vs torch** — for the GEMM numerics (`C == A_inbox @ B`).

`play.py` is the whole pipeline written out in plain Python. It's the source of truth the kernel is graded against.

---

## How it works

One **persistent kernel**. Every CTA pulls a work *ticket* from a single global atomic counter:

```
task = atomic_add(counter, 1)
total = num_dispatch + num_gemm_tiles
while task < total:
    if task < num_dispatch:  dispatch_tile(task)            # comm
    else:                    gemm_tile(task - num_dispatch)  # compute
    task = atomic_add(counter, 1)
```

Two "crews" share the grid, coupled **only** by symmetric-memory flags ("bells"):

- **dispatch (comm)** — `put`s each token copy to the GPU that owns its expert, then rings that expert's bell.
- **grouped GEMM (compute)** — waits on an expert's bells, then matmuls the rows it received against the expert weights.

A token's GEMM can't start until its tokens have arrived — so the bell is the one handshake between comm and compute.

The pieces that make it correct and fast (all in `mega_kernel.py`):

- **whole-CTA dispatch** — every warp of the CTA issues puts, so the communication runs at full bandwidth instead of bottlenecking the GEMM. The main perf lever.
- **dynamic ticket schedule** — one atomic counter feeds both crews; a shared-memory slot + a named barrier keep the load warp and the MMA warps on the same ticket.
- **bf16** — Hopper does bf16 natively; the dispatch moves raw bytes (it's dtype-agnostic), so it just works once the GEMM dtype gate allows it.

### Performance

End to end: **~300 TF/s/GPU**.

With dispatch turned off (GEMM only), the same kernel runs at **~560 TF/s** — so the GEMM isn't the bottleneck; the gap is communication that isn't fully hidden yet.

### Two-stage (this branch)

The one-stage kernel sends each (token, expert) copy as its own network put. The reference also has a **two-stage** path (`NUM_TAIL_SMS > 0`): send each token to a destination GPU only *once* (dedup) into a staging buffer, then have receiver-side "tail" workers copy it locally into each of its experts' rows. With top-8 over 8 GPUs a token hits ~5.25 *distinct* GPUs on average, so deduping same-GPU copies removes **~34% of the network sends**.

This branch implements it, faithful to the reference:

- a deduped **SEND** stage — a per-rank `d_rank` table sends each (token, dest-GPU) once, then a value-carrying acquire/release handshake records the landing slot in the receiver's indirection table;
- a **COPY** stage — `num_tail` tail workers spin on the handshake, replicate staging → final locally, and ring the bells the GEMM waits on.

`num_tail` is the COPY-worker count (a `constexpr` knob, like the reference's `NUM_TAIL_SMS`).

**Correctness.** Two-stage produces the *identical* final layout to one-stage, so the same oracle applies — verified against `play.py` across the e2e check, the 15-routing sweep (empty experts, dropped tokens, idle ranks), and the full 8-GPU real shape, in fp16 and bf16.

**Performance — an honest tradeoff.** At the real shape two-stage runs **~184 TF/s/GPU** (best `num_tail`) versus one-stage's ~300. This isn't a structural shortfall: the COPY stage is drawn from the *same* dynamic ticket pool as in the reference (its `NUM_TAIL_SMS` names tail *tickets*, not dedicated SMs — `if task_id < num_dispatch: dispatch_two_stage(...)`, then `pid >= num_pid - num_tail` does the COPY). Two-stage simply does **more total work** — SEND + a local COPY + GEMM, versus one-stage's SEND + GEMM — and only pays off when the layer is **communication-bound** (NVLink-saturated at production scale), where deduping ~34% of the sends wins back more than the local copy costs. At this shape it doesn't, and the reference's two-stage carries the same tradeoff. So this is the faithful two-stage: correct, structurally matched, with the perf characterization that says *use it when you're comm-bound, not otherwise*.

---

## Run it

You need **one node of 8× H100** (NVLink), CUDA 12, Python 3.11, and MPI.

**1. Install the pinned environment** (versions matter — see `pyproject.toml`):

```bash
uv sync -p 3.11 --extra cute --extra torch --no-install-project
```

**2. Run the whole suite:**

```bash
bash run_tests.sh
```

That brings up MPI + NVSHMEM and runs every test, printing `PASS`/`FAIL` and the throughput.

**Run a single test** (e.g. the end-to-end check on 2 GPUs):

```bash
NVL=$PWD/.venv/lib/python3.11/site-packages/nvidia/nvshmem/lib
mpirun -np 2 -x PYTHONPATH=$PWD -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
  -x LD_LIBRARY_PATH=$NVL .venv/bin/python tests/mega_d_test.py
```

### The tests

| test | GPUs | what it checks |
|------|------|----------------|
| `mega_d_test.py` | 2 | end-to-end, fp16 |
| `mega_bf16_test.py` | 2 | end-to-end, bf16 |
| `mega_sweep_test.py` | 2 | 15 randomized routings (empty experts, dropped tokens, …) |
| `mega_weight_test.py` | 2 | the optional gating-weight transfer |
| `mega_scale_test.py` | 8 | the real architecture: 128 experts, top-8, K2048/N1536, bf16 |
| `mega_perf_test.py` | 8 | throughput (`GEMM_ONLY=1` for the ~560 diagnostic; `NTOK=` to resize) |
| `mega_twocall_test.py`, `mega_m0_test.py` | 2, 1 | regressions |

---

## Layout

```
mega_kernel.py    the fused kernel + its host driver
play.py           the plain-Python oracle the kernel is graded against
tests/            the suite above
run_tests.sh      runs everything
pyproject.toml    the pinned environment
uv.lock
```

---

## References

A re-implementation — the algorithm and structure aren't ours. Please cite the originals:

- **UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training** — the work this kernel comes from. [arXiv:2604.19241](https://arxiv.org/abs/2604.19241)
- **Triton-distributed** (ByteDance-Seed) — reference implementation: [`mega_kernel_dispatch_token_moe_grouped_gemm`](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/ep_all2all_fused.py). [repo](https://github.com/ByteDance-Seed/Triton-distributed) · [paper](https://arxiv.org/abs/2504.19442)
- **[CuteDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)** — the GPU kernel DSL this is written in; the GEMM machinery starts from its Hopper grouped-GEMM example. See `LICENSE`.

## License

BSD-3-Clause. The GEMM machinery derives from NVIDIA's CUTLASS example (copyright preserved in `mega_kernel.py` and `LICENSE`); the fusion is added under the same license.
