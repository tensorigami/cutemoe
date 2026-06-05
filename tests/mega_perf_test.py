# Perf: the full fused kernel at the real Qwen3-30B-A3B shape, bf16, 8 GPUs. Reports
# gemm TFLOP/s over the FUSED wall-time (dispatch hidden under compute => approaches the
# gemm-solo number if comm is well hidden). Compile-once, launch-many (run(bench_iters=...)).
#
# Per timed launch only the ticket COUNTER is reset (reset_fn); bells stay set after warmup
# so there's no cross-rank lost-signal race, and dispatch still does all its puts each launch
# (the put precedes the ring), so comm IS in the measured time. The bell-wait stall (if any)
# is paid once in warmup, not timed.
#
# NTOK is tokens/GPU (env NTOK, default 4096 — already saturates: M_e~2048 -> 16 M-tiles
# x 6 N-tiles x 16 experts >> 132 SMs). Real workload is 65536; set NTOK=65536 to confirm.
#
# run:  NTOK=4096 mpirun -np 8 -x NTOK -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
#         -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_perf_test.py

import os
import random

import cutlass
import mega_kernel
import nvshmem.core as nv
import nvshmem.core.interop.torch as nvt
import play
import torch
from cuda.core import Device
from cutlass.cute.runtime import from_dlpack
from mpi4py import MPI

GPUS, EXPERTS, TOP_K = 8, 128, 8
EPR = EXPERTS // GPUS
K, Nint = 2048, 1536
NTOK = int(os.environ.get("NTOK", "4096"))
NUM_DISPATCH = 132
BENCH_ITERS = int(os.environ.get("BENCH_ITERS", "30"))
BENCH_WARMUP = int(os.environ.get("BENCH_WARMUP", "8"))
# diagnostic: GEMM_ONLY=1 sets num_dispatch=0 + pre-set bells, so the kernel runs ONLY
# gemm tickets through the same per-tile ticket loop. Isolates the ticket-barrier cost
# (vs the dispatch comm). If gemm-only ~= fused, the barrier is the wall, not the comm.
GEMM_ONLY = os.environ.get("GEMM_ONLY", "0") == "1"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
assert comm.Get_size() == GPUS
dev = Device(rank)
dev.set_current()
nv.init(device=dev, mpi_comm=comm, initializer_method="mpi")


def gen_state(rng):
    for _ in range(100):
        st = {g: {t: rng.sample(range(EXPERTS), TOP_K) for t in range(NTOK)} for g in range(GPUS)}
        load = [0] * GPUS
        for g in st:
            for t, es in st[g].items():
                for e in es:
                    load[e // EPR] += 1
        if all(c > 0 for c in load):
            return st
    raise RuntimeError("no valid routing")


if rank == 0:
    print(f"building routing tables (NTOK={NTOK}/GPU)...", flush=True)
STATE = gen_state(random.Random(7))
inbox = play.build_inbox(GPUS, EXPERTS, STATE)
scatter = play.build_dispatch_tables(GPUS, EXPERTS, inbox, STATE)
_, schedules = play.make_tables(GPUS, EXPERTS, TOP_K, 128, STATE)
n_slots = {g: len(schedules[g]["a_rows"]) for g in range(GPUS)}
MAX_SLOTS = max(n_slots.values())
my_experts = sorted(inbox[rank].keys())
Ms = [len(inbox[rank][e]) for e in my_experts]

slot_of = {(tok, e): slot for (tok, e, dest, slot) in scatter[rank]}
TOTAL_COPIES = NTOK * TOP_K
topk_flat = [-1] * TOTAL_COPIES
scat_flat = [-1] * TOTAL_COPIES
splits = [0] * EXPERTS
for tok, experts in STATE[rank].items():
    for k, e in enumerate(experts):
        so = tok * TOP_K + k
        topk_flat[so], scat_flat[so] = e, slot_of[(tok, e)]
        splits[e] += 1

in_t = nvt.tensor((NTOK, K), dtype=torch.bfloat16)
out_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.bfloat16)
bar_t = nvt.tensor((EPR * GPUS, 1), dtype=torch.int64)
in_t.fill_(1.0)
out_t.zero_()
bar_t.fill_(1) if GEMM_ONLY else bar_t.zero_()   # GEMM_ONLY: bells pre-set, no dispatch
topk_t = torch.tensor(topk_flat, dtype=torch.int32, device="cuda")
scat_t = torch.tensor(scat_flat, dtype=torch.int32, device="cuda")
splits_t = torch.tensor(splits, dtype=torch.int32, device="cuda")
ctr_t = torch.zeros((EXPERTS, 1), dtype=torch.int32, device="cuda")
task_t = torch.zeros(1, dtype=torch.int32, device="cuda")
torch.cuda.synchronize()
comm.Barrier()

in_c = from_dlpack(in_t.view(torch.float16))  # bf16 dispatch via fp16 view (byte move)
out_c = from_dlpack(out_t.view(torch.float16))
bar_c = from_dlpack(bar_t)
topk_c, scat_c, splits_c = from_dlpack(topk_t), from_dlpack(scat_t), from_dlpack(splits_t)
ctr_c, task_c = from_dlpack(ctr_t), from_dlpack(task_t)

tot_fmas = sum(m * Nint * K for m in Ms)
print(
    f"[G{rank}] perf: sum(M)={sum(Ms)} gemm={2 * tot_fmas / 1e9:.1f} GFLOP/GPU/launch", flush=True
)
comm.Barrier()

res = mega_kernel.run(
    num_groups=EPR,
    problem_sizes_mnkl=[(m, Nint, K, 1) for m in Ms],
    a_dtype=cutlass.BFloat16,
    b_dtype=cutlass.BFloat16,
    c_dtype=cutlass.BFloat16,
    acc_dtype=cutlass.Float32,
    a_major="k",
    b_major="k",
    c_major="n",
    tile_shape_mn=(128, 256),
    cluster_shape_mn=(1, 1),
    tolerance=1e-1,
    iterations=0,
    skip_ref_check=True,
    barriers_cute=bar_c,
    task_counter_cute=task_c,
    world_size=GPUS,
    num_dispatch=0 if GEMM_ONLY else NUM_DISPATCH,
    d_topk_cute=topk_c,
    d_scatter_cute=scat_c,
    d_splits_cute=splits_c,
    d_bellctr_cute=ctr_c,
    d_input_cute=in_c,
    d_output_cute=out_c,
    num_experts=EXPERTS,
    top_k=TOP_K,
    bench_iters=BENCH_ITERS,
    bench_warmup=BENCH_WARMUP,
    reset_fn=lambda: task_t.zero_(),  # counter-only reset (race-free across ranks)
)
torch.cuda.synchronize()
comm.Barrier()
if res is not None:
    ms, tf = res
    print(
        f"[G{rank}] RESULT: {ms:.4f} ms/launch  {tf:.1f} TFLOP/s/GPU (gemm, fused wall-time)",
        flush=True,
    )

comm.Barrier()
nvt.free_tensor(in_t)
nvt.free_tensor(out_t)
nvt.free_tensor(bar_t)
nv.finalize()
