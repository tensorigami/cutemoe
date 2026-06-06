# Two-stage perf at the real Qwen3-30B-A3B shape, bf16, 8 GPUs. Same harness as
# mega_perf_test.py but with the two-stage path (num_tail>0): deduped SEND to staging +
# tail COPY workers that replicate into the per-expert rows. Sweep NUM_TAIL to balance
# SEND/COPY/GEMM. Target: beat the one-stage ~300, climb toward the ~560 gemm-only ceiling.
#
# Per timed launch the ticket COUNTER and the local dedup table d_rank are reset (reset_fn,
# both local => race-free); d_stage/d_indirect are rewritten by each launch's SEND, so the
# full send+copy work is in the measured time. Bells/indirect spin is paid once in warmup.
#
# run:  NTOK=4096 NUM_TAIL=16 mpirun -np 8 -x NTOK -x NUM_TAIL -x NVSHMEM_REMOTE_TRANSPORT=none \
#         -x PATH -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_twostage_perf_test.py

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
NUM_TAIL = int(os.environ.get("NUM_TAIL", "16"))
BENCH_ITERS = int(os.environ.get("BENCH_ITERS", "30"))
BENCH_WARMUP = int(os.environ.get("BENCH_WARMUP", "8"))

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
    print(f"building routing tables (NTOK={NTOK}/GPU, num_tail={NUM_TAIL})...", flush=True)
STATE = gen_state(random.Random(7))
inbox = play.build_inbox(GPUS, EXPERTS, STATE)
scatter = play.build_dispatch_tables(GPUS, EXPERTS, inbox, STATE)
_, schedules = play.make_tables(GPUS, EXPERTS, TOP_K, 128, STATE)
n_slots = {g: len(schedules[g]["a_rows"]) for g in range(GPUS)}
MAX_SLOTS = max(n_slots.values())
my_experts = sorted(inbox[rank].keys())
Ms = [len(inbox[rank][e]) for e in my_experts]
erb = schedules[rank]["expert_row_begin"]
row_begins = [erb[e] for e in my_experts]

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

ELEM = 2  # bf16 bytes
in_t = nvt.tensor((NTOK, K), dtype=torch.bfloat16)
stage_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.bfloat16)
final_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.bfloat16)
out_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.bfloat16)  # one-stage inbox (unused)
bar_t = nvt.tensor((EPR * GPUS, 1), dtype=torch.int64)
indir_t = nvt.tensor((MAX_SLOTS, 1), dtype=torch.int32)
in_t.fill_(1.0)
stage_t.zero_()
final_t.zero_()
out_t.zero_()
bar_t.zero_()
indir_t.fill_(-1)
rank_t = torch.full((NTOK, GPUS), -1, dtype=torch.int32, device="cuda")
topk_t = torch.tensor(topk_flat, dtype=torch.int32, device="cuda")
scat_t = torch.tensor(scat_flat, dtype=torch.int32, device="cuda")
splits_t = torch.tensor(splits, dtype=torch.int32, device="cuda")
ctr_t = torch.zeros((EXPERTS, 1), dtype=torch.int32, device="cuda")
task_t = torch.zeros(1, dtype=torch.int32, device="cuda")
torch.cuda.synchronize()
comm.Barrier()

# bf16 dispatch via fp16 view (nvshmem rma has no bf16 — pure byte movement)
in_c = from_dlpack(in_t.view(torch.float16))
stage_c = from_dlpack(stage_t.view(torch.float16))
final_c = from_dlpack(final_t.view(torch.float16))
out_c = from_dlpack(out_t.view(torch.float16))
bar_c = from_dlpack(bar_t)
indir_c = from_dlpack(indir_t)
rank_c = from_dlpack(rank_t)
topk_c, scat_c, splits_c = from_dlpack(topk_t), from_dlpack(scat_t), from_dlpack(splits_t)
ctr_c, task_c = from_dlpack(ctr_t), from_dlpack(task_t)

a_base_ptrs = [final_t.data_ptr() + rb * K * ELEM for rb in row_begins]

tot_fmas = sum(m * Nint * K for m in Ms)
print(f"[G{rank}] two-stage perf: sum(M)={sum(Ms)} gemm={2 * tot_fmas / 1e9:.1f} GFLOP/GPU/launch", flush=True)
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
    num_dispatch=NUM_DISPATCH,
    d_topk_cute=topk_c,
    d_scatter_cute=scat_c,
    d_splits_cute=splits_c,
    d_bellctr_cute=ctr_c,
    d_input_cute=in_c,
    d_output_cute=out_c,
    num_experts=EXPERTS,
    top_k=TOP_K,
    a_base_ptrs=a_base_ptrs,
    num_tail=NUM_TAIL,
    d_stage_cute=stage_c,
    d_final_cute=final_c,
    d_rank_cute=rank_c,
    d_indirect_cute=indir_c,
    bench_iters=BENCH_ITERS,
    bench_warmup=BENCH_WARMUP,
    reset_fn=lambda: (task_t.zero_(), rank_t.fill_(-1)),  # both local => race-free
)
torch.cuda.synchronize()
comm.Barrier()
if res is not None:
    ms, tf = res
    print(
        f"[G{rank}] RESULT (num_tail={NUM_TAIL}): {ms:.4f} ms/launch  {tf:.1f} TFLOP/s/GPU "
        f"(gemm, fused wall-time)",
        flush=True,
    )

comm.Barrier()
for t in (in_t, stage_t, final_t, out_t, bar_t, indir_t):
    nvt.free_tensor(t)
nv.finalize()
