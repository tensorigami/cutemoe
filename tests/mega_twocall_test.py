# Minimal repro: the sweep's failing routing (some-experts-empty -> Ms=[6,0]) with the
# FULL dispatch path + a_base_ptrs, on 2 GPUs. M=0 alone is fine (mega_m0_test passes
# [6,0] at num_dispatch=0); this isolates M=0 *combined with* real dispatch / inbox
# aliasing. Run with CUDA_LAUNCH_BLOCKING=1 to surface the exact faulting launch.
#
# Routing: every token (3/GPU) -> e0,e2.  e0 gets 6, e1 gets 0, e2 gets 6, e3 gets 0.
#   G0 owns e0,e1 -> Ms=[6,0];  G1 owns e2,e3 -> Ms=[6,0].
#
# run:  CUDA_LAUNCH_BLOCKING=1 mpirun -np 2 -x CUDA_LAUNCH_BLOCKING -x NVSHMEM_REMOTE_TRANSPORT=none \
#         -x PATH -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_m0disp_test.py

import cutlass
import mega_kernel
import nvshmem.core as nv
import nvshmem.core.interop.torch as nvt
import play
import torch
from cuda.core import Device
from cutlass.cute.runtime import from_dlpack
from mpi4py import MPI

GPUS, EXPERTS, TOP_K = 2, 4, 2
EPR = EXPERTS // GPUS
K, Nint = 256, 256
STATE = {0: {0: [0, 1], 1: [1, 2]}, 1: {0: [0, 3], 1: [1, 2]}}  # baseline, no M=0
MAXTOK = max(len(t) for t in STATE.values())
TOTAL_COPIES = MAXTOK * TOP_K
NUM_DISPATCH = 8
ELEM = 2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
assert comm.Get_size() == GPUS
dev = Device(rank)
dev.set_current()
nv.init(device=dev, mpi_comm=comm, initializer_method="mpi")

inbox = play.build_inbox(GPUS, EXPERTS, STATE)
scatter = play.build_dispatch_tables(GPUS, EXPERTS, inbox, STATE)
schedules = {g: play.build_gemm_schedule(inbox[g], 128) for g in range(GPUS)}
_A, bar_gold = play.run_dispatch(GPUS, EXPERTS, scatter, inbox, STATE)
golden = {g: schedules[g]["a_rows"] for g in range(GPUS)}
n_slots = {g: len(golden[g]) for g in range(GPUS)}
MAX_SLOTS = max(max(n_slots.values()), 1)
my_experts = sorted(inbox[rank].keys())
Ms = [len(inbox[rank][e]) for e in my_experts]
erb = schedules[rank]["expert_row_begin"]
row_begins = [erb[e] for e in my_experts]

slot_of = {(tok, e): slot for (tok, e, dest, slot) in scatter[rank]}
topk_flat = [-1] * TOTAL_COPIES
scat_flat = [-1] * TOTAL_COPIES
splits = [0] * EXPERTS
for tok, experts in STATE[rank].items():
    for k, e in enumerate(experts):
        so = tok * TOP_K + k
        topk_flat[so], scat_flat[so] = e, slot_of[(tok, e)]
        splits[e] += 1


def code(sender, token):
    return float(sender * MAXTOK + token + 1)


in_t = nvt.tensor((MAXTOK, K), dtype=torch.float16)
out_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.float16)
bar_t = nvt.tensor((EPR * GPUS, 1), dtype=torch.int64)
for tok in range(MAXTOK):
    in_t[tok].fill_(code(rank, tok))
out_t.zero_()
bar_t.zero_()
topk_t = torch.tensor(topk_flat, dtype=torch.int32, device="cuda")
scat_t = torch.tensor(scat_flat, dtype=torch.int32, device="cuda")
splits_t = torch.tensor(splits, dtype=torch.int32, device="cuda")
ctr_t = torch.zeros((EXPERTS, 1), dtype=torch.int32, device="cuda")
task_t = torch.zeros(1, dtype=torch.int32, device="cuda")
torch.cuda.synchronize()
comm.Barrier()

in_c, out_c, bar_c = from_dlpack(in_t), from_dlpack(out_t), from_dlpack(bar_t)
topk_c, scat_c, splits_c = from_dlpack(topk_t), from_dlpack(scat_t), from_dlpack(splits_t)
ctr_c, task_c = from_dlpack(ctr_t), from_dlpack(task_t)
a_base_ptrs = [out_t.data_ptr() + rb * K * ELEM for rb in row_begins]

print(f"[G{rank}] experts={my_experts} Ms={Ms} (calling run TWICE in one process)", flush=True)
for _call in range(2):
    out_t.zero_()
    bar_t.zero_()
    ctr_t.zero_()
    task_t.zero_()
    torch.cuda.synchronize()
    comm.Barrier()
    print(f"[G{rank}] --- run() call {_call} ---", flush=True)
    mega_kernel.run(
        num_groups=EPR,
        problem_sizes_mnkl=[(m, Nint, K, 1) for m in Ms],
        a_dtype=cutlass.Float16,
        b_dtype=cutlass.Float16,
        c_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        a_major="k",
        b_major="k",
        c_major="n",
        tile_shape_mn=(128, 256),
        cluster_shape_mn=(1, 1),
        tolerance=1e-1,
        iterations=0,
        skip_ref_check=False,
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
    )
    torch.cuda.synchronize()
    comm.Barrier()
    print(f"[G{rank}] survived call {_call}", flush=True)
nvt.free_tensor(in_t)
nvt.free_tensor(out_t)
nvt.free_tensor(bar_t)
nv.finalize()
