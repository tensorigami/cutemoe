# Step D test: the FULL one-stage fused kernel, end-to-end, on 2 GPUs.
# Dispatch tickets push tokens into the symmetric inbox + ring bells; gemm tickets
# wait on the bells then matmul reading A = THE DISPATCHED INBOX (a_base_ptrs point
# each group's A into the inbox). One kernel: dispatch -> bell -> gemm-on-dispatched-data.
#
# Dual check (the methodology):
#   - inbox (d_output) == play.py golden placement   (integer index logic)
#   - bells            == play.py golden bells         (runtime-atomic gate)
#   - C == torch(inbox @ B)                            (the gemm consumed the dispatched
#                                                       rows, gated by the real bell)
#
# run:  mpirun -np 2 -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
#         -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_d_test.py

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
STATE = {0: {0: [0, 1], 1: [1, 2]}, 1: {0: [0, 3], 1: [1, 2]}}
MAXTOK = max(len(t) for t in STATE.values())
TOTAL_COPIES = MAXTOK * TOP_K
NUM_DISPATCH = 8
ELEM = 2  # fp16 bytes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
assert comm.Get_size() == GPUS
dev = Device(rank)
dev.set_current()
nv.init(device=dev, mpi_comm=comm, initializer_method="mpi")

inbox = play.build_inbox(GPUS, EXPERTS, STATE)
scatter = play.build_dispatch_tables(GPUS, EXPERTS, inbox, STATE)
_, schedules = play.make_tables(GPUS, EXPERTS, TOP_K, 1, STATE)
_, bar_gold = play.run_dispatch(GPUS, EXPERTS, scatter, inbox, STATE)
golden = {g: schedules[g]["a_rows"] for g in range(GPUS)}
n_slots = {g: len(golden[g]) for g in range(GPUS)}
MAX_SLOTS = max(n_slots.values())
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
    # small distinct id (1..GPUS*MAXTOK) so A stays small -> C is fp16-precise for
    # the numeric check, while still uniquely decoding (sender, token) for placement.
    return float(sender * MAXTOK + token + 1)


in_t = nvt.tensor((MAXTOK, K), dtype=torch.float16)
out_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.float16)  # inbox == gemm A
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

# A base ptr for group g = inbox base + expert_row_begin[g] * K elems
a_base_ptrs = [out_t.data_ptr() + rb * K * ELEM for rb in row_begins]

print(f"[G{rank}] D: experts={my_experts} Ms={Ms} row_begins={row_begins}", flush=True)
bc_tensors = mega_kernel.run(
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

# ── checks ──────────────────────────────────────────────────────────────────
got = out_t.cpu()
inbox_ok = all(
    (int(round(got[slot, 0].item())) - 1) // MAXTOK == golden[rank][slot][0]
    and (int(round(got[slot, 0].item())) - 1) % MAXTOK == golden[rank][slot][1]
    for slot in range(n_slots[rank])
)
bells = bar_t.cpu().flatten()
bell_ok = all(
    int(bells[ei * GPUS + s].item()) == bar_gold[rank][rank * EPR + ei][s]
    for ei in range(EPR)
    for s in range(GPUS)
)

# C == torch(inbox @ B): A = the dispatched inbox rows the gemm actually read
gemm_ok = True
for g, (rb, m) in enumerate(zip(row_begins, Ms)):
    a_ref = got[rb : rb + m, :].to(torch.float32).unsqueeze(-1)  # [M,K,1] inbox rows
    b_ref = mega_kernel._to_reference_operand_fp32(bc_tensors[g][1], cutlass.Float16)
    ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
    c_got = bc_tensors[g][2].cpu().to(torch.float32)
    try:
        torch.testing.assert_close(c_got, ref, atol=2e-1, rtol=1e-2)
    except AssertionError:
        gemm_ok = False

ok = inbox_ok and bell_ok and gemm_ok
print(
    f"[G{rank}] inbox={'ok' if inbox_ok else 'BAD'} bells={'ok' if bell_ok else 'BAD'} "
    f"C==torch(inbox@B)={'ok' if gemm_ok else 'BAD'} -> {'PASS' if ok else 'FAIL'}",
    flush=True,
)

comm.Barrier()
nvt.free_tensor(in_t)
nvt.free_tensor(out_t)
nvt.free_tensor(bar_t)
nv.finalize()
