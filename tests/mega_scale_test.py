# Real-shape correctness: the full fused kernel at the Qwen3-30B-A3B architecture, bf16,
# on 8 GPUs. hidden K=2048, intermediate N=1536, 128 experts (16/GPU), top-8. A tractable
# token count (play.py is the golden generator in pure Python) — full 65536/GPU is for the
# perf run, not this correctness check. One random routing, dual-checked:
#   inbox == play.py  +  bells == play.py  +  C == torch(inbox @ B)
#
# bf16 dispatch: the nvshmem cute RMA lacks a bf16 entry, so the dispatch token buffers
# are passed as fp16 VIEWS of the bf16 memory (byte movement, ref's putmem is byte-based);
# the gemm reads bf16 via a_base_ptrs, the decode reads bf16.
#
# run:  mpirun -np 8 -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
#         -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_scale_test.py

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

# ── real architecture ────────────────────────────────────────────────────────
GPUS, EXPERTS, TOP_K = 8, 128, 8
EPR = EXPERTS // GPUS  # 16 experts/GPU
K, Nint = 2048, 1536  # hidden, up-proj intermediate (SwiGLU N)
NTOK = 256  # tokens/GPU for the correctness check (perf uses 65536)
NUM_DISPATCH = 64
ELEM = 2  # bf16 bytes

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
assert comm.Get_size() == GPUS, f"need {GPUS} ranks"
dev = Device(rank)
dev.set_current()
nv.init(device=dev, mpi_comm=comm, initializer_method="mpi")


def gen_state(rng):
    # identical across ranks (shared seed); each token -> TOP_K distinct experts;
    # resample until every rank's owned range has load (no all-zero gemm grid).
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


STATE = gen_state(random.Random(2024))

inbox = play.build_inbox(GPUS, EXPERTS, STATE)
scatter = play.build_dispatch_tables(GPUS, EXPERTS, inbox, STATE)
_, schedules = play.make_tables(GPUS, EXPERTS, TOP_K, 128, STATE)
_, bar_gold = play.run_dispatch(GPUS, EXPERTS, scatter, inbox, STATE)
golden = {g: schedules[g]["a_rows"] for g in range(GPUS)}
n_slots = {g: len(golden[g]) for g in range(GPUS)}
MAX_SLOTS = max(n_slots.values())
my_experts = sorted(inbox[rank].keys())  # this rank's 16 experts
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


# Encode identity in 2 columns (both <= 256 -> exact in bf16), fill the rest with the
# token id so the gemm A is non-trivial (C check is meaningful, not ~0):
#   col 0 = token id, col 1 = sender id, cols 2.. = token id.
in_t = nvt.tensor((NTOK, K), dtype=torch.bfloat16)
out_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.bfloat16)
bar_t = nvt.tensor((EPR * GPUS, 1), dtype=torch.int64)
for tok in range(NTOK):
    in_t[tok].fill_(float(tok + 1))  # whole row = token id (<=256, exact)
    in_t[tok, 1].fill_(float(rank + 1))  # col 1 = sender id (<=8, exact)
out_t.zero_()
bar_t.zero_()
topk_t = torch.tensor(topk_flat, dtype=torch.int32, device="cuda")
scat_t = torch.tensor(scat_flat, dtype=torch.int32, device="cuda")
splits_t = torch.tensor(splits, dtype=torch.int32, device="cuda")
ctr_t = torch.zeros((EXPERTS, 1), dtype=torch.int32, device="cuda")
task_t = torch.zeros(1, dtype=torch.int32, device="cuda")
torch.cuda.synchronize()
comm.Barrier()

# bf16 dispatch via fp16 view (byte movement); gemm + decode use bf16
in_c = from_dlpack(in_t.view(torch.float16))
out_c = from_dlpack(out_t.view(torch.float16))
bar_c = from_dlpack(bar_t)
topk_c, scat_c, splits_c = from_dlpack(topk_t), from_dlpack(scat_t), from_dlpack(splits_t)
ctr_c, task_c = from_dlpack(ctr_t), from_dlpack(task_t)
a_base_ptrs = [out_t.data_ptr() + rb * K * ELEM for rb in row_begins]

print(
    f"[G{rank}] scale: {EXPERTS}e/{GPUS}gpu top{TOP_K} K{K} N{Nint} | my Ms sum={sum(Ms)} "
    f"slots={n_slots[rank]} empties={sum(1 for m in Ms if m == 0)}",
    flush=True,
)
bc = mega_kernel.run(
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
got = out_t.cpu().to(torch.float32)
inbox_ok = True
for slot in range(n_slots[rank]):
    tok_id = int(round(got[slot, 0].item())) - 1
    snd_id = int(round(got[slot, 1].item())) - 1
    if (snd_id, tok_id) != golden[rank][slot]:
        inbox_ok = False
        break
bells = bar_t.cpu().flatten()
bell_ok = all(
    int(bells[ei * GPUS + s].item()) == bar_gold[rank][rank * EPR + ei][s]
    for ei in range(EPR)
    for s in range(GPUS)
)
gemm_ok = True
nbad = 0
for g, (rb, m) in enumerate(zip(row_begins, Ms)):
    if m == 0:
        continue
    a_ref = got[rb : rb + m, :].unsqueeze(-1)
    b_ref = mega_kernel._to_reference_operand_fp32(bc[g][1], cutlass.BFloat16)
    ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
    c_got = bc[g][2].cpu().to(torch.float32)
    try:
        torch.testing.assert_close(c_got, ref, atol=8.0, rtol=1e-1)
    except AssertionError:
        gemm_ok = False
        nbad += 1

ok = inbox_ok and bell_ok and gemm_ok
print(
    f"[G{rank}] inbox={'ok' if inbox_ok else 'BAD'} bells={'ok' if bell_ok else 'BAD'} "
    f"C={'ok' if gemm_ok else f'BAD({nbad})'} -> {'PASS' if ok else 'FAIL'}",
    flush=True,
)

comm.Barrier()
nvt.free_tensor(in_t)
nvt.free_tensor(out_t)
nvt.free_tensor(bar_t)
nv.finalize()
