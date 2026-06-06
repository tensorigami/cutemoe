# Two-stage real-shape correctness: the full two-stage kernel at Qwen3-30B-A3B, bf16, 8
# GPUs. K=2048, N=1536, 128 experts (16/GPU), top-8. Deduped SEND to staging + tail COPY.
# play.py is UNCHANGED (final layout identical to one-stage); read d_final.
#   d_final == play.py  +  d_indirect populated  +  bells rung  +  C == torch(d_final @ B)
#
# run:  NUM_TAIL=16 mpirun -np 8 -x NUM_TAIL -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
#         -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_twostage_scale_test.py

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
NTOK = 256
NUM_DISPATCH = 64
NUM_TAIL = int(os.environ.get("NUM_TAIL", "16"))
ELEM = 2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
assert comm.Get_size() == GPUS, f"need {GPUS} ranks"
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


STATE = gen_state(random.Random(2024))
inbox = play.build_inbox(GPUS, EXPERTS, STATE)
scatter = play.build_dispatch_tables(GPUS, EXPERTS, inbox, STATE)
_, schedules = play.make_tables(GPUS, EXPERTS, TOP_K, 128, STATE)
golden = {g: schedules[g]["a_rows"] for g in range(GPUS)}
n_slots = {g: len(golden[g]) for g in range(GPUS)}
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

in_t = nvt.tensor((NTOK, K), dtype=torch.bfloat16)
stage_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.bfloat16)
final_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.bfloat16)
out_t = nvt.tensor((MAX_SLOTS, K), dtype=torch.bfloat16)  # one-stage inbox (unused)
bar_t = nvt.tensor((EPR * GPUS, 1), dtype=torch.int64)
indir_t = nvt.tensor((MAX_SLOTS, 1), dtype=torch.int32)
for tok in range(NTOK):
    in_t[tok].fill_(float(tok + 1))
    in_t[tok, 1].fill_(float(rank + 1))
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

print(
    f"[G{rank}] two-stage scale: {EXPERTS}e/{GPUS}gpu top{TOP_K} K{K} N{Nint} num_tail={NUM_TAIL} | "
    f"my Ms sum={sum(Ms)} slots={n_slots[rank]} empties={sum(1 for m in Ms if m == 0)}",
    flush=True,
)
bc = mega_kernel.run(
    num_groups=EPR,
    problem_sizes_mnkl=[(m, Nint, K, 1) for m in Ms],
    a_dtype=cutlass.BFloat16, b_dtype=cutlass.BFloat16, c_dtype=cutlass.BFloat16,
    acc_dtype=cutlass.Float32, a_major="k", b_major="k", c_major="n",
    tile_shape_mn=(128, 256), cluster_shape_mn=(1, 1), tolerance=1e-1,
    iterations=0, skip_ref_check=False,
    barriers_cute=bar_c, task_counter_cute=task_c, world_size=GPUS,
    num_dispatch=NUM_DISPATCH, d_topk_cute=topk_c, d_scatter_cute=scat_c,
    d_splits_cute=splits_c, d_bellctr_cute=ctr_c, d_input_cute=in_c, d_output_cute=out_c,
    num_experts=EXPERTS, top_k=TOP_K, a_base_ptrs=a_base_ptrs,
    num_tail=NUM_TAIL, d_stage_cute=stage_c, d_final_cute=final_c,
    d_rank_cute=rank_c, d_indirect_cute=indir_c,
)
torch.cuda.synchronize()
comm.Barrier()

got = final_t.cpu().to(torch.float32)
final_ok = True
for slot in range(n_slots[rank]):
    tok_id = int(round(got[slot, 0].item())) - 1
    snd_id = int(round(got[slot, 1].item())) - 1
    if (snd_id, tok_id) != golden[rank][slot]:
        final_ok = False
        break
indir = indir_t.cpu().flatten()
indir_ok = all(int(indir[slot].item()) >= 0 for slot in range(n_slots[rank]))
bells = bar_t.cpu().flatten()
bell_ok = all(int(bells[ei].item()) == 1 for ei in range(EPR))  # 1 bell/expert (two-stage)
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

ok = final_ok and indir_ok and bell_ok and gemm_ok
print(
    f"[G{rank}] final={'ok' if final_ok else 'BAD'} indir={'ok' if indir_ok else 'BAD'} "
    f"bells={'ok' if bell_ok else 'BAD'} C={'ok' if gemm_ok else f'BAD({nbad})'} "
    f"-> {'PASS' if ok else 'FAIL'}",
    flush=True,
)

comm.Barrier()
for t in (in_t, stage_t, final_t, out_t, bar_t, indir_t):
    nvt.free_tensor(t)
nv.finalize()
