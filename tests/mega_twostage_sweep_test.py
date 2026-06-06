# Two-stage randomized routing sweep: the FULL two-stage kernel vs play.py across MANY
# routings. Same battery as mega_sweep_test.py (named edge cases + random draws, incl.
# M=0 experts, dropped -1 tokens, one-rank-idle), but num_tail>0: deduped SEND to staging
# + tail COPY workers. play.py is UNCHANGED — the final layout is identical, so the same
# goldens apply, read from d_final instead of d_output.
#
# Per draw, dual-checked:
#   - d_final == play golden inbox       (SEND+COPY index logic, all routings)
#   - d_indirect populated (no -1)       (every final slot got its landing slot)
#   - bells all rung (== 1)              (COPY rang every owned expert, incl. M=0)
#   - C == torch(d_final @ B)            (gemm consumed the copied rows)
#
# run:  mpirun -np 2 -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
#         -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_twostage_sweep_test.py

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

GPUS, EXPERTS, TOP_K = 2, 4, 2
EPR = EXPERTS // GPUS
K, Nint = 256, 256
MAXTOK = 6
TOTAL_COPIES = MAXTOK * TOP_K
UB_SLOTS = GPUS * MAXTOK * TOP_K
NUM_DISPATCH = 8
NUM_TAIL = 2
BLOCK_M = 128
ELEM = 2
NUM_RANDOM = 10

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
assert comm.Get_size() == GPUS
dev = Device(rank)
dev.set_current()
nv.init(device=dev, mpi_comm=comm, initializer_method="mpi")


def code(sender, token):
    return float(sender * MAXTOK + token + 1)


def _layout_from_choice(choice, drop_set):
    slots = [(-1 if k in drop_set else e) for k, e in enumerate(choice)]
    real = [e for k, e in enumerate(choice) if k not in drop_set]
    return slots, real


def _every_rank_has_load(state_real):
    got = [0] * GPUS
    for g in state_real:
        for tok, experts in state_real[g].items():
            for e in experts:
                got[e // EPR] += 1
    return all(c > 0 for c in got)


def gen_random(rng):
    for _ in range(100):
        state_real, topk_layout = {}, {}
        for g in range(GPUS):
            ntok = rng.randint(1, MAXTOK)
            sr, tl = {}, {}
            for tok in range(ntok):
                choice = rng.sample(range(EXPERTS), TOP_K)
                drop_set = {k for k in range(TOP_K) if rng.random() < 0.2}
                slots, real = _layout_from_choice(choice, drop_set)
                tl[tok], sr[tok] = slots, real
            state_real[g], topk_layout[g] = sr, tl
        if _every_rank_has_load(state_real):
            return state_real, topk_layout
    raise RuntimeError("could not generate a routing with load on every rank")


def named_scenarios():
    out = []
    sr = {0: {0: [0, 1], 1: [1, 2]}, 1: {0: [0, 3], 1: [1, 2]}}
    out.append(("baseline", sr, {g: {t: list(v) for t, v in sr[g].items()} for g in sr}))
    sr = {g: {t: [0, 2] for t in range(3)} for g in range(GPUS)}
    out.append(("some-experts-empty", sr, {g: {t: list(v) for t, v in sr[g].items()} for g in sr}))
    sr = {g: {t: [1, 3] for t in range(MAXTOK)} for g in range(GPUS)}
    out.append(("max-overlap", sr, {g: {t: list(v) for t, v in sr[g].items()} for g in sr}))
    sr, tl = {}, {}
    for g in range(GPUS):
        sr[g], tl[g] = {}, {}
        for t in range(3):
            e0 = t % EXPERTS
            tl[g][t] = [e0, -1]
            sr[g][t] = [e0]
    out.append(("all-2nd-dropped", sr, tl))
    sr = {0: {0: [0, 2], 1: [1, 3]}, 1: {}}
    out.append(("one-rank-idle", sr, {0: {0: [0, 2], 1: [1, 3]}, 1: {}}))
    return out


def build_draw(state_real, topk_layout):
    inbox = play.build_inbox(GPUS, EXPERTS, state_real)
    scatter = play.build_dispatch_tables(GPUS, EXPERTS, inbox, state_real)
    schedules = {g: play.build_gemm_schedule(inbox[g], BLOCK_M) for g in range(GPUS)}
    _A, bar_gold = play.run_dispatch(GPUS, EXPERTS, scatter, inbox, state_real)
    golden = {g: schedules[g]["a_rows"] for g in range(GPUS)}
    my_experts = sorted(inbox[rank].keys())
    Ms = [len(inbox[rank][e]) for e in my_experts]
    erb = schedules[rank]["expert_row_begin"]
    row_begins = [erb[e] for e in my_experts]
    slot_of = {(tok, e): slot for (tok, e, dest, slot) in scatter[rank]}
    topk_flat = [-1] * TOTAL_COPIES
    scat_flat = [-1] * TOTAL_COPIES
    splits = [0] * EXPERTS
    for tok, slots in topk_layout[rank].items():
        for k, e in enumerate(slots):
            so = tok * TOP_K + k
            if e >= 0:
                topk_flat[so] = e
                scat_flat[so] = slot_of[(tok, e)]
                splits[e] += 1
    return dict(
        golden=golden, my_experts=my_experts, Ms=Ms, row_begins=row_begins,
        topk_flat=topk_flat, scat_flat=scat_flat, splits=splits, n_slots=len(golden[rank]),
    )


# ── buffers (upper bound, reused) ─────────────────────────────────────────────
in_t = nvt.tensor((MAXTOK, K), dtype=torch.float16)
stage_t = nvt.tensor((UB_SLOTS, K), dtype=torch.float16)
final_t = nvt.tensor((UB_SLOTS, K), dtype=torch.float16)
out_t = nvt.tensor((UB_SLOTS, K), dtype=torch.float16)  # one-stage inbox (unused)
bar_t = nvt.tensor((EPR * GPUS, 1), dtype=torch.int64)
indir_t = nvt.tensor((UB_SLOTS, 1), dtype=torch.int32)
for tok in range(MAXTOK):
    in_t[tok].fill_(code(rank, tok))
rank_t = torch.full((MAXTOK, GPUS), -1, dtype=torch.int32, device="cuda")
topk_t = torch.zeros(TOTAL_COPIES, dtype=torch.int32, device="cuda")
scat_t = torch.zeros(TOTAL_COPIES, dtype=torch.int32, device="cuda")
splits_t = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda")
ctr_t = torch.zeros((EXPERTS, 1), dtype=torch.int32, device="cuda")
task_t = torch.zeros(1, dtype=torch.int32, device="cuda")

in_c, stage_c, final_c = from_dlpack(in_t), from_dlpack(stage_t), from_dlpack(final_t)
out_c, bar_c, indir_c = from_dlpack(out_t), from_dlpack(bar_t), from_dlpack(indir_t)
rank_c = from_dlpack(rank_t)
topk_c, scat_c, splits_c = from_dlpack(topk_t), from_dlpack(scat_t), from_dlpack(splits_t)
ctr_c, task_c = from_dlpack(ctr_t), from_dlpack(task_t)


def run_draw(name, draw):
    final_t.zero_()
    stage_t.zero_()
    out_t.zero_()
    bar_t.zero_()
    ctr_t.zero_()
    task_t.zero_()
    indir_t.fill_(-1)
    rank_t.fill_(-1)
    topk_t.copy_(torch.tensor(draw["topk_flat"], dtype=torch.int32, device="cuda"))
    scat_t.copy_(torch.tensor(draw["scat_flat"], dtype=torch.int32, device="cuda"))
    splits_t.copy_(torch.tensor(draw["splits"], dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    comm.Barrier()

    Ms, row_begins = draw["Ms"], draw["row_begins"]
    a_base_ptrs = [final_t.data_ptr() + rb * K * ELEM for rb in row_begins]
    bc = mega_kernel.run(
        num_groups=EPR,
        problem_sizes_mnkl=[(m, Nint, K, 1) for m in Ms],
        a_dtype=cutlass.Float16, b_dtype=cutlass.Float16, c_dtype=cutlass.Float16,
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

    got = final_t.cpu()
    golden, n_slots = draw["golden"], draw["n_slots"]
    final_ok = True
    for slot in range(n_slots):
        c = int(round(got[slot, 0].item())) - 1
        final_ok = final_ok and (c // MAXTOK, c % MAXTOK) == golden[rank][slot]
    indir = indir_t.cpu().flatten()
    indir_ok = all(int(indir[slot].item()) >= 0 for slot in range(n_slots))
    bells = bar_t.cpu().flatten()
    bell_ok = all(int(bells[ei].item()) == 1 for ei in range(EPR))  # 1 bell/expert (two-stage)

    gemm_ok = True
    for g, (rb, m) in enumerate(zip(row_begins, Ms)):
        if m == 0:
            continue
        a_ref = got[rb : rb + m, :].to(torch.float32).unsqueeze(-1)
        b_ref = mega_kernel._to_reference_operand_fp32(bc[g][1], cutlass.Float16)
        ref = torch.einsum("mkl,nkl->mnl", a_ref, b_ref)
        c_got = bc[g][2].cpu().to(torch.float32)
        try:
            torch.testing.assert_close(c_got, ref, atol=2e-1, rtol=1e-2)
        except AssertionError:
            gemm_ok = False

    ok = final_ok and indir_ok and bell_ok and gemm_ok
    tag = (
        f"final={'ok' if final_ok else 'BAD'} indir={'ok' if indir_ok else 'BAD'} "
        f"bells={'ok' if bell_ok else 'BAD'} C={'ok' if gemm_ok else 'BAD'}"
    )
    print(f"[G{rank}] {name:20s} Ms={Ms} -> {tag} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


draws = [(nm, build_draw(sr, tl)) for (nm, sr, tl) in named_scenarios()]
shared = random.Random(987)
rand_states = [gen_random(shared) for _ in range(NUM_RANDOM)]
draws += [(f"random#{i}", build_draw(sr, tl)) for i, (sr, tl) in enumerate(rand_states)]

all_ok = True
for name, draw in draws:
    all_ok = run_draw(name, draw) and all_ok

comm.Barrier()
print(
    f"[G{rank}] {'TWO-STAGE SWEEP ALL PASS' if all_ok else 'TWO-STAGE SWEEP FAIL'}: "
    f"{len(draws)} routings vs play.py (final + indirect + bells + C==torch)",
    flush=True,
)

for t in (in_t, stage_t, final_t, out_t, bar_t, indir_t):
    nvt.free_tensor(t)
nv.finalize()
