# Randomized routing sweep: the FULL fused kernel vs play.py across MANY routings.
#
# mega_d_test pins ONE hand-built routing. This sweeps a battery — named edge cases
# + random draws — so the integer/index logic (inbox placement, runtime-atomic bells,
# the empty-expert eager ring, AND the dropped-token -1 path) is exercised broadly,
# not on a single lucky scenario. play.py is the golden generator for every draw.
#
# Each draw, ONE fused kernel does dispatch -> fence -> bell -> gemm(A = the dispatched
# inbox), dual-checked:
#   - inbox (d_output) == play.py golden placement   (index logic, all routings)
#   - bells (barriers)  == play.py golden bells        (runtime-atomic gate + eager ring)
#   - C == torch(inbox @ B)                            (gemm consumed the dispatched rows)
#
# FIXED across draws (so ONE buffer set + stable num_groups; loads still recompile since
# total_tasks is constexpr — fine for a correctness sweep): GPUS, EXPERTS, TOP_K, MAXTOK,
# K, Nint, num_dispatch, and every tensor SHAPE. Only the routing CONTENT + per-expert M
# vary. Symmetric buffers are sized to the upper bound and reused (zeroed) each draw.
#
# Dropped tokens: a (token, k) slot whose expert is -1 (ref L114 `if e >= 0`). The kernel
# skips it; play never sees it. We build play's golden from the REAL experts only
# (state_real), and lay the kernel's topk array as tok-major with -1 in the dropped slots.
# A real copy at so=tok*TOP_K+k still decodes _tok = so//TOP_K correctly; -1 slots no-op.
#
# run:  mpirun -np 2 -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
#         -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_sweep_test.py

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

# ── fixed config (shapes constant so buffers + num_groups are stable) ─────────
GPUS, EXPERTS, TOP_K = 2, 4, 2
EPR = EXPERTS // GPUS
K, Nint = 256, 256
MAXTOK = 6  # max tokens/GPU; in_t rows + topk stride space
TOTAL_COPIES = MAXTOK * TOP_K  # topk/scatter array length (tok-major, fixed)
UB_SLOTS = GPUS * MAXTOK * TOP_K  # inbox upper bound: every sender -> every copy -> me
NUM_DISPATCH = 8
BLOCK_M = 128
ELEM = 2  # fp16 bytes
NUM_RANDOM = 10  # random draws after the named edge cases

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
assert comm.Get_size() == GPUS
dev = Device(rank)
dev.set_current()
nv.init(device=dev, mpi_comm=comm, initializer_method="mpi")


def code(sender, token):
    # small distinct id so A stays fp16-precise while uniquely decoding (sender, token)
    return float(sender * MAXTOK + token + 1)


# ── routing generators: each returns (state_real, topk_layout) per GPU ────────
#   state_real[g][tok]  = list of REAL (non-dropped) experts  -> drives play golden
#   topk_layout[g][tok] = length-TOP_K list, expert or -1 (dropped) -> kernel topk array
def _layout_from_choice(choice, drop_set):
    """choice = list of TOP_K distinct experts; drop_set = indices k to drop (-1)."""
    slots = [(-1 if k in drop_set else e) for k, e in enumerate(choice)]
    real = [e for k, e in enumerate(choice) if k not in drop_set]
    return slots, real


def _every_rank_has_load(state_real):
    """True iff each rank's owned expert range [r*EPR,(r+1)*EPR) receives >=1 copy.
    We require this so no rank gets an all-zero gemm grid (unsupported degenerate)."""
    got = [0] * GPUS
    for g in state_real:
        for tok, experts in state_real[g].items():
            for e in experts:
                got[e // EPR] += 1
    return all(c > 0 for c in got)


def gen_random(rng):
    # resample until every rank has load (excludes the all-zero-grid degenerate; SOME
    # experts may still be empty -> M=0 groups, which IS exercised)
    for _ in range(100):
        state_real, topk_layout = {}, {}
        for g in range(GPUS):
            ntok = rng.randint(1, MAXTOK)
            sr, tl = {}, {}
            for tok in range(ntok):
                choice = rng.sample(range(EXPERTS), TOP_K)  # distinct experts
                # drop each slot with prob 0.2 (exercises the -1 path + variable load)
                drop_set = {k for k in range(TOP_K) if rng.random() < 0.2}
                slots, real = _layout_from_choice(choice, drop_set)
                tl[tok], sr[tok] = slots, real
            state_real[g], topk_layout[g] = sr, tl
        if _every_rank_has_load(state_real):
            return state_real, topk_layout
    raise RuntimeError("could not generate a routing with load on every rank")


def named_scenarios():
    """Deterministic edge cases, each a clear story."""
    out = []

    # 1. the mega_d baseline (sanity: matches the pinned test)
    sr = {0: {0: [0, 1], 1: [1, 2]}, 1: {0: [0, 3], 1: [1, 2]}}
    out.append(("baseline", sr, {g: {t: list(v) for t, v in sr[g].items()} for g in sr}))

    # 2. one HOT expert per rank, the rank's other owned expert(s) get 0 tokens (M=0
    #    group, total>0). Both ranks keep load -> exercises the realistic empty-expert
    #    case. (A rank whose EVERY owned expert is empty -> all-zero gemm grid -> the
    #    CUTLASS scheduler can't build a 0-tile grid; that degenerate is excluded here,
    #    documented as unsupported. At Qwen3-30B scale a rank owns 16 experts over
    #    65536*8 copies, so an all-empty rank never occurs.)
    sr = {g: {t: [0, 2] for t in range(3)} for g in range(GPUS)}  # only e0 (G0) + e2 (G1)
    out.append(("some-experts-empty", sr, {g: {t: list(v) for t, v in sr[g].items()} for g in sr}))

    # 3. every token routes to the SAME pair -> max overlap, one hot expert per rank
    #    (e1 on G0, e3 on G1; e0,e2 stay empty -> M=0 groups). Both ranks keep load,
    #    so no all-zero grid. Exercises non-zero row_begin (the hot expert is index 1).
    sr = {g: {t: [1, 3] for t in range(MAXTOK)} for g in range(GPUS)}
    out.append(("max-overlap", sr, {g: {t: list(v) for t, v in sr[g].items()} for g in sr}))

    # 4. dropped tokens: every token drops its 2nd slot (-1) -> only e in slot0 lands
    sr, tl = {}, {}
    for g in range(GPUS):
        sr[g], tl[g] = {}, {}
        for t in range(3):
            e0, e1 = (t % EXPERTS), ((t + 1) % EXPERTS)
            tl[g][t] = [e0, -1]  # slot 1 dropped
            sr[g][t] = [e0]
    out.append(("all-2nd-dropped", sr, tl))

    # 5. one rank totally idle (0 tokens) -> the other does all the dispatch
    sr = {0: {0: [0, 2], 1: [1, 3]}, 1: {}}
    out.append(("one-rank-idle", sr, {0: {0: [0, 2], 1: [1, 3]}, 1: {}}))

    return out


def build_draw(state_real, topk_layout):
    """play golden + per-rank kernel tables for one routing. Returns None if this rank
    has no inbox rows AND no copies (nothing to check / degenerate)."""
    inbox = play.build_inbox(GPUS, EXPERTS, state_real)
    scatter = play.build_dispatch_tables(GPUS, EXPERTS, inbox, state_real)
    schedules = {g: play.build_gemm_schedule(inbox[g], BLOCK_M) for g in range(GPUS)}
    _A, bar_gold = play.run_dispatch(GPUS, EXPERTS, scatter, inbox, state_real)
    golden = {g: schedules[g]["a_rows"] for g in range(GPUS)}

    my_experts = sorted(inbox[rank].keys())  # always EPR experts (some M=0)
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
        golden=golden,
        bar_gold=bar_gold,
        my_experts=my_experts,
        Ms=Ms,
        row_begins=row_begins,
        topk_flat=topk_flat,
        scat_flat=scat_flat,
        splits=splits,
        n_slots=len(golden[rank]),
    )


# ── symmetric + local buffers, allocated ONCE at the upper bound ─────────────
in_t = nvt.tensor((MAXTOK, K), dtype=torch.float16)
out_t = nvt.tensor((UB_SLOTS, K), dtype=torch.float16)  # inbox == gemm A
bar_t = nvt.tensor((EPR * GPUS, 1), dtype=torch.int64)
for tok in range(MAXTOK):
    in_t[tok].fill_(code(rank, tok))

topk_t = torch.zeros(TOTAL_COPIES, dtype=torch.int32, device="cuda")
scat_t = torch.zeros(TOTAL_COPIES, dtype=torch.int32, device="cuda")
splits_t = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda")
ctr_t = torch.zeros((EXPERTS, 1), dtype=torch.int32, device="cuda")
task_t = torch.zeros(1, dtype=torch.int32, device="cuda")

in_c, out_c, bar_c = from_dlpack(in_t), from_dlpack(out_t), from_dlpack(bar_t)
topk_c, scat_c, splits_c = from_dlpack(topk_t), from_dlpack(scat_t), from_dlpack(splits_t)
ctr_c, task_c = from_dlpack(ctr_t), from_dlpack(task_t)


def run_draw(name, draw):
    # reset symmetric + counter state, load this draw's tables
    out_t.zero_()
    bar_t.zero_()
    ctr_t.zero_()
    task_t.zero_()
    topk_t.copy_(torch.tensor(draw["topk_flat"], dtype=torch.int32, device="cuda"))
    scat_t.copy_(torch.tensor(draw["scat_flat"], dtype=torch.int32, device="cuda"))
    splits_t.copy_(torch.tensor(draw["splits"], dtype=torch.int32, device="cuda"))
    torch.cuda.synchronize()
    comm.Barrier()

    Ms, row_begins = draw["Ms"], draw["row_begins"]
    a_base_ptrs = [out_t.data_ptr() + rb * K * ELEM for rb in row_begins]
    bc = mega_kernel.run(
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

    # ── checks ──────────────────────────────────────────────────────────────
    got = out_t.cpu()
    golden, n_slots = draw["golden"], draw["n_slots"]
    inbox_ok = True
    for slot in range(n_slots):
        c = int(round(got[slot, 0].item())) - 1
        inbox_ok = inbox_ok and (c // MAXTOK, c % MAXTOK) == golden[rank][slot]

    bells = bar_t.cpu().flatten()
    bell_ok = all(
        int(bells[ei * GPUS + s].item()) == draw["bar_gold"][rank][rank * EPR + ei][s]
        for ei in range(EPR)
        for s in range(GPUS)
    )

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

    ok = inbox_ok and bell_ok and gemm_ok
    tag = (
        f"inbox={'ok' if inbox_ok else 'BAD'} bells={'ok' if bell_ok else 'BAD'} "
        f"C={'ok' if gemm_ok else 'BAD'}"
    )
    print(f"[G{rank}] {name:20s} Ms={Ms} -> {tag} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# ── run the battery: named edge cases, then random draws ─────────────────────
draws = [(nm, build_draw(sr, tl)) for (nm, sr, tl) in named_scenarios()]
# Random states MUST be identical across ranks (both GPUs agree on the routing), so
# generate them from a rank-independent shared RNG.
shared = random.Random(987)
rand_states = [gen_random(shared) for _ in range(NUM_RANDOM)]
draws += [(f"random#{i}", build_draw(sr, tl)) for i, (sr, tl) in enumerate(rand_states)]

all_ok = True
for name, draw in draws:
    all_ok = run_draw(name, draw) and all_ok

comm.Barrier()
print(
    f"[G{rank}] {'SWEEP ALL PASS' if all_ok else 'SWEEP FAIL'}: "
    f"{len(draws)} routings vs play.py (inbox + bells + C==torch)",
    flush=True,
)

nvt.free_tensor(in_t)
nvt.free_tensor(out_t)
nvt.free_tensor(bar_t)
nv.finalize()
