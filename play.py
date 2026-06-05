# start with 8 GPU's, each with a set of E experts and N tokens where we do top K
# can see these as a nested dict:
#   state_dict :  GPU -> Token -> [experts it routes to]
#
# An expert lives on exactly ONE GPU:  owner(e) = e // experts_per_gpu
# A token-copy atom is (sender_gpu, token) — sender kept so atoms are unique
# AND because the deterministic order is *by sender rank*.
#
# Pipeline (Stage 1 = dispatch + up-GEMM):
#   state_dict ─► build_inbox ───────────────────────────► GOLDEN inbox
#             └► build_dispatch_tables ─► run_dispatch ──► A_buffer  (assert == golden)
#                                                          + barriers (the bells)
#                build_gemm_schedule ─► run_gemm(A, barriers) ─► C   (symbolic Wᵢ·atom)

from __future__ import annotations

StateDict = dict[int, dict[int, list[int]]]  # state_dict[gpu][token] = [experts]
Atom = tuple[int, int]  # (sender_gpu, token)


def owner(expert: int, experts_per_gpu: int) -> int:
    """Which GPU physically holds this (global) expert."""
    return expert // experts_per_gpu


def build_inbox(GPUS: int, EXPERTS: int, state_dict: StateDict) -> dict[int, dict[int, list[Atom]]]:
    """
    For each GPU, for each expert it OWNS, the list of (sender, token) copies
    routed to it — in the pinned deterministic order:
        group by expert, then by sender rank, then by the sender's token order.
    Order falls out of the loop nesting: sender-outer, token-inner.

    This IS index-prep's answer — the GOLDEN that run_dispatch must reproduce.
    """
    experts_per_gpu = EXPERTS // GPUS
    inbox: dict[int, dict[int, list[Atom]]] = {
        gpu: {e: [] for e in range(gpu * experts_per_gpu, (gpu + 1) * experts_per_gpu)}
        for gpu in range(GPUS)
    }
    for gpu in range(GPUS):
        for e in inbox[gpu]:  # only experts this GPU owns
            for sender in range(GPUS):  # sender-rank order  (pinned)
                for token, expert_list in state_dict[sender].items():  # token order (pinned)
                    if e in expert_list:
                        inbox[gpu][e].append((sender, token))
    return inbox


def build_gemm_schedule(inbox_gpu: dict[int, list[Atom]], BLOCK_M: int) -> dict:
    """One GPU's inbox -> the grouped-GEMM tile schedule it consumes."""
    experts = sorted(inbox_gpu.keys())
    split_size = {e: len(inbox_gpu[e]) for e in experts}

    a_rows: list[Atom] = []  # flattened A buffer, experts ascending
    expert_row_begin: dict[int, int] = {}
    for e in experts:
        expert_row_begin[e] = len(a_rows)
        a_rows.extend(inbox_gpu[e])

    expert_ids: list[int] = []
    split_size_cum: list[int] = []  # expert block-start row (broadcast)
    tile_num: list[int] = []  # expert's #tiles            (broadcast)
    tile_num_cum: list[int] = []  # inclusive prefix of #tiles (broadcast)
    cum_tiles = 0
    for e in experts:
        n_tiles = -(-split_size[e] // BLOCK_M)  # cdiv; 0 tokens -> 0 tiles
        cum_tiles += n_tiles
        for _ in range(n_tiles):
            expert_ids.append(e)
            split_size_cum.append(expert_row_begin[e])
            tile_num.append(n_tiles)
            tile_num_cum.append(cum_tiles)

    return {
        "a_rows": a_rows,
        "split_size": split_size,
        "expert_row_begin": expert_row_begin,
        "expert_ids": expert_ids,
        "split_size_cum": split_size_cum,
        "tile_num": tile_num,
        "tile_num_cum": tile_num_cum,
        "num_total_tiles": len(expert_ids),
    }


def decode_tile(sched: dict, pid_m: int, BLOCK_M: int) -> dict:
    """The per-tile decode the GEMM kernel runs (self-consistency check on the tables)."""
    expert_id = sched["expert_ids"][pid_m]
    row_begin = sched["split_size_cum"][pid_m]
    tile_begin = sched["tile_num_cum"][pid_m] - sched["tile_num"][pid_m]
    local_pid_m = pid_m - tile_begin
    row_remain = sched["split_size"][expert_id] - local_pid_m * BLOCK_M
    rows = [row_begin + local_pid_m * BLOCK_M + i for i in range(min(BLOCK_M, row_remain))]
    return {"expert_id": expert_id, "rows": rows, "row_remain": row_remain}


# ── Dispatch (Team A) ──────────────────────────────────────────────────────


def build_dispatch_tables(GPUS: int, EXPERTS: int, inbox, state_dict: StateDict) -> dict:
    """
    Per sender GPU: each copy's (token, expert, dest, slot), in pinned walk order.
    slot = expert's block-start in dest's A buffer + this copy's position in that block.
    """
    experts_per_gpu = EXPERTS // GPUS
    row_begin: dict[tuple[int, int], int] = {}  # (gpu, expert) -> first row
    for gpu in range(GPUS):
        cum = 0
        for e in sorted(inbox[gpu]):
            row_begin[(gpu, e)] = cum
            cum += len(inbox[gpu][e])

    scatter: dict[int, list[tuple[int, int, int, int]]] = {}
    for sender in range(GPUS):
        copies = [(token, e) for token, experts in state_dict[sender].items() for e in experts]
        copies.sort(key=lambda te: te[1])  # stable: group by expert, keep token order
        walk = []
        for token, e in copies:
            dest = owner(e, experts_per_gpu)
            slot = row_begin[(dest, e)] + inbox[dest][e].index((sender, token))
            walk.append((token, e, dest, slot))
        scatter[sender] = walk
    return scatter


def run_dispatch(GPUS: int, EXPERTS: int, scatter, inbox, state_dict: StateDict):
    """Push every copy into its dest A buffer; ring barriers[dest][expert][sender] per group."""
    experts_per_gpu = EXPERTS // GPUS
    A = {gpu: [None] * sum(len(inbox[gpu][e]) for e in inbox[gpu]) for gpu in range(GPUS)}
    barriers = {gpu: {e: {s: 0 for s in range(GPUS)} for e in inbox[gpu]} for gpu in range(GPUS)}

    # how many copies each sender owes each expert
    owed: dict[tuple[int, int], int] = {}
    for sender in range(GPUS):
        for _token, experts in state_dict[sender].items():
            for e in experts:
                owed[(sender, e)] = owed.get((sender, e), 0) + 1

    # empty-expert tail: a sender that owes 0 to an owned expert rings eagerly
    for dest in range(GPUS):
        for e in inbox[dest]:
            for s in range(GPUS):
                if owed.get((s, e), 0) == 0:
                    barriers[dest][e][s] = 1

    sent: dict[tuple[int, int], int] = {}
    for sender in range(GPUS):
        for token, e, dest, slot in scatter[sender]:
            A[dest][slot] = (sender, token)  # PUSH
            sent[(sender, e)] = sent.get((sender, e), 0) + 1
            if sent[(sender, e)] == owed[(sender, e)]:  # last copy for this (expert,sender)
                barriers[dest][e][sender] = 1  # RING
    return A, barriers


# ── GEMM (Team B) ──────────────────────────────────────────────────────────


def run_gemm(GPUS: int, schedules, A, barriers, BLOCK_M: int, gpu: int):
    """Consume the schedule + A buffer; gate on bells; emit C with atoms left symbolic."""
    s = schedules[gpu]
    C: list = [None] * len(A[gpu])
    for pid_m in range(s["num_total_tiles"]):
        d = decode_tile(s, pid_m, BLOCK_M)
        e = d["expert_id"]
        assert all(barriers[gpu][e][snd] == 1 for snd in range(GPUS)), f"bell not set for e{e}"
        for r in d["rows"]:
            C[r] = (f"W{e}", A[gpu][r])  # symbolic  Wᵢ·atom
    return C


def make_tables(GPUS: int, EXPERTS: int, TOP_K: int, BLOCK_M: int, state_dict: StateDict):
    for gpu, toks in state_dict.items():
        for token, experts in toks.items():
            assert len(experts) == TOP_K, f"gpu {gpu} token {token}: {len(experts)} != TOP_K"
    inbox = build_inbox(GPUS, EXPERTS, state_dict)
    schedules = {gpu: build_gemm_schedule(inbox[gpu], BLOCK_M) for gpu in range(GPUS)}
    return inbox, schedules


if __name__ == "__main__":
    # the x/y trace, executable:
    #   G0 owns e0,e1   G1 owns e2,e3   topk=2   BLOCK_M=2
    #   G0 tokens A=0,B=1   G1 tokens C=0,D=1
    #   A->{e0,e1}  B->{e1,e2}  C->{e0,e3}  D->{e1,e2}
    GPUS, EXPERTS, TOP_K, BLOCK_M = 2, 4, 2, 2
    state = {
        0: {0: [0, 1], 1: [1, 2]},  # A->e0,e1   B->e1,e2
        1: {0: [0, 3], 1: [1, 2]},  # C->e0,e3   D->e1,e2
    }
    NAME: dict[tuple[int, int], str] = {
        (0, 0): "A",
        (0, 1): "B",
        (1, 0): "C",
        (1, 1): "D",
    }  # atom -> letter, for display

    inbox, schedules = make_tables(GPUS, EXPERTS, TOP_K, BLOCK_M, state)
    scatter = build_dispatch_tables(GPUS, EXPERTS, inbox, state)
    A, barriers = run_dispatch(GPUS, EXPERTS, scatter, inbox, state)

    print("G0 inbox (GOLDEN):")
    for e, rows in inbox[0].items():
        print(f"  e{e}: {[NAME[a] for a in rows]}")

    print("\nDispatch — G0 fills its A buffer:")
    print(f"  A[G0] = {[NAME[a] for a in A[0]]}")
    # B2 check: dispatch must reproduce the golden inbox, slot for slot
    for gpu in range(GPUS):
        assert A[gpu] == schedules[gpu]["a_rows"], f"dispatch != golden on G{gpu}"
    print("  ✓ A == golden inbox (exact permutation) on every GPU")

    print("\nBells on G0:")
    for e, snds in barriers[0].items():
        print(f"  e{e}: {snds}")

    print("\nGEMM on G0:")
    C = run_gemm(GPUS, schedules, A, barriers, BLOCK_M, gpu=0)
    for r, (w, atom) in enumerate(C):
        print(f"  C[{r}] = {w}·{NAME[atom]}")
