# M=0 probe: does the gemm tolerate a group with ZERO rows (an expert with 0 tokens)?
# Pure-gemm path (num_dispatch=0, pre-set bells, no dispatch, no a_base_ptrs) so this
# isolates the grouped-GEMM tile scheduler from the dispatch/inbox machinery.
#
# The sweep crashed the moment a group had M=0 (Group 1: 0x256x256). This pins whether
# that is the BASE grouped GEMM (M=0 unsupported) or the fused scheduler interaction.
#
# run:  mpirun -np 1 -x NVSHMEM_REMOTE_TRANSPORT=none -x PATH \
#         -x LD_LIBRARY_PATH=<nvshmem/lib> .venv/bin/python mega_m0_test.py

import sys

import cutlass
import mega_kernel
import nvshmem.core as nv
import nvshmem.core.interop.torch as nvt
import torch
from cuda.core import Device
from cutlass.cute.runtime import from_dlpack
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
dev = Device(rank)
dev.set_current()
nv.init(device=dev, mpi_comm=comm, initializer_method="mpi")

N, K = 256, 256
# a few group lists with M=0 in various positions. ONE per process (argv idx): an
# illegal access corrupts the CUDA context, so cases must not share a process.
CASES = [
    [5, 0, 3],  # zero in the middle
    [0, 5, 3],  # zero first
    [5, 3, 0],  # zero last
    [6, 0],  # exactly the sweep's failing shape
]
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 3

all_ok = True
for Ms in [CASES[idx]]:
    G = len(Ms)
    bar_t = nvt.tensor((G, 1), dtype=torch.int64)
    bar_t.fill_(1)  # pre-set bells: every group's gate already open
    ctr_t = torch.zeros(1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    bar_c, ctr_c = from_dlpack(bar_t), from_dlpack(ctr_t)

    print(f"\n=== M0 probe: Ms={Ms} ===", flush=True)
    try:
        mega_kernel.run(
            num_groups=G,
            problem_sizes_mnkl=[(m, N, K, 1) for m in Ms],
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
            task_counter_cute=ctr_c,
            world_size=1,
            num_dispatch=0,
        )
        torch.cuda.synchronize()
        print(f"--> Ms={Ms}: OK (gemm C==torch passed, no fault)", flush=True)
    except Exception as e:
        all_ok = False
        print(f"--> Ms={Ms}: FAULT {type(e).__name__}: {str(e)[:80]}", flush=True)
    nvt.free_tensor(bar_t)

print(f"\n{'M0 ALL OK' if all_ok else 'M0 FAULT'}: base grouped GEMM with M=0 groups", flush=True)
nv.finalize()
