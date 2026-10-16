"""GRACE locomotion x Embodied-LoT pipeline.

Each worker process runs single-threaded numpy. Without this, OpenBLAS
reserves per-thread buffers for all 32 cores in every process (~800 MB of
committed memory each), and parallel runs exhaust the Windows commit limit.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
