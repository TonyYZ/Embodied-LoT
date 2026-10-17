# Retired learners (legacy)

The first, recognition-oriented learners, retired on 2026-10-17 in favour of the GRACE learner
(`grace_lot/inference/`: tempered Metropolis-Hastings over program trees, scored on rendered videos).

| File | What it is |
| --- | --- |
| `concept_learning.py` | Metropolis-Hastings over observing concept graphs (scanners under an input-portal root), scored on scalar responses |
| `dreamcoder_lot.py` | DreamCoder-style wake/compress learner over flat scanner sets, with a fragment library |
| `learning_data.py` | the toy `Observation` / `Trajectory` data and scoring both share |
| `mcmc_demos.py`, `dreamcoder_demos.py` | their examples |

They predate ludo dynamics, motion kinds, parameter-free bodies and input/output windows, and use the
old `InputNode` portal. Run from the repository root:

    python legacy/learners/mcmc_demos.py
    python legacy/learners/dreamcoder_demos.py
