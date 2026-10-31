#!/bin/bash
# Five manners scored on the full descriptors (inference.descriptors="full", fixed a priori 30 Oct): every rhythm from
# once per clip to twice per cycle, and each part's posture (average position, size, axis). A short fit (6000
# iterations, a quarter of desc_fix_5's) to finish within hours; otherwise as desc_fix_5.
#   sbatch -p gpu-p2 -c 12 --mem=50G cluster/old/full_5.sbatch
#SBATCH --job-name=elot-full
#SBATCH --time=0-06:00:00
#SBATCH --output=/home/yzhou/workdir/Embodied-LoT/results/grace_v7_parts/slurm-%x-%j.out
set -euo pipefail
cd ~/workdir/Embodied-LoT
module load python/3.12.1
out=results/grace_v7_parts/full_5
if [ -e "$out" ]; then echo "refusing to overwrite $out" >&2; exit 1; fi
.venv/bin/python -u -m grace_lot.inference.run --config configs/grace_v7_parts.toml --split train --manners bowing jumping scurrying shuffling swinging --out $out \
  --set inference.lambda_size=1.0 inference.phases=8 inference.crossover=0.5 \
        inference.chains=2 inference.population=2 inference.replicas=12 inference.ladder='"adaptive"' inference.tau_min=0.05 \
        inference.iterations=6000 inference.burn_in=3000 inference.select_every=1000 inference.progress_every=1000 \
        inference.parallel_ladders=True lot.max_nodes=32 inference.likelihood=\"descriptors\" inference.descriptors=\"full\" \
        inference.temperature=1.0 > $out.log 2>&1
.venv/bin/python -m grace_lot.inference.predictive $out >> $out.log 2>&1
echo "done: $(tail -1 $out.log)"
