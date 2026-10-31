#!/bin/bash
# Five manners with a time-warp token level (each video its own timing, +-2 frames, steps 0/2 at 0.1 each, set a
# priori), otherwise as all26_current (its control: same 12 x 12 board, so held-out densities compare).
#   sbatch -p gpu-p2 -c 12 --mem=50G --dependency=afterany:89824 cluster/old/warp2_5.sbatch
#SBATCH --job-name=elot-warp
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yzhou/workdir/Embodied-LoT/results/grace_v7_parts/slurm-%x-%j.out
set -euo pipefail
cd ~/workdir/Embodied-LoT
module load python/3.12.1
out=results/grace_v7_parts/warp2_5
if [ -e "$out" ]; then echo "refusing to overwrite $out" >&2; exit 1; fi
.venv/bin/python -u -m grace_lot.inference.run --config configs/grace_v7_parts.toml --split train --manners bowing jumping scurrying shuffling swinging --out $out \
  --set inference.lambda_size=1.0 inference.temperature=0.03125 inference.phases=8 inference.crossover=0.5 \
        inference.chains=2 inference.population=2 inference.replicas=12 inference.ladder='"adaptive"' inference.tau_min=0.05 \
        inference.iterations=24000 inference.burn_in=12000 inference.select_every=1000 \
        inference.parallel_ladders=True lot.max_nodes=32 inference.warp_radius=2 inference.warp_slack=0.1 > $out.log 2>&1
.venv/bin/python -m grace_lot.inference.predictive $out >> $out.log 2>&1
echo "done: $(tail -1 $out.log)"
