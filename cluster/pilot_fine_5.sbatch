#!/bin/bash
# Pilot (31 Oct): round 0's settings with the blob baseline (inference.baseline), cropped windows (lot.window_crop), fine motion
# (lot.fine_motion = 3) and lot.max_depth = 9; no library; the 9 trigram leaves. 6000 iterations.
#   sbatch -p gpu-p2 -c 22 --mem=60G cluster/pilot_fine_5.sbatch   (20 ladder processes; 22 CPUs approved for this pilot, 31 Oct)
#SBATCH --job-name=elot-pilot_fine_5
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yzhou/workdir/Embodied-LoT/results/grace_v7_parts/slurm-%x-%j.out
set -euo pipefail
cd ~/workdir/Embodied-LoT
module load python/3.12.1
out=results/grace_v7_parts/pilot_fine_5
if [ -e "$out" ]; then echo "refusing to overwrite $out" >&2; exit 1; fi
.venv/bin/python -u -m grace_lot.inference.run --config configs/grace_v7_parts.toml --split train --manners bowing jumping scurrying shuffling swinging --out $out \
  --set inference.lambda_size=1.0 inference.phases=8 inference.crossover=0.5 \
        inference.chains=2 inference.population=2 inference.replicas=12 inference.ladder='"adaptive"' inference.tau_min=0.05 \
        inference.iterations=6000 inference.burn_in=3000 inference.select_every=1000 inference.progress_every=1000 \
        inference.parallel_ladders=True lot.max_nodes=32 lot.max_depth=9 inference.likelihood=\"descriptors\" inference.temperature=1.0 inference.open_cells=\"expected\" lot.subcell=1 inference.descriptors=\"full\" inference.warmup=16 inference.baseline=\"blobs\" lot.window_crop=True lot.fine_motion=3 lot.primitives=[\"top\",\"center\",\"upper\",\"bottom\",\"periphery\",\"lower\",\"universal\",\"void\",\"neutral\"] > $out.log 2>&1
.venv/bin/python -m grace_lot.inference.predictive $out >> $out.log 2>&1
echo "done: $(tail -1 $out.log)"
