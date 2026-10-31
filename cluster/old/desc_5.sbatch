#!/bin/bash
# Five manners scored on limb-motion descriptors (grace.descriptors; temperature 1, the plain density; set a
# priori), otherwise as all26_current. Densities differ from the Gaussian scan: compare recognition, moving, size.
#   sbatch -p gpu-p2 -c 12 --mem=50G --dependency=afterany:89826 cluster/old/desc_5.sbatch
#SBATCH --job-name=elot-desc
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/yzhou/workdir/Embodied-LoT/results/grace_v7_parts/slurm-%x-%j.out
set -euo pipefail
cd ~/workdir/Embodied-LoT
module load python/3.12.1
out=results/grace_v7_parts/desc_5
if [ -e "$out" ]; then echo "refusing to overwrite $out" >&2; exit 1; fi
.venv/bin/python -u -m grace_lot.inference.run --config configs/grace_v7_parts.toml --split train --manners bowing jumping scurrying shuffling swinging --out $out \
  --set inference.lambda_size=1.0 inference.phases=8 inference.crossover=0.5 \
        inference.chains=2 inference.population=2 inference.replicas=12 inference.ladder='"adaptive"' inference.tau_min=0.05 \
        inference.iterations=24000 inference.burn_in=12000 inference.select_every=1000 \
        inference.parallel_ladders=True lot.max_nodes=32 inference.likelihood=\"descriptors\" inference.temperature=1.0 > $out.log 2>&1
.venv/bin/python -m grace_lot.inference.predictive $out >> $out.log 2>&1
echo "done: $(tail -1 $out.log)"
