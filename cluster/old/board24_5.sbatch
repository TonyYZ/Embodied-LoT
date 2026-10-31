#!/bin/bash
# Five manners on a 24 x 24 board, otherwise as all26_current (its 12 x 12 control), 29 Oct 2026. Two fits in turn:
#   fine24_5   body frame, cells half as big both ways (edge_samples 8: the fewest with no gaps along a limb)
#   world24_5  world frame (the walk across the screen kept): columns as wide as now over 2.6 body heights,
#              rows half as tall over the same 1.3 (edge_samples 8, the same rule)
#   sbatch -p gpu-p2 -c 12 --mem=50G cluster/old/board24_5.sbatch
#SBATCH --job-name=elot-board24
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/yzhou/workdir/Embodied-LoT/results/grace_v7_parts/slurm-%x-%j.out
set -euo pipefail
set -f                                      # the overrides hold [ ], not globs
cd ~/workdir/Embodied-LoT
module load python/3.12.1
common="inference.lambda_size=1.0 inference.temperature=0.03125 inference.phases=8 inference.crossover=0.5
        inference.chains=2 inference.population=2 inference.replicas=12 inference.ladder=\"adaptive\" inference.tau_min=0.05
        inference.iterations=24000 inference.burn_in=12000 inference.select_every=1000
        inference.parallel_ladders=True lot.max_nodes=32 preprocess.field_size=24 interface.edge_samples=8"
for v in fine24_5 world24_5; do
  out=results/grace_v7_parts/$v
  if [ -e "$out" ]; then echo "refusing to overwrite $out" >&2; exit 1; fi
  extra=""
  [ $v = world24_5 ] && extra='preprocess.window_x=[-1.3,1.3] preprocess.frame="world"'
  .venv/bin/python -u -m grace_lot.inference.run --config configs/grace_v7_parts.toml --split train \
    --manners bowing jumping scurrying shuffling swinging --out $out --set $common $extra > $out.log 2>&1
  .venv/bin/python -m grace_lot.inference.predictive $out >> $out.log 2>&1
  echo "done $v: $(tail -1 $out.log)"
done
