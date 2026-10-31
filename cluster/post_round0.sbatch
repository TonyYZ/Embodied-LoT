#!/bin/bash
# What the results page reads for a fit, once it has finished: gallery, part-by-part motion views, score comparison.
#   RUN=results/grace_v7_parts/round0_5 BASE=results/grace_v7_parts/claims_5 sbatch -p gpu-p2 -c 6 --mem=24G --dependency=afterok:<fit> cluster/post_round0.sbatch
#SBATCH --job-name=elot-post
#SBATCH --time=06:00:00
#SBATCH --output=/home/yzhou/workdir/Embodied-LoT/results/grace_v7_parts/slurm-%x-%j.out
set -uo pipefail
cd ~/workdir/Embodied-LoT
module load python/3.12.1
.venv/bin/python -W ignore -m grace_lot.analysis.gallery --run $RUN > $RUN/gallery.log 2>&1; echo "gallery $?"
.venv/bin/python -W ignore -m grace_lot.analysis.motion_view --run $RUN > $RUN/motion_view.log 2>&1; echo "motion_view $?"
.venv/bin/python -W ignore -m grace_lot.analysis.compare_scoring $RUN $BASE > $RUN/compare.log 2>&1; echo "compare $?"
