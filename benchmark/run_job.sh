#!/usr/bin/env bash
# One benchmark run end to end, independent of the scheduler (benchmark/gpu_queue.sh on a shared GPU
# node, or inside a Slurm allocation with <gpu> = slurm):
# 1. train, render the test views, extract meshes (benchmark/run_method.sh, quick frame evaluation)
# 2. the shared reference-free alignment to the SfM tie points (evaluation/align_to_sfm.py)
# 3. main-table evaluation of the raw and the aligned main mesh (frame + similarity, 1e6 samples)
# 4. novel-view metrics on the held-out test images (evaluation/nvs_metrics.py, LPIPS on the GPU)
# usage: run_job.sh <method> <body> <split> <gpu|slurm> [variant|-] [extra train flags...]
# Steps whose outputs exist are skipped, so a run can be restarted.
set -eo pipefail
METHOD=$1; BODY=$2; SPLIT=$3; GPU=$4; VARIANT=${5:--}; shift 5 2>/dev/null || shift $#
[ "$VARIANT" = "-" ] && VARIANT=""
# paths: the repository root from this file; machine-specific settings (conda prefix, data) in
# benchmark/env.local.sh, see benchmark/env.example.sh
D=${AM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -f "$D/benchmark/env.local.sh" ] && source "$D/benchmark/env.local.sh"
R=${BENCH_PREFIX:-$HOME}
PY_EVAL=${PY_EVAL:-$R/miniconda3/envs/gs/bin/python}
TAG=$METHOD${VARIANT:+-$VARIANT}
M=$D/results/img/$BODY/${TAG}_s$SPLIT
[ "$GPU" = slurm ] || export CUDA_VISIBLE_DEVICES=$GPU
# variants named *camext* keep the repository's default camera-ring scene extent (no OBJECT_EXTENT)
case "$VARIANT" in *camext*) export EXTENT=camera ;; esac
echo "$(date -Is) $(hostname) RUN $TAG $BODY s$SPLIT gpu=${CUDA_VISIBLE_DEVICES:-?} flags: $*"

FAST=1 bash $D/benchmark/run_method.sh $METHOD $BODY $SPLIT $GPU "$VARIANT" "$@"

case $METHOD in                      # the mesh each method is compared with
  2dgs) MESH=mesh_obj ;;        gof) MESH=mesh_gof ;;       ags) MESH=mesh_ags ;;
  pgsr) MESH=mesh_pgsr ;;       pcgs) MESH=mesh_mvpoisson_fairoff ;;
  3dgs|mip) MESH=mesh_tsdf ;;   rade) MESH=mesh_rade ;;     sugar) MESH=mesh_sugar ;;
  *) echo "unknown method $METHOD"; exit 1 ;;
esac
SC=$D/data/img/$BODY/scene
# the evaluation env's own libstdc++ (open3d -> PIL -> libLerc needs GLIBCXX_3.4.29)
export LD_LIBRARY_PATH=$(dirname $(dirname $PY_EVAL))/lib:${LD_LIBRARY_PATH:-}
[ -f $M/${MESH}_sfm.ply ] || $PY_EVAL $D/evaluation/align_to_sfm.py $SC $M/$MESH.ply $M/${MESH}_sfm.ply
# PCGS: Taubin smoothing (20 iterations) as the last extraction step; sub-GSD vertex noise of the
# fused Poisson mesh, chosen on split 0 (normals -1.2..-1.5 deg, distances unchanged)
EXTRA_MESH=""
if [ $METHOD = pcgs ]; then
  [ -f $M/${MESH}_sfm_t20.ply ] || $PY_EVAL $D/evaluation/smooth_mesh.py $M/${MESH}_sfm.ply $M/${MESH}_sfm_t20.ply 20
  EXTRA_MESH=${MESH}_sfm_t20
fi
for mesh in $MESH ${MESH}_sfm $EXTRA_MESH; do
  [ -f $M/table_$mesh.json ] || $PY_EVAL $D/evaluation/eval_mesh.py --rec $M/$mesh.ply \
    --ref $D/data/img/$BODY/ref_mesh_km.ply --out $M/table_$mesh.json --keep-largest \
    --frame-tf $SC/frame_to_ref.json --align similarity frame --samples 1000000
done
$PY_EVAL $D/evaluation/nvs_metrics.py --run $M --device cuda --threads 8
echo "$(date -Is) JOB_DONE $TAG $BODY s$SPLIT"
