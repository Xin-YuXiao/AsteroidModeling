#!/usr/bin/env bash
# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
# One PCGS run end to end on one split of one campaign, as in the paper:
#   1. train (train.py, flags from configs/<campaign>.sh), 30k iterations
#   2. render the held-out views (pcgs/render_views.py: albedo x lunar-Lambert shading, no cast shadows)
#   3. extract the watertight mesh: multi-view-consistent depth fusion + screened Poisson
#      (extraction/mvfuse.py), hole fairing (extraction/fair_holes.py), SfM offset
#      (extraction/offset_to_sfm.py) -> mesh_mvpoisson_fairoff.ply, the PCGS shape model
#   4. evaluation protocol shared with the baselines: translation + normal offset to the SfM tie
#      points (evaluation/align_to_sfm.py), Taubin smoothing x20 (evaluation/smooth_mesh.py),
#      metrics against the reference with the shared frame transform (evaluation/eval_mesh.py)
#      -> table_mesh_mvpoisson_fairoff_sfm_t20.json; novel-view metrics (evaluation/nvs_metrics.py)
# Steps whose outputs exist are skipped, so a run can be restarted.
# usage: bash scripts/run_campaign.sh <itokawa|ryugu_7deg|ryugu_19deg|ryugu_40deg|bennu> <split 0|1|2> [gpu=0]
# env:   PCGS_DATA (default <repo>/datasets), PCGS_OUTPUT (default <repo>/output), PYTHON (default python),
#        EVAL_ALL=1 also evaluates the mesh before / after the SfM alignment alone (appendix tables),
#        EXTRA_FLAGS appended to the training flags (a later value overrides, e.g. "--lambda_mv 0"),
#        DROP_FLAGS switch flags to remove (e.g. "--sil_night"), TAG output name (default pcgs):
#        DROP_FLAGS=--sil_night TAG=pcgs_noband bash scripts/run_campaign.sh bennu 0
set -eo pipefail
CAMPAIGN=$1; SPLIT=$2; GPU=${3:-0}
[ -n "$CAMPAIGN" ] && [ -n "$SPLIT" ] || { sed -n '2,19p' "$0"; exit 1; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=${PCGS_DATA:-$REPO/datasets}
OUT=${PCGS_OUTPUT:-$REPO/output}
PY=${PYTHON:-python}
[ -f "$REPO/configs/$CAMPAIGN.sh" ] || { echo "unknown campaign $CAMPAIGN (configs/)"; exit 1; }
source "$REPO/configs/$CAMPAIGN.sh"                     # DATA_DIR, FLAGS
for f in ${DROP_FLAGS:-}; do FLAGS=$(echo " $FLAGS " | sed "s/ $f / /"); done
FLAGS="$FLAGS ${EXTRA_FLAGS:-}"
SC=$DATA/$DATA_DIR/scene                                # shared scene (images, masks, SfM, meta.json)
S=$DATA/$DATA_DIR/scene_split$SPLIT                     # this split (links to the scene + test_images.txt)
REF=$DATA/$DATA_DIR/ref_mesh_km.ply
M=$OUT/$CAMPAIGN/${TAG:-pcgs}_s$SPLIT
[ -f "$S/test_images.txt" ] || { echo "missing $S (scripts/prepare_data.sh $CAMPAIGN)"; exit 1; }
mkdir -p "$M"
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
# conda envs with conda-forge image libraries need the env's own libstdc++ ahead of the system one
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
# object-centric extent: 1.1 x the 99th-percentile radius of the scene's SfM points (km); it replaces
# the camera-ring radius that 2DGS uses to scale densification and the position learning rate
export OBJECT_EXTENT=$($PY -c "import numpy as np; p=np.loadtxt('$S/sparse/0/points3D.txt', usecols=(1,2,3), comments='#'); print(round(1.1*np.percentile(np.linalg.norm(p,axis=1),99),5))")
PORT=$((6100 + RANDOM % 3000))                          # train.py binds a viewer port
echo "$(date -Is) PCGS $CAMPAIGN split $SPLIT gpu=$GPU OBJECT_EXTENT=$OBJECT_EXTENT -> $M"
cd "$REPO"

# 1-3. train, render the held-out views, extract the mesh
T0=$(date +%s)
[ -f "$M/point_cloud/iteration_30000/point_cloud.ply" ] || \
  $PY train.py -s "$S" -m "$M" --eval --quiet --port $PORT $FLAGS
T1=$(date +%s)
[ -d "$M/test/ours_30000/renders" ] || $PY pcgs/render_views.py -s "$S" -m "$M" --sun_shading
[ -f "$M/mesh_mvpoisson.ply" ] || $PY extraction/mvfuse.py -s "$S" -m "$M"
[ -f "$M/mesh_mvpoisson_fair.ply" ] || $PY extraction/fair_holes.py "$M"
[ -f "$M/mesh_mvpoisson_fairoff.ply" ] || \
  $PY extraction/offset_to_sfm.py "$S" "$M/mesh_mvpoisson_fair.ply" "$M/mesh_mvpoisson_fairoff.ply"
T2=$(date +%s)
[ -f "$M/timing.json" ] || echo "{\"train_s\": $((T1-T0)), \"total_s\": $((T2-T0)), \"flags\": \"$FLAGS\"}" > "$M/timing.json"

# 4. evaluation protocol of the paper (identical for every method)
MESH=mesh_mvpoisson_fairoff
[ -f "$M/${MESH}_sfm.ply" ] || $PY evaluation/align_to_sfm.py "$SC" "$M/$MESH.ply" "$M/${MESH}_sfm.ply"
[ -f "$M/${MESH}_sfm_t20.ply" ] || $PY evaluation/smooth_mesh.py "$M/${MESH}_sfm.ply" "$M/${MESH}_sfm_t20.ply" 20
EVAL_MESHES="${MESH}_sfm_t20"
[ "${EVAL_ALL:-0}" = 1 ] && EVAL_MESHES="$MESH ${MESH}_sfm ${MESH}_sfm_t20"
for mesh in $EVAL_MESHES; do
  [ -f "$M/table_$mesh.json" ] || $PY evaluation/eval_mesh.py --rec "$M/$mesh.ply" --ref "$REF" \
    --out "$M/table_$mesh.json" --keep-largest --frame-tf "$SC/frame_to_ref.json" --align similarity frame \
    --samples 1000000
done
# held-out views (nvs_metrics.py copies the lit masks of the held-out images in render order)
$PY evaluation/nvs_metrics.py --run "$M" --device cuda --threads ${OMP_NUM_THREADS}
echo "$(date -Is) DONE $CAMPAIGN split $SPLIT: $M/table_${MESH}_sfm_t20.json, $M/nvs_table.json"
