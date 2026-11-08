#!/usr/bin/env bash
# Train + render test views + extract mesh + evaluate one method on one split.
# usage: run_method.sh <2dgs|gof|ags|pgsr|pcgs|3dgs|mip|rade|sugar> <body> <split> <gpu|slurm> [variant] [extra train flags...]
# Output: results/img/<body>/<method>[-<variant>]_s<split>/ (model, renders, meshes, eval_*.json)
set -e
METHOD=$1; BODY=$2; SPLIT=$3; GPU=$4; VARIANT=${5:-}; shift 5 2>/dev/null || shift $#
EXTRA="$@"
# paths: the repository root from this file; machine-specific settings (conda prefix, data) in
# benchmark/env.local.sh, see benchmark/env.example.sh
D=${AM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -f "$D/benchmark/env.local.sh" ] && source "$D/benchmark/env.local.sh"
R=${BENCH_PREFIX:-$HOME}
PY_GS=${PY_GS:-$R/miniconda3/envs/gs/bin/python}
PY_AGS=${PY_AGS:-$R/miniconda3/envs/gsast/bin/python}
PY_PGSR=${PY_PGSR:-$R/miniconda3/envs/pgsr/bin/python}
PY_3DGS=${PY_3DGS:-$R/venvs/v_3dgs/bin/python}     # benchmark/baselines/setup_new_baselines.sh
PY_MIP=${PY_MIP:-$R/venvs/v_mip/bin/python}
PY_RADE=${PY_RADE:-$R/venvs/v_rade/bin/python}
PY_SUGAR=${PY_SUGAR:-$R/venvs/v_sugar/bin/python}
PY_EVAL=${PY_EVAL:-$PY_GS}
S=$D/data/img/$BODY/scene_split$SPLIT
TAG=$METHOD${VARIANT:+-$VARIANT}
M=$D/results/img/$BODY/${TAG}_s$SPLIT
REF=$D/data/img/$BODY/ref_mesh_km.ply
PORT=$((6100 + RANDOM % 3000))   # each 3DGS-lineage train.py binds a GUI port
mkdir -p $M
# GPU: an index, or "slurm" to keep the allocation's CUDA_VISIBLE_DEVICES
[ "$GPU" = slurm ] || export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=8
# conda-forge CGAL deps (libtiff/libLerc) need the env libstdc++ (GLIBCXX_3.4.29), not the system one
[ -d $R/miniconda3/envs/gs/lib ] && export LD_LIBRARY_PATH=$R/miniconda3/envs/gs/lib:$R/miniconda3/envs/gsast/lib:$LD_LIBRARY_PATH
# object-centric extent (see img_patch_splits.py): 1.1 x p99 radius of the scene's own SfM /
# visual-hull points; set EXTENT=camera to keep the repos' default camera-ring extent
if [ "${EXTENT:-object}" = object ]; then
  export OBJECT_EXTENT=$($PY_EVAL -c "import numpy as np; p=np.loadtxt('$S/sparse/0/points3D.txt', usecols=(1,2,3), comments='#'); print(round(1.1*np.percentile(np.linalg.norm(p,axis=1),99),5))")
else
  unset OBJECT_EXTENT
fi
echo "OBJECT_EXTENT=${OBJECT_EXTENT:-camera-default}"
# camera range of this body (SPICE): TSDF depth truncation = farthest camera + 2 extents (km);
# PGSR's neighbour-distance threshold = half the median range (its DTU default 1.5 is in
# DTU units; half the range keeps the same ~30 deg neighbourhood)
read DTRUNC DHALF < <($PY_EVAL -c "import json, numpy as np; d=np.array([v['distance_km'] for v in json.load(open('$D/data/img/$BODY/scene/meta.json'))['images'].values()]); print(round(d.max()+2*${OBJECT_EXTENT:-0.5},3), round(0.5*np.median(d),3))")
echo "DTRUNC=$DTRUNC DHALF=$DHALF"
T0=$(date +%s)
case $METHOD in
  2dgs)
    PY=$PY_GS; cd $D/repos/2d-gaussian-splatting
    [ -f $M/point_cloud/iteration_30000/point_cloud.ply ] || $PY train.py -s $S -m $M --eval --quiet --port $PORT $EXTRA
    T1=$(date +%s)
    DR=$(echo "$EXTRA" | grep -oP '(?<=--depth_ratio )[0-9.]+' || echo 0)
    # (a) repo defaults, as used for the 2DGS row of Asteroid-GS (voxel = 2*radius/1024)
    [ -f $M/mesh_default.ply ] || { $PY render.py -s $S -m $M --skip_train --quiet --depth_ratio $DR && \
      cp $M/train/ours_30000/fuse_post.ply $M/mesh_default.ply; }
    # (b) object protocol: foreground masks (RGBA alpha, 2DGS's own mask_backgrond path),
    #     1 m voxels, per-body depth truncation, largest cluster (as the 2DGS DTU script)
    [ -f $M/mesh_obj.ply ] || { $PY render.py -s $S -m $M -i images_rgba --skip_train --skip_test --quiet \
      --depth_ratio $DR --depth_trunc $DTRUNC --voxel_size 0.001 --sdf_trunc 0.004 --num_cluster 1 && \
      cp $M/train/ours_30000/fuse_post.ply $M/mesh_obj.ply; }
    MESHES="mesh_default mesh_obj" ;;
  pgsr)
    # PGSR (zju3dv/PGSR @ de24f1a, adapted by benchmark/baselines/patch_pgsr.py); RGBA input only
    # supplies the extraction mask, training uses RGB. Meshing as its DTU script: TSDF, num_cluster 1.
    PY=$PY_PGSR; cd $D/repos/PGSR
    [ -d $R/miniconda3/envs/pgsr/lib ] && export LD_LIBRARY_PATH=$R/miniconda3/envs/pgsr/lib:$LD_LIBRARY_PATH
    [ -f $M/point_cloud/iteration_30000/point_cloud.ply ] || \
      $PY train.py -s $S -m $M -i images_rgba --eval --quiet --port $PORT --multi_view_max_dis $DHALF $EXTRA
    T1=$(date +%s)
    [ -f $M/mesh_pgsr.ply ] || { $PY render.py -m $M --quiet --max_depth $DTRUNC --voxel_size 0.001 --num_cluster 1 && \
      cp $M/mesh/tsdf_fusion_post.ply $M/mesh_pgsr.ply; }
    MESHES="mesh_pgsr" ;;
  gof)
    PY=$PY_GS; cd $D/repos/gaussian-opacity-fields
    [ -f $M/point_cloud/iteration_30000/point_cloud.ply ] || $PY train.py -s $S -m $M --eval --quiet --port $PORT
    T1=$(date +%s)
    $PY render.py -s $S -m $M --skip_train --quiet
    [ -f $M/mesh_gof.ply ] || { $PY extract_mesh.py -s $S -m $M --iteration 30000 --quiet && \
      cp $(ls -t $(find $M -name "mesh_binary_search_*.ply") | head -1) $M/mesh_gof.ply; }
    MESHES="mesh_gof" ;;
  ags)
    PY=$PY_AGS; cd $D/repos/Asteroid-GS
    # flags as in the authors' train_eval_render_mesh.sh
    [ -f $M/point_cloud/iteration_30000/point_cloud.ply ] || \
      $PY train.py -s $S -m $M -r 1 --use_decoupled_appearance --eval --quiet --port $PORT
    T1=$(date +%s)
    $PY render.py -s $S -m $M --skip_train --quiet
    [ -f $M/mesh_ags.ply ] || { $PY mesh_extract_gof.py -s $S -m $M --iteration 30000 --quiet && \
      cp $M/final_model.ply $M/mesh_ags.ply; }
    MESHES="mesh_ags" ;;
  3dgs)
    # 3DGS (graphdeco-inria/gaussian-splatting); no extraction of its own: shared TSDF of its rendered depth
    PY=${PY_3DGS:?}; cd $D/repos/gaussian-splatting
    [ -f $M/point_cloud/iteration_30000/point_cloud.ply ] || \
      $PY train.py -s $S -m $M --eval --quiet --disable_viewer $EXTRA
    T1=$(date +%s)
    $PY render.py -m $M --skip_train --quiet
    [ -f $M/mesh_tsdf.ply ] || $PY $D/benchmark/baselines/gs_tsdf.py --method 3dgs -s $S -m $M --depth_trunc $DTRUNC --out $M/mesh_tsdf.ply
    MESHES="mesh_tsdf" ;;
  mip)
    # Mip-Splatting; depth rendered with GOF's renderer (GOF loads Mip-Splatting models), shared TSDF
    PY=${PY_MIP:?}; cd $D/repos/mip-splatting
    [ -f $M/point_cloud/iteration_30000/point_cloud.ply ] || \
      $PY train.py -s $S -m $M --eval --quiet --port $PORT $EXTRA
    T1=$(date +%s)
    $PY render.py -m $M --skip_train --quiet
    [ -f $M/mesh_tsdf.ply ] || (cd $D/repos/gaussian-opacity-fields && \
      $PY_GS $D/benchmark/baselines/gs_tsdf.py --method gof -s $S -m $M --depth_trunc $DTRUNC --out $M/mesh_tsdf.ply)
    MESHES="mesh_tsdf" ;;
  rade)
    # RaDe-GS; its TSDF extraction with our voxel/truncation (shared gs_tsdf.py)
    PY=${PY_RADE:?}; cd $D/repos/RaDe-GS
    [ -f $M/point_cloud/iteration_30000/point_cloud.ply ] || \
      $PY train.py -s $S -m $M --eval --quiet --port $PORT $EXTRA
    T1=$(date +%s)
    $PY render.py -m $M --skip_train --quiet
    [ -f $M/mesh_rade.ply ] || $PY $D/benchmark/baselines/gs_tsdf.py --method rade -s $S -m $M --depth_trunc $DTRUNC --out $M/mesh_rade.ply
    MESHES="mesh_rade" ;;
  sugar)
    # SuGaR (Anttwo/SuGaR, dn_consistency, high poly). Its vanilla-3DGS stage ignores --eval, so we
    # train that stage ourselves on the training images only; the scene is linked under a unique
    # name (SuGaR names its outputs after the scene folder); bbox = the object, not the cameras.
    PY=${PY_SUGAR:?}; cd $D/repos/SuGaR
    export PATH=$(dirname $PY):$PATH         # SuGaR calls `python` internally
    SN=$M/scene_${BODY}_s$SPLIT; [ -e $SN ] || ln -s $S $SN
    [ -f $M/vanilla_gs/point_cloud/iteration_7000/point_cloud.ply ] || \
      $PY gaussian_splatting/train.py -s $SN -m $M/vanilla_gs --eval --iterations 7000 --quiet --port $PORT
    T1=$(date +%s)
    BB=$($PY_EVAL -c "r=1.2*${OBJECT_EXTENT:-0.5}; print(f'({-r},{-r},{-r}) ({r},{r},{r})')")
    if [ ! -f $M/mesh_sugar.ply ]; then
      $PY train_full_pipeline.py -s $SN --gs_output_dir $M/vanilla_gs -r dn_consistency --high_poly True \
        --refinement_time long --export_obj True --export_ply False --eval True \
        --bboxmin "${BB% *}" --bboxmax "${BB#* }" --center_bbox False
      OUT=$(ls -t output/refined_mesh/scene_${BODY}_s$SPLIT/*.obj 2>/dev/null | head -1)
      [ -n "$OUT" ] || OUT=$(ls -t output/coarse_mesh/scene_${BODY}_s$SPLIT/*.ply | head -1)
      $PY_EVAL -c "import trimesh; trimesh.load('$OUT', force='mesh', process=False).export('$M/mesh_sugar.ply')"
      echo "SuGaR mesh from $OUT" > $M/mesh_sugar.txt
    fi
    MESHES="mesh_sugar" ;;
  pcgs)
    PY=$PY_GS; cd $D                     # the PCGS code of this repository
    [ -f $M/point_cloud/iteration_30000/point_cloud.ply ] || \
      $PY train.py -s $S -m $M --eval --quiet --port $PORT $EXTRA
    T1=$(date +%s)
    SHADE=""; case "$EXTRA" in *--sun_shading*) SHADE="--sun_shading";; esac
    [ -d $M/test/ours_30000/renders ] || $PY pcgs/render_views.py -s $S -m $M $SHADE
    # multi-view-consistent fusion (depths confirmed by >= 3 neighbouring views)
    [ -f $M/mesh_mvpoisson.ply ] || $PY extraction/mvfuse.py -s $S -m $M
    # watertight post-processing (extraction/): fairing of unsupported regions + SfM offset
    [ -f $M/mesh_mvpoisson_fair.ply ] || $PY $D/extraction/fair_holes.py $M
    [ -f $M/mesh_mvpoisson_fairoff.ply ] || $PY $D/extraction/offset_to_sfm.py $S $M/mesh_mvpoisson_fair.ply $M/mesh_mvpoisson_fairoff.ply
    MESHES="mesh_mvtsdf mesh_mvpoisson mesh_mvpoisson_fairoff" ;;
esac
T2=$(date +%s)
echo "{\"train_s\": $((T1-T0)), \"total_s\": $((T2-T0)), \"flags\": \"$EXTRA\"}" > $M/timing.json
RENDERS=$M/test/ours_30000

# renders are saved as 00000.png...; masks follow the sorted test-image order
PYE=$PY_EVAL
$PYE - <<EOF
import os, shutil
names = sorted(l.strip() for l in open("$S/test_images.txt") if l.strip())
os.makedirs("$M/test_masks", exist_ok=True)
for i, n in enumerate(names):
    shutil.copy(os.path.join("$S/masks", n), "$M/test_masks/%05d.png" % i)
EOF
FTF=$D/data/img/$BODY/scene/frame_to_ref.json
# FAST=1: iteration mode (frame registration only, 300k samples); default: full tables
if [ "${FAST:-0}" = 1 ]; then EVALOPTS="--align frame --samples 300000"; else EVALOPTS="--align none similarity frame --samples 1000000"; fi
# NVS only where test renders follow our 00000.png naming (PGSR names them by image)
NVSOPTS=""
[ -f $RENDERS/renders/00000.png ] && NVSOPTS="--renders $RENDERS/renders --gt-images $RENDERS/gt --masks $M/test_masks"
for mesh in $MESHES; do
  # re-evaluate results that predate the shared frame registration
  [ -f $M/eval_$mesh.json ] && ! grep -q '"frame"' $M/eval_$mesh.json && rm -f $M/eval_$mesh.json
  [ -f $M/eval_$mesh.json ] || $PYE $D/evaluation/eval_mesh.py --rec $M/$mesh.ply --ref $REF \
    --out $M/eval_$mesh.json $NVSOPTS --frame-tf $FTF $EVALOPTS
done
echo "BASELINE_DONE $TAG $BODY s$SPLIT"
