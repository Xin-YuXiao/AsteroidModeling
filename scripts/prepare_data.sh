#!/usr/bin/env bash
# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
# Download the public inputs of one campaign and build its scene (run in the data-preparation
# environment, data/requirements.txt):
#   SPICE geometry -> 60 frames, 8-bit PNG images and body masks -> SIFT features (COLMAP) ->
#   SPICE-initialised triangulation + bundle adjustment -> Sim(3) to the SPICE camera centres ->
#   three random 50/10 splits -> evaluation-only frame transform to the reference shape model.
# Then checks that the selected frames and splits are those of the paper (data/lists/<campaign>/).
# usage: bash scripts/prepare_data.sh <itokawa|ryugu_7deg|ryugu_19deg|ryugu_40deg|bennu>
# env:   PCGS_DATA (default <repo>/datasets), PYTHON (default python), OMP_NUM_THREADS (COLMAP threads)
set -eo pipefail
CAMPAIGN=$1
REPO=$(cd "$(dirname "$0")/.." && pwd)
export PCGS_DATA=${PCGS_DATA:-$REPO/datasets}
PY=${PYTHON:-python}
D=$PCGS_DATA/$CAMPAIGN
B="$PY $REPO/data/build_scene.py $CAMPAIGN"
case $CAMPAIGN in
  itokawa)
    bash "$REPO/data/fetch_itokawa.sh"
    [ -f "$D/spice_geometry.json" ] || $PY "$REPO/data/itokawa_spice.py"
    [ -f "$D/scene/meta.json" ] || $PY "$REPO/data/itokawa_prep.py"
    grep -q '"sfm"' "$D/scene/meta.json" || $PY "$REPO/data/itokawa_refine.py"
    $B splits ;;
  ryugu_7deg|ryugu_19deg|ryugu_40deg)
    bash "$REPO/data/fetch_ryugu.sh"
    # the 19-deg campaign fixes the pixel orientation and writes the reference mesh shared by all three
    [ -f "$PCGS_DATA/ryugu_19deg/orientation.json" ] || $PY "$REPO/data/build_scene.py" ryugu_19deg geometry
    [ -f "$D/spice_geometry.json" ] || $B geometry
    [ -f "$D/scene/meta.json" ] || $B prep
    grep -q '"sfm"' "$D/scene/meta.json" || $B refine ;;
  bennu)
    ls "$D"/shape/g_00880mm_alt_obj_0000n00000_v021.obj >/dev/null 2>&1 || [ -f "$D/ref_mesh_km.ply" ] || {
      echo "place the OLA global shape model v21 (g_00880mm_alt_obj_0000n00000_v021.obj) in $D/shape/ (README)"
      exit 1; }
    bash "$REPO/data/fetch_bennu.sh"
    [ -f "$D/spice_geometry.json" ] || $B geometry
    [ -f "$D/scene/meta.json" ] || $B prep
    grep -q '"sfm"' "$D/scene/meta.json" || $B refine ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
[ -f "$D/scene/frame_to_ref.json" ] || $B register

# same frames and splits as the paper?
$PY - "$D/scene" "$REPO/data/lists/$CAMPAIGN" <<'EOF'
import json, os, sys
scene, lists = sys.argv[1:3]
meta = json.load(open(os.path.join(scene, "meta.json")))["images"]
got = sorted(os.path.basename(v["file"]) for v in meta.values())
ref = sorted(l.strip() for l in open(os.path.join(lists, "frames.txt")) if l.strip())
ok = got == ref
print(f"frames: {len(got)} selected, {len(set(got) & set(ref))} of the paper's {len(ref)}")
for k in range(3):
    a = sorted(l.strip() for l in open(os.path.join(scene, "splits", f"test_split{k}.txt")) if l.strip())
    b = sorted(l.strip() for l in open(os.path.join(lists, f"test_split{k}.txt")) if l.strip())
    print(f"split {k}: {'same as the paper' if a == b else 'DIFFERS from the paper'}")
    ok &= a == b
print("SCENE_OK" if ok else "SCENE_DIFFERS (see data/lists/ for the paper's frames and splits)")
EOF
