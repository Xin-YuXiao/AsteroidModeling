#!/usr/bin/env bash
# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
# Download the Bennu inputs: the OSIRIS-REx SPICE kernels (NAIF PDS4 archive) and the 60 MapCam
# panchromatic I/F frames of the paper (data/lists/bennu/frames.txt) through the PDS Small Bodies Image
# Browser (data/fetch_sbib.py). The reference, the OLA global shape model v21, has to be placed by hand
# (see below and README.md). Idempotent.
# usage: bash data/fetch_bennu.sh          (into $PCGS_DATA/bennu, default <repo>/datasets/bennu)
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=${PCGS_DATA:-$REPO/datasets}/bennu
K=https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels
PY=${PYTHON:-python}
mkdir -p "$D/spice" "$D/shape" "$D/frames"

cd "$D/spice"
for k in lsk/naif0012.tls sclk/orx_sclkscet_00093.tsc fk/orx_v14.tf ik/orx_ocams_v07.ti pck/bennu_v17.tpc \
  spk/de424.bsp spk/bennu_refdrmc_v1.bsp spk/orx_180801_190302_181218_od077_v1.bsp \
  spk/orx_181203_190302_190104_od085_v1.bsp spk/orx_struct_v04.bsp \
  ck/orx_sc_rel_181029_181104_v02.bc ck/orx_sc_rel_181105_181111_v02.bc ck/orx_sc_rel_181112_181118_v01.bc \
  ck/orx_sc_rel_181119_181125_v01.bc ck/orx_sc_rel_181126_181202_v01.bc ck/orx_sc_rel_181203_181209_v01.bc \
  ck/orx_sc_rel_181210_181216_v01.bc ck/orx_sc_rel_181217_181223_v01.bc ck/orx_struct_mapcam_v01.bc; do
  f=$(basename $k); [ -s $f ] || curl -fsSL --retry 3 -m 3600 -o $f $K/$k
done
echo "bennu: $(ls | wc -l) SPICE kernels"

sed 's/\.fits$//' "$REPO/data/lists/bennu/frames.txt" > "$D/frames.txt"
$PY "$REPO/data/fetch_sbib.py" PDS-Bennu "$D/frames.txt" "$D/frames"

f=g_00880mm_alt_obj_0000n00000_v021.obj
if [ -s "$D/shape/$f" ]; then
  echo "bennu: reference $f"
else
  echo "bennu: place the OLA global shape model v21 ($f, 0.88 m facets, km, IAU_BENNU)"
  echo "       from the OSIRIS-REx shape-model archive of the PDS Small Bodies Node in $D/shape/"
fi
echo FETCH_BENNU_DONE
