#!/usr/bin/env bash
# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
# Download the Ryugu inputs from JAXA DARTS: the calibrated (L2c) Hayabusa2 ONC-T archives of the three
# hovering campaigns (2018-07-10: 19 deg, 2018-08-31: 40 deg, 2019-01-15: 7 deg), the Hayabusa2 SPICE
# kernels plus the NAIF generic planetary kernel, and the SfM reference shape of Watanabe et al. (2019).
# Kernels and reference live in ryugu_19deg/; the other two campaigns link them. Idempotent.
# usage: bash data/fetch_ryugu.sh          (into $PCGS_DATA/ryugu_*, default <repo>/datasets)
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=${PCGS_DATA:-$REPO/datasets}
B=https://data.darts.isas.jaxa.jp/pub/hayabusa2/onc_bundle/data/v03c/data_calibrated_l2c
K=https://data.darts.isas.jaxa.jp/pub/hayabusa2/old/2020/spice_bundle/spice_kernels
W=https://data.darts.isas.jaxa.jp/pub/hayabusa2/paper/Watanabe_2019
N=https://naif.jpl.nasa.gov/pub/naif/generic_kernels

mkdir -p "$D/ryugu_19deg/spice" "$D/ryugu_19deg/shape"
cd "$D/ryugu_19deg/spice"
for k in lsk/naif0012.tls pck/pck00010.tpc pck/hyb2_ryugu_shape_v20190328.tpc fk/hyb2_v14.tf fk/hyb2_hp_v01.tf \
  fk/hyb2_ryugu_v01.tf ik/hyb2_onc_v05.ti sclk/hyb2_20141203-20191231_v01.tsc spk/2162173_Ryugu.bsp \
  spk/2162173_ryugu_20180601-20191230_0060_20181221.bsp spk/hyb2_approach_od_v20180811114238.bsp \
  spk/hyb2_hpk_20180627_20190213_v01.bsp spk/lidar_derived_trj_20191114_20180630053224_20190213030000_v02.bsp \
  ck/hyb2_hkattrpt_2018_v02.bc ck/hyb2_aocsc_2018_v02.bc \
  ck/hyb2_hkattrpt_20190101000000_20190213030000_v02.bc ck/hyb2_aocsc_20190101000000_20190213030000_v02.bc; do
  f=$(basename $k); [ -s $f ] || curl -fsSL --retry 3 -m 1800 -o $f $K/$k
done
[ -s de440s.bsp ] || curl -fsSL --retry 3 -m 1800 -o de440s.bsp $N/spk/planets/de440s.bsp
echo "ryugu: $(ls | wc -l) SPICE kernels"
cd "$D/ryugu_19deg/shape"
f=SHAPE_SFM_3M_v20180804.obj
[ -s $f ] || curl -fsSL --retry 3 -m 3600 -o $f $W/$f
echo "ryugu: reference $f"

for pair in ryugu_19deg:20180710 ryugu_40deg:20180831 ryugu_7deg:20190115; do
  n=${pair%%:*}; d=${pair#*:}
  mkdir -p "$D/$n/raw"
  if [ $n != ryugu_19deg ]; then              # shared kernels, reference and pixel orientation
    for l in spice shape ref_mesh_km.ply orientation.json; do
      [ -L "$D/$n/$l" ] || [ -e "$D/$n/$l" ] || ln -s ../ryugu_19deg/$l "$D/$n/$l"
    done
  fi
  cd "$D/$n/raw"
  [ -s ${d}_l2c_v03c.tgz ] || curl -fsSL --retry 3 -m 3600 -o ${d}_l2c_v03c.tgz $B/${d}_l2c_v03c.tgz
  [ -d $d ] || tar xzf ${d}_l2c_v03c.tgz
  miss=$(for f in $(cat "$REPO/data/lists/$n/frames.txt"); do [ -s $d/$f ] || echo $f; done | wc -l)
  echo "$n: $(ls $d | grep -c tvf) ONC-T v-band (tvf) frames on $d; $miss of the paper's frames missing"
done
echo FETCH_RYUGU_DONE
