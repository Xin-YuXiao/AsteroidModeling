#!/usr/bin/env bash
# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
# Download the Itokawa inputs: Hayabusa AMICA v-band FITS frames of 2005-09-28..10-02 (JAXA DARTS),
# the Hayabusa SPICE kernels (DARTS) plus the NAIF generic leap-second and planetary kernels, and the
# Gaskell SPC reference shape (3.1 M facets). Idempotent: existing files are skipped.
# usage: bash data/fetch_itokawa.sh        (into $PCGS_DATA/itokawa, default <repo>/datasets/itokawa)
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=${PCGS_DATA:-$REPO/datasets}/itokawa
B=https://data.darts.isas.jaxa.jp/pub/hayabusa
N=https://naif.jpl.nasa.gov/pub/naif/generic_kernels
mkdir -p "$D/fits" "$D/spice" "$D/shape"

cd "$D/fits"
for day in 20050928 20050929 20050930 20051001 20051002; do
  for f in $(curl -fsSL --retry 3 -m 60 $B/amica/$day/ | grep -oE "ST_[0-9]+_v\.fits" | sort -u); do
    [ -s $f ] || curl -fsSL --retry 3 -m 120 -o $f $B/amica/$day/$f
  done
done
echo "itokawa: $(ls *.fits | wc -l) AMICA v-band frames"
miss=$(cd "$D/fits" && for f in $(cat "$REPO/data/lists/itokawa/frames.txt"); do [ -s $f ] || echo $f; done | wc -l)
[ "$miss" = 0 ] || echo "WARNING: $miss of the paper's frames (data/lists/itokawa/frames.txt) are missing"

cd "$D/spice"
for k in sclk/hayabusa.tsc fk/itokawa_fixed.tf fk/HAYABUSA_HP.TF ik/amica31.ti pck/itokawa_gaskell.tpc \
         spk/itokawa_1989-2010.bsp spk/hayabusa_itokawarendezvous_v01.bsp ck/hayabusa_itokawarendezvous_v01.bc; do
  f=$(basename $k); [ -s $f ] || curl -fsSL --retry 3 -m 600 -o $f $B/spice/$k
done
[ -s naif0012.tls ] || curl -fsSL --retry 3 -m 600 -o naif0012.tls $N/lsk/naif0012.tls
[ -s de440s.bsp ] || curl -fsSL --retry 3 -m 1800 -o de440s.bsp $N/spk/planets/de440s.bsp
echo "itokawa: $(ls | wc -l) SPICE kernels"

cd "$D/shape"
f=itokawa_f3145728.tri
[ -s $f ] || { curl -fsSL --retry 3 -m 1800 -o $f.gz $B/shape/gaskell/$f.gz && gunzip -f $f.gz; }
echo "itokawa: reference $(ls)"
echo FETCH_ITOKAWA_DONE
