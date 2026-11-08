#!/usr/bin/env bash
# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
# The benchmark scripts read the scenes from data/img/<body> under the body names used during the
# experiments; this links them to the image sets prepared by scripts/prepare_data.sh (datasets/).
#   itokawa -> itokawa   ryugu_lp -> ryugu_7deg   ryugu -> ryugu_19deg   ryugu_b2 -> ryugu_40deg   bennu -> bennu
# usage: bash benchmark/link_datasets.sh        (env PCGS_DATA, default <repo>/datasets)
set -eo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${PCGS_DATA:-$REPO/datasets}
mkdir -p "$REPO/data/img"
for pair in itokawa:itokawa ryugu_lp:ryugu_7deg ryugu:ryugu_19deg ryugu_b2:ryugu_40deg bennu:bennu; do
  body=${pair%%:*}; set_=${pair#*:}
  [ -d "$DATA/$set_" ] || { echo "skip $body: $DATA/$set_ not prepared"; continue; }
  [ -e "$REPO/data/img/$body" ] || ln -s "$DATA/$set_" "$REPO/data/img/$body"
  echo "data/img/$body -> $DATA/$set_"
done
