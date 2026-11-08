#!/usr/bin/env bash
# PCGS: photometrically constrained Gaussian splatting. Copyright (C) 2026 Xin-Yu Xiao and Fei He.
# Free for non-commercial, research and evaluation use under LICENSE.md (Gaussian-Splatting License);
# see NOTICE.md.
# Reproduce the PCGS results of the paper, one run at a time on one GPU, then a summary table.
# The paper trained three splits of itokawa, ryugu_19deg and bennu, splits 0-1 of ryugu_40deg and
# split 0 of ryugu_7deg (12 runs); ALL_SPLITS=1 runs splits 0-2 everywhere. Each run takes about
# 20-50 min of training on an A100 (depending on the load of the GPU) plus 5-10 min of extraction and
# 10 min of evaluation, and writes 0.8-1.5 GB.
# Data must be prepared first (scripts/prepare_data.sh <campaign>, in the data environment).
# usage: bash scripts/reproduce_paper.sh [gpu=0] [campaign ...]
#        default campaigns: itokawa ryugu_7deg ryugu_19deg ryugu_40deg bennu
set -eo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${PCGS_OUTPUT:-$REPO/output}
mkdir -p "$OUT"
GPU=${1:-0}; shift || true
CAMPAIGNS=${*:-itokawa ryugu_7deg ryugu_19deg ryugu_40deg bennu}
for c in $CAMPAIGNS; do
  case "${ALL_SPLITS:-0}:$c" in
    0:ryugu_7deg)  SPLITS="0" ;;
    0:ryugu_40deg) SPLITS="0 1" ;;
    *)             SPLITS="0 1 2" ;;
  esac
  for split in $SPLITS; do
    bash "$REPO/scripts/run_campaign.sh" $c $split $GPU 2>&1 | tee -a "$OUT/reproduce_$c.log"
  done
done
${PYTHON:-python} "$REPO/scripts/summarize.py" $CAMPAIGNS
