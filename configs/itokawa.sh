# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
# Itokawa, Hayabusa AMICA (v band), 2005 Sep 29 - Oct 2, 7.3-8.0 km, phase 9.3-10.4 deg.
# 60 images, 3 random splits of 50 training / 10 held-out images. Reference: Gaskell SPC model (3.1 M facets).
# Phase < 15 deg: the albedo total-variation prior is active, cast shadows are not.
# Sourced by scripts/run_campaign.sh. The settings are identical for every campaign (paper, Table 1);
# PCGS in the paper includes --sil_night; drop it for the variant without the night-side band.
DATA_DIR=itokawa
FLAGS="--depth_ratio 1 --lambda_mask 0.05 --sun_shading --sh_degree 0 --lambda_mv 0.15 --mv_mode ncc --pixfix --limb_halfmax --lambda_albedo_tv 0.1 --tv_max_phase 15 --cast_shadows --shadow_slope 1.0 --shadow_gate 0.9 --shadow_min_phase 15 --sil_night"
