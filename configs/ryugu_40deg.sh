# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
# Ryugu, Hayabusa2 ONC-T, 2018 Aug 31, 21.77-21.89 km, phase 39.9-40.3 deg.
# 48 images, 3 random splits of 40 training / 8 held-out images. Reference: SfM model of Watanabe et al. (2019).
# Phase >= 15 deg: cast shadows are active, the albedo total-variation prior is not.
# Sourced by scripts/run_campaign.sh. The settings are identical for every campaign (paper, Table 1);
# PCGS in the paper includes --sil_night; drop it for the variant without the night-side band.
DATA_DIR=ryugu_40deg
FLAGS="--depth_ratio 1 --lambda_mask 0.05 --sun_shading --sh_degree 0 --lambda_mv 0.15 --mv_mode ncc --pixfix --limb_halfmax --lambda_albedo_tv 0.1 --tv_max_phase 15 --cast_shadows --shadow_slope 1.0 --shadow_gate 0.9 --shadow_min_phase 15 --sil_night"
