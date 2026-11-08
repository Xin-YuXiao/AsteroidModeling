# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
# Bennu, OSIRIS-REx MapCam (panchromatic), 2018 Nov 30 - Dec 13 (Approach, Preliminary Survey), 9.07-38.31 km,
# phase 26.1-59.9 deg. 60 images, 3 random splits of 50 training / 10 held-out images.
# Reference: OLA laser-altimeter global shape model v21 (independent of the images).
# Phase >= 15 deg: cast shadows are active, the albedo total-variation prior is not. At these phase angles
# the night-side silhouette band (--sil_night) matters most.
# Sourced by scripts/run_campaign.sh. The settings are identical for every campaign (paper, Table 1).
DATA_DIR=bennu
FLAGS="--depth_ratio 1 --lambda_mask 0.05 --sun_shading --sh_degree 0 --lambda_mv 0.15 --mv_mode ncc --pixfix --limb_halfmax --lambda_albedo_tv 0.1 --tv_max_phase 15 --cast_shadows --shadow_slope 1.0 --shadow_gate 0.9 --shadow_min_phase 15 --sil_night"
