# PCGS: photometrically constrained Gaussian splatting. Copyright (c) 2026 Xin-Yu Xiao and Fei He.
# SPDX-License-Identifier: MIT
"""PCGS: photometrically constrained Gaussian splatting for asteroid shape reconstruction.

extras.py        photometry, cast shadows, phase-dependent priors, silhouette and multi-view losses,
                 pixel convention (--pixfix), multi-view-consistent depth fusion; imported by train.py
render_views.py  held-out-view renders under the SPICE Sun direction (script)
decompose.py     albedo / shading / normal decomposition of chosen views (script, figures)
"""
