# Notice

PCGS is built on **2D Gaussian Splatting** (2DGS; Huang et al., SIGGRAPH 2024,
<https://github.com/hbb1/2d-gaussian-splatting>, commit `f3e3b9f`), which is itself built on
**3D Gaussian Splatting** (Kerbl et al., SIGGRAPH 2023, <https://github.com/graphdeco-inria/gaussian-splatting>).

- The PCGS code (`pcgs/`, `extraction/`, `evaluation/`, `data/`, `configs/`, `scripts/`, `benchmark/`,
  `tests/`, `docs/`) is released under the MIT License ([`LICENSE`](LICENSE)).
- The files taken from 2DGS, modified or not (listed below), and the two submodules remain under the
  Gaussian-Splatting License ([`LICENSE.md`](LICENSE.md), reproduced
  without modification): research and evaluation use only, no commercial use without the consent of the
  licensors. Their copyright notices are retained. Training with PCGS uses these files, so the complete
  training pipeline is subject to that license.

## What comes from 2DGS

| Path | Status |
|---|---|
| `gaussian_renderer/`, `scene/cameras.py`, `scene/colmap_loader.py`, `scene/gaussian_model.py`, `utils/` | unmodified |
| `render.py` | unmodified apart from a header comment |
| `train.py` | modified: the PCGS rendering and losses (every call into `pcgs.extras`), the `phot.pt` checkpoint of the photometric model |
| `arguments/__init__.py` | modified: the PCGS options (all off by default, so the defaults train plain 2DGS) |
| `scene/__init__.py` | modified: object-centric scene extent from the `OBJECT_EXTENT` environment variable |
| `scene/dataset_readers.py` | modified: fixed train/test split from `<scene>/test_images.txt` |
| `submodules/diff-surfel-rasterization`, `submodules/simple-knn` | git submodules at the commits pinned by 2DGS (`e0ed020`, `f155ec0`), not modified |

Not included from 2DGS: the DTU / Tanks-and-Temples / Mip-NeRF 360 evaluation scripts (`scripts/`),
`convert.py`, `metrics.py`, `lpipsPyTorch/`, `view.py` and `assets/`.

## What is new in PCGS

Copyright (C) 2026 Xin-Yu Xiao and Fei He (Institute of Geology and Geophysics, Chinese Academy of
Sciences; University of Chinese Academy of Sciences), released under the MIT License:

- `pcgs/`: photometric rendering under the SPICE Sun direction (lunar-Lambert), cast shadows by
  shadow mapping with the darkness gate, albedo total variation, the phase rule, the limb and night-side
  silhouette terms, multi-view NCC, the COLMAP pixel convention (`--pixfix`), multi-view-consistent
  depth fusion; held-out-view rendering and the albedo/shading decomposition
- `extraction/`: mesh extraction (fusion + screened Poisson, hole fairing, SfM offset)
- `evaluation/`: SfM alignment, Taubin smoothing, shape and image metrics
- `data/`: download of the public mission data and construction of the scenes (SPICE geometry,
  COLMAP bundle adjustment, masks, splits)
- `configs/`, `scripts/`, `tests/`, `docs/`
- `benchmark/`: orchestration of the comparison with the baselines, the patches that make the baselines read
  the fixed splits, the COLMAP model and the object extent (their training and meshing code is not changed;
  the baselines themselves are cloned into `repos/` and keep their own licences), the tables and figures
- `paper/`: the LaTeX source of the manuscript; `aa.cls` and `aa.bst` are the class and bibliography style of
  Astronomy & Astrophysics (EDP Sciences)

The mission data (JAXA DARTS, NASA PDS, NAIF) are not redistributed here; their own terms of use apply.
