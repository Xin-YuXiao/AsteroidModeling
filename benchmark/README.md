# Benchmark

This directory reproduces the comparison in the paper. PCGS and the Gaussian splatting baselines are trained on the
same images, cameras, lit masks and splits. Every mesh goes through the same reference-free post-processing and is
evaluated with the same protocol. The directory also produces the tables and figures of the manuscript (`paper/`).

PCGS itself, the evaluation and the data preparation are the code at the repository root (`train.py`, `pcgs/`,
`extraction/`, `evaluation/`, `data/`). This directory adds the baselines and the orchestration around them.

## Contents

```
benchmark/
  run_method.sh        train one method on one split, render the held-out views, extract its mesh
  run_job.sh           run_method.sh + shared SfM alignment + Taubin smoothing (PCGS) + shape and image metrics
  gpu_queue.sh         run a job list one run at a time on an idle GPU, with RAM / VRAM watchdog and core pinning
  usage_monitor.sh     resource use of this project's processes against caps (for long unattended runs)
  link_datasets.sh     link the prepared image sets (datasets/) to the scene names used here (data/img/)
  env.example.sh       machine-specific settings; copy to env.local.sh
  baselines/           environment setup, split and scene-extent patches, TSDF extraction for the baselines
  jobs/                the job lists of every experiment, in the order they were run
  paper_tables.py      LaTeX tables of the manuscript from logs/all_metrics.json
  figures/             metrics collection and the figures of the manuscript and the project page
```

## Methods

| Key | Method | Code | Commit | Mesh evaluated |
|---|---|---|---|---|
| `pcgs` | PCGS (this work) | this repository | | `mesh_mvpoisson_fairoff` |
| `2dgs` | 2DGS (Huang et al. 2024) | [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) | `f3e3b9f` | `mesh_obj` |
| `gof` | GOF (Yu et al. 2024) | [autonomousvision/gaussian-opacity-fields](https://github.com/autonomousvision/gaussian-opacity-fields) | `5245b20` | `mesh_gof` |
| `pgsr` | PGSR (Chen et al. 2024) | [zju3dv/PGSR](https://github.com/zju3dv/PGSR) | `de24f1a` | `mesh_pgsr` |
| `ags` | Asteroid-GS (Zhang et al. 2025) | [zhangxiaojie1243fei-crypto/Asteroid-GS](https://github.com/zhangxiaojie1243fei-crypto/Asteroid-GS) | `6ce62c1` | `mesh_ags` |
| `mip` | Mip-Splatting (Yu et al. 2024) | [autonomousvision/mip-splatting](https://github.com/autonomousvision/mip-splatting) | `dda02ab` | `mesh_tsdf` |

`run_method.sh` also supports 3DGS (`3dgs`, graphdeco-inria/gaussian-splatting `54c035f`), RaDe-GS (`rade`,
`d72f207`) and SuGaR (`sugar`, `7c10c4a`), which were set up but are not reported in the paper. Each method uses
the mesh extraction its authors recommend; methods without one at this scale use the TSDF fusion of
`baselines/gs_tsdf.py`.

## Setup

1. Prepare the image sets with `scripts/prepare_data.sh` (repository README), then link them to the scene names of
   the benchmark:

   ```bash
   bash benchmark/link_datasets.sh     # data/img/{itokawa,ryugu_lp,ryugu,ryugu_b2,bennu} -> datasets/...
   ```

   | Scene name | Image set |
   |---|---|
   | `itokawa` | `itokawa` (10&deg;) |
   | `ryugu_lp` | `ryugu_7deg` |
   | `ryugu` | `ryugu_19deg` |
   | `ryugu_b2` | `ryugu_40deg` |
   | `bennu` | `bennu` |

2. Machine settings: `cp benchmark/env.example.sh benchmark/env.local.sh` and set `BENCH_PREFIX`, the prefix of the
   conda installation and of the baseline venvs, and, if needed, the interpreters `PY_GS`, `PY_AGS` and `PY_PGSR`.

3. Baselines, cloned into `repos/` at the commits above:

   ```bash
   bash benchmark/baselines/img_env_setup.sh            # conda envs gs (2DGS, GOF, PCGS) and gsast (Asteroid-GS)
   python benchmark/baselines/img_patch_splits.py       # fixed splits and object extent for 2DGS, GOF, Asteroid-GS
   python benchmark/baselines/patch_pgsr.py             # the same for PGSR (env pgsr, installed per its README)
   bash benchmark/baselines/setup_new_baselines.sh      # venvs for 3DGS, Mip-Splatting, RaDe-GS, SuGaR
   ```

   The patches change only how the scenes are read (the fixed splits, the COLMAP model, and the object extent of
   Sect. 3.4 of the paper). The training and meshing code of the baselines is left untouched.

## Running

One run, end to end:

```bash
bash benchmark/run_job.sh <method> <scene> <split> <gpu> [variant|-] [extra training flags]
bash benchmark/run_job.sh pgsr itokawa 0 0
```

The outputs go to `results/img/<scene>/<method>[-<variant>]_s<split>/`: the model, the held-out renders, the meshes,
`table_<mesh>_sfm[_t20].json` (shape metrics) and `nvs_table.json` (image metrics). Steps whose outputs exist are
skipped, so a run can be restarted. Variants named `*camext*` keep the camera-ring scene extent of the original code;
this is Asteroid-GS&dagger; in the paper.

A job list runs one run at a time on one idle GPU. Several queues can share a node, one card each:

```bash
GPUS="2" CORES=24-29 setsid nohup bash benchmark/gpu_queue.sh benchmark/jobs/jobs_main_table.txt \
    > logs/queue_main.log 2>&1 < /dev/null &
MAX_GPUS=2 bash benchmark/usage_monitor.sh          # one line per interval, immediately if a cap is exceeded
```

`gpu_queue.sh` waits for a card with less than 500 MiB in use and no load, pins the run to `CORES` at low priority,
and stops it if its resident memory exceeds `MAX_GB` (32) or its GPU memory exceeds `MAX_VRAM_GB` (24).

### Run tags of the paper

| Tag | In the paper |
|---|---|
| `pcgs-v18n` | PCGS, the final model (`configs/*.sh`) |
| `pcgs-v18` | PCGS without the night-side band |
| `pcgs-v10` | PCGS without the phase-dependent priors and without the night-side band |
| `pcgs-v18n-<x>` | leave-one-out ablations: `noshade`, `nomv`, `nomask`, `nopix`, `noprior`, and `sh3` (free view-dependent colour) |
| `ags` / `ags-camext` | Asteroid-GS with the object extent / with the scene extent of its code (Asteroid-GS&dagger;) |

The job lists in `jobs/` record every experiment in the order in which it was run, including the development rounds
that led to the final model.

## Tables and figures

```bash
python benchmark/figures/collect_metrics.py                       # results/img -> logs/all_metrics.json
python benchmark/paper_tables.py logs/all_metrics.json paper/tables
```

`paper_tables.py` reads every number from the evaluation files; nothing is typed in by hand. The figures of the
manuscript are made by the scripts below. Each script states its inputs in its first lines. The render panels are
computed on CPU from the meshes in `results/` and composed locally.

| Figure | Script |
|---|---|
| 1, overview | `figures/nc_fig_overview.py` (decomposition by `pcgs/decompose.py`) |
| 2, image sets | `figures/aa_figs.py campaigns` |
| 3, accuracy of all methods | `figures/nc_fig_numbers.py` |
| 4, cumulative distances and latitude | `figures/nc_fig_quant.py` (profiles by `figures/dist_profiles.py`) |
| 5, reconstructions | `figures/nc_fig_recon.py` (panels by `figures/aa_render.py`) |
| 6, night side | `figures/aa_figs.py night` (panels by `figures/aa_render2.py`) |
| 7, global error maps | `figures/nc_fig_maps.py` (maps by `figures/err_maps.py`) |
| 8, Ryugu at three phase angles | `figures/aa_figs.py ryugu_phase` |
| 9, lunar-Lambert parameter | `figures/aa_figs.py lambert` |
| 10, phase rule | `figures/nc_fig_steps.py` |
| 11, held-out images | `figures/nc_fig_nvs.py` |
| project-page videos | `figures/aa_turntable.py` |
