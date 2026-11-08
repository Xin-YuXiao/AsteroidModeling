#!/usr/bin/env bash
# Build the `gs` env for image-domain Gaussian-splatting baselines
# (2DGS, GOF, Asteroid-GS) and our method. torch 2.1.2 + CUDA 11.8 matches the
# local nvcc (/usr/local/cuda-11.8); download.pytorch.org is blocked here, so
# torch comes from the conda pytorch channel. Idempotent: rerun after failures.
set -ex
# paths: the repository root from this file; machine-specific settings (conda prefix, data) in
# benchmark/env.local.sh, see benchmark/env.example.sh
D=${AM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
[ -f "$D/benchmark/env.local.sh" ] && source "$D/benchmark/env.local.sh"
R=${BENCH_PREFIX:-$HOME}
C=$R/miniconda3/bin/conda
ENV=$R/miniconda3/envs/gs
export CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=16

[ -x $ENV/bin/python ] || $C create -y -p $ENV python=3.10
# mkl pin: pytorch-channel 2.1 builds break on mkl>=2024.1 (undefined iJIT_NotifyEvent)
$ENV/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
  $C install -y -p $ENV pytorch=2.1.2 torchvision=0.16.2 pytorch-cuda=11.8 mkl=2023.1.0 -c pytorch -c nvidia
# CGAL/GMP/cmake for GOF's tetrahedra triangulation
$C install -y -p $ENV -c conda-forge cgal-cpp gmp cmake=3.27 eigen
P=$ENV/bin/pip
$P install -q plyfile tqdm opencv-python-headless open3d trimesh scipy lpips scikit-image \
  astropy spiceypy matplotlib ninja rtree mediapy "numpy<2"
# torch 2.1 cpp_extension imports pkg_resources (removed from recent setuptools)
$P install -q "setuptools<70" wheel

mkdir -p $D/repos && cd $D/repos
[ -d 2d-gaussian-splatting ] || git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git
[ -d gaussian-opacity-fields ] || git clone --recursive https://github.com/autonomousvision/gaussian-opacity-fields.git
[ -d Asteroid-GS ] || git clone --recursive https://github.com/zhangxiaojie1243fei-crypto/Asteroid-GS.git || true

# GOF and Asteroid-GS both ship a fork named diff_gaussian_rasterization, so
# Asteroid-GS gets a clone of this env (gsast). 2DGS's rasterizer has its own name.
NBI="--no-build-isolation"
cd $D/repos/2d-gaussian-splatting
$P install -q $NBI submodules/diff-surfel-rasterization submodules/simple-knn
[ -x $R/miniconda3/envs/gsast/bin/python ] || $C create -y -p $R/miniconda3/envs/gsast --clone $ENV
cd $D/repos/gaussian-opacity-fields
$P install -q $NBI submodules/diff-gaussian-rasterization
cd submodules/tetra-triangulation
cmake -DCMAKE_PREFIX_PATH=$ENV . && make -j16 && $P install -q $NBI -e .
PA=$R/miniconda3/envs/gsast/bin/pip
cd $D/repos/Asteroid-GS
$PA install -q $NBI submodules/diff-gaussian-rasterization submodules/simple-knn
cd submodules/tetra_triangulation
cmake -DCMAKE_PREFIX_PATH=$R/miniconda3/envs/gsast . && make -j16 && $PA install -q $NBI -e .
$ENV/bin/python -c "import torch, diff_surfel_rasterization, diff_gaussian_rasterization, simple_knn, tetranerf; print('GS_ENV_OK', torch.__version__, torch.cuda.is_available())"
$R/miniconda3/envs/gsast/bin/python -c "import torch, diff_gaussian_rasterization, simple_knn, tetranerf; print('GSAST_ENV_OK')"
