#!/usr/bin/env bash
# Environments for the baselines added on 2026-11-03 (3DGS, Mip-Splatting, RaDe-GS, SuGaR) on the
# one venv per method on top of the `gs` env (torch 2.1.2 + CUDA 11.8, built by
# img_env_setup.sh), because each ships its own `diff_gaussian_rasterization`. Compiled with
# /usr/local/cuda-11.8, 8 cores at low priority. Idempotent: rerun after failures.
# usage: taskset -c 0-7 nice -n 19 bash benchmark/baselines/setup_new_baselines.sh [venv ...] > logs/setup_new_baselines.log 2>&1
set -eo pipefail
# paths: the repository root from this file; machine-specific settings (conda prefix, data) in
# benchmark/env.local.sh, see benchmark/env.example.sh
D=${AM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
[ -f "$D/benchmark/env.local.sh" ] && source "$D/benchmark/env.local.sh"
R=${BENCH_PREFIX:-$HOME}
B=$R/miniconda3/envs/gs
V=$R/venvs
WANT=${*:-"v_3dgs v_mip v_rade v_sugar"}
export CUDA_HOME=/usr/local/cuda-11.8 PATH=/usr/local/cuda-11.8/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=${MAX_JOBS:-6} FORCE_CUDA=1
export LD_LIBRARY_PATH=$B/lib:${LD_LIBRARY_PATH:-}

mkdir -p $D/repos $V && cd $D/repos
# shallow clones; submodules except the (large, unused) SIBR viewers, fetched in full because the
# pinned commits are not branch tips; simple-knn (gitlab.inria.fr, unreachable from some clusters) is
# the identical copy from the 2DGS checkout
retry() { for i in 1 2 3 4 5; do "$@" && return 0; echo "retry $i: $*"; sleep 60; done; return 1; }
clone() {  # url dir [branch]
  [ -d $2/.git ] || retry bash -c "rm -rf $2 && LD_LIBRARY_PATH= git clone --depth 1 ${3:+--branch $3} $1 $2"
  (cd $2 && for p in $(git config -f .gitmodules --get-regexp path 2>/dev/null | awk '{print $2}' | grep -v SIBR); do
     case $p in
       *simple-knn) [ -f $p/simple_knn.cu ] || { rm -rf $p; cp -r $D/repos/2d-gaussian-splatting/submodules/simple-knn $p; rm -rf $p/.git; } ;;
       *) retry env LD_LIBRARY_PATH= git submodule update --init --recursive $p ;;
     esac
   done)
}
clone https://github.com/graphdeco-inria/gaussian-splatting.git gaussian-splatting
clone https://github.com/autonomousvision/mip-splatting.git mip-splatting
clone https://github.com/BaowenZ/RaDe-GS.git RaDe-GS
clone https://github.com/Anttwo/SuGaR.git SuGaR
for r in gaussian-splatting mip-splatting RaDe-GS SuGaR; do
  echo "$r $(LD_LIBRARY_PATH= git -C $r rev-parse --short HEAD) $(LD_LIBRARY_PATH= git -C $r remote get-url origin)"
done > COMMITS_new_baselines.txt
cat COMMITS_new_baselines.txt
# fixed splits + object extent, as for the other baselines
$B/bin/python $D/benchmark/baselines/img_patch_splits.py gaussian-splatting mip-splatting RaDe-GS SuGaR/gaussian_splatting

mkv() { [ -x $V/$1/bin/python ] || $B/bin/python -m venv --system-site-packages $V/$1; }
pipx() { $V/$1/bin/pip install -q --no-build-isolation "${@:2}"; }
for v in $WANT; do
  echo "=== $v"; mkv $v
  case $v in
    v_3dgs)  pipx $v $D/repos/gaussian-splatting/submodules/diff-gaussian-rasterization \
                     $D/repos/gaussian-splatting/submodules/simple-knn \
                     $D/repos/gaussian-splatting/submodules/fused-ssim ;;
    v_mip)   pipx $v $D/repos/mip-splatting/submodules/diff-gaussian-rasterization \
                     $D/repos/mip-splatting/submodules/simple-knn ;;
    v_rade)  pipx $v $D/repos/RaDe-GS/submodules/diff-gaussian-rasterization \
                     $D/repos/RaDe-GS/submodules/simple-knn ;;
    v_sugar) pipx $v $D/repos/SuGaR/gaussian_splatting/submodules/diff-gaussian-rasterization \
                     $D/repos/SuGaR/gaussian_splatting/submodules/simple-knn
             # local clones: pip's own git would run with the env's LD_LIBRARY_PATH (libffi clash)
             clone https://github.com/facebookresearch/pytorch3d.git pytorch3d v0.7.5
             clone https://github.com/NVlabs/nvdiffrast.git nvdiffrast
             $V/$v/bin/python -c "import pytorch3d" 2>/dev/null || pipx $v $D/repos/pytorch3d
             $V/$v/bin/python -c "import nvdiffrast.torch" 2>/dev/null || pipx $v $D/repos/nvdiffrast ;;
  esac
  $V/$v/bin/python -c "import torch, diff_gaussian_rasterization, simple_knn; print('$v import ok', torch.__version__)"
done
echo NEW_BASELINES_SETUP_DONE
