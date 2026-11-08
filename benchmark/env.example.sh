# Machine-specific settings of the benchmark scripts. Copy to benchmark/env.local.sh (not tracked) and edit;
# run_method.sh, run_job.sh, gpu_queue.sh and the baseline setup scripts source it if it exists.

# prefix under which the conda installation (miniconda3/envs/...) and the baseline venvs (venvs/...) live
export BENCH_PREFIX=$HOME

# interpreters (defaults: $BENCH_PREFIX/miniconda3/envs/<env>/bin/python, see run_method.sh)
# export PY_GS=$BENCH_PREFIX/miniconda3/envs/gs/bin/python        # 2DGS, GOF, PCGS, evaluation
# export PY_AGS=$BENCH_PREFIX/miniconda3/envs/gsast/bin/python    # Asteroid-GS
# export PY_PGSR=$BENCH_PREFIX/miniconda3/envs/pgsr/bin/python    # PGSR
# export PY_3DGS=$BENCH_PREFIX/venvs/v_3dgs/bin/python            # 3DGS, Mip-Splatting (setup_new_baselines.sh)
# export PY_MIP=$BENCH_PREFIX/venvs/v_mip/bin/python

# the CUDA toolkit used to build the extensions
# export CUDA_HOME=/usr/local/cuda-11.8
