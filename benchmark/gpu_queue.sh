#!/usr/bin/env bash
# Polite single-GPU queue for a shared GPU node: runs the job list one run at a
# time (benchmark/run_job.sh) on ONE idle GPU, with hard limits so a run cannot crowd other users out.
# - GPU: before each run, picks an idle card (< 500 MiB used, 0% utilisation) among $GPUS; if
#   none is idle it waits. Never more than one card at a time.
# - Memory: a watchdog kills the run if the resident memory of its process group exceeds MAX_GB
#   (default 32) or its GPU memory exceeds MAX_VRAM_GB (default 24); the event is logged.
# - CPU: confined to CORES (default 8 cores) at nice 10.
# Job list lines: <method> <body> <split> [variant|-] [extra train flags...]  (# comments allowed)
# usage: GPUS="1 4" setsid nohup bash benchmark/gpu_queue.sh <joblist> > logs/gpu_queue_<name>.log 2>&1 < /dev/null &
LIST=$1
# paths: the repository root from this file; machine-specific settings (conda prefix, data) in
# benchmark/env.local.sh, see benchmark/env.example.sh
D=${AM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -f "$D/benchmark/env.local.sh" ] && source "$D/benchmark/env.local.sh"
R=${BENCH_PREFIX:-$HOME}
GPUS=${GPUS:-"1 4"}; MAX_GB=${MAX_GB:-32}; MAX_VRAM_GB=${MAX_VRAM_GB:-24}; CORES=${CORES:-0-7}
idle_gpu() {
  for g in $GPUS; do
    read -r used util < <(nvidia-smi -i $g --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
    [ "${used:-99999}" -lt 500 ] && [ "${util:-100}" -eq 0 ] && { echo $g; return; }
  done
}
grep -v '^\s*#' "$LIST" | grep -v '^\s*$' | while read -r METHOD BODY SPLIT VARIANT EXTRA; do
  TAG=$METHOD${VARIANT:+-$VARIANT}; [ "$VARIANT" = "-" ] && TAG=$METHOD
  if grep -q '"matched"' $D/results/img/$BODY/${TAG}_s$SPLIT/nvs_table.json 2>/dev/null && \
     ls $D/results/img/$BODY/${TAG}_s$SPLIT/table_*_sfm.json > /dev/null 2>&1; then
    echo "$(date -Is) SKIP done $TAG $BODY s$SPLIT"; continue
  fi
  G=$(idle_gpu)
  while [ -z "$G" ]; do echo "$(date -Is) waiting for an idle GPU among: $GPUS"; sleep 300; G=$(idle_gpu); done
  setsid taskset -c $CORES nice -n 10 bash $D/benchmark/run_job.sh $METHOD $BODY $SPLIT $G ${VARIANT:--} $EXTRA < /dev/null &
  PID=$!; SID=$PID; peak=0; vpeak=0; T0=$(date +%s)
  while kill -0 $PID 2>/dev/null; do
    rss=$(ps -o rss= -s $SID 2>/dev/null | awk '{s+=$1} END {print s+0}')
    pids=$(ps -o pid= -s $SID 2>/dev/null | tr -d ' ' | paste -sd'|')
    vram=$(nvidia-smi -i $G --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | \
           awk -F', ' -v p="^($pids)$" '$1 ~ p {s+=$2} END {print s+0}')
    [ $rss -gt $peak ] && peak=$rss; [ $vram -gt $vpeak ] && vpeak=$vram
    if [ $rss -gt $((MAX_GB * 1048576)) ] || [ $vram -gt $((MAX_VRAM_GB * 1024)) ]; then
      echo "$(date -Is) KILLED $TAG $BODY s$SPLIT: RAM $((rss / 1048576)) GB (cap $MAX_GB), VRAM $((vram / 1024)) GB (cap $MAX_VRAM_GB)"
      kill -- -$SID 2>/dev/null; sleep 5; kill -9 -- -$SID 2>/dev/null
    fi
    sleep 10
  done
  wait $PID; rc=$?
  echo "$(date -Is) END $TAG $BODY s$SPLIT gpu $G exit $rc, $(( ($(date +%s) - T0) / 60 )) min, peak RAM $((peak / 1048576)) GB, peak VRAM $((vpeak / 1024)) GB"
done
echo "$(date -Is) QUEUE_DONE $LIST"
