#!/usr/bin/env bash
# Resource use of THIS project's processes on a shared node (only processes of this user whose
# command line points at AsteroidModeling, its venvs, or the CUDA compiler during env builds).
# Prints one line every SUMMARY_S seconds (default 900) and immediately whenever a limit is
# exceeded: RAM > MAX_RAM_GB (48), more than MAX_GPUS cards (1), VRAM > MAX_VRAM_GB (24),
# CPU > MAX_CPU_PCT (1700 % = 17 cores).
# usage: bash benchmark/usage_monitor.sh        (for the Monitor tool: one line = one notification)
MAX_RAM_GB=${MAX_RAM_GB:-48}; MAX_GPUS=${MAX_GPUS:-1}; MAX_VRAM_GB=${MAX_VRAM_GB:-24}
MAX_CPU_PCT=${MAX_CPU_PCT:-1700}; SUMMARY_S=${SUMMARY_S:-900}
PAT='AsteroidModeling|/venvs/v_|repos/pcgs|cuda-11.8/bin|setup_new_baselines|gpu_queue'
D=${AM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# a process belongs to the project if its command line matches PAT or it runs inside the project
# directory (jobs started with relative paths, e.g. benchmark/figures/*.py from $D)
ours() { case "$(readlink /proc/$1/cwd 2>/dev/null)" in "$D"*) return 0 ;; esac
         { tr '\0' ' ' < /proc/$1/cmdline; } 2>/dev/null | grep -qE "$PAT"; }
last=0
while true; do
  read -r rss cpu < <(ps -u "$USER" -o pid=,rss=,pcpu=,args= | grep -v -E "usage_monitor" | \
                      while read -r p r c a; do ours $p && echo "$r $c"; done | \
                      awk '{r+=$1; c+=$2} END {printf "%d %d\n", r/1048576, c}')
  gpu=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=', ' read -r u p m; do
          [ "$(stat -c %U /proc/$p 2>/dev/null)" = "$USER" ] && ours $p && echo "$u $m"
        done)
  ngpu=$(echo "$gpu" | awk 'NF {print $1}' | sort -u | wc -l)
  vram=$(echo "$gpu" | awk 'NF {s+=$2} END {printf "%d", s/1024}')
  alert=""
  [ "$rss" -gt "$MAX_RAM_GB" ] && alert="$alert RAM"
  [ "$ngpu" -gt "$MAX_GPUS" ] && alert="$alert GPUS"
  [ "$vram" -gt "$MAX_VRAM_GB" ] && alert="$alert VRAM"
  [ "$cpu" -gt "$MAX_CPU_PCT" ] && alert="$alert CPU"
  now=$(date +%s)
  if [ -n "$alert" ] || [ $((now - last)) -ge "$SUMMARY_S" ]; then
    echo "$(date +%H:%M) $(hostname) RAM ${rss}G CPU ${cpu}% GPUs ${ngpu} VRAM ${vram}G${alert:+  ALERT:$alert}"
    last=$now
  fi
  sleep 60
done
