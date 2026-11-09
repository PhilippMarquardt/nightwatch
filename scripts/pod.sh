#!/usr/bin/env bash
# Single entry point for everything on the GPU pod (one approval covers all of it).
#   bash scripts/pod.sh sync              # push code (nightwatch/, scripts/, docs) to the pod
#   bash scripts/pod.sh run "<cmd>"       # run a shell command in /root/nightwatch on the pod
#   bash scripts/pod.sh bg <name> "<cmd>" # start a command in a new tmux window (logs/<name>.log)
#   bash scripts/pod.sh status            # GPU, tmux windows, recent log lines, finished runs
#   bash scripts/pod.sh pull              # fetch results (runs/*.json*, report/) back to this machine
set -euo pipefail
if [ "${POD:-1}" = 2 ]; then NW_POD_HOST=${NW_POD_HOST:-root@POD2_IP}; NW_POD_PORT=${NW_POD_PORT:-POD2_PORT}; fi   # second pod (2026-11-09)
HOST=${NW_POD_HOST:-root@POD1_IP}
PORT=${NW_POD_PORT:-POD1_PORT}
KEY=${NW_POD_KEY:-$HOME/.ssh/runpod_ed25519}
REMOTE=/root/nightwatch
cd "$(dirname "$0")/.."
SSH=(ssh -p "$PORT" -i "$KEY" -o ConnectTimeout=20 "$HOST")

case "${1:-status}" in
  sync)
    tar czf - nightwatch scripts requirements.txt README.md RUNBOOK.md | "${SSH[@]}" "cd $REMOTE && tar xzf - --no-same-owner" && echo synced ;;
  run)
    "${SSH[@]}" "cd $REMOTE && $2" ;;
  bg)
    "${SSH[@]}" "cd $REMOTE && mkdir -p logs && tmux new-window -t nw -n $2 \"bash -lc '$3' 2>&1 | tee logs/$2.log; exec bash\"" && echo "started $2" ;;
  status)
    "${SSH[@]}" "cd $REMOTE && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; tmux list-windows -t nw -F '#W' | tr '\n' ' '; echo;
      for f in \$(ls -t logs/*.log 2>/dev/null | head -4); do echo \"== \$f\"; tail -c 1500 \$f | tr '\r' '\n' | grep -v -E 'Loading weights|Warning|warn' | tail -4 | cut -c1-220; done;
      echo '== finished:'; ls runs/*/train_summary.json runs/*/*_metrics.json 2>/dev/null | tr '\n' ' '; echo" ;;
  pull)
    mkdir -p runs report
    "${SSH[@]}" "cd $REMOTE && mkdir -p report && tar czf - --exclude='*.safetensors' --exclude='*.pt' --exclude='*.engine' --exclude='*.onnx' runs report 2>/dev/null" | tar xzf - && echo pulled ;;
  *)
    echo "unknown command $1"; exit 2 ;;
esac
