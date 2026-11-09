# GPU runbook (tested target: RTX PRO 6000 96 GB / H200 / A100-80GB)

```bash
# 1. ship code + pre-rendered TriSAR frames (from the Windows box)
scp nightwatch_code.tgz trisar_fused.tgz user@gpu:~/
# 2. on the GPU box
mkdir -p ~/nightwatch && cd ~/nightwatch && tar xzf ~/nightwatch_code.tgz && tar xzf ~/trisar_fused.tgz -C data/
tmux new -s nw
bash scripts/setup.sh 2>&1 | tee logs_setup.txt     # deps, LLVIP 4 GB, Qwen3.5-9B 19 GB, DINOv3 weights
WORKERS=16 bash scripts/run_all.sh                  # LLVIP stage: zero-shot, 3x SFT, eval, pseudo-labels, fast heads, report
WORKERS=16 bash scripts/run_trisar.sh               # TriSAR stage: RGB -> +thermal -> +LiDAR ladder, slow + fast
python -m nightwatch.report.make_report             # one report with both stages
# optional: STAGES="grpo" bash scripts/run_slow.sh ; bigger slow head: NW_MODEL=Qwen/Qwen3.6-27B
```

Without `trisar_fused.tgz`, `run_trisar.sh` downloads the 21.8 GB zip from TUdatalib and renders it itself (~2 min on 16 cores).

Rough wall-clock estimates: H200 about 5 h for both stages; RTX PRO 6000 about 6-7 h (lower memory bandwidth slows generation);
A100-80GB about 10 h (use `EPOCHS=1`). Every stage is resumable.

Watch: `tail -f logs/slow.log | grep -E "step=|img/s|Error"`. Bring back: `runs/*/*.json*`, `runs/*/log.jsonl`, `report/`.
