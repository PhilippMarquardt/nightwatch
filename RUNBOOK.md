# GPU runbook

```bash
# 1. ship the code (from the Windows box)
scp C:/Users/pmarq/source/repos/nightwatch_code.tgz user@gpu:~/
# 2. on the GPU box
mkdir -p ~/nightwatch && cd ~/nightwatch && tar xzf ~/nightwatch_code.tgz
tmux new -s nw
bash scripts/setup.sh 2>&1 | tee logs_setup.txt          # ~10 min (LLVIP 4 GB + Qwen3.5-9B 19 GB)
WORKERS=16 bash scripts/run_all.sh                       # resumable; Ctrl-b d to detach
# bigger slow head on an H200 (slower, ~3x SFT time): NW_MODEL=Qwen/Qwen3.6-27B
# optional, if time is left afterwards:
STAGES="grpo" bash scripts/run_slow.sh && python -m nightwatch.report.make_report
```

Estimated wall-clock on one H200: zero-shot 20 min, 3x SFT ~75 min, eval ~35 min, pseudo-labels ~15 min,
fast heads in parallel. Total about 2.5-3 h. On an A100-80GB, roughly double; use `EPOCHS=1` to halve SFT.

Watch: `tail -f logs/slow.log | grep -E "step=|img/s|Error"`, `tail -f logs/fast_a.log`.
Bring back: `runs/*/{*.json,*.jsonl,log.jsonl}` and `report/` (skip `*.safetensors`, `best.pt`, engines).
