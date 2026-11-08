#!/usr/bin/env bash
# Full Night Watch pipeline on one GPU box. Resumable: finished stages are skipped.
#   bash scripts/setup.sh && bash scripts/run_all.sh
# Logs: logs/*.log    Results: runs/    Report: report/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# The fast head is tiny: train its GT/ablation variants in parallel with the slow head.
( VARIANTS="oracle gt_labeled oracle_rgb oracle_t" bash scripts/run_fast.sh > logs/fast_a.log 2>&1 ) &
FAST_PID=$!

STAGES="zeroshot sft eval pseudo" bash scripts/run_slow.sh 2>&1 | tee logs/slow.log

wait $FAST_PID || echo "fast ablations had an error - see logs/fast_a.log"
VARIANTS="distill" bash scripts/run_fast.sh 2>&1 | tee logs/fast_b.log

python -m nightwatch.eval.escalation 2>&1 | tee logs/escalation.log || true
python -m nightwatch.eval.gate --candidate runs/sft_fusion/pred_test_metrics.json --baseline runs/sft_dual/pred_test_metrics.json --out runs/gate.json \
    2>&1 | tee logs/gate.log || true
python -m nightwatch.eval.gate --candidate runs/sft_fusion/pred_test_robust_metrics.json     --baseline runs/sft_dual/pred_test_robust_metrics.json --out runs/gate_robust.json 2>&1 | tee -a logs/gate.log || true
python -m nightwatch.report.make_report 2>&1 | tee logs/report.log
echo "ALL DONE -> report/index.html"
