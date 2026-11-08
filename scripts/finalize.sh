#!/usr/bin/env bash
# CPU-only wrap-up: re-score everything with the right thresholds, escalation curves, regression gates, report.
# Safe to re-run at any time.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
{
STAGES=none bash scripts/run_slow.sh       # re-score LLVIP slow runs
STAGES=none bash scripts/run_trisar.sh     # re-score TriSAR runs
for r in fast_gt_labeled fast_oracle fast_distill fast_oracle_rgb fast_oracle_t; do
  [ -s runs/$r/train_summary.json ] || continue
  thr=$(python -c "import json;print(json.load(open('runs/$r/train_summary.json'))['score_thr'])")
  python -m nightwatch.eval.metrics --pred runs/$r/pred_test.jsonl --score-thr $thr --out runs/$r/pred_test_metrics.json > /dev/null
done
python -m nightwatch.eval.escalation || true
[ -f nightwatch/eval/escalation_trisar.py ] && python -m nightwatch.eval.escalation_trisar || true
g() { [ -s runs/$1 ] && [ -s runs/$2 ] && python -m nightwatch.eval.gate --candidate runs/$1 --baseline runs/$2 --out runs/$3 > /dev/null; true; }
g sft_fusion/pred_test_metrics.json        sft_rgb/pred_test_metrics.json         gate_fusion_vs_rgb.json
g sft_fusion/pred_test_robust_metrics.json sft_rgb/pred_test_robust_metrics.json  gate_fusion_vs_rgb_robust.json
g sft_fusion/pred_test_metrics.json        sft_dual/pred_test_metrics.json        gate.json
g sft_fusion/pred_test_robust_metrics.json sft_dual/pred_test_robust_metrics.json gate_robust.json
g fast_distill/pred_test_metrics.json      fast_gt_labeled/pred_test_metrics.json gate_distill.json
g t_sft_fusion_l/pred_test_metrics.json    t_sft_rgb/pred_test_metrics.json       gate_trisar.json
python -m nightwatch.report.make_report
} >> logs/finalize.log 2>&1
echo "FINALIZE_DONE $(date)" | tee -a logs/finalize.log
