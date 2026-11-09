#!/usr/bin/env bash
# CPU-only wrap-up: re-score everything with the right thresholds, same-subset comparisons, escalation curves,
# regression gates, report. Safe to re-run at any time.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
thr_of() { python -c "import json;print(json.load(open('runs/$1/train_summary.json'))['score_thr'])"; }
{
STAGES=none bash scripts/run_slow.sh       # re-score LLVIP slow runs
STAGES=none bash scripts/run_trisar.sh     # re-score TriSAR runs (finer range bins, observability, false alarms)

# LLVIP fast heads: full test, plus the SAME frames the slow heads were scored on (1500 normal / 800 robust subsets)
for r in fast_gt_labeled fast_gt_labeled_long fast_oracle fast_distill fast_distill_consensus fast_selfdistill fast_distill_rgbteacher fast_oracle_rgb fast_oracle_t; do
  [ -s runs/$r/train_summary.json ] || continue
  thr=$(thr_of $r)
  python -m nightwatch.eval.metrics --pred runs/$r/pred_test.jsonl --score-thr $thr --out runs/$r/pred_test_metrics.json > /dev/null
  [ -s runs/sft_fusion/pred_test.jsonl ] && python -m nightwatch.eval.metrics --pred runs/$r/pred_test.jsonl --score-thr $thr \
      --ids-from runs/sft_fusion/pred_test.jsonl --out runs/$r/pred_test_sub1500_metrics.json > /dev/null
  [ -s runs/sft_fusion/pred_test_robust.jsonl ] && python -m nightwatch.eval.metrics --pred runs/$r/pred_test.jsonl --score-thr $thr \
      --ids-from runs/sft_fusion/pred_test_robust.jsonl --out runs/$r/pred_test_sub800_metrics.json > /dev/null
done
# TriSAR fast heads on the slow heads' 600-frame robustness/QA subset
for r in tf_rgbtl tf_tl tf_rgb tf_rgbt; do
  [ -s runs/$r/train_summary.json ] || continue
  [ -s runs/t_sft_fusion_l/pred_test_robust.jsonl ] && python -m nightwatch.eval.metrics_trisar --pred runs/$r/pred_test.jsonl \
      --score-thr $(thr_of $r) --ids-from runs/t_sft_fusion_l/pred_test_robust.jsonl --out runs/$r/pred_test_sub600_metrics.json > /dev/null
done

python -m nightwatch.eval.escalation || true
python -m nightwatch.eval.router_trisar || true
[ -f nightwatch/eval/escalation_trisar.py ] && python -m nightwatch.eval.escalation_trisar || true
g() { [ -s runs/$1 ] && [ -s runs/$2 ] && python -m nightwatch.eval.gate --candidate runs/$1 --baseline runs/$2 --out runs/$3 > /dev/null; true; }
g sft_fusion/pred_test_metrics.json         sft_rgb/pred_test_metrics.json           gate_fusion_vs_rgb.json
g sft_fusion/pred_test_robust_metrics.json  sft_rgb/pred_test_robust_metrics.json    gate_fusion_vs_rgb_robust.json
g sft_fusion/pred_test_metrics.json         sft_thermal/pred_test_metrics.json       gate_fusion_vs_thermal.json
g sft_fusion/pred_test_metrics.json         sft_dual/pred_test_metrics.json          gate.json
g sft_fusion/pred_test_robust_metrics.json  sft_dual/pred_test_robust_metrics.json   gate_robust.json
g fast_distill/pred_test_metrics.json       fast_gt_labeled/pred_test_metrics.json   gate_distill.json
g fast_distill/pred_test_metrics.json       fast_selfdistill/pred_test_metrics.json  gate_distill_vs_self.json
g t_sft_fusion_l/pred_test_metrics.json     t_sft_rgb/pred_test_metrics.json         gate_trisar.json
g t_sft_fusion_l_cd50/pred_test_metrics.json t_sft_fusion_l/pred_test_metrics.json gate_trisar_cd50.json
g t_sft_fusion_l_cd50/pred_test_robust_metrics.json t_sft_fusion_l/pred_test_robust_metrics.json gate_trisar_cd50_robust.json
python -m nightwatch.eval.site_bootstrap --fig --runs zs_rgb zs_dual sft_rgb sft_thermal sft_dual sft_fusion     fast_gt_labeled fast_gt_labeled_s1 fast_gt_labeled_s2 fast_gt_labeled_long fast_selfdistill fast_selfdistill_s1 fast_selfdistill_s2 fast_distill fast_distill_s1 fast_distill_s2 fast_distill_rgbteacher fast_distill_consensus fast_distill_consensus_s1 fast_distill_consensus_s2 fast_oracle     --pairs sft_fusion:sft_rgb sft_fusion:sft_thermal sft_fusion:sft_dual fast_distill:fast_gt_labeled fast_distill_s1:fast_gt_labeled_s1 fast_distill_s2:fast_gt_labeled_s2     fast_distill:fast_selfdistill fast_distill_s1:fast_selfdistill_s1 fast_distill_s2:fast_selfdistill_s2 fast_distill:fast_gt_labeled_long fast_distill:fast_distill_rgbteacher fast_distill_consensus:fast_gt_labeled fast_distill_consensus_s1:fast_gt_labeled_s1 fast_distill_consensus_s2:fast_gt_labeled_s2 fast_distill_consensus:fast_distill fast_oracle:fast_gt_labeled || true
python -m nightwatch.report.thr_sweep || true
python -m nightwatch.report.teacher_sites || true
python -m nightwatch.report.make_report
} >> logs/finalize.log 2>&1
echo "FINALIZE_DONE $(date)" | tee -a logs/finalize.log
