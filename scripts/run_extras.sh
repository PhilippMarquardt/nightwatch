#!/usr/bin/env bash
# Extra controls. Resumable; each step skipped if its output exists.
#   stretch : test-time thermal contrast stretch on the TriSAR test set (hypothesis 2 for the forest failure:
#             the forest is a thermal *distribution shift*, not an uninformative sensor)
#   self    : self-training control for distillation (student trained on its own pseudo-labels)
set -uo pipefail
cd "$(dirname "$0")/.."
EXTRAS=${EXTRAS:-"stretch self"}
has() { [ -s "$1" ]; }
if [[ " $EXTRAS " == *" stretch "* ]]; then
  for r in tf_rgbtl tf_tl; do
    [ -s runs/$r/train_summary.json ] || continue
    has runs/$r/pred_test_stretch.jsonl || python -m nightwatch.fast.trisar_eval --name $r --conds thermal_stretch \
        --out runs/$r/pred_test_stretch.jsonl
    thr=$(python -c "import json;print(json.load(open('runs/$r/train_summary.json'))['score_thr'])")
    python -m nightwatch.eval.metrics_trisar --pred runs/$r/pred_test_stretch.jsonl --score-thr $thr \
        --out runs/$r/pred_test_stretch_metrics.json > /dev/null
  done
  for r in t_sft_fusion_l t_sft_fusion_l_cd50; do
    [ -s runs/$r/train_summary.json ] || continue
    has runs/$r/pred_test_stretch.jsonl || python -m nightwatch.slow.infer --dataset trisar --fused-dir data/trisar_fused \
        --mode fusion_l --adapter runs/$r --split test --tasks report --conds thermal_stretch,rgb_off --limit 600 \
        --bs ${IBS:-48} --workers 12 --out runs/$r/pred_test_stretch.jsonl
    python -m nightwatch.eval.metrics_trisar --pred runs/$r/pred_test_stretch.jsonl --out runs/$r/pred_test_stretch_metrics.json > /dev/null
  done
fi
if [[ " $EXTRAS " == *" self "* ]] && [ -s runs/fast_gt_labeled/train_summary.json ]; then
  has runs/pseudo_labels_unlabeled_self.json || python -m nightwatch.fast.label_split --run runs/fast_gt_labeled \
      --split unlabeled --out runs/pseudo_labels_unlabeled_self.json
  VARIANTS=selfdistill bash scripts/run_fast.sh
fi
if [[ " $EXTRAS " == *" shuffle "* ]]; then   # aligned vs misaligned sensors: is the fusion used *spatially*?
  if [ -s runs/sft_fusion/train_summary.json ]; then
    has runs/sft_fusion/pred_test_shuffle.jsonl || python -m nightwatch.slow.infer --mode fusion --adapter runs/sft_fusion         --split test --tasks detect --conds thermal_shuffle --limit 800 --bs ${IBS:-48} --workers 12 --out runs/sft_fusion/pred_test_shuffle.jsonl
    python -m nightwatch.eval.metrics --pred runs/sft_fusion/pred_test_shuffle.jsonl --out runs/sft_fusion/pred_test_shuffle_metrics.json > /dev/null
  fi
  for r in t_sft_fusion_l t_sft_fusion_l_cd50; do
    [ -s runs/$r/train_summary.json ] || continue
    has runs/$r/pred_test_shuffle.jsonl || python -m nightwatch.slow.infer --dataset trisar --fused-dir data/trisar_fused         --mode fusion_l --adapter runs/$r --split test --tasks report --conds sensors_shuffle --limit 600         --bs ${IBS:-48} --workers 12 --out runs/$r/pred_test_shuffle.jsonl
    python -m nightwatch.eval.metrics_trisar --pred runs/$r/pred_test_shuffle.jsonl --out runs/$r/pred_test_shuffle_metrics.json > /dev/null
  done
fi
if [[ " $EXTRAS " == *" matched "* ]]; then   # compute-matched labelled-only student (same optimiser updates as distill)
  VARIANTS=gt_labeled_long bash scripts/run_fast.sh
fi
echo EXTRAS_DONE
