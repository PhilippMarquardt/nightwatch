#!/usr/bin/env bash
# Pre-registered Track A criteria for one run: normal vs sensor-off on the fixed 600-frame robustness subset.
#   bash scripts/score_criteria.sh t_sft_tri
cd "$(dirname "$0")/.."
R=runs/$1
python -m nightwatch.eval.metrics_trisar --pred $R/pred_test.jsonl --ids-from $R/pred_test_robust.jsonl --out /tmp/$1_n600.json > /dev/null
python - "$1" <<'PY'
import json, sys
r = sys.argv[1]
n = json.load(open(f"/tmp/{r}_n600.json"))["report/normal"]
rb = json.load(open(f"runs/{r}/pred_test_robust_metrics.json"))
full = json.load(open(f"runs/{r}/pred_test_metrics.json"))["report/normal"]
g = lambda d, s: d.get(s, {}).get("f1", float("nan"))
lab0, for0 = g(n, "loc_lab"), g(n, "loc_forest")
row = {c: (round(g(rb[f"report/{c}"], "loc_lab"), 3), round(g(rb[f"report/{c}"], "loc_forest"), 3))
       for c in ("rgb_off", "thermal_off", "lidar_off", "sensors_shuffle") if f"report/{c}" in rb}
print(r, "normal(600) lab", round(lab0, 3), "forest", round(for0, 3), "| sensor-off (lab, forest):", row)
print("  A1 camera-off lab >= 0.50:", row["rgb_off"][0])
print("  A2 lab drop: thermal_off", round(lab0 - row["thermal_off"][0], 3), "lidar_off", round(lab0 - row["lidar_off"][0], 3),
      "shuffle", round(lab0 - row["sensors_shuffle"][0], 3), "| forest drop: lidar_off", round(for0 - row["lidar_off"][1], 3))
fp = full["loc_forest"]
print("  A3 all-frame P", round(full["all"]["precision"], 3), "forest P", round(fp["precision"], 3), "forest F1", round(fp["f1"], 3))
PY
