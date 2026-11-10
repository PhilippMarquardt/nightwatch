"""Print the tri_pc3 stage-1 gate (G1) for one run dir: lidar_only report F1 on the validation lab sequences.
python scripts/pc3_gate.py runs/t_sft_tri_pc3a [--check]     (--check: exit code 0 if F1 >= 0.30, else 3)
"""
import json
import sys

run = sys.argv[1]
m = json.load(open(run + "/pred_val_metrics.json"))["report/lidar_only"]
lab = m.get("loc_lab", m["all"])
print("G1_VAL", run, "lidar_only val lab F1", round(lab["f1"], 4), "| all F1", round(m["all"]["f1"], 4),
      "P", round(m["all"]["precision"], 3), "R", round(m["all"]["recall"], 3), flush=True)
if "--check" in sys.argv:
    sys.exit(0 if lab["f1"] >= 0.30 else 3)
