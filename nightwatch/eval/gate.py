"""Regression gate: a candidate that is worse on ANY slice does not ship.

"Worse" = below the baseline by more than the sampling noise of that slice
(2 x binomial standard error on the slice's F1 / accuracy, floor 0.5 points), so
small slices can't flap the gate and large slices can't hide a real regression.

python -m nightwatch.eval.gate --candidate runs/a/pred_test_metrics.json --baseline runs/b/pred_test_metrics.json
Exit code 1 on any regression -> drop into CI as-is.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

KEY_METRIC = {"detect": "f1", "report": "f1", "count": "count_exact", "zone": "accuracy", "lying": "accuracy",
              "nearest": "nearest_mae_m"}
LOWER_BETTER = {"nearest_mae_m", "dist_mae_m", "count_mae"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--floor", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cand, base = json.load(open(a.candidate)), json.load(open(a.baseline))
    rows, fails = [], 0
    for key in sorted(set(cand) & set(base)):
        metric = KEY_METRIC.get(key.split("/")[0].split("_")[0])
        if metric is None:
            continue
        for sl in sorted(set(cand[key]) & set(base[key])):
            c, b = cand[key][sl].get(metric), base[key][sl].get(metric)
            if c is None or b is None or c != c or b != b:
                continue
            n = max(1, cand[key][sl].get("n_images", 1))
            if metric in LOWER_BETTER:          # errors in metres etc.: lower is better, absolute floor 0.25
                tol = max(0.25, 0.1 * abs(b))
                status = "FAIL" if c > b + tol else ("better" if c < b - tol else "same")
            else:
                tol = max(a.floor, 2 * math.sqrt(max(b * (1 - b), 1e-4) / n))
                status = "FAIL" if c < b - tol else ("better" if c > b + tol else "same")
            fails += status == "FAIL"
            rows.append({"check": key, "slice": sl, "metric": metric, "candidate": c, "baseline": b, "tol": tol, "status": status})
    w = max(len(r["check"] + r["slice"]) for r in rows) + 3 if rows else 20
    for r in rows:
        print(f"{r['check'] + ' | ' + r['slice']:<{w}} {r['metric']:<12} {r['candidate']:.4f} vs {r['baseline']:.4f} "
              f"(tol {r['tol']:.4f})  {r['status']}")
    verdict = "SHIP" if fails == 0 else f"BLOCKED ({fails} slice regressions)"
    print("\nGATE:", verdict)
    if a.out:
        json.dump({"verdict": verdict, "rows": rows}, open(a.out, "w"), indent=1)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
