"""Re-evaluate a trained TriSAR fast head under extra test-time conditions (e.g. thermal_stretch).

python -m nightwatch.fast.trisar_eval --name tf_rgbtl --conds thermal_stretch --out runs/tf_rgbtl/pred_test_stretch.jsonl
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from nightwatch.fast.trisar import CH, CenterNetTriSAR, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", default="data/cache_trisar")
    ap.add_argument("--conds", default="thermal_stretch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = f"runs/{a.name}"
    summ = json.load(open(f"{d}/train_summary.json"))
    idx = CH[summ["modal"]]
    dev = torch.device("cuda")
    net = CenterNetTriSAR(len(idx), pretrained=False).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(f"{d}/best.pt", map_location=dev))
    X = torch.from_numpy(np.load(f"{a.cache}/test.npy", mmap_mode="r")[:])
    ids = json.load(open(f"{a.cache}/test_ids.json"))
    with open(a.out, "w") as fh:
        for cond in a.conds.split(","):
            for iid, ps in zip(ids, predict(net, X, idx, cond)):
                fh.write(json.dumps({"id": iid, "task": "report", "cond": cond, "persons": ps}) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
