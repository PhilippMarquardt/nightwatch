"""Pack the LiDAR clouds that the point-cloud encoder needs into a small upload (new pod, slow dataset server).

Per frame in data/trisar_fused/frames.json: xyz quantised to 5 mm (int16, +-163 m) and intensity (uint16 of [0, 1]),
plus the thermal image size that pc_encode.project() reads. 23 GB of PCD -> ~3 GB.

python scripts/pc_pack.py            (writes data/trisar_pcpack/<id>.npz and data/trisar_pcpack/meta.json)
On the pod: NW_PC_PACK=data/trisar_pcpack python -m nightwatch.slow.pc_encode ...
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nightwatch.data.trisar import read_pcd  # noqa: E402

OUT = "data/trisar_pcpack"
Q = 0.005


def one(rec):
    dst = os.path.join(OUT, rec["id"] + ".npz")
    if not os.path.exists(dst):
        xyz, inten = read_pcd(rec["lidar"])
        q = np.round(xyz / Q)
        assert np.abs(q).max() < 32767, rec["id"]
        np.savez_compressed(dst + ".tmp.npz", xyz=q.astype(np.int16),
                            inten=np.round(np.clip(inten, 0, 1) * 65535).astype(np.uint16))
        os.replace(dst + ".tmp.npz", dst)
    return rec["id"], list(Image.open(rec["thermal"]).size)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    idx = {r["id"]: r for r in json.load(open("data/trisar_index.json"))}
    ids = [f["id"] for f in json.load(open("data/trisar_fused/frames.json"))]
    with Pool(12) as pool:
        sizes = dict(pool.imap_unordered(one, [idx[i] for i in ids], chunksize=32))
    json.dump({"quant_m": Q, "thermal_size": sizes}, open(os.path.join(OUT, "meta.json"), "w"))
    print("PACK_DONE", len(sizes))
