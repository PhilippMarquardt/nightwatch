"""LTDv2 (Long-term Thermal Drift v2, Aalborg harbour, CC BY-NC 4.0): one fixed thermal security camera, 8 months.

We use the labelled Valid split (401 two-minute clips at 1 fps, 43.8k frames, every month May-Aug 2020 + Jan-Apr 2021),
fetched clip-by-clip with HTTP range requests into the 48 GB frames.zip (one request per clip, ~5.7 MB).
Each frame carries 8 weather fields (temperature, humidity, precipitation, sun radiation, wind, ...).

Splits are by MONTH (the official splits are random over the whole period, so they cannot measure drift):
    det_train  2020-05, 2020-06          fast thermal detector trains here, then is tested month by month
    wm_train   2020-05 .. 2021-01        site-memory world model learns "normal" here
    wm_calib   2021-02                   per-context surprise normalisation
    wm_test    2021-03, 2021-04          held-out future months
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

URL = "https://huggingface.co/datasets/vapaau/LTDv2/resolve/main/data/frames.zip"
CD_OFF, CD_SIZE = 47808832683, 141977253          # zip64 central directory of frames.zip (verified)
W, H = 384, 288
CLASSES = {1: "person", 2: "bicycle", 3: "motorcycle", 4: "vehicle"}
COND_KEYS = ["Temperature", "Humidity", "Precipitation latest 10 min", "Sun Radiation Intensity", "Wind Speed"]


def central_directory(cache):
    if not os.path.exists(cache):
        req = urllib.request.Request(URL, headers={"Range": f"bytes={CD_OFF}-{CD_OFF + CD_SIZE - 1}"})
        open(cache, "wb").write(urllib.request.urlopen(req, timeout=600).read())
    cd = open(cache, "rb").read()
    p, clips = 0, collections.defaultdict(list)
    while p < len(cd):
        meth, csz, usz, nl, el, cl = struct.unpack("<H8xIIHHH", cd[p + 10:p + 34])
        off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + nl].decode()
        extra, q = cd[p + 46 + nl:p + 46 + nl + el], 0
        while q < len(extra):                                        # zip64 extra field
            hid, hl = struct.unpack("<HH", extra[q:q + 4])
            body, k = extra[q + 4:q + 4 + hl], 0
            if hid == 1:
                if usz == 0xFFFFFFFF:
                    usz = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
                if csz == 0xFFFFFFFF:
                    csz = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
                if off == 0xFFFFFFFF:
                    off = struct.unpack("<Q", body[k:k + 8])[0]; k += 8
            q += 4 + hl
        if name.endswith(".jpg"):
            clips["/".join(name.split("/")[1:3])].append((name, off, csz, meth))
        p += 46 + nl + el + cl
    return clips


def fetch_clip(entries, out_dir, retries=4):
    entries = sorted(entries, key=lambda e: e[1])
    a, b = entries[0][1], entries[-1][1] + entries[-1][2] + 4096
    for attempt in range(retries):
        try:
            blob = urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"}), timeout=120).read()
            break
        except Exception:
            if attempt == retries - 1:
                raise
    for name, off, csz, meth in entries:
        o = off - a
        nl, el = struct.unpack("<HH", blob[o + 26:o + 30])
        data = blob[o + 30 + nl + el:o + 30 + nl + el + csz]
        raw = zlib.decompress(data, -15) if meth == 8 else data
        path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(raw)
    return len(entries)


def split_of(month: str) -> dict:
    return {"det_train": month in ("202005", "202006"),
            "wm": "train" if month <= "202101" else ("calib" if month == "202102" else "test")}


def build(raw_dir, out_dir, max_clips=0, workers=16):
    """Download the Valid clips, build frames.json + a uint8 cache (N, 288, 384)."""
    from huggingface_hub import hf_hub_download
    os.makedirs(out_dir, exist_ok=True)
    ann_path = hf_hub_download("vapaau/LTDv2", "data/Valid.json", repo_type="dataset", local_dir=raw_dir)
    d = json.load(open(ann_path))
    boxes = collections.defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        boxes[a["image_id"]].append({"box": [x, y, x + w, y + h], "cls": CLASSES.get(a["category_id"], "other"),
                                     "uid": a.get("uid")})
    frames = []
    for im in d["images"]:
        fn = im["file_name"]                      # frames/20200514/clip_21_2239/image_0015.jpg
        _, day, clip, img = fn.split("/")
        ts = im["date_captured"]
        hour = int(ts[11:13]) + int(ts[14:16]) / 60
        frames.append({"id": str(im["id"]), "file": fn, "clip": f"{day}/{clip}", "t": int(img[6:10]), "day": day,
                       "month": day[:6], "hour": hour, "doy": int(np.datetime64(ts[:10], "D").astype(object).timetuple().tm_yday),
                       "cond": [float(im["meta"].get(k, 0.0) or 0.0) for k in COND_KEYS],
                       "boxes": boxes.get(im["id"], []), **split_of(day[:6])})
    frames.sort(key=lambda f: (f["clip"], f["t"]))
    clips = sorted({f["clip"] for f in frames})
    if max_clips:                                    # dev subset: spread over months
        by_m = collections.defaultdict(list)
        for c in clips:
            by_m[c[:6]].append(c)
        per = max(1, max_clips // len(by_m))
        clips = sorted(c for m in by_m.values() for c in m[:: max(1, len(m) // per)][:per])
        keep = set(clips)
        frames = [f for f in frames if f["clip"] in keep]
    cdir = central_directory(os.path.join(raw_dir, "cd.bin"))
    todo = [c for c in clips if not os.path.exists(os.path.join(raw_dir, "frames", c, "image_0000.jpg"))]
    print(f"{len(clips)} clips / {len(frames)} frames; downloading {len(todo)} clips", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        n = sum(ex.map(lambda c: fetch_clip(cdir[c], raw_dir), todo))
    print("downloaded", n, "jpgs", flush=True)
    frames = [f for f in frames if os.path.exists(os.path.join(raw_dir, f["file"]))]
    arr = np.lib.format.open_memmap(os.path.join(out_dir, "frames.npy"), mode="w+", dtype=np.uint8, shape=(len(frames), H, W))

    def load(i):
        arr[i] = np.asarray(Image.open(os.path.join(raw_dir, frames[i]["file"])).convert("L"), dtype=np.uint8)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(load, range(len(frames))))
    arr.flush()
    for i, f in enumerate(frames):
        f["idx"] = i
    json.dump(frames, open(os.path.join(out_dir, "frames.json"), "w"))
    c = collections.Counter((f["month"], f["wm"]) for f in frames)
    print("frames per month/split:", sorted(c.items()))
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/ltd")
    ap.add_argument("--out", default="data/ltd")
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    build(a.raw, a.out, a.max_clips, a.workers)
