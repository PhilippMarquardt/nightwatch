"""Night Watch demo video: TriSAR held-out test sequences, three panels per frame.

  [ camera (warped into thermal frame) ] [ thermal + LiDAR (range-coloured) + ground truth ] [ system output ]

System output = union router (nightwatch/eval/router_trisar.py): slow-head (VLM) persons with posture + range, plus
fast-head detections it did not report. Rendered from saved predictions only (CPU).

python -m nightwatch.report.video --scene rosbag2_2025_11_17-17_59_40_out --n 240 --out report/demo_forest.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

import matplotlib
import numpy as np
from matplotlib import cm, font_manager
from PIL import Image, ImageDraw, ImageFont

RUNS = os.environ.get("NW_RUNS", "runs")
FUSED = os.environ.get("NW_TRISAR_FUSED", "data/trisar_fused")
FONT = font_manager.findfont("DejaVu Sans")
FONT_B = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
BG = (24, 24, 24)
SLOW_C, FAST_C, GT_C, INK = (57, 135, 229), (25, 158, 112), (240, 239, 236), (225, 225, 225)
W, H, HEAD, FOOT = 640, 512, 64, 54


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


def load(path, thr=None):
    out = {}
    for d in map(json.loads, open(path)):
        if d["task"] == "report" and d["cond"] == "normal":
            p = d.get("persons", [])
            if thr is not None:
                p = [q for q in p if q["score"] >= thr]
            out[d["id"]] = p
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def lidar_overlay(th, li):
    """Thermal in grey, LiDAR returns coloured by range (turbo: near = red, far = blue)."""
    base = np.stack([th] * 3, -1).astype(np.float32) * 0.85
    hit = li[..., 2] > 0
    rng = 510.0 / np.maximum(li[..., 0].astype(np.float32), 1.0)          # inverse of the 1/r * 2 * 255 encoding
    t = np.clip(1 - np.log(np.clip(rng, 1, 60)) / np.log(60), 0, 1)       # 1 m -> 1, 60 m -> 0
    col = cm.turbo(t)[..., :3] * 255
    base[hit] = 0.25 * base[hit] + 0.75 * col[hit]
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def label(d, box, text, color, f, placed=None):
    tw = d.textlength(text, font=f)
    for y0 in (box[1] - 20, box[3] + 1, box[1] + 1):                      # above, below, inside: first free slot
        x0 = max(0, min(W - tw - 8, box[0])); y0 = max(0, min(H - 20, y0))
        r = (x0, y0, x0 + tw + 8, y0 + 19)
        if placed is None or all(r[2] < p[0] or r[0] > p[2] or r[3] < p[1] or r[1] > p[3] for p in placed):
            break
    if placed is not None:
        placed.append(r)
    d.rectangle([x0, y0, x0 + tw + 8, y0 + 19], fill=color)
    d.text((x0 + 4, y0 + 1), text, fill=(255, 255, 255), font=f)


def fmt(q):
    pose = q.get("pose") or "person"
    return f'{pose} {q["distance"]:.0f} m' if q.get("distance") is not None else pose


def render(fid, frame, slow, fast, t_s, title):
    rgb = Image.open(f"{FUSED}/{fid}_rgb.jpg").convert("RGB")
    th = np.asarray(Image.open(f"{FUSED}/{fid}_t.png"))
    li = np.asarray(Image.open(f"{FUSED}/{fid}_l.png"))
    mid = lidar_overlay(th, li)
    out = Image.fromarray(np.stack([th] * 3, -1))
    f14, f15 = font(14), font(15, True)
    dm = ImageDraw.Draw(mid)
    for q in frame["persons"]:
        dm.rectangle(q["box"], outline=GT_C, width=2)
    do = ImageDraw.Draw(out)
    keep = []                                              # display-only de-duplication (scored output has no NMS)
    for q in sorted(fast, key=lambda q: -q["score"]):
        if all(iou(q["box"], r["box"]) < 0.5 for r in keep):
            keep.append(q)
    fast_only = [q for q in keep if all(iou(q["box"], s["box"]) < 0.3 for s in slow)]
    placed = []
    for q in slow:
        do.rectangle(q["box"], outline=SLOW_C, width=3); label(do, q["box"], fmt(q), SLOW_C, f14, placed)
    for q in fast_only:
        do.rectangle(q["box"], outline=FAST_C, width=3); label(do, q["box"], fmt(q), FAST_C, f14, placed)

    canvas = Image.new("RGB", (3 * W + 40, H + HEAD + FOOT), BG)
    d = ImageDraw.Draw(canvas)
    dark = np.asarray(rgb.convert("L")).mean() < 25
    heads = ["camera (dark: nearly no signal)" if dark else "camera", "thermal + LiDAR  (white = ground truth)", "Night Watch output"]
    for k, im in enumerate([rgb, mid, out]):
        x = 10 + k * (W + 10)
        canvas.paste(im, (x, HEAD))
        d.text((x, HEAD - 24), heads[k], fill=INK, font=font(16, True))
    d.text((10, 8), title, fill=INK, font=font(18, True))
    d.text((3 * W + 30, 10), f"t = {t_s:5.1f} s", fill=INK, font=font(16), anchor="ra")
    if slow:
        verdict, vc = f"slow head (VLM) reports {len(slow)}" + (f", fast head adds {len(fast_only)}" if fast_only else ""), SLOW_C
    elif fast_only:
        verdict, vc = f'slow head: "nobody visible", fast head (LiDAR) finds {len(fast_only)}', FAST_C
    else:
        verdict, vc = "no detections", (140, 140, 140)
    xo = 10 + 2 * (W + 10)
    d.rectangle([xo, H + HEAD + 8, xo + 14, H + HEAD + 22], fill=vc)
    d.text((xo + 22, H + HEAD + 6), verdict, fill=INK, font=f15)
    d.text((10, H + HEAD + 6), f"ground truth: {len(frame['persons'])} person(s)", fill=INK, font=f15)
    # legend
    lx = 10 + (W + 10)
    for k, (c, s) in enumerate([(SLOW_C, "slow head: 9B VLM, posture + range"), (FAST_C, "fast head: 12M CenterNet, 0.7 ms")]):
        d.rectangle([lx + k * 330, H + HEAD + 32, lx + k * 330 + 14, H + HEAD + 46], fill=c)
        d.text((lx + k * 330 + 22, H + HEAD + 30), s, fill=INK, font=font(14))
    # range colour bar
    bar = (cm.turbo(np.linspace(1, 0, 200))[:, :3] * 255).astype(np.uint8)[None].repeat(10, 0)
    canvas.paste(Image.fromarray(bar), (10, H + HEAD + 36))
    d.text((215, H + HEAD + 30), "LiDAR range 1 m → 60 m", fill=INK, font=font(13))
    d.text((3 * W + 30, H + HEAD + 32), "fast boxes de-duplicated for display (NMS 0.5)", fill=(150, 150, 150), font=font(12), anchor="ra")
    return canvas


def stamp_index():
    """frame id -> capture time (ns, from the LiDAR sample filename)."""
    idx = json.load(open(os.environ.get("NW_TRISAR_INDEX", "data/trisar_index.json")))
    return {r["id"]: int(re.findall(r"(\d{16,})", r["lidar"])[-1]) for r in idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, default=-1, help="first frame index in the scene; -1 = densest window of persons")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--slow", default="t_sft_fusion_l_cd50")
    ap.add_argument("--fast", default="tf_rgbtl")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--title", default="")
    ap.add_argument("--note", default="", help="appended to the title, e.g. how the window was chosen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif", default=None, help="also write a small GIF of the first --gif-n frames")
    ap.add_argument("--gif-n", type=int, default=64)
    a = ap.parse_args()
    frames = {f["id"]: f for f in json.load(open(f"{FUSED}/frames.json"))}
    ts = stamp_index()
    ids = sorted([i for i, f in frames.items() if f["scene"] == a.scene], key=lambda i: ts[i])
    if a.start < 0:
        cnt = np.convolve([len(frames[i]["persons"]) for i in ids], np.ones(a.n), "valid")
        a.start = int(np.argmax(cnt)) if len(cnt) else 0
    ids = ids[a.start:a.start + a.n]
    thr = json.load(open(f"{RUNS}/{a.fast}/train_summary.json"))["score_thr"]
    slow, fast = load(f"{RUNS}/{a.slow}/pred_test.jsonl"), load(f"{RUNS}/{a.fast}/pred_test.jsonl", thr)
    f0 = frames[ids[0]]
    title = a.title or f'TriSAR-3D held-out {f0["location"]}, {"night" if f0["night"] else "day"}  ·  {a.scene.replace("rosbag2_", "").replace("_out", "")}  ·  never seen in training'
    if a.note:
        title += f"  ·  {a.note}"
    tmp = a.out + "_frames"
    os.makedirs(tmp, exist_ok=True)
    for k, i in enumerate(ids):
        render(i, frames[i], slow.get(i, []), fast.get(i, []), (ts[i] - ts[ids[0]]) / 1e9, title).save(f"{tmp}/{k:05d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", a.out],
                   check=True)
    if a.gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/%05d.png",
                        "-frames:v", str(a.gif_n), "-vf",
                        "fps=6,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                        a.gif], check=True)
    import shutil
    shutil.rmtree(tmp)
    print("wrote", a.out, len(ids), "frames")


if __name__ == "__main__":
    main()
