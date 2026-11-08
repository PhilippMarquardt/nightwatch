"""Synthesise plausible run outputs from GT so the report/eval code can be tested without a GPU."""
import json, os, random, sys
import numpy as np
from nightwatch.data.llvip import load_index
R = sys.argv[1]
recs = load_index("data/index.json", ("test",))
rng = random.Random(0)
def noisy(boxes, p_miss, jit, p_fp):
    out = []
    for b in boxes:
        if rng.random() < p_miss: continue
        w, h = b[2]-b[0], b[3]-b[1]
        out.append([b[0]+rng.gauss(0,jit*w), b[1]+rng.gauss(0,jit*h), b[2]+rng.gauss(0,jit*w), b[3]+rng.gauss(0,jit*h)])
    if rng.random() < p_fp:
        x, y = rng.uniform(0,1100), rng.uniform(0,800); out.append([x,y,x+80,y+200])
    return out
spec = {"zs_rgb":(.45,.15,.3),"zs_dual":(.35,.12,.3),"sft_rgb":(.2,.06,.1),"sft_dual":(.12,.05,.08),"sft_fusion":(.08,.05,.06),
        "fast_gt_labeled":(.2,.07,.2),"fast_distill":(.1,.06,.1),"fast_oracle":(.07,.05,.08),"fast_oracle_rgb":(.25,.07,.2),"fast_oracle_t":(.1,.06,.1)}
for run,(pm,j,pf) in spec.items():
    os.makedirs(f"{R}/{run}", exist_ok=True)
    fast = run.startswith("fast")
    with open(f"{R}/{run}/pred_test.jsonl","w") as f:
        for cond, extra in [("normal",0),("rgb_off",.5 if "rgb" in run else .05),("thermal_off",.4)]:
            if not fast and cond!="normal": continue
            for r in recs:
                b = noisy(r.boxes, min(.95,pm+extra), j, pf)
                d = {"id":r.id,"task":"detect","cond":cond,"boxes":b}
                if fast:
                    d["scores"] = [rng.uniform(.4,.95) for _ in b] + [rng.uniform(.05,.4) for _ in range(3)]
                    d["boxes"] = b + [[10,10,60,120]]*3
                f.write(json.dumps(d)+"\n")
    if not fast:
        with open(f"{R}/{run}/pred_test_robust.jsonl","w") as f:
            for cond, extra in [("rgb_off",.6 if "rgb" in run else .1),("thermal_off",.25)]:
                for r in recs[:800]:
                    f.write(json.dumps({"id":r.id,"task":"detect","cond":cond,"boxes":noisy(r.boxes,min(.95,pm+extra),j,pf)})+"\n")
        json.dump([{"task":"detect","cond":"normal","n":3463,"seconds":600}],open(f"{R}/{run}/pred_test_timing.json","w"))
        with open(f"{R}/{run}/log.jsonl","w") as f:
            for s in range(1,431,5):
                f.write(json.dumps({"step":s,"loss":0.3+0.5*np.exp(-s/60)+rng.random()*.05,"gate_main":0.3*(1-np.exp(-s/100)),"gate_deep0":0.1*(1-np.exp(-s/80)),"gate_deep1":0.05,"gate_deep2":-0.02})+"\n")
    else:
        json.dump({"score_thr":0.35,"modal":"rgbt"},open(f"{R}/{run}/train_summary.json","w"))
        json.dump({"gpu":"NVIDIA H200","torch_fp32_ms":2.1,"torch_fp16_ms":1.6,"tensorrt_fp16_ms":0.62,"tensorrt_fp16_fps":1612},open(f"{R}/{run}/deploy.json","w"))
print("fake runs in", R)
