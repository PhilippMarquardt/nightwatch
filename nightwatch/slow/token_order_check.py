"""Which merged visual token does a 32-px image cell map to in Qwen3.5? Verifies the row-major 16x20 grid assumption
used by PCTokens (LiDAR point-cloud tokens replace the LiDAR image slot cell by cell)."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

M = "Qwen/Qwen3.5-9B"
proc = AutoProcessor.from_pretrained(M)
model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16).cuda().eval()
vis = model.model.visual


_cap = {}
vis.patch_embed.register_forward_hook(lambda m, a, o: _cap.__setitem__("pe", o.detach().float().cpu()))


def tokens(img):
    """Patch embeddings BEFORE any attention, grouped by 4 consecutive patches = one merged token (spatial merge 2x2)."""
    enc = proc.image_processor(images=[img], return_tensors="pt")
    with torch.no_grad():
        vis(enc["pixel_values"].cuda().to(torch.bfloat16), grid_thw=enc["image_grid_thw"].cuda())
    pe = _cap["pe"]
    return pe.view(pe.shape[0] // 4, 4, -1).reshape(pe.shape[0] // 4, -1), enc["image_grid_thw"].tolist()


base, thw = tokens(Image.new("RGB", (640, 512)))
print("grid_thw", thw, "tokens", base.shape)
ok = 0
cells = [(0, 0), (0, 19), (2, 15), (7, 4), (15, 0), (15, 19), (9, 11)]
for r, c in cells:
    a = np.zeros((512, 640, 3), np.uint8)
    a[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = 255
    t, _ = tokens(Image.fromarray(a))
    d = (t - base).norm(dim=-1)
    got = int(d.argmax()); exp = r * 20 + c
    ok += got == exp
    print(f"cell (r={r}, c={c}): expected token {exp}, most-changed token {got} ({'OK' if got == exp else 'MISMATCH'}); top3 {d.topk(3).indices.tolist()}")
print("TOKEN_ORDER", "OK" if ok == len(cells) else f"MISMATCH {ok}/{len(cells)}")
