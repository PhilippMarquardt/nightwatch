# Night Watch — RGB + thermal fusion for a VLM slow head, distilled into a real-time fast head

A small-scale, end-to-end version of a multi-sensor perception stack for security robots:

1. **Native sensor encoders**: the VLM's own ViT for the visible camera, plus a DINOv3 ConvNeXt with a
   1-channel stem for raw thermal.
2. **Fusion**: thermal features land exactly on the VLM's 2×2-merged visual token grid (stride 32 ↔ 32 px
   tokens), and are added onto the RGB tokens (and onto DeepStack levels where the VLM has them) through
   zero-initialised output projections (ControlNet-style; a zero *gate* was tested and starves the
   projection of gradient, so the model never used thermal). The result is one spatially aligned token stream with the same token budget as RGB alone, and
   at step 0 the model is exactly the pretrained VLM.
3. **Slow head**: Qwen3.5-9B (natively multimodal, Feb 2026; any Qwen3.5/3.6 or Qwen3-VL works via `NW_MODEL`), LoRA-tuned to reason over the fused stream (grounding, counting, zone
   checks). It is trained on 7 labeled sites only.
4. **Fast head**: a 12M-parameter CenterNet on early-fused RGB+T, distilled from the slow head's labels on
   11 unlabeled sites. It has no NMS, exports to ONNX/TensorRT, and sees every frame. Only ambiguous
   frames escalate to the slow head.
5. **Evaluation**: every test number comes from **held-out sites** (LLVIP test scenes share no location
   with training). The slices are darkest-25%, crowded, per-site, far/mid/near persons, *visible camera
   dead* and *thermal dead*. A regression gate blocks any candidate that is worse on any slice.

## Data: LLVIP, split by scene

| split | scenes | frames | used for |
|---|---|---|---|
| labeled | 01–07 | 3,438 | slow-head SFT (the only human labels the slow head sees) |
| val | 25 | 423 | checkpoint + threshold selection |
| unlabeled | 08–18 | 8,164 | "field data": the slow head labels it, the fast head learns from that |
| test | 19–24, 26 | 3,463 | held-out sites, reporting only |

## Experiments

| id | what | question |
|---|---|---|
| `zs_rgb`, `zs_dual` | zero-shot Qwen3.5-9B | how far does the open model get with no training? |
| `sft_rgb` | LoRA, visible only | how much does fine-tuning alone buy? |
| `sft_dual` | LoRA, thermal as a 2nd image | strong simple baseline (2× tokens) |
| `sft_fusion` | **ours**: native thermal encoder + gated token fusion | same tokens as RGB alone. Better? More robust? |
| `fast_gt_labeled` | fast head on the 7 labeled sites | what the labels alone buy |
| `fast_distill` | fast head on labeled + slow-head pseudo-labels | does the slow head's knowledge transfer? |
| `fast_oracle` | fast head on all GT | upper bound |
| `fast_oracle_rgb/_t` | single-sensor ablations | what each sensor contributes |

## Run

```bash
bash scripts/setup.sh        # deps, LLVIP (4 GB), Qwen3.5-9B, DINOv3 weights, index
bash scripts/run_all.sh      # everything, resumable; logs in logs/, results in runs/, report in report/
```

Multi-GPU slow head: `torchrun --nproc_per_node 8 -m nightwatch.slow.train --mode fusion ...`.
