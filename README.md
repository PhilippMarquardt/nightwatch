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

## Stage 2 — TriSAR-3D: RGB + thermal + LiDAR (CC-BY-4.0, TU Darmstadt, Oct 2026)

19.7k synchronised frames from a 128-beam Ouster LiDAR, an RGB camera and a FLIR thermal camera, with 52k 3D person
boxes and posture tags (standing, walking, sitting, lying). Everything is rendered into one reference frame, the
undistorted thermal camera:

- RGB is warped in with the infinite-depth homography. The cameras are about 10 cm apart, so residual parallax is a few
  pixels, well under one visual token.
- LiDAR is z-buffer projected into inverse-range, intensity and hit maps.
- 3D boxes are projected to tight 2D boxes and keep their posture and metric range.

LiDAR therefore enters the slow head exactly like thermal: its own DINOv3 encoder, added onto the same visual token grid
through a zero-initialised projection.

**Split (ours, by sequence):** test is the whole forest (an unseen site at night) plus two held-out lab sequences with
lying persons. The official split is frame-level within sequences, which leaks near-duplicate frames.

**Task:** an incident report per person, `{"bbox_2d", "pose": upright|sitting|lying, "distance_m"}`, plus "is anyone
lying down?" and "how far is the nearest person?". Every answer is checked against the 3D labels.

| id | sensors | question |
|---|---|---|
| `t_zs_rgb` | RGB, zero-shot | what does the stock VLM get? |
| `t_sft_rgb` → `t_sft_fusion` → `t_sft_fusion_l` | RGB → +thermal → +LiDAR | what does each sensor add? |
| `tf_rgb`, `tf_rgbt`, `tf_tl`, `tf_rgbtl` | fast head, same ladder | does the fast head need the camera at all? |

Run it with `bash scripts/run_trisar.sh`.

## Run

```bash
bash scripts/setup.sh        # deps, LLVIP (4 GB), Qwen3.5-9B, DINOv3 weights, index
bash scripts/run_all.sh      # everything, resumable; logs in logs/, results in runs/, report in report/
```

Multi-GPU slow head: `torchrun --nproc_per_node 8 -m nightwatch.slow.train --mode fusion ...`.
