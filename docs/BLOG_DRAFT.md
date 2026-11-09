# Night Watch: Camera, Thermal and LiDAR Perception After Dark

<!-- DRAFT for personal-site/blog-posts/nightwatch.md. Media go to personal-site/public/nightwatch/.
     Sections marked PENDING are filled from the overnight run (LLVIP distillation, controls, dual baseline). -->

A security robot does most of its work at night, at sites it has never seen. That breaks two habits of computer vision. The camera is the weakest sensor in the dark, and the usual test split shares locations with training. This project builds a small version of a full multi-sensor perception stack and evaluates it only on held-out sites:

1. **Per-sensor encoders, fused into one token stream.** The camera goes through the VLM's own ViT. Thermal and LiDAR each get a DINOv3 ConvNeXt encoder whose features land on the same token grid.
2. **A slow head**: Qwen3.5-9B with LoRA, which writes a JSON incident report per person (box, posture, distance in metres).
3. **A fast head**: a 12M-parameter CenterNet on the raw 7-channel frame. It runs in 0.7 ms with TensorRT FP16 and learns from the slow head's labels on unlabeled sites.
4. **Evaluation on unseen sites**: a forest at night, held-out lab sequences, sensor-failure conditions, per-site confidence intervals and regression gates that fail closed.

Everything ran on one rented RTX PRO 6000 (96 GB) overnight, as two GPU lanes plus a seed lane, each with a memory cap.

<video src="/nightwatch/demo_forest_best.mp4" controls muted loop playsinline preload="metadata"></video>

*An unseen forest at night. Left: the camera is almost black. Middle: thermal with LiDAR coloured by range, and the ground truth in white. Right: the system's output. The VLM reports "nobody visible", and the LiDAR-driven fast head finds both people with posture and range. This is the forest's best window; the forest's overall F1 is 0.31.*

![System diagram](/nightwatch/architecture.png)

## 1. Data: two datasets, both split by site

**LLVIP** has 15,488 aligned visible/thermal frames from street cameras at night. I split it by scene, not by frame:

<div class="table-wrap">
<table>
<thead><tr><th>Split</th><th>Scenes</th><th>Frames</th><th>Used for</th></tr></thead>
<tbody>
<tr><td>Labelled</td><td>01–07</td><td>3,438</td><td>slow-head fine-tuning (the only human labels it sees)</td></tr>
<tr><td>Val</td><td>25</td><td>423</td><td>checkpoint and threshold selection</td></tr>
<tr><td>Unlabeled</td><td>08–18</td><td>8,164</td><td>"field data": the slow head labels it, the fast head learns from it</td></tr>
<tr><td>Test</td><td>19–24, 26</td><td>3,463</td><td>unseen sites, reporting only</td></tr>
</tbody>
</table>
</div>

**TriSAR-3D** (TU Darmstadt, CC-BY-4.0, October 2026) has 19.7k synchronised frames from a 128-beam LiDAR, an RGB camera and a thermal camera. It comes with 3D person boxes and posture tags (standing, walking, sitting, lying). I render everything into the thermal camera's view:
- RGB is warped in by homography; the cameras are 10 cm apart, so residual parallax stays under one token.
- LiDAR is z-buffered into inverse-range, intensity and hit-mask channels.
- The 3D boxes become tight 2D boxes that keep posture and metric range.

The official split is frame-level inside sequences, which leaks near-duplicates. My test set is instead the whole forest (four night sequences the model never sees) plus two held-out lab sequences.

## 2. Fusion that starts as the pretrained model

Thermal and LiDAR features are added onto the VLM's 2×2-merged visual tokens through projections whose last layer is initialised to zero. At step 0 the model is exactly the pretrained VLM, and the token budget is the same as RGB alone. My first version used a tanh gate initialised to zero instead. It starved the projection of gradient and the model never used thermal, which a logged contribution ratio caught early.

## 3. LLVIP: does fusion beat the best single sensor?

On 1,500 frames from the 7 test sites, with confidence intervals from resampling whole sites (frames from one camera are correlated, so frame-level intervals are far too narrow). Every model saw the same frames and the same fine-tuning recipe:

<div class="table-wrap">
<table>
<thead><tr><th>Qwen3.5-9B + LoRA</th><th>F1</th><th>95 % site CI</th><th>Darkest 25 %</th><th>Camera dead</th><th>Thermal dead</th></tr></thead>
<tbody>
<tr><td>Zero-shot, RGB</td><td>0.839</td><td>0.80–0.88</td><td>0.832</td><td>0.00</td><td>n/a</td></tr>
<tr><td>Zero-shot, thermal as a second image</td><td>0.696</td><td>0.64–0.78</td><td>–</td><td>0.20</td><td>0.83</td></tr>
<tr><td>Fine-tuned, RGB only</td><td>0.882</td><td>0.83–0.92</td><td>0.820</td><td><b>0.00</b></td><td>n/a</td></tr>
<tr><td><b>Fine-tuned, thermal only</b> (thermal fed as the image)</td><td><b>0.923</b></td><td>0.89–0.95</td><td>0.891</td><td>n/a</td><td><b>0.00</b></td></tr>
<tr><td>Fine-tuned, thermal fused onto the RGB tokens</td><td>0.894</td><td>0.87–0.93</td><td><b>0.902</b></td><td><b>0.80</b></td><td><b>0.88</b></td></tr>
<tr><td>Brightness router (fused head if dark, RGB-only if bright)</td><td>0.910</td><td>0.89–0.94</td><td>0.902</td><td>–</td><td>–</td></tr>
</tbody>
</table>
</div>

I expected the fused model to win. It does not, on average. The **thermal-only** model is the strongest single head: +0.042 over RGB-only (CI +0.007 to +0.074), better on 6 of 7 sites. On LLVIP's clean night thermal, feeding thermal straight through the VLM's own image encoder beats adding it as a side branch. Against thermal-only, fusion is −0.030 (not significant), and my fail-closed regression gate blocks it.

What fusion does buy is the **worst case**. The fused model is the only one that keeps working when either sensor dies: 0.80 with the camera dead, 0.88 with the thermal camera dead. Both single-sensor models drop to 0.00 when their own sensor fails. For a security robot, a dirty lens or a failed thermal core is a when, not an if. So the deployable design is to run the strongest healthy single-sensor head and fall back to the fused head on a sensor fault.

![Average vs worst case](/nightwatch/fig_llvip_worst_case.png)

Two details made this concrete:
- **The fusion gain tracks darkness.** Against RGB-only, it is +0.13 on the darkest site and about −0.035 on the two brightest. The gate blocked the fused head for those two bright sites. A router that sends dark frames to the fused head (threshold: the median brightness of the *training* frames) gets 0.910 and passes the gate. I designed it after seeing the per-site pattern, so I count it as exploratory.
- **The model uses thermal spatially.** Feed it thermal from a *different* frame and it drops to 0.53, far below simply switching thermal off (0.88). A misaligned thermal camera is more dangerous than a dead one, so a production stack needs a cross-sensor consistency check.

![Fusion gain vs scene brightness](/nightwatch/fig_llvip_fusion_vs_lum.png)

## 4. LLVIP: distilling the VLM into the fast head

The fused VLM labelled 8,164 frames from 11 sites whose labels I withheld. A 12M fast head then trained on them. Two controls decide what that proves. **Self-training**: the small head labels the same frames itself. **Compute-matched**: the labelled-only head trained for as many steps. Full test set (3,463 frames, 7 unseen sites); mean and range over three seeds where I have them:

<div class="table-wrap">
<table>
<thead><tr><th>12M fast head, labels on the 11 extra sites</th><th>F1</th><th>AP50</th></tr></thead>
<tbody>
<tr><td>none (7 labelled sites only)</td><td>0.82 (0.81–0.83)</td><td>0.87</td></tr>
<tr><td>none, trained for as many steps (compute-matched)</td><td>0.84 (0.83–0.86)</td><td>0.89</td></tr>
<tr><td>its own labels (self-training)</td><td><b>0.86 (0.86–0.86)</b></td><td>0.89</td></tr>
<tr><td>VLM labels</td><td>0.83 (0.83–0.84)</td><td><b>0.91</b></td></tr>
<tr><td>VLM labels, kept where the small head agrees</td><td>0.85 (0.83–0.87)</td><td><b>0.91</b></td></tr>
<tr><td>human labels (oracle)</td><td>0.86</td><td>0.91</td></tr>
</tbody>
</table>
</div>

The self-training control is the most useful number in this project. Labelling the extra sites closes the gap to the human-label oracle, but the small head's own labels do it as well as the 9B teacher's: +0.03 over labelled-only on 7 of 7 sites, in two seeds. Some of it is just extra training steps. A labelled-only head trained for as many steps gets 0.844 on average, so the self-labelled data adds about +0.015 beyond that, significant on two of three seeds.

The teacher's labels still carry something. Every VLM-labelled student ranks detections as well as the oracle (AP50 0.91 vs 0.89 for self-training). But its operating point, the score threshold tuned on a one-site validation set, does not transfer, so F1 doesn't show it. The teacher's own labels on those sites are F1 0.82 against the withheld ground truth, and at two bright, tree-lined sites it boxes street lamps and foliage as people. On easy pedestrians, that is no better than what the small head already knows. A VLM teacher should pay off where the small model is weak: posture, range and unfamiliar sites, not LLVIP boxes.

![Teacher label quality per site](/nightwatch/fig_llvip_teacher_sites.png)

## 5. TriSAR-3D: the forest at night

This is where it got interesting. Person F1 on 2,751 held-out frames:

<div class="table-wrap">
<table>
<thead><tr><th>System</th><th>All</th><th>Unseen forest</th><th>Held-out lab</th></tr></thead>
<tbody>
<tr><td>VLM fine-tuned, RGB only</td><td>0.25</td><td>0.00</td><td>0.78</td></tr>
<tr><td>VLM fine-tuned, RGB + thermal + LiDAR</td><td>0.17</td><td>0.00</td><td><b>0.79</b></td></tr>
<tr><td>… with camera-dropout 0.5</td><td>0.24</td><td>0.00</td><td>0.76</td></tr>
<tr><td>Fast head, 12M (mean of 3 seeds, range)</td><td>0.36 (0.30–0.40)</td><td>0.30 (0.24–0.36)</td><td>0.56</td></tr>
<tr><td>Fast head + thermal normalisation (3 seeds)</td><td>0.42 (0.37–0.45)</td><td><b>0.37 (0.31–0.42)</b></td><td>0.56</td></tr>
<tr><td><b>RGB-only VLM + normalised fast head, combined</b> (3 seeds)</td><td><b>0.48 (0.44–0.51)</b></td><td>0.37 (0.31–0.42)</td><td><b>0.78</b></td></tr>
</tbody>
</table>
</div>

In the black forest (2,119 frames), the RGB-only VLM reports 6 people: it sees nothing and says so. The VLM with thermal and LiDAR side branches reports **3,373**, nearly all wrong. Two things differ between those runs. The fused one has the extra branches, and it also trained with 20 % camera-dropout that kept the lab people as targets on black frames. A fused run without camera-dropout is training to separate the two. Retraining with 50 % camera-dropout brought it down to 39 reports, only 1 of them correct. Precision over the whole test set went from 0.20 to 0.75.

The TriSAR VLMs never learned to use thermal or LiDAR. Switching either off, or feeding both from another frame, changes held-out-lab F1 by less than 0.03; with the camera off, the model emits the same memorised box on every frame. On LLVIP the same design does use thermal (0.80 with the camera off), so this is a training problem, not an architecture limit: one epoch on 6k frames, two new encoders at once, and a camera that was already enough on the lab training frames.

Abstention is what makes the two heads composable. The combined output keeps the VLM's persons (with posture and range) and adds fast-head detections that do not overlap them. On every one of six fast-head variants it gains +0.054 to +0.066 F1 over the fast head alone, even though the fast head's own level varies a lot by seed (0.30–0.40, and lab validation does not predict which seed holds up in the forest). With the hallucinating VLM, the same combination is *worse* than the fast head alone (0.32 vs 0.37).

The fast head runs on LiDAR: switching LiDAR off drops it from 0.37 to 0.22, and switching the camera off changes nothing. Building a per-frame thermal level match into its training and inference made thermal a second sensor it actually uses. Forest F1 went 0.30 → 0.37 on average, with overlapping seed ranges. With LiDAR off, every normalised seed beat every plain one (0.26–0.31 vs 0.19–0.24).

![Combined vs single heads](/nightwatch/fig_trisar_router.png)

![Accuracy vs compute](/nightwatch/fig_trisar_pareto.png)

## 6. Things that did not work, and what that taught me

- **Uncertainty escalation where the slow head is blind.** On LLVIP, sending the 20 % most uncertain frames to the VLM recovers 61 % of the fast→slow gap (random frames: 28 %). On TriSAR the same rule was *worse* than random, because the fast head is unsure exactly where the VLM cannot see. Routing has to know which head is trustworthy for the scene, not just how unsure the fast head is.
- **Single seeds.** One consensus-student seed gave 0.866 and looked like "the student matches the oracle"; another gave 0.830. One thermal-normalisation seed gave forest 0.42, another 0.31. Every claim above uses three seeds or a paired site bootstrap.
- **Site-memory events I defined by hand.** The "person at the water's edge" region was wrong and fired on almost every frame. I excluded those numbers.

## 7. Site memory

A JEPA-style predictor on LTDv2 (one harbour thermal camera over 8 months) predicts the next frame's frozen DINOv3 tokens from recent frames and a per-site memory. Anomaly is surprise: the peak token prediction error, z-scored per hour and daylight. Inserted persons are found with AUROC 0.70 (pixel baseline 0.52), crowds with 0.85 (token baseline 0.67). The two context types catch different anomalies, and combining them naively averaged the signals away.

## 8. Engineering that kept the numbers honest

- Every comparison uses the same frames; a gate refuses to compare runs with different frame counts.
- Pseudo-labels originally covered only the 1,500-frame test subset, and the student silently fell back to ground truth for the rest. Caught before training.
- Independent AI reviewers checked the work. Fable reviewed the plan and the code before the run, and then the live progress every 45 minutes overnight. Codex reviewed the code once. Every finding was triaged: fixed, deferred or rejected with a reason.

- Every comparison passes through a fail-closed, per-slice regression gate. Overnight it blocked the fused VLM against both single-sensor VLMs, and both distilled students against self-training. It passed the brightness router and the fused VLM under sensor failure. A gate that blocks the author's favourite model is the point.
- Before each control result arrived, I wrote down what outcome would count against the claim. The self-training control then overturned my distillation story.

PENDING (morning): fused vs two-image fine-tuning (dual) row, zero-camera-dropout TriSAR control, edge latency table.
