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

## 3. LLVIP: the fused VLM

On 1,500 frames from the 7 test sites, with confidence intervals from resampling whole sites (frames from one camera are correlated, so frame-level intervals are far too narrow):

<div class="table-wrap">
<table>
<thead><tr><th>Model</th><th>F1</th><th>95 % site CI</th><th>Darkest 25 %</th><th>Worst site</th></tr></thead>
<tbody>
<tr><td>Qwen3.5-9B zero-shot, RGB</td><td>0.839</td><td>0.80–0.88</td><td>0.837</td><td>0.73</td></tr>
<tr><td>Zero-shot, thermal as a second image</td><td>0.696</td><td>0.64–0.78</td><td>–</td><td>0.58</td></tr>
<tr><td><b>Fine-tuned, fused RGB + thermal</b></td><td><b>0.894</b></td><td>0.87–0.93</td><td><b>0.902</b></td><td>0.85</td></tr>
<tr><td>Fast head, all sites hand-labelled (oracle)</td><td>0.857</td><td>0.83–0.88</td><td>–</td><td>0.79</td></tr>
</tbody>
</table>
</div>

Feeding thermal as a second image makes the zero-shot model worse on all 7 sites: it counts people twice across the two images. That is the reason for one fused stream. Fine-tuning with fusion adds +0.055 over zero-shot (CI +0.035 to +0.076), better on 7 of 7 sites.

PENDING: fused vs RGB-only vs thermal-only vs two-image fine-tuning on the same frames (isolates what fusion adds); sensor-off rows.

## 4. LLVIP: distilling the VLM into the fast head

PENDING: labelled-only student vs distilled student vs self-training control vs compute-matched control vs RGB-teacher student vs oracle, with site-bootstrap deltas. The headroom is the oracle − labelled gap: +0.027 (CI +0.011 to +0.049), better on 7/7 sites.

## 5. TriSAR-3D: the forest at night

This is where it got interesting. Person F1 on 2,751 held-out frames:

<div class="table-wrap">
<table>
<thead><tr><th>System</th><th>All</th><th>Unseen forest</th><th>Held-out lab</th></tr></thead>
<tbody>
<tr><td>VLM fine-tuned, RGB + thermal + LiDAR</td><td>0.17</td><td>0.00</td><td><b>0.79</b></td></tr>
<tr><td>… with camera-dropout 0.5</td><td>0.24</td><td>0.00</td><td>0.76</td></tr>
<tr><td>Fast head, 12M (two seeds)</td><td>0.37 / 0.40</td><td><b>0.31 / 0.36</b></td><td>0.56 / 0.55</td></tr>
<tr><td><b>Combined</b></td><td><b>0.43 / 0.46</b></td><td>0.31 / 0.35</td><td>0.76</td></tr>
</tbody>
</table>
</div>

The first fine-tuned VLM reported 3,373 people on 2,119 forest frames, and almost all of them were wrong. It had learned "the lab has people" and kept saying so in the dark. Retraining with the camera blacked out on half of the samples did not teach it to see in the forest. It taught it to say **nobody**: 2,099 of the 2,119 forest frames now get an empty report, and precision across the test set went from 0.20 to 0.75.

That abstention is what makes the two heads composable. The combined output keeps the VLM's persons (with posture and range) and adds fast-head detections that do not overlap them. It gains +0.066 and +0.065 F1 over the fast head alone, for two fast-head seeds. The same combination with the hallucinating VLM is *worse* than the fast head alone (0.32 vs 0.37).

![Combined vs single heads](/nightwatch/fig_trisar_router.png)

Two more experiments pinned down why the VLM fails in the forest:
- **With the camera off, the camera-dropout VLM emits the same box on all 600 test frames.** That is a memorised prior, not perception.
- **Matching the thermal levels to the training distribution does nothing for it** (forest F1 stays 0.00), while the same trick helps the fast head.

Thermal and LiDAR added onto the visual tokens modulate the camera path; they cannot replace it. A system that has to work with a dead camera should route thermal into the image slot itself.

The fast head, on the other hand, runs on LiDAR. Switching LiDAR off drops it from 0.37 to 0.22, and switching the camera off changes nothing.

![Accuracy vs compute](/nightwatch/fig_trisar_pareto.png)

## 6. Things that did not work

- **Escalating the VLM only on uncertain frames.** Picking frames by fast-head uncertainty was *worse* than picking them at random. The fast head is uncertain in the forest, where the VLM abstains anyway; the VLM helps in the lab, where the fast head is confident but loosely localised. A useful escalation policy needs a router trained on validation outcomes.
- **Test-time thermal level-matching for the fast head** (+0.046 forest F1 on one seed). The second seed moved forest F1 by the same amount on its own, so I don't count it until the multi-seed runs are in.
- **Site-memory events I defined by hand.** The "person at the water's edge" region was wrong and fired on almost every frame. I excluded those numbers.

## 7. Site memory

A JEPA-style predictor on LTDv2 (one harbour thermal camera over 8 months) predicts the next frame's frozen DINOv3 tokens from recent frames and a per-site memory. Anomaly is surprise: the peak token prediction error, z-scored per hour and daylight. Inserted persons are found with AUROC 0.70 (pixel baseline 0.52), crowds with 0.85 (token baseline 0.67). The two context types catch different anomalies, and combining them naively averaged the signals away.

## 8. Engineering that kept the numbers honest

- Every comparison uses the same frames; a gate refuses to compare runs with different frame counts.
- Pseudo-labels originally covered only the 1,500-frame test subset, and the student silently fell back to ground truth for the rest. Caught before training.
- Independent AI reviewers checked the work. Fable reviewed the plan and the code before the run, and then the live progress every 45 minutes overnight. Codex reviewed the code once. Every finding was triaged: fixed, deferred or rejected with a reason.

PENDING: edge numbers (TensorRT FP16 latency per head), final gate verdicts.
