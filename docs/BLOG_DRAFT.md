# Night Watch: Camera, Thermal and LiDAR Perception After Dark

A security robot does most of its work at night, at sites it has never seen. That breaks two habits of computer
vision. The camera is the weakest sensor in the dark, and the usual test split shares locations with training.

Night Watch is a small, end-to-end version of the perception stack such a robot needs:
- a **vision-language model as the slow head**: Qwen3.5-9B with LoRA, which reads every sensor and writes an incident
  report per person (box, posture, distance in metres)
- a **12M-parameter detector as the fast head**: it runs on every frame in 0.7 ms (TensorRT FP16) and learns from the
  slow head
- **evaluation only on sites the models never saw**, with regression gates that fail closed

I set out to answer three questions:

<div class="table-wrap">
<table>
<thead><tr><th></th><th>Question</th><th>Short answer</th></tr></thead>
<tbody>
<tr><td><b>Fusion</b></td><td>How should a 9B VLM receive thermal and LiDAR so that it actually uses them?</td><td>As extra images. My custom fusion branch lost on every site, and on the robot data the VLM ignored it.</td></tr>
<tr><td><b>Distillation</b></td><td>Can the VLM teach a small edge detector?</td><td>On familiar data, no better than self-training. On a truly new site, yes: F1 0.13 → 0.46 without a single human label.</td></tr>
<tr><td><b>New sites</b></td><td>Does it hold up where it was never trained?</td><td>Partly. In an unseen forest at night the VLM goes from blind (0.00) to 0.44. At an unseen harbour, 75 % of the gap to human labels is closed.</td></tr>
</tbody>
</table>
</div>

<video src="/nightwatch/demo_forest_best.mp4" controls muted loop playsinline preload="metadata"></video>

*An unseen forest at night. Left: the camera is almost black. Middle: thermal with LiDAR coloured by range, and the
ground truth in white. Right: the system's output. This is the forest's best window for the fast head; its overall
forest F1 is 0.31.*

## 1. Three datasets, each split by place or time

<div class="table-wrap">
<table>
<thead><tr><th></th><th>LLVIP</th><th>TriSAR-3D</th><th>LTDv2</th></tr></thead>
<tbody>
<tr><td>Sensors</td><td>camera + thermal</td><td>camera + thermal + 128-beam LiDAR</td><td>thermal only</td></tr>
<tr><td>Scene</td><td>street surveillance cameras at night</td><td>a robot sensor rig: indoor lab, outdoor forest</td><td>one fixed harbour camera in Denmark</td></tr>
<tr><td>Resolution</td><td>1280×1024</td><td>640×512, all sensors rendered into the thermal view</td><td>384×288, people ~16–19 px tall</td></tr>
<tr><td>Labels</td><td>person boxes</td><td>3D person boxes with posture and range</td><td>person boxes</td></tr>
<tr><td>My split</td><td>by scene: train 01–07, test 19–24 and 26</td><td>by sequence: train in the lab, test the whole night forest + 2 held-out lab sequences</td><td>by time: adapt on May 2020, test on March–April 2021</td></tr>
<tr><td>Size</td><td>3,438 labelled, 8,164 unlabelled, 3,463 test</td><td>9,832 frames; test 2,751 (forest 2,119)</td><td>2,000 unlabelled adaptation frames; test 15,925 frames in 142 clips</td></tr>
</tbody>
</table>
</div>

![The three datasets as the models see them, with ground-truth person boxes](/nightwatch/fig_datasets.png)

The official LLVIP and TriSAR splits put frames from the same place into train and test, which leaks near-duplicates.
I re-split both by site or sequence. In LLVIP I hide the labels of sites 08–18 and treat them as "field data" that the
VLM labels for the small detector. For TriSAR I render everything into the thermal camera's view: RGB is warped in by
homography (the cameras are 10 cm apart, so parallax stays under one token), LiDAR is z-buffered into range,
intensity and hit channels, and the 3D boxes become 2D boxes that keep posture and metric range.

## 2. The system

![System diagram](/nightwatch/architecture.png)

The slow head is Qwen3.5-9B with LoRA. It only ever sees human labels from the training sites. The fast head is a
CenterNet (ResNet18-FPN, 12M parameters) without NMS; it learns from human labels plus the slow head's labels on
unlabelled data, and only uncertain frames escalate to the slow head.

I compared two ways of giving the VLM its extra sensors:
1. **Side branch.** A DINOv3 encoder per extra sensor, whose features are added onto the VLM's visual tokens through
   projections initialised to zero. At step 0 the model is exactly the pretrained VLM, and the token budget is the
   same as camera alone. (My first version used a tanh gate initialised to zero instead. It starved the projection of
   gradient, and a logged contribution ratio caught that early.)
2. **Sensors as images.** Thermal (level-matched per frame) and LiDAR (1/range through a turbo colormap) go into the
   VLM's own image encoder as extra images. More tokens, no new modules.

## 3. How I measured

- **Unseen places only.** No test site, sequence or month was used for training, thresholds or model choice.
- **Intervals from resampling whole sites, sequences or clips.** Frames from one camera are correlated, so
  frame-level intervals are far too narrow.
- **Three seeds** wherever results move with the seed; single-seed rows say so.
- **Same frames** for every compared row.
- **A fail-closed regression gate**: a model ships only if it is no worse on any slice (per site, darkest 25 %,
  crowded, camera dead, thermal dead).
- **Criteria written down before training** for the follow-up experiments, and AI reviewers auditing progress every
  45 minutes. Anything decided after seeing results is labelled post hoc.

## 4. Fusion: how should the VLM see thermal and LiDAR?

### LLVIP: camera + thermal

1,500 frames from the 7 test sites:

<div class="table-wrap">
<table>
<thead><tr><th>Qwen3.5-9B</th><th>F1</th><th>95 % site CI</th><th>Camera dead</th><th>Thermal dead</th></tr></thead>
<tbody>
<tr><td>Zero-shot, camera</td><td>0.839</td><td>0.80–0.88</td><td>0.00</td><td>n/a</td></tr>
<tr><td>Zero-shot, camera + thermal as two images</td><td>0.696</td><td>0.64–0.78</td><td>0.20</td><td>0.83</td></tr>
<tr><td>Fine-tuned, camera only</td><td>0.882</td><td>0.83–0.92</td><td>0.00</td><td>n/a</td></tr>
<tr><td>Fine-tuned, thermal only</td><td>0.923</td><td>0.89–0.95</td><td>n/a</td><td>0.00</td></tr>
<tr><td>Fine-tuned, my side-branch fusion</td><td>0.894</td><td>0.87–0.93</td><td>0.80</td><td>0.88</td></tr>
<tr><td>Fine-tuned, two images, same token budget as the side branch</td><td>0.928</td><td>0.90–0.94</td><td>0.91</td><td>0.88</td></tr>
<tr><td><b>Fine-tuned, two images</b></td><td><b>0.937</b></td><td>0.92–0.95</td><td><b>0.93</b></td><td><b>0.89</b></td></tr>
</tbody>
</table>
</div>

I expected my fused model to win. It did not. The winner is the simplest baseline: fine-tune the VLM on the camera
and thermal frames as two separate images. It beats my side branch by 0.043 on all 7 sites, and keeps working when
either camera dies. At an equal token budget my side branch still loses on all 7 sites (0.894 vs 0.928). Zero-shot,
the same two-image input was the worst model of all (0.696), because it counted people twice. That zero-shot number
is why I built the side branch in the first place, and fine-tuning removed the problem.

My fail-closed regression gate blocked the fused model against every stronger baseline. That is the part I would
bring to a fusion team on day one: a gate that refuses its author's favourite model.

![Average vs worst case](/nightwatch/fig_llvip_worst_case.png)

Two details made this concrete:
- **The fusion gain tracks darkness.** Against camera-only it is +0.13 on the darkest site and about −0.035 on the
  two brightest.
- **The model uses thermal spatially.** Feed it thermal from a *different* frame and it drops to 0.53, far below
  simply switching thermal off (0.88). A misaligned thermal camera is more dangerous than a dead one.

![Fusion gain vs scene brightness](/nightwatch/fig_llvip_fusion_vs_lum.png)

### TriSAR-3D: camera + thermal + LiDAR, lab-trained, tested in a forest at night

<div class="table-wrap">
<table>
<thead><tr><th>Qwen3.5-9B</th><th>Seeds</th><th>All</th><th>Unseen forest</th><th>Held-out lab</th><th>Lab, camera dead</th></tr></thead>
<tbody>
<tr><td>Zero-shot, camera (600-frame subset)</td><td>1</td><td>0.07</td><td>0.00</td><td>0.26</td><td>–</td></tr>
<tr><td>Fine-tuned, camera only</td><td>1</td><td>0.25</td><td>0.00</td><td>0.78</td><td>0.00</td></tr>
<tr><td>Fine-tuned, side-branch fusion of thermal + LiDAR</td><td>2</td><td>0.26</td><td>0.00</td><td>0.81</td><td>0.00</td></tr>
<tr><td>Fine-tuned, thermal as an image</td><td>1</td><td>0.41</td><td>0.33</td><td>0.82</td><td>n/a</td></tr>
<tr><td>Fine-tuned, camera + thermal as images</td><td>1</td><td>0.48</td><td>0.39</td><td><b>0.86</b></td><td>0.78</td></tr>
<tr><td><b>Fine-tuned, camera + thermal + LiDAR as images</b></td><td>2</td><td><b>0.52</b></td><td><b>0.44</b></td><td>0.84</td><td><b>0.79</b></td></tr>
</tbody>
</table>
</div>

![Sensor ladder](/nightwatch/fig_trisar_sensor_ladder.png)

**As a side branch, thermal and LiDAR were ignored.** Switching either off changed held-out-lab F1 by less than 0.03,
and with the camera off the model emitted the same memorised box on every frame. **As images, the same thermal data
lets the VLM see people in the black forest for the first time.**

Before training the three-image model I wrote down three criteria:

<div class="table-wrap">
<table>
<thead><tr><th>Criterion</th><th>Bar</th><th>Result (2 seeds)</th></tr></thead>
<tbody>
<tr><td>Works with the camera dead</td><td>lab F1 ≥ 0.50</td><td><b>pass</b>: 0.82 / 0.77</td></tr>
<tr><td>Each extra sensor matters</td><td>thermal off and LiDAR off each cost ≥ 0.03</td><td><b>fail</b>: thermal matters (forest 0.43 → 0.15 without it), LiDAR does not (≤ 0.01)</td></tr>
<tr><td>Few false alarms</td><td>precision ≥ 0.70</td><td><b>fail</b>: 0.60 / 0.54</td></tr>
</tbody>
</table>
</div>

On the full report, where box, posture and distance must all be right, the camera + thermal model is best (0.32 vs
0.28–0.29 for three images, whose distance error is larger). My hypothesis for the ignored LiDAR is the frozen image
encoder: it was pretrained on photos, and nothing taught it to read a colour-coded range image. A run that also adapts
the vision encoder is training now, with "LiDAR off must cost ≥ 0.03" fixed as its test in advance.

**The fast head on TriSAR runs on LiDAR.** Switching LiDAR off drops it from 0.37 to 0.22; switching the camera off
changes nothing. A per-frame thermal level match in training and inference made thermal a second sensor it uses:
forest F1 0.30 → 0.37 on average over 3 seeds. Combining the two heads (the VLM's people plus non-overlapping fast
detections) gains +0.054 to +0.066 F1 over the fast head alone, but only with a VLM that abstains in the dark.

![Combined vs single heads](/nightwatch/fig_trisar_router.png)

## 5. Distillation: can the VLM teach the edge detector?

On LLVIP the VLM labelled 8,164 frames from 11 sites whose labels I withheld, and a 12M fast head trained on them.
Two controls decide what that proves. **Self-training**: the small head labels the same frames itself.
**Compute-matched**: the labelled-only head, trained for as many steps. Full test set (3,463 frames), 3 seeds:

<div class="table-wrap">
<table>
<thead><tr><th>12M fast head, labels on the 11 extra sites</th><th>F1, mean (range)</th><th>AP50</th></tr></thead>
<tbody>
<tr><td>none (7 labelled sites only)</td><td>0.821 (0.805–0.833)</td><td>0.872</td></tr>
<tr><td>none, trained for as many steps</td><td>0.844 (0.827–0.861)</td><td>–</td></tr>
<tr><td><b>its own labels (self-training)</b></td><td><b>0.859</b> (0.855–0.862)</td><td>0.893</td></tr>
<tr><td>the 9B's labels</td><td>0.832 (0.827–0.836)</td><td>0.907</td></tr>
<tr><td>the 9B's labels, kept where the small head agrees</td><td>0.853 (0.830–0.868)</td><td><b>0.915</b></td></tr>
<tr><td>human labels on all 18 sites (upper bound)</td><td>0.855</td><td>0.912</td></tr>
</tbody>
</table>
</div>

The self-training control is the most useful number in this project. Labelling the extra sites closes the gap to
human labels, but the small head's own labels do it as well as the 9B's. The 9B's labels still carry something: every
VLM-labelled student ranks detections as well as human labels (AP50 0.91). On easy LLVIP pedestrians, though, the
teacher knows nothing the small head doesn't. A VLM teacher should pay off where the small model is weak.

Escalation follows the same logic. On LLVIP, sending the 20 % least certain frames to the VLM recovers 61 % of the
fast → slow gap (random frames: 28 %). On TriSAR the same rule was *worse* than random, because the fast head is
unsure exactly where the side-branch VLM cannot see.

![Teacher label quality per site](/nightwatch/fig_llvip_teacher_sites.png)

## 6. A genuinely new site: the LTDv2 harbour

By then the forest was not really "new" any more: I had tuned against it. So I dropped the LLVIP thermal fast head
(0.86 at home) onto LTDv2, a harbour camera in another country at another resolution. It adapts on one unlabelled
month and is tested on months a year later. Every arm gets the same 3,000 fine-tuning steps from the same start model,
and thresholds come from LLVIP only. 3 seeds per arm:

<div class="table-wrap">
<table>
<thead><tr><th>Labels for the new site's first month</th><th>F1 a year later</th><th>AP50</th><th>F1 back home</th></tr></thead>
<tbody>
<tr><td>None (zero-shot)</td><td>0.13</td><td>0.07</td><td>0.86</td></tr>
<tr><td>None, same extra training on the home data</td><td>0.14</td><td>0.07</td><td>0.86</td></tr>
<tr><td>The fast head's own labels (self-training)</td><td>0.14</td><td>0.09</td><td>0.86</td></tr>
<tr><td>Only boxes both 9B and fast head agree on</td><td>0.23</td><td>0.22</td><td>0.86</td></tr>
<tr><td><b>The thermal 9B's labels</b></td><td><b>0.46</b> (0.45–0.47)</td><td><b>0.37</b></td><td>0.87</td></tr>
<tr><td>Human labels (upper bound)</td><td>0.57</td><td>0.52</td><td>0.86</td></tr>
</tbody>
</table>
</div>

Across 142 test clips, the comparison I fixed in advance (9B labels filtered by the fast head vs self-training) is
+0.08 to +0.10 F1 on all three seeds, and every 95 % interval excludes zero. Raw 9B labels do even better, but that
comparison was not pre-registered, so I report it as post hoc.

**With no human labelling at the new site, the 9B's labels close three quarters of the gap to human labels, and no
adapted head gets worse at home.** This is the opposite of the LLVIP result, and the reason is visible in the labels:
at the harbour the small head finds 5 % of people, so its own labels are nearly empty (F1 0.10), and only the 9B
(F1 0.44) can bootstrap it. **The teacher matters when the student is out of its depth.** The "agree on" filter that
helped on LLVIP hurts here, because a student that barely fires vetoes most real people.

## 7. Things that did not work, and what they taught me

- **Camera dropout taught hallucination.** My first fused TriSAR VLM reported **3,373** people in the black forest,
  nearly all wrong. I blamed the new sensor branches. A reviewer pointed out that the run also trained with 20 %
  camera dropout, keeping the lab people as targets on black frames. Without dropout the same model reported **zero**.
  The augmentation taught "black camera ⇒ there are people", and only an unseen site exposed it.
- **My side-branch fusion** lost to plain multi-image input on LLVIP and was ignored on TriSAR.
- **LiDAR is still unused** by the VLM, even as an image, and TriSAR precision (0.54–0.60) is below my bar.
- **Single seeds lie.** One student seed gave 0.866 and looked like "the student matches the oracle"; another gave
  0.830. Every claim here uses three seeds or a paired bootstrap.
- **Hand-defined site-memory events.** A "person at the water's edge" region was wrong and fired on almost every
  frame. I excluded those numbers.

## 8. Site memory

A JEPA-style predictor on LTDv2 (one harbour camera over 8 months) predicts the next frame's frozen DINOv3 tokens from
recent frames and a per-site memory. Anomaly is surprise: the peak token prediction error, z-scored per hour and
daylight. Inserted persons are found with AUROC 0.70 (pixel baseline 0.52), crowds with 0.85 (token baseline 0.67).

## 9. Engineering that kept the numbers honest

- Every comparison uses the same frames; a gate refuses to compare runs with different frame counts.
- Every model passes a fail-closed, per-slice regression gate. It blocked the fused VLM against both single-sensor
  VLMs, and both distilled students against self-training.
- Pseudo-labels once covered only the 1,500-frame test subset, and the student silently fell back to ground truth for
  the rest. Caught before training.
- Before each control result arrived, I wrote down what outcome would count against the claim. The self-training
  control then overturned my distillation story.
- Independent AI reviewers (Fable, Codex) checked the plan, the code and the live progress. Every finding was triaged:
  fixed, deferred or rejected with a reason.

*Still running as of this update: the three-image model with an adapted vision encoder (does LiDAR become useful?) and
extra seeds of the camera + thermal and three-image models.*
