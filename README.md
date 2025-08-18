# Patch‑Informative Visual‑Inertial Odometry (Stereo + IMU)

A research repo for **identifying informative regions/patches** in images, running **patch‑wise optical flow**, and fusing with IMU via a **MAC‑VO‑style backend + pose graph optimization (PGO)**.

> Goal: Spend compute **only where it matters**. Score pixels → pick patches → run light flow → estimate pose with metric‑aware covariance → PGO.

---

## TL;DR

1. Build dataloaders for **TartanAir**, **EuRoC**, and **custom drone** logs (stereo + IMU + GT pose; flow if available).
2. Produce per‑pixel **information scores** (heuristics or learned).
3. **Select patches** under a compute budget.
4. Run **lightweight optical flow** on selected patches (coarse→refine).
5. (Optional) Try **FlowDiffuser** with patch‑scaler/early‑exit ideas.
6. Plug flow + uncertainty into **MAC‑VO backend**; run **PGO**.
7. Evaluate ATE/RPE, EPE, runtime; ablate.

---

## Repo Structure (proposed)

```
patch_vio/
  configs/                 # Hydra/OMEGACONF config trees
  dataio/                  # Datasets + dataloaders
    tartanair.py
    euroc.py
    drone_custom.py
    transforms.py          # rectification, resize, normalize, sync
  scoring/                 # pixel/patch information scoring
    heuristics.py          # Harris/entropy/texture/photometric residuals
    learned.py             # tiny CNN for score map; losses
    combine.py             # weighted fusion of cues → score_map
  selection/               # policies to pick patches given score_map
    nms.py
    topk.py
    slic_superpixels.py
    bandit_rl.py
  flow/                    # patch-wise optical flow wrappers
    raft_small.py
    spy_net.py
    flowformer_lite.py
    utils.py               # tiling, padding, warping, refinements
  diffuser/                # (optional) FlowDiffuser integrations
    flowdiffuser.py
    patch_scaler.py        # early-exit schedules per patch
  backend/                 # VO backend (MAC‑VO style)
    macvo_frontend_bridge.py
    covariance.py          # metric‑aware covariance plumbing
    pgo.py                 # factor graph, robust losses
    imu_preintegration.py
  eval/
    metrics.py             # ATE, RPE, EPE, FPS
    benchmarks.py          # scenario runners
  scripts/
    prepare_data.py
    train_score_net.py
    run_patch_flow.py
    run_macvo_pipeline.py
    evaluate_vio.py
  docs/
    notes.md
  README.md
```

---

## Environment

- Python 3.10+, PyTorch 2.x, CUDA 12.x
- Optional: **Hydra** for configs, **Weights & Biases** for logs, **OpenCV** for vision ops
- Build requirements: `pip install -r requirements.txt`

---

## Data

### Supported Datasets

- **TartanAir**: synthetic, stereo, GT pose, **GT optical flow available for many sequences**.
- **EuRoC MAV**: real, stereo (cam0/cam1), IMU, accurate GT pose, **no GT flow**.
- **Custom Drone**: ROS bags or timestamped folders; assume stereo, IMU, optional motion‑capture pose.

### Folder Layout (expected)

```
DATA_ROOT/
  tartanair/
    {env}/{difficulty}/P0000/
      image_left/000000.png
      image_right/000000.png
      flow/000000.flo        # if available
      pose_gt.txt
      imu.csv
  euroc/
    MH_01_easy/
      cam0/data/*.png
      cam1/data/*.png
      imu/data.csv
      state_groundtruth_estimate.csv
      calib.yaml             # K, D, T_cam_imu, etc.
  drone_custom/
    seq_01/
      left/*.png  right/*.png
      imu.csv     pose_gt.csv (optional)
      calib.yaml
```

### Dataloaders

Each dataloader returns a dict per sample or mini‑sequence:

```python
{
  'left':  (B,3,H,W),
  'right': (B,3,H,W),
  'K':     (B,3,3), 'D': (B,dist_params),
  'T_cam_imu': (B,4,4) or (B,7),
  'imu':   (B,T,6),           # gyro(ax,ay,az), accel(gx,gy,gz) or vice‑versa
  'pose_gt': (B,4,4) or (B,7),
  'flow_gt': (B,2,H,W) or None,
  'timestamps': {...}
}
```

**Substeps & Methods**

1. **Calibration ingest**: parse `calib.yaml`; cache intrinsics/extrinsics; assert stereo rectified or rectify offline.
2. **Time sync**: spline‑interpolate IMU to cam timestamps; handle dropped frames; sanity checks on dt.
3. **Normalization**: intensity scaling, optional CLAHE; resize with intrinsics scaling.
4. **Augmentations (optional)**: mild photometric/geometric; keep stereo consistency.
5. **Flow for EuRoC/custom**: if no GT flow, optionally **synthesize pseudo‑flow** using depth (from stereo disparity) + GT pose, or run a robust pretrained light flow model to bootstrap (flagged as pseudo).

---

## Stage 1 — Learn the Data

- **Sequence stats**: motion ranges, baseline, FPS, blur, brightness changes.
- **IMU sanity**: bias estimation, Allan variance snapshot; unit checks.
- **Quick baselines**: run a full‑frame light flow on a few sequences; log EPE, runtime.

---

## Stage 2 — Per‑Pixel Information Scoring

Produce a **score map** `S(x,y) ∈ [0,1]` indicating usefulness for flow & VO.

### Heuristic Cues (fast)

- **Texture/Corner strength**: Harris, Shi‑Tomasi, gradients, Laplacian variance.
- **Entropy**: local Shannon entropy windows; penalize low‑texture.
- **Photometric stability**: brightness‑constancy residuals via warped patches; lower residual → higher score.
- **Temporal variation / parallax**: gradient change across frames; epipolar consistency.
- **Depth reliability** (stereo): avoid extremely near/far planes that break linearity.
- **Dynamic object suppression**: lightweight semantic mask (cars/person), or simple temporal inconsistency detector.

### Learned Score (tiny CNN)

- Inputs: `[I_t, I_{t+1}, |∇I_t|, |∇I_{t+1}|, (optional) coarse cost‑volume diag, disparity]` → `S`.
- **Losses (proxy)**: encourage high‑S where **reprojection residuals** and **flow EPE** are low, and **triangulation covariances** are small. Can use ranking loss against heuristic targets + self‑supervision via photometric error.

### Score Fusion

Combine cues with learned weights:

$$
S = \sigma\big( w_1 C_{corner} + w_2 H_{entropy} - w_3 R_{photo} + w_4 P_{parallax} - w_5 M_{dynamic} + \dots \big)
$$

Normalize per‑tile; ensure calibration‑invariant scaling.

**Artifacts saved**: `score_map.png`, `score_raw.npz` (float32), per‑cue maps for debugging.

---

## Stage 3 — Patch/Region Selection (Compute‑Aware)

Given `S`, pick patches under a **budget** (e.g., max N patches or FLOP cap).

**Policies**

1. **Top‑K + NMS**: sample top‑K peaks; apply non‑maximum suppression with stride ≈ patch size.
2. **Grid‑aware sampling**: ensure spatial coverage; top‑M per grid cell.
3. **SLIC superpixels**: aggregate `S` per superpixel; select top superpixels; fit patch boxes.
4. **Bandit / RL** (optional): contextual bandit choosing regions maximizing downstream pose improvement; reward = ATE/RPE gain.

**Hyperparams**: patch sizes `{32, 64, 96}`, overlap `~25%`, pyramid levels `{1/2, 1/1}`.\
**Outputs**: list of patch boxes, pyramid level, expected difficulty.

---

## Stage 4 — Patch‑Wise Optical Flow

Run a **light** flow model only on selected patches. Start coarse → refine uncertain ones.

**Candidate models**: RAFT‑small, SPyNet, FlowFormer‑lite, LiteFlowNet.\
**Substeps**

1. **Coarse pass** (lower res): run on all patches to get initial flow and confidence.
2. **Early exit**: accept patches whose residual < τ and confidence > θ.
3. **Refinement pass**: upsample & run only on hard patches; optionally increase iterations.
4. **Border handling**: pad patches; flow cropping; blend back via feathering.
5. **IMU‑aided init (optional)**: seed patch warps using IMU‑derived SE(3) to reduce iterations.

**Artifacts**: `flow_patch_*.npz`, `confidence_patch_*.npz`, mosaic visualization.

---

## Stage 5 — (Optional) Diffusion with Patch‑Scaler / Early Exit

Plug **FlowDiffuser** or similar:

- Assign **shorter denoising schedules to easy patches**, longer to hard ones (per‑patch NFE).
- **Dynamic masks**: skip denoising in unselected regions.
- **Stoppers**: terminate when patch likelihood or residual plateaus.
- Cache per‑patch **context encodings** to avoid recompute across steps.

Deliverables: schedule policy, speed vs. EPE curves; ablation w/ and w/o patch‑scaler.

---

## Stage 6 — Backend: MAC‑VO‑Style Fusion + PGO

Use the MAC‑VO family idea: **metric‑aware covariance** for features/patches → robust VO.

**Plumbing**

1. **Track selection**: convert patch flows to keypoint tracks (e.g., center‑of‑mass peaks or dense‑to‑sparse sampling within each patch).
2. **Covariance**: propagate **flow confidence + photometric residuals + geometry** into **3D keypoint covariance** (anisotropic, depth‑aware).
3. **IMU preintegration**: integrate between frames; manage biases (state vector includes gyro/accel biases).
4. **Factor graph / PGO**: states = poses + velocities + biases; factors = IMU, visual reprojection with covariances; robust losses (Huber/Cauchy/Tukey), Schur complement for landmarks.
5. **Initialization**: stereo triangulation + gravity alignment; or short VIO bootstrapping window.

**Outputs**: pose trajectory, landmark map (optional), per‑frame covariances.

---

## Evaluation & Validation

- **VO**: ATE, RPE (trans/rot), scale drift (mono subsets if needed), % tracking lost.
- **Flow**: EPE on regions with GT (TartanAir), occlusion‑aware metrics.
- **Runtime**: FPS, FLOPs per stage, GPU mem.
- **Ablations**: scoring (heuristic vs learned), selection policy, patch budget, with/without IMU init, with/without diffusion.

**Protocols**

- Hold‑out sequences per dataset.
- Report per‑environment: texture‑poor, dynamic, low‑light.
- Seed control; config hashes; commit SHA in logs.

---

## Logging & Reproducibility

- **Hydra** configs for: dataset, scoring, selection, flow, backend, eval.
- Save: configs, metrics.json, artifacts.
- Determinism flags where feasible; cuDNN benchmark off for eval.

---

## Scripts (reference CLI)

```bash
# 1) Prepare/verify data & calibration
python scripts/prepare_data.py data.root=...</n> dataset=tartanair  rectify=true

# 2) Generate score maps (heuristics)
python scripts/run_patch_flow.py mode=score_only scoring=heuristics selection=topk

# 3) Train learned scorer (optional)
python scripts/train_score_net.py data=... scoring=learned trainer.max_epochs=20

# 4) Patch‑wise flow (coarse→refine)
python scripts/run_patch_flow.py selection=grid budget.n_patches=256 flow=raft_small

# 5) Full pipeline to poses + PGO
python scripts/run_macvo_pipeline.py backend.macvo=true eval=vio_only

# 6) Evaluate
python scripts/evaluate_vio.py dataset=euroc sequence=MH_03_medium
```

---

## “Take Stock” Checklist (cadence)

- **Week 1–2**: Dataloaders 100% + rectification + sync; baseline full‑frame light flow metrics.
- **Week 2–3**: Heuristic score map + patch selection; first patch‑flow results.
- **Week 3–4**: Learned scoring (if time) + early‑exit loop + IMU‑aided init.
- **Week 4–5**: Backend integration; initial PGO trajectories; tune covariances.
- **Week 5–6**: Optional diffusion/patch‑scaler; ablations; paper‑style tables.

---

## Notes & Tips

- Keep **unit conventions** consistent (rad/s, m/s²).
- Visualize everything: score maps, selected patches, per‑patch residuals.
- Start with **deterministic heuristics**; only then add the learned scorer.
- Always budget compute (patch count, iterations) and track wall‑clock.

---

## References (for orientation)

- TartanAir, EuRoC MAV; RAFT/RAFT‑small, SPyNet, LiteFlowNet, FlowFormer‑lite; FlowDiffuser.
- MAC‑VO / DPVO families for learned VO backends with metric‑aware covariance.

