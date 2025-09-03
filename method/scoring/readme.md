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
