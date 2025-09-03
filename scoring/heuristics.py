import cv2
import numpy as np
from skimage.filters import rank
from skimage.morphology import disk
from typing import Tuple, Optional, Dict


class HeuristicScorer:
    def __init__(self, resolution=(32, 32)):
        self.resolution = resolution  # (height, width)

    def compute_corner_strength(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cornerHarris(gray, 2, 3, 0.04)

    def compute_entropy(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        entropy = rank.entropy(gray, disk(3))
        return entropy

    def compute_photometric_residual(self, img1, img2):
        # Simple absolute difference as a placeholder
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))

    def resize_map(self, score_map):
        return cv2.resize(score_map, (self.resolution[1], self.resolution[0]), interpolation=cv2.INTER_AREA)

    def score(self, frame_t_minus_1, frame_t):
        # Compute cues
        corner = self.compute_corner_strength(frame_t)
        entropy = self.compute_entropy(frame_t)
        photo_res = self.compute_photometric_residual(frame_t_minus_1, frame_t)

        # Normalize cues
        corner = (corner - corner.min()) / (corner.ptp() + 1e-8)
        entropy = (entropy - entropy.min()) / (entropy.ptp() + 1e-8)
        photo_res = (photo_res - photo_res.min()) / (photo_res.ptp() + 1e-8)

        # Weighted sum (example weights)
        S = (
            1.0 * corner +
            1.0 * entropy -
            1.0 * photo_res
        )

        # Sigmoid squashing
        S = 1 / (1 + np.exp(-S))

        # Resize to output resolution
        S_resized = self.resize_map(S)

        # Clamp to [0,1]
        S_resized = np.clip(S_resized, 0, 1)

        return S_resized.astype(np.float32)
    


class HeuristicScorerV2:
    """
    Robust heuristic patch scorer for frame pairs.

    Key upgrades vs V1:
      - Photometric compensation + gradient-constancy residual
      - Multi-scale cues (max-fusion across scales)
      - Gradient magnitude + Laplacian response for texture/blur robustness
      - Robust percentile normalization (avoids outliers)
      - Patch aggregation via top-k mean (not raw resize)
      - Optional temporal smoothing (EMA)
      - Optional NMS-based diverse patch selection
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (32, 32),     # (H_patches, W_patches)
        weights: Optional[Dict[str, float]] = None, # fusion weights
        use_multiscale: bool = True,
        multiscale_sigmas: Tuple[int, ...] = (0, 1, 2),  # 0 => original, else Gaussian blur ksize = 2*s+1
        topk_ratio: float = 0.3,                   # for patch aggregation
        ema_alpha: Optional[float] = None,         # 0<alpha<1 to enable EMA temporal smoothing
        percentile_clip: Tuple[float, float] = (2.0, 98.0) # robust normalization
    ):
        self.resolution = resolution
        self.use_multiscale = use_multiscale
        self.multiscale_sigmas = multiscale_sigmas
        self.topk_ratio = topk_ratio
        self.ema_alpha = ema_alpha
        self.percentile_clip = percentile_clip
        self.prev_score_map = None  # for EMA

        self.weights = weights or {
            "corner": 1.0,
            "texture": 1.0,
            "residual": -1.0,  # negative weight: penalize high residual
        }

    # -----------------------------
    # Utility: robust normalization
    # -----------------------------
    @staticmethod
    def robust_minmax(x: np.ndarray, p_lo=2.0, p_hi=98.0, eps=1e-6) -> np.ndarray:
        lo, hi = np.percentile(x, [p_lo, p_hi])
        x = np.clip(x, lo, hi)
        return (x - lo) / (hi - lo + eps)

    # ----------------------------------------
    # Cues: corners, texture, residual signals
    # ----------------------------------------
    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            return img
        else:
            raise ValueError("Unsupported image shape for grayscale conversion.")

    @staticmethod
    def corner_shi_tomasi_response(gray: np.ndarray, blockSize=2, ksize=3) -> np.ndarray:
        """
        Use MinEigen (Shi–Tomasi) response map as a robust corner strength.
        """
        gray_f = gray.astype(np.float32)
        # cv2.cornerMinEigenVal gives per-pixel response
        resp = cv2.cornerMinEigenVal(gray_f, blockSize=blockSize, ksize=ksize)
        return resp

    @staticmethod
    def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        return mag

    @staticmethod
    def laplacian_response(gray: np.ndarray) -> np.ndarray:
        """
        Laplacian magnitude as a blur/edge detector (higher => sharper/edgier).
        """
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        return np.abs(lap)

    @staticmethod
    def gaussian_blur(img: np.ndarray, sigma: int) -> np.ndarray:
        if sigma <= 0:
            return img
        k = 2 * sigma + 1
        return cv2.GaussianBlur(img, (k, k), sigmaX=0, sigmaY=0)

    @staticmethod
    def photometric_compensated_residual(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """
        Find a, b minimizing ||a*g1 + b - g2||^2 (closed-form LS),
        compute absolute residual after compensation.
        """
        g1f = g1.astype(np.float32)
        g2f = g2.astype(np.float32)

        # Solve for a: sum(g1*g2)/sum(g1*g1), and b: mean(g2) - a*mean(g1)
        denom = np.sum(g1f * g1f) + 1e-6
        a = np.sum(g1f * g2f) / denom
        b = float(np.mean(g2f) - a * np.mean(g1f))
        g1p = a * g1f + b
        r_intensity = np.abs(g1p - g2f)

        # Gradient-constancy residual (more illumination-invariant)
        gx1 = cv2.Sobel(g1f, cv2.CV_32F, 1, 0, ksize=3)
        gx2 = cv2.Sobel(g2f, cv2.CV_32F, 1, 0, ksize=3)
        gy1 = cv2.Sobel(g1f, cv2.CV_32F, 0, 1, ksize=3)
        gy2 = cv2.Sobel(g2f, cv2.CV_32F, 0, 1, ksize=3)
        r_grad = np.abs(gx1 - gx2) + np.abs(gy1 - gy2)

        # Blend intensity + gradient residuals
        return 0.5 * r_intensity + 0.5 * r_grad

    # -----------------------------
    # Multi-scale cue computation
    # -----------------------------
    def multiscale_max(self, gray: np.ndarray, compute_fn) -> np.ndarray:
        """
        Compute a cue at multiple Gaussian scales and take per-pixel max.
        """
        acc = None
        for s in self.multiscale_sigmas:
            g = self.gaussian_blur(gray, s)
            val = compute_fn(g)
            acc = val if acc is None else np.maximum(acc, val)
        return acc

    # -----------------------------------
    # Patch aggregation & selection utils
    # -----------------------------------
    def aggregate_to_patches(self, score: np.ndarray) -> np.ndarray:
        """
        Aggregate pixel-level score to (H_res, W_res) patches via top-k mean.
        """
        H_res, W_res = self.resolution
        H, W = score.shape[:2]
        ph = H // H_res
        pw = W // W_res
        # center crop to divisible area
        score = score[:ph * H_res, :pw * W_res]
        # reshape into patches
        patch = score.reshape(H_res, ph, W_res, pw).transpose(0, 2, 1, 3)  # (H_res, W_res, ph, pw)
        flat = patch.reshape(H_res, W_res, ph * pw)

        k = max(1, int(self.topk_ratio * flat.shape[-1]))
        # select top-k along last axis
        # Using argpartition for efficiency
        idx = np.argpartition(flat, -k, axis=-1)[..., -k:]
        # gather top-k values
        topk_vals = np.take_along_axis(flat, idx, axis=-1)
        # mean over top-k
        patch_scores = topk_vals.mean(axis=-1).astype(np.float32)
        return patch_scores

    @staticmethod
    def nms_on_grid(scores: np.ndarray, budget: int, radius: int = 1):
        """
        Simple NMS over a 2D grid of patch scores.
        Returns list of (i,j) indices for selected patches.
        """
        H, W = scores.shape
        selected = []
        suppressed = np.zeros_like(scores, dtype=bool)

        # Flatten indices by score
        flat_idx = np.argsort(scores, axis=None)[::-1]
        coords = np.column_stack(np.unravel_index(flat_idx, (H, W)))

        for (i, j) in coords:
            if suppressed[i, j]:
                continue
            selected.append((i, j))
            if len(selected) >= budget:
                break
            # suppress neighborhood
            i0, i1 = max(0, i - radius), min(H, i + radius + 1)
            j0, j1 = max(0, j - radius), min(W, j + radius + 1)
            suppressed[i0:i1, j0:j1] = True

        return selected

    def score(self, frame_t_minus_1: np.ndarray, frame_t: np.ndarray) -> np.ndarray:
        """
        Returns a (H_res, W_res) patch score map in [0,1].
        """

        g1 = self.to_gray(frame_t_minus_1)
        g2 = self.to_gray(frame_t)

        # CUES (multi-scale)
        corner_map = self.multiscale_max(g2, self.corner_shi_tomasi_response) if self.use_multiscale \
                     else self.corner_shi_tomasi_response(g2)

        grad_mag = self.multiscale_max(g2, self.gradient_magnitude) if self.use_multiscale \
                   else self.gradient_magnitude(g2)

        lap_resp = self.multiscale_max(g2, self.laplacian_response) if self.use_multiscale \
                   else self.laplacian_response(g2)

        # Treat "texture" as a fusion of gradient magnitude and Laplacian
        texture_map = 0.5 * grad_mag + 0.5 * lap_resp

        # Residual (photometric compensated + gradient constancy)
        residual_map = self.photometric_compensated_residual(g1, g2)

        # ROBUST NORMALIZATION
        p_lo, p_hi = self.percentile_clip
        corner_n = self.robust_minmax(corner_map, p_lo, p_hi)
        texture_n = self.robust_minmax(texture_map, p_lo, p_hi)
        residual_n = self.robust_minmax(residual_map, p_lo, p_hi)

        # FUSION (affine + sigmoid)
        S = (
            self.weights["corner"] * corner_n +
            self.weights["texture"] * texture_n +
            self.weights["residual"] * residual_n
        )

        S = 1.0 / (1.0 + np.exp(-S))  # logistic squashing

        # OPTIONAL EMA TEMPORAL SMOOTHING (on pixel map before patching)
        if self.ema_alpha is not None and 0.0 < self.ema_alpha < 1.0:
            if self.prev_score_map is None or self.prev_score_map.shape != S.shape:
                self.prev_score_map = S.copy()
            else:
                S = self.ema_alpha * self.prev_score_map + (1.0 - self.ema_alpha) * S
                self.prev_score_map = S.copy()

        # PATCH AGGREGATION (top-k mean per patch)
        patch_scores = self.aggregate_to_patches(S)

        # Clamp to [0,1]
        patch_scores = np.clip(patch_scores, 0.0, 1.0).astype(np.float32)
        return patch_scores

    # ------------------------------------
    # Convenience: get top patches by NMS
    # ------------------------------------
    def select_patches(
        self,
        patch_scores: np.ndarray,
        budget: int,
        nms_radius: int = 1
    ):
        """
        Returns:
            selected_indices: List[(i, j)] top patches after NMS
        """
        return self.nms_on_grid(patch_scores, budget=budget, radius=nms_radius)
