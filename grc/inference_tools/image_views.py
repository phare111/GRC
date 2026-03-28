from typing import List

import cv2
import numpy as np


def resize(img: np.ndarray, min_side: int = 96, max_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    scale_up = max(min_side / max(1, min(h, w)), 1.0)
    scale_down = min(max_side / max(1, max(h, w)), 1.0)
    scale = min(scale_up, 10.0)
    scale = min(scale, scale_down)

    if abs(scale - 1.0) < 1e-6:
        scaled = img
    else:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)

    factor = 32
    new_h, new_w = scaled.shape[:2]
    pad_h = (factor - new_h % factor) % factor
    pad_w = (factor - new_w % factor) % factor
    if pad_h > 0 or pad_w > 0:
        scaled = cv2.copyMakeBorder(
            scaled,
            0,
            pad_h,
            0,
            pad_w,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
    return scaled


def _warp_affine_with_white(
    img_bgr: np.ndarray,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
) -> np.ndarray:
    if img_bgr is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Centered similarity transform.
    m = np.array(
        [
            [scale, 0.0, (1.0 - scale) * cx + tx],
            [0.0, scale, (1.0 - scale) * cy + ty],
        ],
        dtype=np.float32,
    )

    return cv2.warpAffine(
        img_bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def build_determ_views(img_bgr: np.ndarray, k: int) -> List[np.ndarray]:
    if img_bgr is None:
        return []

    h, w = img_bgr.shape[:2]

    # Mild geometric perturbations that should preserve nominal text.
    dx = max(1, int(round(0.04 * w)))
    dy = max(1, int(round(0.02 * h)))

    base = img_bgr
    views = [
        base,
        _warp_affine_with_white(base, scale=1.00, tx=-dx, ty=0.0),   # left shift
        _warp_affine_with_white(base, scale=1.00, tx=dx, ty=0.0),    # right shift
        _warp_affine_with_white(base, scale=1.06, tx=0.0, ty=-dy),   # mild zoom-in
        _warp_affine_with_white(base, scale=0.94, tx=0.0, ty=dy),    # mild zoom-out
    ]

    if k <= len(views):
        return views[:k]
    return [views[i % len(views)] for i in range(k)]