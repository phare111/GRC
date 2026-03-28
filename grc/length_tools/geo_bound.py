import cv2
import numpy as np


def cal_geo_bound(image_path, alpha=2, safe_buffer=2):
    # Estimate a conservative character-length bound from foreground geometry.
    img = cv2.imread(image_path)
    if img is None:
        return -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h == 0 or w == 0:
        return safe_buffer

    target_h = 64
    scale = target_h / h
    target_w = int(w * scale)
    gray = cv2.resize(gray, (target_w, target_h))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    points = cv2.findNonZero(bw)
    if points is None:
        return safe_buffer

    pts = points.reshape(-1, 2)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    d01 = np.linalg.norm(box[0] - box[1])
    d12 = np.linalg.norm(box[1] - box[2])

    if d01 > d12:
        vec = box[1] - box[0]
        box_h = d12
    else:
        vec = box[2] - box[1]
        box_h = d01

    if np.linalg.norm(vec) == 0 or box_h == 0:
        return safe_buffer

    vec_unit = vec / np.linalg.norm(vec)
    projections = np.dot(pts, vec_unit)
    min_p, max_p = np.min(projections), np.max(projections)
    bins = max(1, int(np.ceil(max_p - min_p)))
    hist, _ = np.histogram(projections, bins=bins)
    effective_w = np.count_nonzero(hist > 1)
    effective_aspect_ratio = effective_w / max(1.0, float(box_h))
    L_est = int(np.ceil(effective_aspect_ratio * alpha)) + safe_buffer
    return max(safe_buffer, L_est)
