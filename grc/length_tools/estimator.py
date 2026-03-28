from pathlib import Path

import pandas as pd

from .geo_bound import cal_geo_bound


def _resolve_image_path(rel_path: str, input_csv: str, image_root: str = "") -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path

    candidates = [path]
    csv_parent = Path(input_csv).resolve().parent
    candidates.append(csv_parent / path)

    if image_root:
        root = Path(image_root)
        candidates.append(root / path)
        candidates.append(csv_parent / root / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def add_len_est(input_csv, output_csv=None, image_root="", alpha=2, safe_buffer=2):
    """Read a CSV with columns like `image,gt` and append `L_est`."""
    df = pd.read_csv(input_csv)
    df = df.copy()

    if "image" not in df.columns:
        raise ValueError("The input CSV must contain an 'image' column.")

    estimates = []

    for rel_path in df["image"].astype(str):
        image_path = _resolve_image_path(rel_path, input_csv=input_csv, image_root=image_root)
        L_est = cal_geo_bound(
            str(image_path),
            alpha=alpha,
            safe_buffer=safe_buffer,
        )
        estimates.append(L_est)

    df["L_est"] = estimates

    if output_csv is None:
        output_csv = str(Path(input_csv).with_name(Path(input_csv).stem + "_with_L_est.csv"))

    df.to_csv(output_csv, index=False)
    return df
