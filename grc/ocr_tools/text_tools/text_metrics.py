import re
import unicodedata

import numpy as np
from rapidfuzz.distance import Levenshtein


def normalize_text(text, case_sensitive: bool = False) -> str:
    """Normalize OCR strings before comparison."""
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r", "\n").replace("\t", " ")
    s = s.strip()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'") or (s[0] == s[-1] == "`")):
        s = s[1:-1].strip()
    if not case_sensitive:
        s = s.casefold()
    return s


def calculate_cer(pred: str, gt: str, case_sensitive: bool = False) -> float:
    """Compute normalized character error rate."""
    p = normalize_text(pred, case_sensitive=case_sensitive)
    g = normalize_text(gt, case_sensitive=case_sensitive)
    if not g:
        return 1.0 if p else 0.0
    dist = Levenshtein.distance(p, g)
    return dist / max(1, len(g))


def winsorized_mean(values, percentile: float) -> float:
    """Compute winsorized mean with an upper cap."""
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float32)
    cap = np.percentile(arr, percentile)
    arr = np.clip(arr, 0.0, cap)
    return float(arr.mean())
