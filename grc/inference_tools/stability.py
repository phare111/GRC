from typing import List, Tuple

import numpy as np
from rapidfuzz.distance import Levenshtein

from ocr_tools import normalize_text


def text_similarity(a: str, b: str) -> float:
    aa = normalize_text(a, case_sensitive=False)
    bb = normalize_text(b, case_sensitive=False)
    if aa == "" and bb == "":
        return 1.0
    if aa == "" or bb == "":
        return 0.0
    dist = Levenshtein.distance(aa, bb)
    denom = max(len(aa), len(bb), 1)
    return float(1.0 - dist / denom)


def consensus_by_medoid(preds: List[str]) -> Tuple[str, float, float]:
    if not preds:
        return "", 0.0, 0.0

    sims = []
    for i in range(len(preds)):
        row = []
        for j in range(len(preds)):
            row.append(1.0 if i == j else text_similarity(preds[i], preds[j]))
        sims.append(row)

    sims = np.array(sims, dtype=np.float32)
    avg = sims.mean(axis=1)
    best_i = int(np.argmax(avg))
    consensus = preds[best_i]
    agreement = float(avg[best_i])

    c_norm = normalize_text(consensus, case_sensitive=False)
    votes = 0
    for p in preds:
        if normalize_text(p, case_sensitive=False) == c_norm:
            votes += 1
    vote_frac = float(votes / max(1, len(preds)))
    return consensus, agreement, vote_frac
