import unicodedata


def repetition_meltdown_score(s: str, k: int = 6) -> float:
    if len(s) < 2 * k:
        return 1.0
    grams = [s[i:i + k] for i in range(0, len(s) - k + 1)]
    if not grams:
        return 1.0
    return len(set(grams)) / max(1, len(grams))


def is_meltdown(pred_raw_or_text: str, max_len: int = 256) -> bool:
    s = str(pred_raw_or_text) if pred_raw_or_text is not None else ""
    s = unicodedata.normalize("NFKC", s)
    if len(s) > max_len:
        return True
    if len(s) >= 40 and repetition_meltdown_score(s, k=6) < 0.55:
        return True
    return False
