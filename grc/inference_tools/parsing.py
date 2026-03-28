import json
import re


def strip_reasoning_prefix(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<analysis>.*?</analysis>", "", s, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"done thinking\s*", s, flags=re.IGNORECASE)
    if m:
        s = s[m.end():]
    return s.strip()


def extract_ocr_answer(raw: str) -> str:
    if raw is None:
        return ""
    s0 = str(raw).strip()
    if not s0:
        return ""

    try:
        obj = json.loads(s0)
        if isinstance(obj, dict) and "text" in obj:
            return str(obj.get("text", "")).strip()
    except Exception:
        pass

    s = strip_reasoning_prefix(s0)
    if not s:
        return ""

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return ""

    pat = re.compile(r"[A-Za-z0-9\$\.\,\-\+\(\)\#\&\%\/]")
    cand = ""
    for ln in reversed(lines):
        if pat.search(ln):
            cand = ln
            break
    if not cand:
        cand = lines[-1]

    if len(cand) >= 2 and ((cand[0] == cand[-1] == '"') or (cand[0] == cand[-1] == "'") or (cand[0] == cand[-1] == "`")):
        cand = cand[1:-1].strip()
    return cand.strip()
