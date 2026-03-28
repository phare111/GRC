from typing import Any, Dict, List, Optional, Tuple

from ocr_tools import normalize_text


class Verifier:
    @staticmethod
    def verify(
        parsed: Dict[str, Any],
        img_h: int,
        img_w: int,
        stability_tau: float = 0.60,
        vote_tau: float = 0.40,
    ) -> Tuple[bool, Optional[List[int]], Optional[int]]:
        if not parsed or not isinstance(parsed, dict):
            return False, None, None
        if "text" not in parsed or "certificate" not in parsed:
            return False, None, None

        text = parsed.get("text", "")
        cert = parsed.get("certificate", {}) if isinstance(parsed.get("certificate", {}), dict) else {}

        if "\n" in str(text):
            return False, None, None

        text_norm = normalize_text(text, case_sensitive=False)
        if text_norm == "" or text_norm in {"<unk>", "unk", "<?>", "???"}:
            return False, None, None

        try:
            agreement = float(cert.get("agreement", 0.0))
            vote_frac = float(cert.get("vote_frac", 0.0))
        except Exception:
            return False, None, None

        if not (agreement >= float(stability_tau) and vote_frac >= float(vote_tau)):
            return False, None, None

        return True, None, None
