import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .parsing import extract_ocr_answer
from .stability import consensus_by_medoid

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, model_name: str, sleep: float = 0.0, max_retries: int = 2, timeout_backoff: float = 0.8):
        self.model_name = model_name
        self.sleep = sleep
        self.max_retries = max_retries
        self.timeout_backoff = timeout_backoff

        self.baseline_system_prompt = (
            "You are an OCR engine.\n"
            "Output ONLY the text in the image.\n"
            "No explanation, no quotes, no newlines.\n"
            "If illegible, output <UNK>.\n"
            "Do NOT output anything else."
        )
        self.cert_system_prompt = (
            "You are a strict OCR engine.\n"
            "Output ONLY the text in the image.\n"
            "No explanation, no quotes, no newlines.\n"
            "If illegible, output <UNK>.\n"
            "Do NOT output anything else."
        )

    def _chat(
        self,
        messages: List[Dict[str, Any]],
        fmt_schema: Optional[Dict[str, Any]],
        num_predict: int,
        stop: Optional[List[str]],
    ) -> str:
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.sleep > 0:
                    time.sleep(self.sleep)
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": int(num_predict),
                    },
                }
                if stop is not None:
                    kwargs["options"]["stop"] = stop
                if fmt_schema is not None:
                    kwargs["format"] = fmt_schema
                import ollama

                resp = ollama.chat(**kwargs)
                return resp["message"]["content"]
            except Exception as e:
                last_err = e
                logger.warning("Ollama chat error (attempt %s/%s): %s", attempt + 1, self.max_retries + 1, e)
                time.sleep(self.timeout_backoff * (attempt + 1))
        raise RuntimeError(f"Ollama chat failed after retries: {last_err}")

    def infer_baseline(self, image_path: str, num_predict: int = 64) -> Tuple[str, str]:
        raw = self._chat(
            messages=[
                {"role": "system", "content": self.baseline_system_prompt},
                {"role": "user", "content": "Read the text in the image.", "images": [image_path]},
            ],
            fmt_schema=None,
            num_predict=num_predict,
            stop=None,
        )
        return raw, extract_ocr_answer(raw)

    def infer_system_stability(self, image_paths: List[str], num_predict: int = 64) -> Tuple[str, Dict[str, Any], str]:
        raws = []
        preds = []
        for path in image_paths:
            raw = self._chat(
                messages=[
                    {"role": "system", "content": self.cert_system_prompt},
                    {"role": "user", "content": "Read the text in the image.", "images": [path]},
                ],
                fmt_schema=None,
                num_predict=num_predict,
                stop=None,
            )
            raws.append(raw)
            preds.append(extract_ocr_answer(raw))

        consensus, agreement, vote_frac = consensus_by_medoid(preds)
        parsed = {
            "text": consensus,
            "certificate": {
                "k_views": int(len(image_paths)),
                "agreement": float(agreement),
                "vote_frac": float(vote_frac),
                "preds": [str(x) for x in preds],
            },
        }
        raw_join = "\n---VIEW---\n".join(str(x) for x in raws)
        return raw_join, parsed, consensus
