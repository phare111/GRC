from typing import List

from pydantic import BaseModel, Field


class OCRCertificate(BaseModel):
    k_views: int = Field(..., description="Number of deterministic views used for stability check.")
    agreement: float = Field(..., description="Stability score in [0,1].")
    vote_frac: float = Field(..., description="Fraction of views matching the consensus in [0,1].")
    preds: List[str] = Field(..., description="Per-view predictions for audit.")


class OCRResponse(BaseModel):
    text: str = Field(..., description="The transcribed text. Use <UNK> if illegible.")
    certificate: OCRCertificate = Field(..., description="Evidence supporting the transcription.")
