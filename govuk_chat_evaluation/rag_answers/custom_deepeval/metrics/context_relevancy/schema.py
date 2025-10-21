from typing import List, Literal
from pydantic import BaseModel, Field


class Verdict(BaseModel):
    verdict: Literal["yes", "idk", "no"]
    reason: str | None = Field(default=None)


class VerdictCollection(BaseModel):
    verdicts: List[Verdict]


class Truth(BaseModel):
    context: str
    facts: List[str]


class TruthCollection(BaseModel):
    truths: List[Truth]


class InformationNeedsCollection(BaseModel):
    information_needs: List[str]


class ScoreReason(BaseModel):
    reason: str
