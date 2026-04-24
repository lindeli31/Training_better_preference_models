from typing import List

from pydantic import BaseModel

from core.Message import Message


class PermPair(BaseModel):
    context: List[Message] = []
    answer_candidate_a: str
    answer_candidate_b: str
    ground_truth: int
    metadata: dict = {}