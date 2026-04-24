from typing import List

from pydantic import BaseModel

from src.core.Message import Message


class Prompt(BaseModel):
    messages: List[Message] = []