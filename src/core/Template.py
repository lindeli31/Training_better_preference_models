from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from core.PermPair import PermPair
from core.Prompt import Prompt


class Template(BaseModel, ABC):
    @abstractmethod
    def render(self, perm_pair: PermPair) -> List[Prompt]:
        raise NotImplementedError("Method render() must be implemented in the subclass.")