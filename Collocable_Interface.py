from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Collocable(ABC):

    @abstractmethod
    def get_collocation_matrix(self)-> np.ndarray:
        pass