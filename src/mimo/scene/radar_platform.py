from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from dataclasses import dataclass
from numpy.typing import NDArray
from mimo.geometry.state_provider import StateProvider

        
class RadarPlatform(ABC):
    __slots__ = ("_id", "_state_provider", "_active")
    
    def __init__(
        self,
        state_provider: StateProvider,
        platform_id: str | None,
        active: bool = True,

    ):
        self._id = platform_id
        self._state_provider = state_provider
        self._active = active
    
    @property
    def id(self) -> str | None:
        return self._id
    
    @property
    def active(self) -> bool:
        return self._active
    
    def get_state(self, time: float):
        return self._state_provider.get_state(time)
    
    




