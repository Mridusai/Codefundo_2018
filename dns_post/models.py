from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimeWindow:
    label: str
    start: Optional[float] = None
    end: Optional[float] = None

    def contains(self, time: float) -> bool:
        if self.start is not None and time < self.start:
            return False
        if self.end is not None and time > self.end:
            return False
        return True


@dataclass(frozen=True)
class ComparisonSpec:
    label: str
    reference: str
