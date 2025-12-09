# drives.py

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Any, Optional
import numpy as np


class DriveTier(Enum):
    VIABILITY = auto()       # 절대 삭제 불가
    AFFECT_CORE = auto()     # 감정 핵심 축 (삭제 불가, 비활성만 가능)
    AFFECT_DERIVED = auto()  # 감정 개념 기반 파생 채널
    INSTRUMENTAL = auto()    # 도구적/태스크 기반 채널


@dataclass
class Drive:
    drive_id: str
    tier: DriveTier
    name: str
    weight: float
    fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], float]
    mutable: bool = True       # 삭제 가능 여부
    alive: bool = True         # 비활성 플래그
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(
        self,
        h_t: np.ndarray,
        a_t: np.ndarray,
        ctx: Dict[str, Any],
    ) -> float:
        """
        h_t: 내부 상태 벡터 (R^n)
        a_t: 감정 벡터 (R^d)
        ctx: 기타 컨텍스트 (viability, tool 결과 등)
        """
        if not self.alive:
            return 0.0
        val = float(self.fn(h_t, a_t, ctx))
        return self.weight * val
