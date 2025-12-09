# reward_system.py

from __future__ import annotations
from typing import Dict, Any, Callable, Literal, List, Optional
from dataclasses import dataclass, field
import numpy as np

from .drives import Drive, DriveTier
from .affect_kernel import AffectKernel

AffectCoreMode = Literal["homeostatic", "maximize", "minimize"]


@dataclass
class AffectAxisConfig:
    """
    AffectKernel의 각 축에 대해:
    - mode: homeostatic / maximize / minimize
    - setpoint: 목표값 (보통 [-1, 1] 안)
    - band: homeostatic 모드용 허용 오차
    - exponent: 페널티 곡률
    - weight: Drive.weight 기본값
    - adaptive_gain: 편차가 클 때 가중치를 얼마나 증폭할지 (Urgency)
    - soft_penalty: Softplus 기반 부드러운 페널티 사용 여부
    """
    name: str
    mode: AffectCoreMode = "homeostatic"
    setpoint: float = 0.0
    band: float = 0.3
    exponent: float = 2.0
    weight: float = 1.0
    adaptive_gain: float = 0.0
    soft_penalty: bool = True


@dataclass
class AffectOverrideConfig:
    """
    런타임 오버라이드 설정 객체.
    ctx['affect_override'] 에 주입하여 사용.
    """
    # axis_index -> {param_name: value}
    axis_overrides: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class RewardSystem:
    """
    - AffectKernel: 항상 존재하는 감정 좌표계
    - Drive 집합: 티어별로 관리 (생존/감정코어/파생/도구적)
    """

    def __init__(
        self,
        hidden_dim: int,
        affect_dim: int = 4,
        affect_core_axes: Optional[List[AffectAxisConfig]] = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.affect_kernel = AffectKernel(dim=affect_dim, hidden_dim=hidden_dim)
        self.drives: Dict[str, Drive] = {}
        self.last_affect: np.ndarray | None = None
        self.last_cost_breakdown: Dict[str, float] = {}  # 시각화/디버깅용

        # 기본 축 설정이 없으면 디폴트로 채운다
        if affect_core_axes is None:
            affect_core_axes = [
                # 예시: 축 0 = valence (0 근처, 너무 양극단은 싫다)
                AffectAxisConfig(name="valence", mode="homeostatic",
                                 setpoint=0.0, band=0.4, exponent=2.0, weight=0.5,
                                 adaptive_gain=0.5, soft_penalty=True),
                # 축 1 = arousal (중간 수준 유지)
                AffectAxisConfig(name="arousal", mode="homeostatic",
                                 setpoint=0.0, band=0.5, exponent=2.0, weight=0.5,
                                 adaptive_gain=0.5, soft_penalty=True),
                # 축 2 = tension (낮을수록 좋음, 긴급성 높음)
                AffectAxisConfig(name="tension", mode="minimize",
                                 setpoint=0.0, band=0.0, exponent=2.0, weight=1.0,
                                 adaptive_gain=2.0, soft_penalty=True),
                # 축 3 = drive/motivation (높을수록 좋음)
                AffectAxisConfig(name="drive", mode="maximize",
                                 setpoint=0.3, band=0.0, exponent=2.0, weight=0.7,
                                 adaptive_gain=0.2, soft_penalty=True),
            ][:affect_dim]

        self._affect_core_axes = affect_core_axes

        # 레벨 0: 생존 드라이브 등록
        self._register_viability_drive()

        # 레벨 1: 감정 코어 드라이브 등록
        self._register_affect_core_drives()

    # -------- 초기 드라이브 등록 --------

    def _register_viability_drive(self) -> None:
        """
        레벨 0: 생존 드라이브.
        삭제 불가(mutable=False).
        """
        def viability_fn(h_t: np.ndarray,
                         a_t: np.ndarray,
                         ctx: Dict[str, Any]) -> float:
            # 예: ctx["viability_cost"]에 미리 계산된 비용을 넣어둔다.
            return float(ctx.get("viability_cost", 0.0))

        d = Drive(
            drive_id="viability",
            tier=DriveTier.VIABILITY,
            name="viability",
            weight=1.0,
            fn=viability_fn,
            mutable=False,
        )
        self.drives[d.drive_id] = d

    def _register_affect_core_drives(self) -> None:
        """
        레벨 1: AffectKernel 축에 대응되는 코어 드라이브.
        - 삭제 불가 (mutable=False)
        - 축별 mode/setpoint/band/exponent/weight 로 기본 목적 정의
        - ctx 를 통해 런타임 override 가능
        """
        d = self.affect_kernel.dim
        
        for idx in range(d):
            # If we have fewer configs than dims, reuse the last one or a default
            if idx < len(self._affect_core_axes):
                cfg = self._affect_core_axes[idx]
            else:
                cfg = AffectAxisConfig(name=f"axis_{idx}", mode="homeostatic", weight=0.1)

            def make_core_fn(axis_idx: int, axis_cfg: AffectAxisConfig) -> Callable:
                def core_affect_fn(
                    h_t: np.ndarray,
                    a_t: np.ndarray,
                    ctx: Dict[str, Any],
                ) -> float:
                    x = float(a_t[axis_idx])

                    # 1. Override Parsing (Explicit Config > Dict keys > Default)
                    override_obj = ctx.get("affect_override")
                    axis_ovr = {}
                    if isinstance(override_obj, AffectOverrideConfig):
                        axis_ovr = override_obj.axis_overrides.get(axis_idx, {})
                    
                    # Helper to get param
                    def get_param(key: str, default: Any) -> Any:
                        # 1. Check AffectOverrideConfig
                        if key in axis_ovr: return axis_ovr[key]
                        # 2. Check flat ctx keys (legacy support)
                        ctx_key = f"affect_{key}_{axis_idx}"
                        if ctx_key in ctx: return ctx[ctx_key]
                        # 3. Default
                        return default

                    mode = str(get_param("mode", axis_cfg.mode))
                    setpoint = float(get_param("setpoint", axis_cfg.setpoint))
                    band = float(get_param("band", axis_cfg.band))
                    exponent = float(get_param("exponent", axis_cfg.exponent))
                    adaptive_gain = float(get_param("adaptive_gain", axis_cfg.adaptive_gain))
                    use_soft = bool(get_param("soft_penalty", axis_cfg.soft_penalty))

                    # 2. Calculate Raw Deviation
                    raw_diff = 0.0
                    if mode == "homeostatic":
                        # |x - s| - b
                        dist = abs(x - setpoint)
                        raw_diff = dist - band
                    elif mode == "maximize":
                        # s - x
                        raw_diff = setpoint - x
                    elif mode == "minimize":
                        # x - s
                        raw_diff = x - setpoint
                    
                    # 3. Apply Thresholding (Soft vs Hard)
                    deviation = 0.0
                    if use_soft:
                        # Softplus: log(1 + exp(beta * raw_diff)) / beta
                        # beta가 클수록 hard threshold에 가까움. 
                        # raw_diff가 음수(band 안쪽)면 0에 수렴하지만 완전히 0은 아님 (Gradient 유지)
                        beta = 10.0 
                        # numerical stability
                        if raw_diff > 10: # linear regime
                            deviation = raw_diff
                        else:
                            deviation = np.log1p(np.exp(beta * raw_diff)) / beta
                    else:
                        deviation = max(0.0, raw_diff)

                    if deviation <= 1e-6:
                        return 0.0

                    # 4. Calculate Base Cost
                    cost = float(deviation ** exponent)

                    # 5. Adaptive Weighting (Urgency)
                    # 편차가 클수록 가중치를 증폭 (Urgency Factor)
                    if adaptive_gain > 0:
                        urgency = 1.0 + (adaptive_gain * deviation)
                        cost *= urgency

                    return cost

                return core_affect_fn

            drive = Drive(
                drive_id=f"affect_core_{idx}",
                tier=DriveTier.AFFECT_CORE,
                name=f"affect_core_{idx}_{cfg.name}",
                weight=cfg.weight,
                fn=make_core_fn(idx, cfg),
                mutable=False,
            )
            self.drives[drive.drive_id] = drive

    # -------- 파생/도구적 드라이브 관리 --------

    def create_derived_drive(
        self,
        drive_id: str,
        name: str,
        fn: Callable[[np.ndarray, np.ndarray, Dict[str, Any]], float],
        tier: DriveTier = DriveTier.AFFECT_DERIVED,
        initial_weight: float = 1.0,
    ) -> None:
        """
        레벨 2: 파생(감정 기반) 또는 도구적 드라이브 생성.
        """
        assert tier in (DriveTier.AFFECT_DERIVED, DriveTier.INSTRUMENTAL)
        d = Drive(
            drive_id=drive_id,
            tier=tier,
            name=name,
            weight=initial_weight,
            fn=fn,
            mutable=True,
        )
        self.drives[drive_id] = d

    def deactivate_drive(self, drive_id: str) -> None:
        """
        파생/도구적 드라이브를 비활성화.
        - VIABILITY / AFFECT_CORE 는 삭제 불가: alive=False로도 안 바꾼다.
        """
        d = self.drives.get(drive_id)
        if d is None:
            return
        if not d.mutable:
            # 삭제 불가 드라이브는 weight만 0으로 줄일 수 있음
            d.weight = 0.0
            return
        d.alive = False

    # -------- 평가 --------

    def evaluate_all(
        self,
        h_t: np.ndarray,
        ctx: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        1. AffectKernel로 a_t 추론
        2. 모든 드라이브 평가
        """
        a_t = self.affect_kernel.infer(h_t, ctx.get("context"))
        self.last_affect = a_t  # Store for observability
        scores: Dict[str, float] = {}
        for drive_id, d in self.drives.items():
            scores[drive_id] = d.evaluate(h_t, a_t, ctx)
        
        # 시각화/디버깅을 위해 저장
        self.last_cost_breakdown = scores.copy()
        
        return scores

    def total_cost(
        self,
        h_t: np.ndarray,
        ctx: Dict[str, Any],
        lambda_V: float = 1.0,
    ) -> float:
        """
        전체 비용:
        J_t = lambda_V * V_t + sum_d R_d(t)
        여기서는 'viability' 드라이브를 V_t에 대응시킨 예시.
        """
        scores = self.evaluate_all(h_t, ctx)
        V_t = scores.get("viability", 0.0)
        total = lambda_V * V_t + sum(
            v for k, v in scores.items() if k != "viability"
        )
        return float(total)
