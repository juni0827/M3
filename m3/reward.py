"""
M3 Reward System Components
Merged from: drives.py, affect_kernel.py, reward_system.py

Contains:
- Drive/DriveTier definitions
- AffectKernel (Emotion Engine)
- RewardSystem (Drive Integration)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Sequence, Union, Callable, List, Literal

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

# ============================================================================
# PART 1: Drives
# ============================================================================

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


# ============================================================================
# PART 2: Affect Kernel
# ============================================================================

ArrayLike = Union[np.ndarray, Sequence[float]]


class _TorchAffectEncoder(nn.Module):
    """
    h_t (and optional context) -> a_t 로 가는 작은 MLP.

    - 입력: 정규화된 hidden_state (및 선택적 context 벡터)
    - 출력: affect 벡터 (dim,)
    """

    def __init__(self, input_dim: int, affect_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim or max(16, min(256, input_dim // 2))

        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.Tanh(),
            nn.Linear(h, affect_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        # x: (B, input_dim)
        return self.net(x)


class AffectKernel:
    """
    h_t, c_t -> a_t 를 추론하는 감정 핵심 모듈.

    핵심 설계 포인트:
    - 감정은 고정된 레이블(arousal/valence/...)이 아니라,
      h_t 에서 추론되는 latent 벡터 a_t ∈ R^dim 로 취급한다.
    - 이 모듈은 항상 존재하지만, 파라미터/의미는 학습으로 갱신된다.
    - torch 가 있으면 작은 MLP, 없으면 numpy 선형+비선형으로 동작한다.
    - h_t 는 내부적으로 러닝 mean/var 기반으로 정규화해서 넣는다.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        context_dim: int = 0,
        use_torch: Optional[bool] = None,
        eps: float = 1e-5,
    ) -> None:
        """
        Args:
            dim: 감정 공간 차원 (예: 5)
            hidden_dim: h_t 차원
            context_dim: c_t 를 실수 벡터로 표현했을 때의 차원 (없으면 0)
            use_torch:
                - True: torch 강제 사용 (없으면 에러)
                - False: numpy 경로 강제 사용
                - None: torch 있으면 torch, 없으면 numpy
            eps: 정규화에 쓰일 epsilon
        """
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.eps = float(eps)

        # 러닝 통계 (h_t 정규화용)
        self._running_mean = np.zeros(self.hidden_dim, dtype=np.float32)
        self._running_var = np.ones(self.hidden_dim, dtype=np.float32)
        self._running_count = self.eps  # 0 나누기 방지

        # torch 사용 여부 결정
        if use_torch is None:
            self._use_torch = torch is not None
        else:
            self._use_torch = bool(use_torch)
            if self._use_torch and torch is None:
                raise RuntimeError("use_torch=True 이지만 torch 가 설치되어 있지 않습니다.")

        input_dim = self.hidden_dim + self.context_dim

        if self._use_torch:
            # torch 기반 MLP encoder
            self._encoder = _TorchAffectEncoder(input_dim=input_dim, affect_dim=dim)
        else:
            # numpy 기반 선형 + tanh encoder
            # (실제론 torch 버전 학습 후 weight를 옮겨오는 용도로 쓰는 게 이상적)
            scale = 1.0 / math.sqrt(input_dim)
            self._W = np.random.randn(dim, input_dim).astype(np.float32) * scale
            self._b = np.zeros(dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def infer(
        self,
        h_t: np.ndarray,
        c_t: Any = None,
    ) -> np.ndarray:
        """
        단일 시점 h_t 에 대한 affect 벡터 a_t 를 반환.

        Args:
            h_t: shape (hidden_dim,)
            c_t: 선택적 컨텍스트. 숫자 벡터 또는 dict 를 지원:
                 - np.ndarray / list / tuple: 그대로 context vector 로 사용
                 - dict: float 로 cast 가능한 값들만 key 정렬 순서대로 추출

        Returns:
            a_t: shape (dim,), np.float32
        """
        h = self._to_1d_numpy(h_t, expected_dim=self.hidden_dim)
        ctx_vec = self._context_to_vec(c_t)

        h_norm = self._normalize(h)
        x = self._concat(h_norm, ctx_vec)  # (input_dim,)

        if self._use_torch:
            return self._infer_torch(x)
        else:
            return self._infer_numpy(x)

    def infer_batch(
        self,
        H: np.ndarray,
        C: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        """
        배치 버전 inference.

        Args:
            H: shape (B, hidden_dim)
            C: 길이 B 의 컨텍스트 시퀀스 (각 원소는 infer 의 c_t 포맷)

        Returns:
            A: shape (B, dim)
        """
        H = np.asarray(H, dtype=np.float32)
        assert H.ndim == 2 and H.shape[1] == self.hidden_dim, (
            f"expected H shape (B, {self.hidden_dim}), got {H.shape}"
        )
        B = H.shape[0]

        if C is None:
            ctx_mat = None
        else:
            assert len(C) == B
            ctx_vecs = [self._context_to_vec(c) for c in C]
            ctx_mat = np.stack(ctx_vecs, axis=0)  # (B, context_dim)

        # 정규화
        H_norm = np.stack([self._normalize(h) for h in H], axis=0)

        if ctx_mat is not None and self.context_dim > 0:
            X = np.concatenate([H_norm, ctx_mat], axis=1)  # (B, input_dim)
        else:
            X = H_norm

        if self._use_torch:
            return self._infer_batch_torch(X)
        else:
            return self._infer_batch_numpy(X)

    def update_running_stats(self, H_batch: np.ndarray) -> None:
        """
        h_t 배치를 넣어서 running mean/var 를 업데이트.
        - training / logging 루프에서 주기적으로 호출하면 됨.

        Args:
            H_batch: shape (B, hidden_dim)
        """
        H = np.asarray(H_batch, dtype=np.float32)
        assert H.ndim == 2 and H.shape[1] == self.hidden_dim

        batch_mean = H.mean(axis=0)
        batch_var = H.var(axis=0)
        batch_count = H.shape[0]

        # Welford-style 병합
        total_count = self._running_count + batch_count
        delta = batch_mean - self._running_mean

        new_mean = self._running_mean + delta * (batch_count / total_count)
        m_a = self._running_var * self._running_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._running_count * batch_count / total_count
        new_var = m2 / total_count

        self._running_mean = new_mean.astype(np.float32)
        self._running_var = np.maximum(new_var.astype(np.float32), self.eps)
        self._running_count = float(total_count)

    # ------------------------------------------------------------------
    # 내부 helper
    # ------------------------------------------------------------------

    def _to_1d_numpy(self, x: ArrayLike, expected_dim: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        assert arr.ndim == 1 and arr.shape[0] == expected_dim, (
            f"expected vector of shape ({expected_dim},), got {arr.shape}"
        )
        return arr

    def _context_to_vec(self, c_t: Any) -> Optional[np.ndarray]:
        if self.context_dim == 0:
            return None
        if c_t is None:
            return np.zeros(self.context_dim, dtype=np.float32)

        # numeric vector
        if isinstance(c_t, (list, tuple, np.ndarray)):
            v = np.asarray(c_t, dtype=np.float32)
            assert v.shape[0] == self.context_dim, (
                f"expected context_dim={self.context_dim}, got {v.shape[0]}"
            )
            return v

        # dict -> key 정렬 후 value 추출
        if isinstance(c_t, dict):
            # float 로 cast 가능한 value만 사용
            keys = sorted(c_t.keys())
            vals = []
            for k in keys:
                v = c_t[k]
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            v = np.asarray(vals, dtype=np.float32)
            if v.shape[0] < self.context_dim:
                # 부족하면 뒤를 0으로 채움
                pad = np.zeros(self.context_dim - v.shape[0], dtype=np.float32)
                v = np.concatenate([v, pad], axis=0)
            elif v.shape[0] > self.context_dim:
                v = v[: self.context_dim]
            return v

        # 그 외 타입은 무시하고 0 벡터
        return np.zeros(self.context_dim, dtype=np.float32)

    def _normalize(self, h: np.ndarray) -> np.ndarray:
        mean = self._running_mean
        var = self._running_var
        return (h - mean) / np.sqrt(var + self.eps)

    def _concat(self, h_norm: np.ndarray, ctx_vec: Optional[np.ndarray]) -> np.ndarray:
        if self.context_dim > 0 and ctx_vec is not None:
            return np.concatenate([h_norm, ctx_vec.astype(np.float32)], axis=0)
        return h_norm

    # ------------------------------------------------------------------
    # backend 별 forward
    # ------------------------------------------------------------------

    def _infer_numpy(self, x: np.ndarray) -> np.ndarray:
        # x: (input_dim,)
        assert x.ndim == 1
        z = self._W @ x + self._b  # (dim,)
        # 비선형성 추가 (tanh)
        a = np.tanh(z)
        return a.astype(np.float32)

    def _infer_batch_numpy(self, X: np.ndarray) -> np.ndarray:
        # X: (B, input_dim)
        z = X @ self._W.T + self._b[None, :]
        a = np.tanh(z)
        return a.astype(np.float32)

    def _infer_torch(self, x: np.ndarray) -> np.ndarray:
        # x: (input_dim,)
        assert torch is not None
        self._encoder.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # (1, input_dim)
            out = self._encoder(t)  # (1, dim)
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    def _infer_batch_torch(self, X: np.ndarray) -> np.ndarray:
        assert torch is not None
        self._encoder.eval()
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32))  # (B, input_dim)
            out = self._encoder(t)                      # (B, dim)
        return out.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # 상태 저장/로드 (학습 후 weight 옮길 때 쓰라고 열어둠)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """
        AffectKernel 자체의 파라미터 + 러닝 통계를 dict 로 export.
        (torch 버전/ numpy 버전 모두 지원)
        """
        state: Dict[str, Any] = {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "running_mean": self._running_mean.copy(),
            "running_var": self._running_var.copy(),
            "running_count": self._running_count,
            "use_torch": self._use_torch,
        }
        if self._use_torch:
            assert torch is not None
            state["encoder"] = self._encoder.state_dict()
        else:
            state["W"] = self._W.copy()
            state["b"] = self._b.copy()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        state_dict 에서 파라미터/통계를 복원.
        dim/hidden_dim/context_dim 이 다르면 에러.
        """
        assert state["dim"] == self.dim
        assert state["hidden_dim"] == self.hidden_dim
        assert state["context_dim"] == self.context_dim

        self._running_mean = np.asarray(state["running_mean"], dtype=np.float32)
        self._running_var = np.asarray(state["running_var"], dtype=np.float32)
        self._running_count = float(state["running_count"])

        use_torch_state = bool(state.get("use_torch", self._use_torch))

        if self._use_torch and use_torch_state:
            assert torch is not None
            self._encoder.load_state_dict(state["encoder"])
        elif (not self._use_torch) and (not use_torch_state):
            self._W = np.asarray(state["W"], dtype=np.float32)
            self._b = np.asarray(state["b"], dtype=np.float32)
        else:
            # torch <-> numpy 혼용해서 로딩하고 싶으면
            # 별도 변환 루틴을 구현해야 한다.
            raise RuntimeError("현재 use_torch 설정과 state_dict 의 use_torch 플래그가 다릅니다.")


# ============================================================================
# PART 3: Reward System
# ============================================================================

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
