from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import os
import re
import logging

import numpy as np
import torch
import torch.nn as nn

_log = logging.getLogger("llm_adapter.m3_control_bridge")


@dataclass
class M3BridgeControls:
    prefix_embeddings: Optional[torch.Tensor]
    layer_gates: Optional[torch.Tensor]
    logit_bias: Optional[torch.Tensor]


class M3ControlBridge(nn.Module):
    """
    Bridge from M3 state vector z_m3 to decoder controls:
    - virtual prefix embeddings (acts as KV-prefix through the model stack)
    - per-layer scalar gates
    - low-rank logit bias
    """

    def __init__(
        self,
        state_dim: int,
        model_hidden_dim: int,
        vocab_size: int,
        num_layers: int,
        prefix_len: int = 8,
        logit_rank: int = 32,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.model_hidden_dim = int(model_hidden_dim)
        self.vocab_size = int(vocab_size)
        self.num_layers = int(max(1, num_layers))
        self.prefix_len = int(max(1, prefix_len))
        self.logit_rank = int(max(1, logit_rank))

        h = max(64, min(1024, self.model_hidden_dim))
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.prefix_head = nn.Linear(h, self.prefix_len * self.model_hidden_dim)
        self.gate_head = nn.Linear(h, self.num_layers)
        self.logit_coeff = nn.Linear(h, self.logit_rank, bias=False)
        self.logit_basis = nn.Parameter(torch.empty(self.logit_rank, self.vocab_size))
        nn.init.normal_(self.logit_basis, std=0.02)

        self.max_gate_delta = 0.45
        self.max_logit_bias = 2.0
        self._maybe_apply_spectral_norm()

    def _maybe_apply_spectral_norm(self) -> None:
        raw = str(os.getenv("M3_STABILITY_SPECTRAL_NORM", "0")).strip().lower()
        enabled = raw in ("1", "true", "yes", "on")
        if not enabled:
            return
        try:
            from torch.nn.utils import spectral_norm
        except Exception:
            return
        for name in ("prefix_head", "gate_head", "logit_coeff"):
            try:
                mod = getattr(self, name, None)
                if isinstance(mod, nn.Linear):
                    setattr(self, name, spectral_norm(mod))
            except Exception:
                continue

    def _prepare_state(self, z_m3: torch.Tensor | np.ndarray | Sequence[float]) -> torch.Tensor:
        if isinstance(z_m3, torch.Tensor):
            z = z_m3.float()
        else:
            z = torch.as_tensor(np.asarray(z_m3, dtype=np.float32))
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2:
            z = z.view(z.size(0), -1)
        if z.size(-1) > self.state_dim:
            z = z[:, : self.state_dim]
        elif z.size(-1) < self.state_dim:
            pad = torch.zeros(z.size(0), self.state_dim - z.size(-1), dtype=z.dtype, device=z.device)
            z = torch.cat([z, pad], dim=-1)
        return z

    def forward(
        self,
        z_m3: torch.Tensor | np.ndarray | Sequence[float],
        strength: float = 1.0,
    ) -> M3BridgeControls:
        z = self._prepare_state(z_m3)
        x = self.trunk(self.state_norm(z))
        s = float(max(0.0, strength))

        prefix = self.prefix_head(x).view(-1, self.prefix_len, self.model_hidden_dim)
        prefix = prefix * s

        gate_raw = torch.tanh(self.gate_head(x))
        layer_gates = 1.0 + (self.max_gate_delta * s) * gate_raw

        coeff = self.logit_coeff(x) * s
        logit_bias = coeff @ self.logit_basis
        logit_bias = torch.tanh(logit_bias / max(1.0, self.max_logit_bias)) * self.max_logit_bias

        return M3BridgeControls(
            prefix_embeddings=prefix,
            layer_gates=layer_gates,
            logit_bias=logit_bias,
        )

    def regularization_loss(
        self,
        controls: Optional[M3BridgeControls],
        gate_reg: float = 1e-3,
        bias_reg: float = 1e-4,
    ) -> torch.Tensor:
        ref = self.logit_basis
        loss = ref.new_tensor(0.0)
        if controls is not None and controls.layer_gates is not None:
            gate = controls.layer_gates
            loss = loss + float(gate_reg) * torch.mean((gate - 1.0) ** 2)
        if controls is not None and controls.logit_bias is not None:
            loss = loss + float(bias_reg) * torch.mean(controls.logit_bias ** 2)
        return loss

    def renorm_parameters(self, max_weight_norm: float = 10.0) -> None:
        lim = float(max(1e-6, max_weight_norm))
        with torch.no_grad():
            for p in self.parameters():
                if p is None or not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    continue
                n = torch.norm(p)
                if torch.isfinite(n) and n.item() > lim:
                    p.mul_(lim / (n + 1e-8))


def _to_module_list(obj: Any) -> Optional[List[nn.Module]]:
    if obj is None:
        return None
    if isinstance(obj, nn.ModuleList):
        return list(obj)
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(m, nn.Module) for m in obj):
        return list(obj)
    return None


def find_decoder_layers(model: nn.Module) -> List[nn.Module]:
    """Best-effort layer discovery across common decoder architectures."""
    candidates: List[Any] = []
    try:
        candidates.append(getattr(getattr(model, "model", None), "layers", None))
    except Exception:
        pass
    try:
        candidates.append(getattr(getattr(model, "transformer", None), "h", None))
    except Exception:
        pass
    try:
        candidates.append(getattr(getattr(model, "gpt_neox", None), "layers", None))
    except Exception:
        pass
    try:
        candidates.append(getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None))
    except Exception:
        pass
    try:
        candidates.append(getattr(model, "layers", None))
    except Exception:
        pass

    for c in candidates:
        out = _to_module_list(c)
        if out:
            return out

    for module in model.modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return list(module)
    return []


class LayerGateRuntime:
    """Attaches per-layer scalar gates using forward hooks."""

    def __init__(self, layers: Sequence[nn.Module]) -> None:
        self.layers = list(layers)
        self._hooks: List[Any] = []

    def close(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def apply(self, gates: torch.Tensor | np.ndarray | Sequence[float]) -> None:
        self.close()
        gate_refs: List[torch.Tensor] = []
        if isinstance(gates, torch.Tensor):
            gflat = gates.view(-1)
            if gflat.numel() == 0:
                return
            gate_refs = [gflat[min(i, gflat.numel() - 1)] for i in range(len(self.layers))]
        else:
            g = np.asarray(gates, dtype=np.float32).ravel()
            if g.size == 0:
                return
            gate_refs = [torch.tensor(float(g[min(i, g.size - 1)]), dtype=torch.float32) for i in range(len(self.layers))]
        if not gate_refs:
            return
        for i, layer in enumerate(self.layers):
            gate_ref = gate_refs[i]

            def _hook(_module, _inputs, output, gate_tensor=gate_ref):
                if isinstance(output, tuple):
                    if not output:
                        return output
                    first = output[0]
                    if torch.is_tensor(first):
                        g = gate_tensor.to(device=first.device, dtype=first.dtype)
                        return (first * g, *output[1:])
                    return output
                if torch.is_tensor(output):
                    g = gate_tensor.to(device=output.device, dtype=output.dtype)
                    return output * g
                return output

            self._hooks.append(layer.register_forward_hook(_hook))


@dataclass
class QualityGateResult:
    score: float
    reject: bool
    reasons: List[str]
    features: Dict[str, float]


class GenerationQualityGate:
    """
    Lightweight critic-like quality gate.
    Keeps runtime cost small while catching numeric dumps and low-language outputs.
    """

    def __init__(self, min_score: float = 0.45) -> None:
        self.min_score = float(min_score)

    @staticmethod
    def _features(text: str) -> Dict[str, float]:
        s = str(text or "")
        n = max(1, len(s))
        letters = sum(ch.isalpha() for ch in s)
        digits = sum(ch.isdigit() for ch in s)
        spaces = sum(ch.isspace() for ch in s)
        punct = sum((not ch.isalnum()) and (not ch.isspace()) for ch in s)

        tokens = [t for t in re.split(r"\s+", s.strip()) if t]
        uniq_ratio = (len(set(tokens)) / max(1, len(tokens))) if tokens else 0.0
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        line_uniq_ratio = (len(set(lines)) / max(1, len(lines))) if lines else 1.0

        return {
            "len": float(len(s)),
            "alpha_ratio": float(letters / n),
            "digit_ratio": float(digits / n),
            "space_ratio": float(spaces / n),
            "punct_ratio": float(punct / n),
            "uniq_token_ratio": float(uniq_ratio),
            "uniq_line_ratio": float(line_uniq_ratio),
        }

    def evaluate(self, text: str) -> QualityGateResult:
        f = self._features(text)
        reasons: List[str] = []
        score = 1.0

        if f["len"] < 8:
            score -= 0.45
            reasons.append("too_short")
        if f["digit_ratio"] > 0.55:
            score -= 0.55
            reasons.append("digit_heavy")
        if f["alpha_ratio"] < 0.20:
            score -= 0.45
            reasons.append("low_language_content")
        if f["uniq_token_ratio"] < 0.25 and f["len"] > 24:
            score -= 0.25
            reasons.append("high_repetition")
        if f["uniq_line_ratio"] < 0.50:
            score -= 0.20
            reasons.append("repeated_lines")

        score = float(max(0.0, min(1.0, score)))
        reject = score < self.min_score
        return QualityGateResult(score=score, reject=reject, reasons=reasons, features=f)


# ============================================================================
# NeuroModulator — weight-level M3 consciousness control
# ============================================================================


@dataclass
class NeuroModControls:
    """Per-layer neural modulation controls produced by :class:`NeuroModulator`."""

    layer_gain: torch.Tensor  # (1, num_layers)
    hidden_bias: torch.Tensor  # (1, num_layers, hidden_dim)
    logit_bias: Optional[torch.Tensor]  # (1, vocab_size)
    phi_gate: float  # overall consciousness gating factor [0, 1]


class NeuroModulator(nn.Module):
    """Maps M3 consciousness state -> per-layer weight modulations.

    Instead of injecting consciousness signals as text prompts, this module
    directly modulates the hidden representations flowing through each
    decoder layer.  The result is that M3's internal state (phi, arousal,
    valence, energy) genuinely *shapes* how the model speaks rather than
    appearing as quoted metadata.

    Per decoder layer
    -----------------
    layer_gain   : multiplicative scale ≈ 1.0 ± ``max_gain_delta``
    hidden_bias  : low-rank additive steering (rank = ``hidden_rank``)

    Global
    ------
    logit_bias   : low-rank output distribution shaping
    phi_gate     : consciousness-proportional gating of all modulations

    Design decisions
    ----------------
    * Initialised to identity (gain=1, bias=0) so the LLM is unaffected
      until the modulator begins learning.
    * Exponential warmup ramp prevents large perturbations early.
    * All deviations are bounded by configurable maxima.
    """

    def __init__(
        self,
        state_dim: int,
        num_layers: int,
        model_hidden_dim: int,
        vocab_size: int,
        trunk_dim: int = 256,
        hidden_rank: int = 16,
        logit_rank: int = 32,
    ) -> None:
        super().__init__()
        self.state_dim = int(max(1, state_dim))
        self.num_layers = int(max(1, num_layers))
        self.model_hidden_dim = int(max(1, model_hidden_dim))
        self.vocab_size = int(max(1, vocab_size))
        self.hidden_rank = int(max(1, hidden_rank))
        self.logit_rank = int(max(1, logit_rank))

        td = int(max(64, min(512, trunk_dim)))

        # ---- state encoding ----
        self.state_norm = nn.LayerNorm(self.state_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.state_dim, td),
            nn.SiLU(),
            nn.Linear(td, td),
            nn.SiLU(),
        )

        # ---- per-layer gain (near 1.0) ----
        self.gain_head = nn.Linear(td, self.num_layers)

        # ---- per-layer hidden bias (low-rank) ----
        self.bias_down = nn.Linear(td, self.num_layers * self.hidden_rank, bias=False)
        self.bias_up = nn.Linear(self.hidden_rank, self.model_hidden_dim, bias=False)

        # ---- logit bias (low-rank) ----
        self.logit_down = nn.Linear(td, self.logit_rank, bias=False)
        self.logit_up = nn.Parameter(torch.empty(self.logit_rank, self.vocab_size))

        # ---- phi gate (consciousness-proportional) ----
        self.phi_gate_head = nn.Linear(td, 1)

        # ---- constraints ----
        self.max_gain_delta: float = 0.3
        self.max_logit_bias: float = 2.0
        self.warmup_total: int = 100
        self._step: int = 0

        # ---- near-identity initialisation ----
        # Small non-zero init so gradient signal exists from the start;
        # warmup factor ensures the effective modulation is still zero
        # during early inference steps.
        nn.init.normal_(self.gain_head.weight, std=0.02)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.normal_(self.bias_down.weight, std=0.02)
        nn.init.normal_(self.bias_up.weight, std=0.01)
        nn.init.normal_(self.logit_down.weight, std=0.02)
        nn.init.normal_(self.logit_up, std=0.01)
        # sigmoid(0) = 0.5 → moderate modulation until learning adjusts
        nn.init.zeros_(self.phi_gate_head.weight)
        nn.init.zeros_(self.phi_gate_head.bias)

    # -- helpers ----------------------------------------------------------

    def _prepare_state(
        self, z_m3: "torch.Tensor | np.ndarray | Sequence[float]"
    ) -> torch.Tensor:
        if isinstance(z_m3, torch.Tensor):
            z = z_m3.float()
        else:
            z = torch.as_tensor(np.asarray(z_m3, dtype=np.float32))
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2:
            z = z.view(z.size(0), -1)
        d = z.size(-1)
        if d > self.state_dim:
            z = z[:, : self.state_dim]
        elif d < self.state_dim:
            pad = torch.zeros(
                z.size(0), self.state_dim - d, dtype=z.dtype, device=z.device
            )
            z = torch.cat([z, pad], dim=-1)
        return z

    def _warmup_factor(self) -> float:
        if self.warmup_total <= 0:
            return 1.0
        # exponential-ish ramp: reaches ~0.95 at warmup_total
        ratio = float(self._step) / float(self.warmup_total)
        return min(1.0, 1.0 - float(np.exp(-3.0 * ratio)))

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        z_m3: "torch.Tensor | np.ndarray | Sequence[float]",
        strength: float = 1.0,
    ) -> NeuroModControls:
        z = self._prepare_state(z_m3)
        h = self.trunk(self.state_norm(z))  # (B, td)

        # Consciousness gate: modulation proportional to phi level
        phi_raw = torch.sigmoid(self.phi_gate_head(h))  # (B, 1)
        phi_g = float(phi_raw.mean().item())

        # Effective strength = external * phi * warmup
        s = float(max(0.0, strength)) * phi_g * self._warmup_factor()

        # Per-layer gain: 1.0 + delta
        gain_delta = torch.tanh(self.gain_head(h))  # (B, L)
        layer_gain = 1.0 + self.max_gain_delta * s * gain_delta

        # Per-layer hidden bias (low-rank)
        bias_low = self.bias_down(h)  # (B, L*R)
        bias_low = bias_low.view(-1, self.num_layers, self.hidden_rank)
        hidden_bias = self.bias_up(bias_low) * s  # (B, L, H)

        # Logit bias (low-rank)
        logit_coeff = self.logit_down(h) * s  # (B, logit_rank)
        logit_bias = logit_coeff @ self.logit_up  # (B, V)
        logit_bias = (
            torch.tanh(logit_bias / max(1.0, self.max_logit_bias))
            * self.max_logit_bias
        )

        self._step += 1

        return NeuroModControls(
            layer_gain=layer_gain,
            hidden_bias=hidden_bias,
            logit_bias=logit_bias,
            phi_gate=phi_g,
        )

    # -- online learning --------------------------------------------------

    def online_loss(
        self,
        z_m3: "torch.Tensor | np.ndarray | Sequence[float]",
        reward: float,
        strength: float = 1.0,
        reg_weight: float = 5e-3,
    ) -> torch.Tensor:
        """Reward-conditioned magnitude targeting for online adaptation.

        The loss drives the *magnitude* of the raw head outputs toward a
        target that is proportional to the reward signal:

        * Positive reward  -> target_mag is high (encourage deviation from
          identity, i.e. allow consciousness to modulate more).
        * Negative reward  -> target_mag → 0 (pull toward identity).
        * Zero reward      -> small exploration magnitude.

        Additionally, the phi-gate is nudged toward predicting reward
        quality, so that over time the consciousness gating itself
        reflects how helpful modulation has been.

        Unlike a pure regularisation loss, this formulation produces a
        non-zero gradient even at (near-)identity initialisation because
        the target magnitude is always > 0 and the raw head outputs are
        initialised with small random values.
        """
        z = self._prepare_state(z_m3)
        h = self.trunk(self.state_norm(z))
        phi_raw = torch.sigmoid(self.phi_gate_head(h))

        # Raw head outputs (before warmup / phi gating)
        gain_raw = self.gain_head(h)
        bias_raw = self.bias_down(h)
        logit_raw = self.logit_down(h)

        # Current output magnitude
        mag = (
            gain_raw.pow(2).mean()
            + bias_raw.pow(2).mean()
            + logit_raw.pow(2).mean()
        )

        clamped_r = float(max(-1.0, min(1.0, reward)))

        # Target magnitude: higher when reward is good
        # reward = -1 -> 0.0;  reward = 0 -> 0.1;  reward = 1 -> 0.6
        target_mag = max(0.0, 0.1 + 0.5 * clamped_r)

        # MSE toward target magnitude
        loss = reg_weight * (mag - target_mag) ** 2

        # Phi-gate reward prediction: teach the gate to reflect quality
        phi_target = max(0.1, min(0.9, 0.5 + 0.3 * clamped_r))
        loss = loss + 0.01 * (phi_raw.mean() - phi_target) ** 2

        return loss

    # -- regularisation helpers -------------------------------------------

    def renorm_parameters(self, max_weight_norm: float = 10.0) -> None:
        lim = float(max(1e-6, max_weight_norm))
        with torch.no_grad():
            for p in self.parameters():
                if p is None or not p.requires_grad:
                    continue
                if p.ndim <= 1:
                    continue
                n = torch.norm(p)
                if torch.isfinite(n) and n.item() > lim:
                    p.mul_(lim / (n + 1e-8))

    def maybe_apply_spectral_norm(self) -> None:
        raw = str(os.getenv("M3_STABILITY_SPECTRAL_NORM", "0")).strip().lower()
        if raw not in ("1", "true", "yes", "on"):
            return
        try:
            from torch.nn.utils import spectral_norm
        except Exception:
            return
        for name in ("gain_head", "bias_down", "logit_down"):
            try:
                mod = getattr(self, name, None)
                if isinstance(mod, nn.Linear):
                    setattr(self, name, spectral_norm(mod))
            except Exception:
                continue


class NeuroModulatorRuntime:
    """Attaches :class:`NeuroModulator` controls to decoder layers via
    forward hooks.

    Each decoder layer's output hidden states are modulated::

        output = output * gain_i + bias_i

    The hooks are removed by calling :meth:`close`.
    """

    def __init__(self, layers: Sequence[nn.Module]) -> None:
        self.layers = list(layers)
        self._hooks: List[Any] = []

    def apply(self, controls: NeuroModControls) -> None:
        """Register forward hooks that apply *controls* to each layer."""
        self.close()
        if controls is None:
            return

        gains = controls.layer_gain.detach()  # (B, L)
        biases = controls.hidden_bias.detach()  # (B, L, H)
        if gains.ndim == 1:
            gains = gains.unsqueeze(0)
        if biases.ndim == 2:
            biases = biases.unsqueeze(0)

        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            gi = min(i, gains.size(-1) - 1)
            bi = min(i, biases.size(1) - 1)
            g_ref = gains[0, gi]  # scalar
            b_ref = biases[0, bi]  # (H,)

            def _hook(
                _mod,
                _inp,
                output,
                g=g_ref,
                b=b_ref,
            ):
                if isinstance(output, tuple):
                    if not output:
                        return output
                    first = output[0]
                    if torch.is_tensor(first):
                        gv = g.to(device=first.device, dtype=first.dtype)
                        bv = b.to(device=first.device, dtype=first.dtype)
                        modulated = first * gv + bv
                        return (modulated,) + output[1:]
                    return output
                if torch.is_tensor(output):
                    gv = g.to(device=output.device, dtype=output.dtype)
                    bv = b.to(device=output.device, dtype=output.dtype)
                    return output * gv + bv
                return output

            self._hooks.append(layer.register_forward_hook(_hook))

    def close(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
