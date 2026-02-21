from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import re

import numpy as np
import torch
import torch.nn as nn


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
