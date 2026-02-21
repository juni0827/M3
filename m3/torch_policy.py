from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore
from m3.device import resolve_torch_device_string


def _resolve_device(device: Optional[str]) -> str:
    return resolve_torch_device_string(
        explicit=device,
        torch_module=torch,
        require_cuda=False,
    )


# Shared representation (MoE-MLP)
class _Expert(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        return self.fc2(F.gelu(self.fc1(x)))


class _MoEFFN(nn.Module):
    def __init__(self, d: int, d_ff: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts = int(n_experts)
        self.top_k = int(max(1, top_k))
        self.gate = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([_Expert(d, d_ff) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, k=min(self.top_k, self.n_experts), dim=-1)
        y = torch.zeros_like(x)
        for k in range(topv.size(-1)):
            idx = topi[:, k]
            w = topv[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                m = (idx == e)
                if m.any():
                    xe = x[m]
                    ye = self.experts[e](xe)
                    y[m] = y[m] + w[m] * ye
        return y


class TorchSharedRepr(nn.Module):
    def __init__(self, dim: int, d_hidden: int = 512, n_layers: int = 2,
                 d_ff: int = 2048, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.dim = int(dim)
        self.proj_in = nn.Linear(self.dim, d_hidden)
        blocks = []
        for _ in range(max(1, int(n_layers))):
            if n_experts <= 0:
                blocks.append(nn.Sequential(nn.LayerNorm(d_hidden), nn.Linear(d_hidden, d_ff), nn.GELU(), nn.Linear(d_ff, d_hidden)))
            else:
                blocks.append(_MoEFFN(d_hidden, d_ff, n_experts=n_experts, top_k=top_k))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_hidden)
        self.proj_out = nn.Linear(d_hidden, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # pyright: ignore[reportInvalidTypeForm]
        h = self.proj_in(x)
        for b in self.blocks:
            if isinstance(b, _MoEFFN):
                h = h + b(h)
            else:
                h = h + b(h)
        h = self.norm(h)
        return self.proj_out(h)


# TorchPolicy (MoE-MLP PPO)
class Expert(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # pyright: ignore[reportInvalidTypeForm]
        return self.fc2(F.gelu(self.fc1(x)))


class MoEFFN(nn.Module):
    def __init__(self, d: int, d_ff: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d, d_ff) for _ in range(n_experts)])
        self.gate = nn.Linear(d, n_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # pyright: ignore[reportInvalidTypeForm]
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=min(self.top_k, self.n_experts), dim=-1)
        lb = (probs.mean(dim=0) ** 2).sum() * float(self.n_experts)
        y = torch.zeros_like(x)
        for k in range(topk_vals.size(-1)):
            idx = topk_idx[:, k]
            w = topk_vals[:, k].unsqueeze(-1)
            for e in range(self.n_experts):
                mask = (idx == e)
                if mask.any():
                    xe = x[mask]
                    ye = self.experts[e](xe)
                    y[mask] = y[mask] + w[mask] * ye
        return y, lb


class PolicyModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, d_model: int = 512, n_layers: int = 4,
                 n_experts: int = 8, d_ff: int = 2048, top_k: int = 2):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList([MoEFFN(d_model, d_ff, n_experts=n_experts, top_k=top_k)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head_mu = nn.Linear(d_model, out_dim)
        self.head_v = nn.Linear(d_model, 1)
        self.log_sigma = nn.Parameter(torch.full((out_dim,), math.log(0.6)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # pyright: ignore[reportInvalidTypeForm]
        h = self.proj_in(x)
        lb_loss = h.new_tensor(0.0)
        for ff in self.layers:
            h_in = h
            h_ff, lb = ff(h)
            h = h_in + h_ff
            lb_loss = lb_loss + lb
        h = self.norm(h)
        mu = self.head_mu(h)
        v = self.head_v(h).squeeze(-1)
        sigma = self.log_sigma.exp().clamp_min(1e-3)
        return mu, v, sigma, lb_loss


@dataclass
class RolloutItem:
    obs: np.ndarray
    act: np.ndarray
    logp: float
    mu: np.ndarray
    rew: float
    # [NEW] Affective state and intrinsic reward
    affect: Optional[np.ndarray] = None
    intrinsic_rew: float = 0.0


class TorchPolicy:
    def __init__(self, in_dim: int, out_dim: int,
                 d_model: int = 2048, n_layers: int = 8,
                 d_ff: int = 8192, n_experts: int = 16, top_k: int = 2,
                 device: Optional[str] = None, fsdp: bool = True,
                 shared_repr: Optional[nn.Module] = None, # type: ignore
                 affect_dim: int = 0) -> None: # pyright: ignore[reportInvalidTypeForm]
        if not TORCH_OK:
            raise RuntimeError('PyTorch is not available; cannot use TorchPolicy')
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.affect_dim = int(affect_dim)
        self.device = torch.device(_resolve_device(device))
        
        # Input dimension includes affect vector if present
        total_in_dim = self.in_dim + self.affect_dim
        
        self.model = PolicyModel(total_in_dim, out_dim, d_model=d_model, n_layers=n_layers,
                                 n_experts=n_experts, d_ff=d_ff, top_k=top_k).to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        self.storage: List[RolloutItem] = []
        self.shared_repr = shared_repr.to(self.device) if shared_repr is not None else None
        self._last_kl: float = 0.0
        self._last_clipped: bool = False

    @property
    def sigma(self) -> float:
        try:
            with torch.no_grad():
                return float(self.model.log_sigma.exp().mean().detach().cpu().item())
        except Exception:
            return 0.6

    def sample(self, obs: np.ndarray, affect: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, np.ndarray, float]:
        self.model.eval()
        
        # Prepare observation tensor
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        if self.shared_repr is not None:
            with torch.no_grad():
                obs_t = self.shared_repr(obs_t)
                
        # Prepare affect tensor
        if self.affect_dim > 0:
            if affect is not None:
                aff_t = torch.as_tensor(affect, dtype=torch.float32, device=self.device).view(1, -1)
            else:
                aff_t = torch.zeros((1, self.affect_dim), dtype=torch.float32, device=self.device)
            # Concatenate observation and affect
            x = torch.cat([obs_t, aff_t], dim=-1)
        else:
            x = obs_t

        with torch.no_grad(), autocast(self.device.type == 'cuda'):
            mu, v, sigma, _ = self.model(x)
            eps = torch.randn_like(mu)
            act = mu + sigma * eps
            logp = -0.5 * torch.sum(((act - mu) / (sigma + 1e-8)) ** 2, dim=-1)
        return act.squeeze(0).cpu().numpy(), float(logp.item()), mu.squeeze(0).cpu().numpy(), float(v.item())

    def record(self, obs: np.ndarray, act: np.ndarray, logp: float, mu: np.ndarray, rew: float, 
               affect: Optional[np.ndarray] = None, intrinsic_rew: float = 0.0) -> None:
        self.storage.append(RolloutItem(obs=np.asarray(obs, np.float32),
                                        act=np.asarray(act, np.float32),
                                        logp=float(logp),
                                        mu=np.asarray(mu, np.float32),
                                        rew=float(rew),
                                        affect=np.asarray(affect, np.float32) if affect is not None else None,
                                        intrinsic_rew=float(intrinsic_rew)))

    def end_batch(self, gamma: float = 0.99, kl_coeff: float = 0.0, lr: float = 3e-4,
                  ppo_epochs: int = 2, clip_range: float = 0.2, target_kl: float = 0.02) -> None:
        if not self.storage:
            return
        self.model.train()
        T = len(self.storage)
        
        # Prepare batch tensors
        obs_list = [torch.as_tensor(it.obs, dtype=torch.float32, device=self.device) for it in self.storage]
        obs = torch.stack(obs_list, dim=0)
        
        if self.shared_repr is not None:
            with torch.no_grad():
                obs = self.shared_repr(obs)
                
        if self.affect_dim > 0:
            aff_list = []
            for it in self.storage:
                if it.affect is not None:
                    aff_list.append(torch.as_tensor(it.affect, dtype=torch.float32, device=self.device))
                else:
                    aff_list.append(torch.zeros(self.affect_dim, dtype=torch.float32, device=self.device))
            aff = torch.stack(aff_list, dim=0)
            x_batch = torch.cat([obs, aff], dim=-1)
        else:
            x_batch = obs

        acts = torch.as_tensor(np.stack([it.act for it in self.storage], axis=0), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(np.array([it.logp for it in self.storage], dtype=np.float32), device=self.device)
        
        # Combine extrinsic and intrinsic rewards
        rews = [it.rew + it.intrinsic_rew for it in self.storage]
        
        G = np.zeros(T, dtype=np.float32)
        run = 0.0
        for t in range(T - 1, -1, -1):
            run = rews[t] + gamma * run
            G[t] = run
        returns = torch.as_tensor(G, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            _, v, _, _ = self.model(x_batch)
        adv = (returns - v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        self.optim.param_groups[0]['lr'] = float(lr)
        last_approx_kl = 0.0
        last_clip_frac = 0.0
        for _ in range(max(1, int(ppo_epochs))):
            with autocast(self.device.type == 'cuda'):
                mu, v, sigma, lb = self.model(x_batch)
                logp = -0.5 * torch.sum(((acts - mu) / (sigma + 1e-8)) ** 2, dim=-1)
                ratio = torch.exp(logp - old_logp)
                pg1 = ratio * adv
                pg2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
                policy_loss = -torch.mean(torch.min(pg1, pg2))
                value_loss = 0.5 * torch.mean((returns - v) ** 2)
                ent = torch.mean(0.5 * torch.log(2 * torch.pi * (sigma ** 2)) + 0.5)
                loss = policy_loss + value_loss - 0.01 * ent + 1e-2 * (lb if isinstance(lb, torch.Tensor) else 0.0)
            self.scaler.scale(loss).backward()
            if self.device.type == 'cuda':
                self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)
            approx_kl = torch.mean(old_logp - logp).item()
            clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).item()
            last_approx_kl = approx_kl
            last_clip_frac = clip_frac
            if approx_kl > float(target_kl):
                break
        self._last_kl = float(last_approx_kl)
        self._last_clipped = bool(last_clip_frac > 0.0)
        self.storage.clear()
        
        return {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'approx_kl': float(last_approx_kl),
            'clip_frac': float(last_clip_frac)
        }

    def update(self, batch=None) -> Dict[str, float]:
        """Alias for end_batch to support UnifiedM3Policy interface."""
        return self.end_batch() or {}


# BRPolicy (Bus-routed experts)
class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # pyright: ignore[reportInvalidTypeForm]
        return self.net(x)


class BRPolicyModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, trunk_dim: int = 512):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_dim), nn.GELU(),
            nn.Linear(trunk_dim, trunk_dim), nn.GELU(),
        )
        self.norm = nn.LayerNorm(trunk_dim)
        self.head_mu = nn.Linear(trunk_dim, out_dim)
        self.head_v = nn.Linear(trunk_dim, 1)
        self.log_sigma = nn.Parameter(torch.full((out_dim,), math.log(0.6)))
        self.gate_mlp = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, 1))
        self.experts: Dict[str, nn.Module] = {} # pyright: ignore[reportInvalidTypeForm]

    def _get_expert(self, name: str, d_spec: int) -> nn.Module: # pyright: ignore[reportInvalidTypeForm]
        key = f'{name}:{d_spec}'
        if key not in self.experts:
            self.experts[key] = MLP(d_spec, self.out_dim, d_hidden=max(64, min(512, d_spec * 2)))
            self.add_module(f'expert_{len(self.experts)}', self.experts[key])
        return self.experts[key]

    @staticmethod
    def _spec_summary(seg: torch.Tensor) -> torch.Tensor: # pyright: ignore[reportInvalidTypeForm]
        mean = seg.mean(dim=-1, keepdim=True)
        std = seg.std(dim=-1, keepdim=True) + 1e-6
        rms = torch.sqrt((seg * seg).mean(dim=-1, keepdim=True))
        d = torch.full_like(mean, float(seg.size(-1)))
        maxi, _ = seg.max(dim=-1, keepdim=True)
        mini, _ = seg.min(dim=-1, keepdim=True)
        span = maxi - mini
        return torch.cat([mean, std, rms, d, maxi, mini, span, (seg[:, :1] if seg.size(-1) > 0 else mean)], dim=-1)

    def forward(self, x: torch.Tensor, active_specs: List[Tuple[str, Tuple[int, int]]], biases: Optional[List[float]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]: # pyright: ignore[reportInvalidTypeForm]
        trunk = self.trunk(x)
        h = self.norm(trunk)
        mu_base = self.head_mu(h)
        v = self.head_v(h).squeeze(-1)
        sigma = self.log_sigma.exp().clamp_min(1e-3)
        B = x.size(0)
        if not active_specs:
            return mu_base, v, sigma, {}
        gate_inputs = []
        spec_contribs: Dict[str, torch.Tensor] = {} # pyright: ignore[reportInvalidTypeForm]
        order: List[str] = []
        for name, (i0, i1) in active_specs:
            seg = x[:, i0:i1]
            expert = self._get_expert(name, seg.size(-1))
            y = expert(seg)
            spec_contribs[name] = y
            order.append(name)
            gate_inputs.append(self._spec_summary(seg))
        G = torch.cat(gate_inputs, dim=1).view(B, len(active_specs), -1)
        g_logits = self.gate_mlp(G).squeeze(-1)
        if biases is not None and len(biases) == g_logits.size(-1):
            b = torch.tensor(biases, dtype=g_logits.dtype, device=g_logits.device).view(1, -1)
            g_logits = g_logits + b
        g = torch.softmax(g_logits, dim=-1)
        mu_exp = torch.zeros_like(mu_base)
        for idx, (name, _) in enumerate(active_specs):
            mu_exp = mu_exp + g[:, idx:idx+1] * spec_contribs[name]
        mu = mu_base + mu_exp
        contribs = [spec_contribs[n] for n in order]
        return mu, v, sigma, { 'g': g, 'mu_base': mu_base, 'mu_exp': mu_exp, 'order': order, 'contribs': contribs }


@dataclass
class BRItem:
    obs: np.ndarray
    act: np.ndarray
    logp: float
    mu: np.ndarray
    rew: float
    active_specs: List[Tuple[str, Tuple[int, int]]]


class BRPolicy:
    def __init__(self, in_dim: int, out_dim: int, trunk_dim: int = 512, device: Optional[str] = None,
                 shared_repr: Optional[nn.Module] = None): # pyright: ignore[reportInvalidTypeForm]
        if not TORCH_OK:
            raise RuntimeError('PyTorch is not available for BRPolicy')
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        dev = torch.device(_resolve_device(device))
        self.device = dev
        self.model = BRPolicyModel(in_dim, out_dim, trunk_dim=trunk_dim).to(dev)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
        self.scaler = GradScaler(enabled=(dev.type == 'cuda'))
        self.storage: List[BRItem] = []
        self._active_specs: List[Tuple[str, Tuple[int, int]]] = []
        self.shared_repr = shared_repr.to(dev) if shared_repr is not None else None
        self._last_kl: float = 0.0
        self._last_clipped: bool = False
        self._gate_bias: Dict[str, float] = {}

    @property
    def sigma(self) -> float:
        try:
            with torch.no_grad():
                return float(self.model.log_sigma.exp().mean().detach().cpu().item())
        except Exception:
            return 0.6

    def set_active_specs(self, active_names: List[str], ranges: Dict[str, Tuple[int, int]]) -> None:
        self._active_specs = [(n, ranges.get(n, (0, 0))) for n in active_names if n in ranges]

    def set_gate_bias(self, biases: Dict[str, float]) -> None:
        self._gate_bias = dict(biases or {})

    def sample(self, obs: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
        self.model.eval()
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad(), autocast(self.device.type == 'cuda'):
            if self.shared_repr is not None:
                x = self.shared_repr(x)
            biases = [float(self._gate_bias.get(n, 0.0)) for n, _ in self._active_specs]
            mu, v, sigma, _ = self.model(x, self._active_specs, biases=biases)
            eps = torch.randn_like(mu)
            act = mu + sigma * eps
            logp = -0.5 * torch.sum(((act - mu) / (sigma + 1e-8)) ** 2, dim=-1)
        return act.squeeze(0).cpu().numpy(), float(logp.item()), mu.squeeze(0).cpu().numpy(), float(v.item())

    def diagnose(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        self.model.eval()
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad(), autocast(self.device.type == 'cuda'):
            if self.shared_repr is not None:
                x = self.shared_repr(x)
            biases = [float(self._gate_bias.get(n, 0.0)) for n, _ in self._active_specs]
            mu, v, sigma, aux = self.model(x, self._active_specs, biases=biases)
        out = {
            'order': [n for n, _ in self._active_specs],
            'g': aux.get('g').squeeze(0).detach().cpu().numpy() if 'g' in aux else None,
            'mu_base': mu.detach().cpu().numpy() if isinstance(mu, torch.Tensor) else None,
        }
        contribs = aux.get('contribs')
        if contribs is not None:
            out['contribs'] = [c.squeeze(0).detach().cpu().numpy() for c in contribs]
        return out

    def record(self, obs: np.ndarray, act: np.ndarray, logp: float, mu: np.ndarray, rew: float) -> None:
        self.storage.append(BRItem(obs=np.asarray(obs, np.float32),
                                   act=np.asarray(act, np.float32),
                                   logp=float(logp),
                                   mu=np.asarray(mu, np.float32),
                                   rew=float(rew),
                                   active_specs=list(self._active_specs)))

    def end_batch(self, gamma: float = 0.99, kl_coeff: float = 0.0, lr: float = 3e-4,
                  ppo_epochs: int = 2, clip_range: float = 0.2, target_kl: float = 0.02) -> None:
        if not self.storage:
            return
        self.model.train()
        T = len(self.storage)
        obs = torch.as_tensor(np.stack([it.obs for it in self.storage], axis=0), dtype=torch.float32, device=self.device)
        if self.shared_repr is not None:
            with torch.no_grad():
                obs = self.shared_repr(obs)
        acts = torch.as_tensor(np.stack([it.act for it in self.storage], axis=0), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(np.array([it.logp for it in self.storage], dtype=np.float32), device=self.device)
        rews = [it.rew for it in self.storage]
        G = np.zeros(T, dtype=np.float32)
        run = 0.0
        for t in range(T - 1, -1, -1):
            run = rews[t] + gamma * run
            G[t] = run
        returns = torch.as_tensor(G, dtype=torch.float32, device=self.device)
        constant_specs = all(self.storage[t].active_specs == self.storage[0].active_specs for t in range(T))
        with torch.no_grad():
            if constant_specs:
                biases = [float(self._gate_bias.get(n, 0.0)) for n, _ in self.storage[0].active_specs]
                _, v, _, _ = self.model(obs, self.storage[0].active_specs, biases=biases)
            else:
                v_list = []
                for t in range(T):
                    x_t = obs[t:t+1]
                    biases_t = [float(self._gate_bias.get(n, 0.0)) for n, _ in self.storage[t].active_specs]
                    _, v_t, _, _ = self.model(x_t, self.storage[t].active_specs, biases=biases_t)
                    v_list.append(v_t)
                v = torch.cat(v_list, dim=0)
        adv = (returns - v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        self.optim.param_groups[0]['lr'] = float(lr)
        last_approx_kl = 0.0
        last_clip_frac = 0.0
        for _ in range(max(1, int(ppo_epochs))):
            with autocast(self.device.type == 'cuda'):
                if constant_specs:
                    biases = [float(self._gate_bias.get(n, 0.0)) for n, _ in self.storage[0].active_specs]
                    mu, v_cur, sigma, _ = self.model(obs, self.storage[0].active_specs, biases=biases)
                    logp = -0.5 * torch.sum(((acts - mu) / (sigma + 1e-8)) ** 2, dim=-1)
                    ratio = torch.exp(logp - old_logp)
                    pg1 = ratio * adv
                    pg2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
                    policy_loss = -torch.mean(torch.min(pg1, pg2))
                    value_loss = 0.5 * torch.mean((returns - v_cur) ** 2)
                    ent = torch.mean(0.5 * torch.log(2 * torch.pi * (sigma ** 2)) + 0.5)
                    loss = policy_loss + value_loss - 0.01 * ent
                    approx_kl = torch.mean(old_logp - logp).item()
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).item()
                else:
                    policy_loss_acc = 0.0
                    value_loss_acc = 0.0
                    ent_acc = 0.0
                    kl_acc = 0.0
                    clip_hits = 0.0
                    for t in range(T):
                        x_t = obs[t:t+1]
                        a_t = acts[t:t+1]
                        biases_t = [float(self._gate_bias.get(n, 0.0)) for n, _ in self.storage[t].active_specs]
                        mu_t, v_t, sigma_t, _ = self.model(x_t, self.storage[t].active_specs, biases=biases_t)
                        logp_t = -0.5 * torch.sum(((a_t - mu_t) / (sigma_t + 1e-8)) ** 2, dim=-1)
                        ratio_t = torch.exp(logp_t - old_logp[t:t+1])
                        pg1_t = ratio_t * adv[t:t+1]
                        pg2_t = torch.clamp(ratio_t, 1.0 - clip_range, 1.0 + clip_range) * adv[t:t+1]
                        policy_loss_acc = policy_loss_acc + (-torch.min(pg1_t, pg2_t)).mean()
                        value_loss_acc = value_loss_acc + 0.5 * ((returns[t:t+1] - v_t) ** 2).mean()
                        ent_acc = ent_acc + (0.5 * torch.log(2 * torch.pi * (sigma_t ** 2)) + 0.5).mean()
                        kl_acc = kl_acc + (old_logp[t:t+1] - logp_t).mean()
                        clip_hits = clip_hits + (torch.abs(ratio_t - 1.0) > clip_range).float().mean()
                    policy_loss = policy_loss_acc / T
                    value_loss = value_loss_acc / T
                    ent = ent_acc / T
                    loss = policy_loss + value_loss - 0.01 * ent
                    approx_kl = float((kl_acc / T).item())
                    clip_frac = float((clip_hits / T).item())
            self.scaler.scale(loss).backward()
            if self.device.type == 'cuda':
                self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)
            last_approx_kl = approx_kl
            last_clip_frac = clip_frac
            if approx_kl > float(target_kl):
                break
        self._last_kl = float(last_approx_kl)
        self._last_clipped = bool(last_clip_frac > 0.0)
        self.storage.clear()

    def update_from_delayed_reward(self, obs: np.ndarray, act: np.ndarray, logp: float, reward: float) -> None:
        """
        Update policy from a single delayed reward signal (e.g. from MessageBus).
        This treats the single step as a mini-episode or adds it to the buffer.
        """
        # For simplicity, we treat this as a single-step trajectory with immediate reward.
        # We add it to storage. If end_batch is called later, it will be used.
        # Note: mu is not available here, we pass zeros or re-compute if needed.
        # Re-computing mu requires the model, which we have.
        
        self.model.eval()
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad(), autocast(self.device.type == 'cuda'):
             if self.shared_repr is not None:
                 x = self.shared_repr(x)
             biases = [float(self._gate_bias.get(n, 0.0)) for n, _ in self._active_specs]
             mu, _, _, _ = self.model(x, self._active_specs, biases=biases)
             mu_np = mu.squeeze(0).cpu().numpy()
             
        self.record(obs, act, logp, mu_np, reward)


