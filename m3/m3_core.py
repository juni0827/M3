from __future__ import annotations

import argparse, os, time, math, json, heapq
import re
import os as _os
import logging
import torch
import torch.nn as nn
from .device import resolve_torch_device_string
from m3.config import QUALIA_CFG, QUALIA_LOG_PATH, _CESConfig
from m3.features import HebbianMemory, FeatureSpec, pack_learned_proj, pack_scalar, pack_spatial_pool, pack_stats_sample, Scope
from m3.visualization import FeatureSummarizer, GlitchEncoder, Retinizer, hilbert_index_to_xy, vector_to_grid
from m3.reward import RewardSystem
from functools import lru_cache

def _write_jsonl_safe(path, obj):
    try:
        import json as _json, io as _io
        with _io.open(path, 'a', encoding='utf-8') as f:
            f.write(_json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _normalize_phi_policy(raw: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    base = {
        "floor": 0.01,
        "low": 0.10,
        "mid": 0.30,
        "high": 0.50,
        "very_high": 0.70,
        "announce_high": 0.50,
    }
    if isinstance(raw, dict):
        for k in list(base.keys()):
            try:
                if k in raw:
                    base[k] = float(raw[k])
            except Exception:
                pass
    base["low"] = float(max(base["floor"] + 1e-6, base["low"]))
    base["mid"] = float(max(base["low"] + 1e-6, base["mid"]))
    base["high"] = float(max(base["mid"] + 1e-6, base["high"]))
    base["very_high"] = float(max(base["high"] + 1e-6, base["very_high"]))
    base["announce_high"] = float(max(base["high"], base["announce_high"]))
    return base


def _compute_phi_policy_from_history(
    phi_history: Optional[List[float]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    policy = _normalize_phi_policy(None)
    if cfg is None:
        cfg = {}
    try:
        enabled = bool(cfg.get("enabled", True))
    except Exception:
        enabled = True
    try:
        floor_min = float(cfg.get("phi_floor_min", 0.005))
        floor_max = float(cfg.get("phi_floor_max", 0.05))
    except Exception:
        floor_min, floor_max = 0.005, 0.05
    if not enabled:
        return policy
    vals = np.asarray(phi_history if phi_history is not None else [], dtype=np.float32)
    if vals.size <= 0:
        return policy
    vals = vals[np.isfinite(vals)]
    if vals.size <= 0:
        return policy
    try:
        warmup = int(max(0, cfg.get("warmup_steps", 256)))
    except Exception:
        warmup = 256
    if vals.size < warmup:
        p75 = float(np.quantile(vals, 0.75))
        policy["floor"] = float(np.clip(0.5 * max(0.0, p75), floor_min, floor_max))
        return _normalize_phi_policy(policy)
    try:
        q_low = float(np.quantile(vals, float(cfg.get("quantile_low", 0.50))))
        q_mid = float(np.quantile(vals, float(cfg.get("quantile_mid", 0.75))))
        q_high = float(np.quantile(vals, float(cfg.get("quantile_high", 0.90))))
        hyst = float(max(0.0, cfg.get("announce_hysteresis", 0.02)))
    except Exception:
        q_low, q_mid, q_high, hyst = 0.10, 0.30, 0.50, 0.02
    policy["floor"] = float(np.clip(0.5 * max(0.0, q_low), floor_min, floor_max))
    policy["low"] = float(max(policy["floor"] + 1e-4, q_low))
    policy["mid"] = float(max(policy["low"] + 1e-4, q_mid))
    policy["high"] = float(max(policy["mid"] + 1e-4, q_high))
    policy["very_high"] = float(max(policy["high"] + 1e-4, q_high + max(0.05, hyst)))
    policy["announce_high"] = float(max(policy["high"], q_high + hyst))
    return _normalize_phi_policy(policy)

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set, Union, Callable, cast
from enum import Enum
from contextlib import contextmanager
import numpy as np


import hashlib
import unicodedata
import pandas as pd
from collections import deque
from itertools import combinations
import threading
import sys
import os
from numpy.random import default_rng, SeedSequence
from PIL import Image, ImageOps
from queue import Queue, Empty

@lru_cache(maxsize=1)
def _default_torch_device():
    """Resolve torch device lazily to avoid import-time side effects."""
    return resolve_torch_device_string(require_cuda=False)
# MetaController is implemented internally in this file to avoid scattering
# core logic across modules. If you need an external implementation for
# testing, keep a copy in tools/ but the core will use the internal class.

# Ensure stdout/stderr can handle Unicode on Windows consoles (CP949, etc.)
try:  # best-effort; never crash on setup
    import io
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            try:
                import traceback as _tb
                _tb.print_exc()
            except Exception:
                pass
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    else:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
    # Try to switch Windows console code page to UTF-8 for correct rendering
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
except Exception:
    pass

# ==================== MULTI-LOOP MESSAGE PASSING INFRASTRUCTURE ====================

@dataclass
class Message:
    """Universal message format for inter-module communication in M3"""
    source: str  # Module name that sent this
    target: str  # Module name (or 'broadcast' for all)
    type: str    # Message semantic type
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.5  # Higher = more urgent (0.0-1.0)

@dataclass
class SpanMeta:
    """Metadata for a temporal decision span (utterance, tool call, etc.)"""
    span_id: str
    source: str          # "llm_adapter", "policy", etc.
    kind: str            # "utter", "toolcall", "action", etc.
    tool: Optional[str]  # Tool name if kind="toolcall"
    ctx_tags: Tuple[str, ...]  # Context tags for filtering
    t_start: float
    t_end: Optional[float] = None
    token_range: Optional[Tuple[int, int]] = None  # (global_idx_start, global_idx_end)

class MessageBus:
    """
    Central message routing system for M3 multi-loop architecture.
    Enables asynchronous, bidirectional communication between all modules.
    """
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        # Per-module inboxes
        self.inboxes: Dict[str, Queue] = {}
        # Global broadcast buffer for debugging/monitoring
        self.broadcast_buffer: deque = deque(maxlen=1000)
        # Module registry
        self.modules: Set[str] = set()
        self._lock = threading.Lock()
        
        # === Temporal Credit Assignment Infrastructure ===
        # credit_router: maps source module to list of its decision spans
        self.credit_router: Dict[str, List[SpanMeta]] = {}
        # span_table: fast lookup by span_id
        self.span_table: Dict[str, SpanMeta] = {}
        
    def register_module(self, name: str):
        """Register a module to receive messages"""
        with self._lock:
            if name not in self.modules:
                self.modules.add(name)
                self.inboxes[name] = Queue(maxsize=self.capacity)
    
    def send(self, msg: Message):
        """Route message to target module(s)"""
        with self._lock:
            if msg.target == 'broadcast':
                self.broadcast_buffer.append(msg)
                # Deliver to all modules except sender
                for name in self.modules:
                    if name != msg.source:
                        try:
                            self.inboxes[name].put_nowait(msg)
                        except:
                            pass  # Queue full, drop message
            else:
                # Direct message to specific module
                if msg.target in self.inboxes:
                    try:
                        self.inboxes[msg.target].put_nowait(msg)
                    except:
                        pass
    
    def receive(self, module_name: str, timeout: float = 0.001) -> Optional[Message]:
        """Non-blocking receive with short timeout"""
        if module_name in self.inboxes:
            try:
                return self.inboxes[module_name].get(timeout=timeout)
            except Empty:
                return None
        return None
    
    def receive_all(self, module_name: str, max_msgs: int = 100) -> List[Message]:
        """Drain inbox up to max_msgs"""
        msgs = []
        for _ in range(max_msgs):
            msg = self.receive(module_name, timeout=0.0001)
            if msg is None:
                break
            msgs.append(msg)
        return msgs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        with self._lock:
            return {
                'modules': len(self.modules),
                'broadcast_buffer_size': len(self.broadcast_buffer),
                'inbox_sizes': {name: self.inboxes[name].qsize() for name in self.modules}
            }
    
    # === Temporal Credit Assignment Methods ===
    
    def register_span(self, span: SpanMeta):
        """Register a decision/generation span for future credit assignment"""
        with self._lock:
            self.span_table[span.span_id] = span
            self.credit_router.setdefault(span.source, []).append(span)
    
    def close_span(self, span_id: str, token_end: Optional[int] = None):
        """Finalize a span with end time and optional token range end"""
        with self._lock:
            span = self.span_table.get(span_id)
            if span:
                span.t_end = time.time()
                if token_end is not None and span.token_range:
                    start, _ = span.token_range
                    span.token_range = (start, token_end)
    
    def route_credit(self, signal: Dict[str, Any]):
        """
        Distribute temporal credit to recent decision spans based on outcome signal.
        
        Args:
            signal: Dict with keys:
                - phi_delta: float (change in integrated information)
                - stability_delta: float (change in belief stability)
                - tool_success: float (1.0 if tool succeeded, 0.0 otherwise)
                - lambda: float (eligibility trace decay, default 0.8)
                - gamma: float (temporal discount, default 0.98)
                - window: int (number of recent spans to credit, default 8)
        
        Sends Message(type='credit') to source modules with span-level credits.
        """
        lam = float(signal.get('lambda', 0.8))
        gamma = float(signal.get('gamma', 0.98))
        window = int(signal.get('window', 8))
        
        # Composite value: φ + stability + tool success + drive reward
        value = (
            float(signal.get('phi_delta', 0.0)) +
            0.5 * float(signal.get('stability_delta', 0.0)) +
            1.0 * float(signal.get('tool_success', 0.0)) +
            1.0 * float(signal.get('drive_reward', 0.0))
        )
        
        with self._lock:
            for source, spans in self.credit_router.items():
                # Credit most recent W spans with exponential decay
                recent_spans = spans[-window:] if len(spans) > window else spans
                for k, span in enumerate(reversed(recent_spans)):
                    # Exponential decay: more recent spans get more credit
                    credit = (gamma ** k) * (lam ** k) * value
                    
                    # Send credit message to source module
                    credit_msg = Message(
                        source='bus',
                        target=source,
                        type='credit',
                        payload={
                            'span_id': span.span_id,
                            'credit': credit,
                            'signal': signal,
                            'decay_factor': (gamma ** k) * (lam ** k)
                        }
                    )
                    
                    # Direct send (bypass queue for immediate delivery)
                    if source in self.inboxes:
                        try:
                            self.inboxes[source].put_nowait(credit_msg)
                        except:
                            pass  # Queue full, drop message

@dataclass
class RNGRegistry:
    def __init__(self, root_seed: int | None):
        self.root_seed = int(root_seed) if root_seed is not None else None
        self.root_ss = SeedSequence(self.root_seed) if self.root_seed is not None else SeedSequence()
        self.cache = {}

    def get(self, name: str):
        if name not in self.cache:
            mix = int.from_bytes(hashlib.blake2b(name.encode('utf-8'), digest_size=8).digest(), 'little')
            child_ss = SeedSequence(self.root_seed if self.root_seed is not None else 0, spawn_key=(mix,))
            self.cache[name] = default_rng(child_ss)
        return self.cache[name]
    def episode(self, key: str, ep: int):
        return self.get(f"{key}:episode:{int(ep)}")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
for _p in (current_dir, project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

class PolicyMLP(nn.Module):
    """Two-layer MLP Gaussian policy with optional adoption from linear weights.

    Provides a similar interface: sample(), record(), end_batch().
    Re-implemented using PyTorch for stability and speed.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, sigma: float = 0.6, rng=None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        # sigma is now a parameter or fixed, handled in log_prob
        self._initial_sigma = float(sigma)
        self.rng = rng or np.random.default_rng()

        # Define Network
        self.base = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.Tanh()
        )
        
        # Heads
        self.mu_head = nn.Linear(self.hidden, self.out_dim)
        self.value_head = nn.Linear(self.hidden, 1)
        
        # Learnable log_std
        self.log_std = nn.Parameter(torch.ones(self.out_dim) * np.log(self._initial_sigma))
        
        self.to(_default_torch_device())
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        
        # buffers for RL
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[np.ndarray] = []
        self._lp_buf: list[float] = []      # log_probs from old policy
        self._rew_buf: list[float] = []
        
        # Diagnostics
        self._last_kl: float = 0.0
        self._last_loss: float = 0.0

    def adopt_linear(self, theta_linear: np.ndarray) -> None:
        """Initialize MLP to approximate a linear policy."""
        # Simple initialization for compatibility, though full logic is harder in Deep RL
        # We'll just re-initialize weights to be small
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns mu, std, value"""
        h = self.base(x)
        mu = self.mu_head(h)
        value = self.value_head(h)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std, value

    def sample(self, obs: np.ndarray, **kwargs) -> tuple[np.ndarray, float, np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(_default_torch_device()).unsqueeze(0) # (1, dim)
            mu, std, v = self.forward(x)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
            
            act_np = action.cpu().numpy().squeeze(0)
            logp_np = logp.cpu().item()
            mu_np = mu.cpu().numpy().squeeze(0)
            v_np = v.cpu().item()
            
        return act_np, logp_np, mu_np, v_np

    def record(self, obs: np.ndarray, act: np.ndarray, logp: float, mu: np.ndarray, rew: float) -> None:
        self._obs_buf.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        self._act_buf.append(np.asarray(act, dtype=np.float32).reshape(-1))
        self._lp_buf.append(float(logp))
        self._rew_buf.append(float(rew))

    def end_batch(
        self,
        gamma: float = 0.97,
        kl_coeff: float = 0.02,
        lr: float = 1e-2, # This lr is ignored, using optimizer's lr
        target_entropy_per_dim: float = 1.5,
        kl_budget: float = 0.02, # Not strictly used in PPO-style clip, but kept for compat
        max_backtrack: int = 6, # Ignored
        global_clip: float = 1.0,
    ):
        if not self._obs_buf:
            return
            
        # 1. Prepare Data
        obs_np = np.stack(self._obs_buf)
        act_np = np.stack(self._act_buf)
        rew_np = np.array(self._rew_buf, dtype=np.float32)
        old_logp_np = np.array(self._lp_buf, dtype=np.float32)
        
        # Calculate Returns & Advantages (GAE can be added later, simple returns for now)
        returns = np.zeros_like(rew_np)
        G = 0
        for t in reversed(range(len(rew_np))):
            G = rew_np[t] + gamma * G
            returns[t] = G
        
        returns = torch.from_numpy(returns).to(_default_torch_device()).float()
        # Normalization
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        obs = torch.from_numpy(obs_np).to(_default_torch_device()).float()
        act = torch.from_numpy(act_np).to(_default_torch_device()).float()
        old_logp = torch.from_numpy(old_logp_np).to(_default_torch_device()).float()
        
        # 2. Update Loop (PPO-style or simple Actor-Critic)
        self.train()
        
        # Single PPO update step style
        # Get current policy outputs
        mu, std, values = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        new_logp = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1).mean()
        
        # Ratio
        ratio = torch.exp(new_logp - old_logp)
        
        # Advantage (using returns - value)
        advantage = returns - values.squeeze(-1).detach()
        
        # Surrogate Loss similar to PPO
        eps_clip = 0.2
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value Loss
        value_loss = nn.MSELoss()(values.squeeze(-1), returns)
        
        # Total Loss
        loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), global_clip)
        self.optimizer.step()
        
        self._last_loss = loss.item()
        
        # Approx KL for logging
        with torch.no_grad():
            self._last_kl = (old_logp - new_logp).mean().item()

        # Clear buffers
        self._obs_buf.clear()
        self._act_buf.clear()
        self._lp_buf.clear()
        self._rew_buf.clear()

    def _approx_kl(self, *args, **kwargs) -> float:
        return self._last_kl

    def spectral_clip(self, **kwargs) -> None:
        pass  # AdamW & LayerNorm handles stability better

    def resize_input(self, new_in_dim: int) -> None:
        new_in_dim = int(new_in_dim)
        if new_in_dim == self.in_dim:
            return
        
        # Expand input layer
        old_layer = self.base[0]
        new_layer = nn.Linear(new_in_dim, self.hidden).to(_default_torch_device())
        
        with torch.no_grad():
            # Copy compatible weights
            min_dim = min(self.in_dim, new_in_dim)
            new_layer.weight[:, :min_dim] = old_layer.weight[:, :min_dim]
            if new_in_dim > self.in_dim:
                # Initialize new weights near zero
                 nn.init.normal_(new_layer.weight[:, min_dim:], std=0.01)
            new_layer.bias.copy_(old_layer.bias)
            
        self.base[0] = new_layer
        self.in_dim = new_in_dim
        # Re-init optimizer to track new params
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

    def resize_hidden(self, new_h: int) -> None:
        # Complex to support with Sequential, keeping simple for now or implement Net2Net
        pass 

    def resize_output(self, new_out: int) -> None:
        new_out = int(new_out)
        if new_out == self.out_dim:
            return
            
        old_mu = self.mu_head
        new_mu = nn.Linear(self.hidden, new_out).to(_default_torch_device())
        
        # Log STD resize
        old_log_std = self.log_std
        new_log_std = nn.Parameter(torch.ones(new_out).to(_default_torch_device()) * np.log(self._initial_sigma))
        
        with torch.no_grad():
            min_dim = min(self.out_dim, new_out)
            new_mu.weight[:min_dim, :] = old_mu.weight[:min_dim, :]
            new_mu.bias[:min_dim] = old_mu.bias[:min_dim]
            new_log_std[:min_dim] = old_log_std[:min_dim]
            
        self.mu_head = new_mu
        self.log_std = new_log_std
        self.out_dim = new_out
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)


class FeatureBank:
    """Build a fixed-length observation vector with optional input mask and running normalization."""
    def __init__(self, max_dim: int = 128, embed_dim: int = 32):
        self.max_dim = int(max_dim)
        self.embed_dim = int(embed_dim)
        self.input_mask = np.ones(self.max_dim, dtype=np.float32)
        self.mu = np.zeros(self.max_dim, dtype=np.float32)
        self.s2 = np.ones(self.max_dim, dtype=np.float32)
        # Welford counters for unbiased variance
        self._count: int = 0
        # Mapping from spec.name -> (start_idx, end_idx) in the assembled vector
        self._ranges: Dict[str, Tuple[int, int]] = {}
        # AB testing state and last-built cache
        self._ab_trials: Dict[str, Dict[str, Any]] = {}
        self._last_applied_specs: List[str] = []
        self._last_built_z: np.ndarray | None = None
        self.use_bus_gating: bool = False
        self._provenance_path: Optional[str] = None
        self._metrics_path: Optional[str] = None
        # Auto-namer and last outputs per spec for token broadcasting
        self._namer = AutoNamer()
        self._last_outputs: Dict[str, np.ndarray] = {}
        # FeatureSpec registry: ordered list of FeatureSpec
        self.specs: List[FeatureSpec] = []
        # initialize default specs to preserve legacy behavior
        try:
            self._register_default_specs()
        except Exception:
            # non-fatal: leave specs empty and fallback to legacy build
            self.specs = []
        
        # === AUTONOMOUS GROWTH CONTROL ===
        self.growth_history: deque = deque(maxlen=256)
        self.complexity_metrics: deque = deque(maxlen=256)
        self.last_growth_time: float = 0.0
        self.growth_cooldown: float = 200.0
        self.adaptive_growth_threshold: float = 0.75

    def set_mask(self, mask: np.ndarray | list[float] | list[int]) -> None:
        m = np.asarray(mask, dtype=np.float32).reshape(-1)
        if m.size != self.max_dim:
            raise ValueError("mask size mismatch")
        self.input_mask = np.clip(m, 0.0, 1.0)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        # Unbiased variance estimate
        var = self.s2 / max(1, self._count - 1)
        s = np.sqrt(np.maximum(var, 1e-6))
        return (v - self.mu) / s

    def _update_stats(self, v: np.ndarray) -> None:
        # Welford online update
        x = np.asarray(v, dtype=np.float32).reshape(-1)
        if x.size != self.mu.size:
            # best-effort shape guard
            L = min(x.size, self.mu.size)
            x = np.pad(x[:L], (0, self.mu.size - L), mode='constant') if L < self.mu.size else x[:self.mu.size]
        self._count += 1
        delta = x - self.mu
        self.mu += delta / float(self._count)
        self.s2 += delta * (x - self.mu)

    def _hash_embed(self, text: str, dim: int) -> np.ndarray:
        # Simple feature hashing with signed bins
        vec = np.zeros(dim, dtype=np.float32)
        for tok in re.split(r"[^\w]+", text.lower()):
            if not tok:
                continue
            h = int.from_bytes(hashlib.blake2b(tok.encode('utf-8'), digest_size=8).digest(), 'little')
            idx = h % dim
            sgn = 1.0 if ((h >> 8) & 1) == 1 else -1.0
            vec[idx] += sgn
        if np.linalg.norm(vec) > 0:
            vec = vec / (np.linalg.norm(vec) + 1e-6)
        return vec.astype(np.float32)

    def build(self, core: "M3ConsciousnessCore") -> np.ndarray:
        """Build observation vector using FeatureSpec registry when available.

        If no specs are registered, fall back to legacy compact observation.
        """
        # If no specs registered, keep small legacy fallback
        if not getattr(self, 'specs', None):
            # Fallback to previous compact behavior (12D-ish)
            try:
                retina = core._vision_build_retina(size=(32, 32), foveate=True)
                contrast, entropy, edge_density, depth_cue = core._vision_features(retina)
                r_mean = float(np.mean(retina))
            except Exception:
                contrast = entropy = edge_density = 0.0
                r_mean = 0.0
            try:
                last_phi = float(core.phi_calculator.phi_history[-1]) if core.phi_calculator.phi_history else 0.0
            except Exception:
                last_phi = 0.0
            obs = np.array([
                r_mean, contrast, entropy, edge_density,
                last_phi,
                float(core.self_model.meta_awareness),
                float(core.energy_ctrl.activation_level),
                float(core.unified_subject.unity_score),
                float(core.world_state.get('stability', 0.5)) if hasattr(core, 'world_state') else 0.5,
                float(core.qualia.arousal), float(core.qualia.valence), float(core.qualia.entropy)
            ], dtype=np.float32)
            obs = np.clip((obs - 0.5) * 2.0, -1.0, 1.0)
            return obs

        # Build a small context of commonly used derived values to avoid repeated computation
        ctx: Dict[str, Any] = {}
        try:
            ctx['vision.retina'] = core._vision_build_retina(size=(32, 32), foveate=True)
            try:
                c, e, ed, dc = core._vision_features(ctx['vision.retina'])
            except Exception:
                c = e = ed = dc = 0.0
            ctx['vision.contrast'] = c
            ctx['vision.entropy'] = e
            ctx['vision.edge_density'] = ed
            ctx['vision.depth_cue'] = dc
            ctx['vision.r_mean'] = float(np.mean(ctx['vision.retina'])) if ctx['vision.retina'] is not None else 0.0
        except Exception:
            ctx['vision.retina'] = None
            ctx['vision.contrast'] = ctx['vision.entropy'] = ctx['vision.edge_density'] = ctx['vision.depth_cue'] = 0.0
            ctx['vision.r_mean'] = 0.0
        try:
            ph = list(core.phi_calculator.phi_history) if hasattr(core, 'phi_calculator') else []
            ctx['phi.last'] = float(ph[-1]) if ph else 0.0
            ctx['phi.mean10'] = float(np.mean(ph[-10:])) if len(ph) >= 10 else ctx['phi.last']
            ctx['phi.delta'] = float(ph[-1] - ph[-2]) if len(ph) >= 2 else 0.0
        except Exception:
            ctx['phi.last'] = ctx['phi.mean10'] = ctx['phi.delta'] = 0.0
        try:
            ctx['self.meta_awareness'] = float(getattr(core.self_model, 'meta_awareness', 0.0))
        except Exception:
            ctx['self.meta_awareness'] = 0.0
        try:
            ctx['self.meta_confidence'] = float(getattr(core.self_model, 'meta_confidence', 0.5))
        except Exception:
            ctx['self.meta_confidence'] = 0.5
        try:
            ctx['energy.activation'] = float(getattr(core.energy_ctrl, 'activation_level', 0.0))
            ctx['energy.ratio'] = float(core.energy_ctrl.cognitive_energy / core.energy_ctrl.energy_capacity)
        except Exception:
            ctx['energy.activation'] = 0.0
            ctx['energy.ratio'] = 0.5
        try:
            ctx['unity'] = float(getattr(core.unified_subject, 'unity_score', 0.0))
        except Exception:
            ctx['unity'] = 0.0
        try:
            ctx['stability'] = float(getattr(core, 'world_state', {}).get('stability', 0.5))
            ctx['delta_hat'] = float(getattr(core, 'world_state', {}).get('delta_hat', 0.5))
        except Exception:
            ctx['stability'] = 0.5
            ctx['delta_hat'] = 0.5
        try:
            ctx['qualia.arousal'] = float(getattr(core.qualia, 'arousal', 0.0))
            ctx['qualia.valence'] = float(getattr(core.qualia, 'valence', 0.0))
            ctx['qualia.entropy'] = float(getattr(core.qualia, 'entropy', 0.0))
            ctx['qualia.engagement'] = float(getattr(core.qualia, 'engagement', 0.0))
            ctx['qualia.frustration'] = float(getattr(core.qualia, 'frustration', 0.0))
        except Exception:
            ctx['qualia.arousal'] = ctx['qualia.valence'] = ctx['qualia.entropy'] = ctx['qualia.engagement'] = ctx['qualia.frustration'] = 0.0

        # Language embedding
        try:
            utt = ''
            try:
                rep = core.self_model.introspect()
                if isinstance(rep, dict):
                    import json as _json
                    utt = _json.dumps(rep, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                else:
                    utt = str(rep)
            except Exception:
                utt = ''
            lang = self._hash_embed(utt, self.embed_dim)
        except Exception:
            lang = np.zeros(self.embed_dim, dtype=np.float32)
        # Now iterate specs and pack deterministically until capacity
        capacity = max(0, self.max_dim - self.embed_dim)
        packed: list[float] = []
        self._ranges.clear()
        applied: List[str] = []
        # Optional gating via Bus tokens (match by spec.name)
        gated_names: Optional[Set[str]] = None
        try:
            if self.use_bus_gating and getattr(core, 'bus', None) is not None:
                gated_names = set(core.bus.top_keys())
                # If only generic 'policy' present, disable gating to avoid empty inputs
                if not gated_names or gated_names.issubset({'policy'}):
                    gated_names = None
        except Exception:
            gated_names = None
        for spec in self.specs:
            if not spec.active:
                continue
            if gated_names is not None and spec.name not in gated_names:
                continue
            # extract value
            try:
                if isinstance(spec.source, str):
                    val = ctx.get(spec.source, None)
                elif callable(spec.source):
                    val = spec.source(core)
                else:
                    val = None
            except Exception:
                val = None
            # pack
            try:
                if spec.packer == 'scalar':
                    outp = pack_scalar(val, spec.params)
                elif spec.packer == 'stats_sample':
                    outp = pack_stats_sample(val, spec.params)
                elif spec.packer == 'spatial_pool':
                    outp = pack_spatial_pool(val, spec.params)
                elif spec.packer == 'learned_proj':
                    outp = pack_learned_proj(val, spec.params)
                else:
                    outp = pack_scalar(val, spec.params)
            except Exception:
                outp = np.zeros(spec.produced_dim if spec.produced_dim > 0 else 1, dtype=np.float32)
            # ensure length
            try:
                ln = outp.size
                if spec.produced_dim != ln:
                    # align declared produced_dim with actual output
                    spec.produced_dim = int(ln)
            except Exception:
                pass
            # A/B trial application: include spec output with probability epsilon when trialing
            apply_this = True
            tr = self._ab_trials.get(spec.name)
            if tr is not None and tr.get('active', False):
                eps = float(tr.get('epsilon', 0.3))
                apply_this = bool((core.rng.random() if hasattr(core, 'rng') else np.random.random()) < eps)
            if apply_this:
                start = len(packed)
                packed.extend(outp.ravel().tolist())
                end = len(packed)
                applied.append(spec.name)
            else:
                start = len(packed)
                end = len(packed)
            try:
                self._ranges[spec.name] = (start, end)
            except Exception:
                pass
            if len(packed) >= capacity:
                break

        raw = np.asarray(packed[:capacity], dtype=np.float32)
        out = np.zeros(self.max_dim, dtype=np.float32)
        L = min(raw.size, capacity)
        if L > 0:
            out[:L] = raw[:L]
        out[L:L + min(self.embed_dim, self.max_dim - L)] = lang[:min(self.embed_dim, self.max_dim - L)]
        self._update_stats(out)
        z = self._norm(out)
        z *= self.input_mask
        z = np.clip(z, -3.0, 3.0)
        self._last_built_z = z.astype(np.float32)
        self._last_applied_specs = applied
        # Broadcast top-K spec tokens to Bus with English keys (spec.name)
        try:
            if getattr(core, 'bus', None) is not None:
                sal_list = []
                for spec in self.specs:
                    try:
                        rng = self._ranges.get(spec.name)
                        if not rng:
                            continue
                        i0, i1 = rng
                        if i1 <= i0:
                            continue
                        vec = out[i0:i1].astype(np.float32)
                        sal = float(np.sqrt(np.mean((vec * vec))))
                        sal_list.append((sal, spec.name, vec))
                    except Exception:
                        continue
                sal_list.sort(key=lambda x: x[0], reverse=True)
                for sal, name, vec in sal_list[:4]:
                    try:
                        core.bus.push('feature', name, vec, salience=float(sal), confidence=0.8, ttl=5)
                    except Exception:
                        pass
                core.bus.step()
        except Exception:
            pass
        return z.astype(np.float32)

    def panels(self, core: "M3ConsciousnessCore") -> list[np.ndarray]:
        """
        Φ_total을 의미층별 '패널'로 쪼개어 D차 벡터 리스트로 반환한다.
        각 패널 = 메모리 토큰 1개로 사용.
        반환: [v1, v2, ..., vM], 각 v_i.shape == (D,)
        """
        D = int(getattr(self, "embed_dim", 32))  # 패널 임베딩 차원
        
        def _to_panel(x: np.ndarray | list[float] | float) -> np.ndarray:
            """입력 스칼라/벡터를 길이 D의 float32 패널로 변환(정규화+패딩/절단)."""
            arr = np.atleast_1d(np.array(x, dtype=np.float32)).ravel()
            if arr.size == 0:
                arr = np.zeros(1, dtype=np.float32)
            # 간단 정규화(robust): 평균-분산 정규화 후 tanh로 스케일 억제
            mu = float(np.mean(arr))
            sd = float(np.std(arr) + 1e-6)
            z = (arr - mu) / sd
            z = np.tanh(z)
            # 길이 맞추기
            if z.size < D:
                out = np.zeros(D, dtype=np.float32)
                out[:z.size] = z
                return out
            else:
                return z[:D].astype(np.float32)

        # ---- 공통 컨텍스트(기존 build()와 유사) ----
        ctx: dict[str, float | np.ndarray] = {}
        # Vision/환경 파생량
        try:
            retina = core._vision_build_retina(size=(32, 32), foveate=True)
            c, e, ed, dc = core._vision_features(retina)
            r_mean = float(np.mean(retina)) if retina is not None else 0.0
        except Exception:
            c = e = ed = dc = 0.0
            r_mean = 0.0
        ctx["vision.contrast"] = float(c)
        ctx["vision.entropy"] = float(e)
        ctx["vision.edge_density"] = float(ed)
        ctx["vision.depth_cue"] = float(dc)
        ctx["vision.r_mean"] = float(r_mean)

        # φ/안정성/에너지/주체/정신상태
        try:
            ph = list(core.phi_calculator.phi_history) if hasattr(core, "phi_calculator") else []
            phi_last = float(ph[-1]) if ph else 0.0
            phi_delta = float(ph[-1] - ph[-2]) if len(ph) >= 2 else 0.0
            phi_mean10 = float(np.mean(ph[-10:])) if len(ph) >= 10 else phi_last
        except Exception:
            phi_last = phi_delta = phi_mean10 = 0.0
        ctx["phi.last"] = phi_last
        ctx["phi.delta"] = phi_delta
        ctx["phi.mean10"] = phi_mean10

        try:
            stability = float(getattr(core, "world_state", {}).get("stability", 0.5))
            delta_hat = float(getattr(core, "world_state", {}).get("delta_hat", 0.5))
        except Exception:
            stability = 0.5
            delta_hat = 0.5
        ctx["stability"] = stability
        ctx["delta_hat"] = delta_hat

        try:
            energy_act = float(getattr(core.energy_ctrl, "activation_level", 0.0))
            energy_ratio = float(core.energy_ctrl.cognitive_energy / core.energy_ctrl.energy_capacity)
        except Exception:
            energy_act = 0.0
            energy_ratio = 0.5
        ctx["energy.activation"] = energy_act
        ctx["energy.ratio"] = energy_ratio

        try:
            unity = float(getattr(core.unified_subject, "unity_score", 0.0))
        except Exception:
            unity = 0.0
        ctx["unity"] = unity

        try:
            arousal = float(getattr(core.qualia, "arousal", 0.0))
            valence = float(getattr(core.qualia, "valence", 0.0))
            qent = float(getattr(core.qualia, "entropy", 0.0))
            engage = float(getattr(core.qualia, "engagement", 0.0))
            frustr = float(getattr(core.qualia, "frustration", 0.0))
        except Exception:
            arousal = valence = qent = engage = frustr = 0.0
        ctx["qualia.arousal"] = arousal
        ctx["qualia.valence"] = valence
        ctx["qualia.entropy"] = qent
        ctx["qualia.engagement"] = engage
        ctx["qualia.frustration"] = frustr

        # 최근 RPE/엔트로피/메타 등 시계열(있으면)
        try:
            rpe_hist = np.asarray(getattr(core, "rpe_history", [])[-32:], dtype=np.float32)
        except Exception:
            rpe_hist = np.zeros(0, dtype=np.float32)
        try:
            ent_hist = np.asarray(getattr(core, "entropy_history", [])[-32:], dtype=np.float32)
        except Exception:
            ent_hist = np.zeros(0, dtype=np.float32)

        # 툴콜/버스 이벤트 통계(있으면)
        tool_counts = np.zeros(8, dtype=np.float32)
        try:
            # 최근 N개 툴콜 로그에서 성공/실패/지연을 요약 (핵심 통계만)
            logs = list(getattr(core, "tool_logs", [])[-64:])
            succ = sum(1 for L in logs if L.get("ok"))
            fail = sum(1 for L in logs if not L.get("ok"))
            lat = np.array([float(L.get("latency_ms", 0.0)) for L in logs], dtype=np.float32)
            tool_counts[0] = float(succ)
            tool_counts[1] = float(fail)
            tool_counts[2] = float(np.mean(lat) if lat.size else 0.0)
            tool_counts[3] = float(np.std(lat) if lat.size else 0.0)
            # 간단 토픽/툴 태그 상위 빈도(최대 4개만 카운팅처럼)
            tags = {}
            for L in logs:
                t = str(L.get("tool", "unk"))[:12]
                tags[t] = tags.get(t, 0) + 1
            top = sorted(tags.items(), key=lambda x: x[1], reverse=True)[:4]
            for i, (_, v) in enumerate(top, start=4):
                tool_counts[i] = float(v)
        except Exception:
            pass

        bus_counts = np.zeros(6, dtype=np.float32)
        try:
            # MessageBus 상위 키/타입/지연 등 요약 (있으면)
            bus = getattr(core, "bus", None)
            if bus is not None:
                # 안전하게 접근 (top_keys()가 있으면 사용)
                if hasattr(bus, "top_keys"):
                    klist = list(bus.top_keys())[:4]
                    for i, _ in enumerate(klist):
                        bus_counts[i] = 1.0  # 존재 플래그
                if hasattr(bus, "latency_ms"):
                    bus_counts[4] = float(getattr(bus, "latency_ms", 0.0))
                if hasattr(bus, "depth"):
                    bus_counts[5] = float(getattr(bus, "depth", 0.0))
        except Exception:
            pass

        # ---- 패널 구성: 각기 다른 의미층을 D차 벡터로 ----
        stability_panel = _to_panel([
            ctx["stability"], ctx["delta_hat"], ctx["energy.activation"], ctx["energy.ratio"], ctx["unity"],
            ctx["qualia.arousal"], ctx["qualia.valence"], ctx["qualia.entropy"]
        ])
        topo_panel = _to_panel([
            ctx["vision.contrast"], ctx["vision.entropy"], ctx["vision.edge_density"], ctx["vision.depth_cue"],
            ctx["phi.last"], ctx["phi.delta"], ctx["phi.mean10"], ctx["vision.r_mean"]
        ])
        time_panel = _to_panel(np.concatenate([
            # 최근 시계열 2종을 간단히 평균/분산/최댓값 등 요약 + 원본 일부
            np.array([np.mean(rpe_hist) if rpe_hist.size else 0.0,
                      np.std(rpe_hist) if rpe_hist.size else 0.0,
                      np.max(rpe_hist) if rpe_hist.size else 0.0,
                      np.mean(ent_hist) if ent_hist.size else 0.0,
                      np.std(ent_hist) if ent_hist.size else 0.0,
                      np.max(ent_hist) if ent_hist.size else 0.0], dtype=np.float32),
            rpe_hist[-8:] if rpe_hist.size >= 8 else np.zeros(8, dtype=np.float32),
            ent_hist[-8:] if ent_hist.size >= 8 else np.zeros(8, dtype=np.float32)
        ], axis=0))
        tool_panel = _to_panel(tool_counts)
        bus_panel = _to_panel(bus_counts)

        # (옵션) 주제/토픽 패널: 간단히 최근 언어 임베딩/키워드 요약이 있으면 사용
        try:
            lang = getattr(core, "language_embed", None)
            topic_panel = _to_panel(lang if isinstance(lang, np.ndarray) else np.zeros(D, dtype=np.float32))
        except Exception:
            topic_panel = _to_panel(np.zeros(D, dtype=np.float32))

        panels: list[np.ndarray] = [stability_panel, topo_panel, time_panel, tool_panel, bus_panel, topic_panel]
        return panels


# ---- Simple Skills manager from Bus patterns -> gate biases ----

    def _feature_scores(self, criterion: str = 'util'):
        """Compute per-feature scores for selection.
        Supported: 'util', 'entropy', 'mi' (mutual information). Fallback to uniform.
        Expects attributes like self.utilization, self.entropy, self.mi (list/array-like).
        """
        import numpy as _np
        n = len(getattr(self, 'active_features', [])) if hasattr(self, 'active_features') else 0
        if n == 0:
            return _np.zeros(0, dtype=float)
        def _to_arr(name):
            v = getattr(self, name, None)
            try:
                a = _np.asarray(v, dtype=float)
                if a.ndim == 1 and len(a) >= n:
                    return a[:n]
            except Exception:
                pass
            return None
        if criterion == 'util':
            a = _to_arr('utilization')
        elif criterion == 'entropy':
            a = _to_arr('entropy')
        elif criterion == 'mi':
            a = _to_arr('mi')
        else:
            a = None
        if a is None:
            a = _np.ones(n, dtype=float)
        # sanitize
        a = _np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return a

    def shrink(self, spec: dict | None = None):
        """Prune low-utility or redundant features.
        spec: {'target': int, 'min_keep': int, 'criterion': 'util'|'entropy'|'mi'}
        Behavior: keeps top-K by score; prunes others (active_features, weights, and aligned tensors).
        """
        import numpy as _np
        spec = spec or {}
        crit = spec.get('criterion', 'util')
        scores = self._feature_scores(crit)
        n = int(scores.shape[0])
        if n <= 1:
            return
        target = int(spec.get('target', max(1, n // 2)))
        min_keep = int(spec.get('min_keep', min(8, n)))
        k = max(min_keep, min(target, n))
        ranks = list(_np.argsort(-scores))  # desc
        keep_idx = set(int(i) for i in ranks[:k])

        # helpers
        def _filter_list(lst):
            return [v for i, v in enumerate(lst) if i in keep_idx]
        def _filter_2d(mat):
            A = _np.asarray(mat)
            if A.ndim == 2 and A.shape[0] >= n:
                return A[ranks[:k]].tolist()
            return mat

        # apply to common fields
        if hasattr(self, 'active_features') and isinstance(self.active_features, list):
            self.active_features = _filter_list(self.active_features)
        if hasattr(self, 'features') and isinstance(self.features, list):
            self.features = _filter_list(self.features)
        if hasattr(self, 'weights') and getattr(self, 'weights', None) is not None:
            self.weights = _filter_2d(self.weights)
        # optional aligned arrays
        for name in ('utilization', 'entropy', 'mi'):
            if hasattr(self, name):
                arr = getattr(self, name)
                try:
                    a = _np.asarray(arr)
                    if a.ndim == 1 and len(a) >= n:
                        setattr(self, name, a[ranks[:k]].tolist())
                except Exception:
                    pass
    def merge(self, spec: dict | None = None):
        """
        Merge highly similar features (cosine sim on weight rows) and merge aligned stats.
        spec: {'sim_threshold': float(0,1), 'max_pairs': int}
        """
        import numpy as _np
        spec = spec or {}
        # Adaptive, data-driven threshold selection (no magic number):
        # If 'intent' provided (0..1), pick target number of merges m_target = ceil(intent * n/2),
        # and set threshold to achieve approximately m_target non-overlapping pairs with highest cosine similarity.
        intent = float(spec.get('intent', -1.0))
        adaptive_pairs = None
        adaptive_thr = None
        if intent >= 0.0:
            intent = max(0.0, min(1.0, intent))

        thr = float(spec.get('sim_threshold', 0.95))
        max_pairs = int(spec.get('max_pairs', 128))
        W = getattr(self, 'weights', None)
        if W is None:
            return
        A = _np.asarray(W, dtype=float)
        if A.ndim != 2 or A.shape[0] < 2:
            return
        n = A.shape[0]
        counts = getattr(self, 'feat_counts', None)
        if counts is None or len(counts) < n:
            counts = [1.0] * n
        counts = _np.asarray(counts, dtype=float)[:n]

        def _row_norm(x):
            nrm = _np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
            return x / nrm
        X = _row_norm(A)
        S = X @ X.T

        merged = _np.zeros(n, dtype=bool)
        new_rows = []
        new_counts = []
        map_keep = []
        pairs = 0
        i = 0
        while i < n and pairs < max_pairs:
            if merged[i]:
                i += 1; continue
            j = int(_np.argmax(S[i]))
            if i != j and S[i, j] >= thr and not merged[j]:
                c1, c2 = counts[i], counts[j]
                w = (c1 * A[i] + c2 * A[j]) / (c1 + c2 + 1e-12)
                new_rows.append(w); new_counts.append(c1 + c2)
                merged[i] = True; merged[j] = True
                map_keep.append((i, j)); pairs += 1
            else:
                new_rows.append(A[i]); new_counts.append(counts[i])
                merged[i] = True; map_keep.append((i,))
            i += 1
        A2 = _np.vstack(new_rows)
        self.weights = A2.tolist()
        self.feat_counts = _np.asarray(new_counts, dtype=float).tolist()

        k = A2.shape[0]
        def _realign_1d(name, reducer='mean'):
            if not hasattr(self, name): return
            arr = _np.asarray(getattr(self, name))
            out = []
            for keep in map_keep:
                if len(keep) == 2:
                    i,j = keep
                    if reducer == 'sum':
                        out.append(float(arr[i]) + float(arr[j]))
                    elif reducer == 'max':
                        out.append(float(max(arr[i], arr[j])))
                    else:
                        out.append(float(0.5*(arr[i]+arr[j])))
                else:
                    out.append(float(arr[keep[0]]))
            setattr(self, name, out[:k])

        for nm in ('utilization', 'entropy', 'mi'):
            _realign_1d(nm, reducer='mean')

        for name in ('active_features', 'features'):
            if hasattr(self, name) and isinstance(getattr(self, name), list):
                lst = getattr(self, name)
                out = []
                for keep in map_keep:
                    out.append(lst[keep[0]] if len(keep)==1 else f"{lst[keep[0]]}+{lst[keep[1]]}")
                setattr(self, name, out[:k])

        def _row_norm(x):
            n = _np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
            return x / n
        X = _row_norm(A)
        S = X @ X.T  # cosine
        n = S.shape[0]
        merged = _np.zeros(n, dtype=bool)
        new_rows = []
        pairs = 0
        i = 0
        while i < n and pairs < max_pairs:
            if merged[i]:
                i += 1
                continue
            # find best partner
            j = int(_np.argmax(S[i]))
            if i != j and S[i, j] >= thr and not merged[j]:
                # average
                new_rows.append(0.5 * (A[i] + A[j]))
                merged[i] = True; merged[j] = True
                pairs += 1
            else:
                # keep single
                new_rows.append(A[i])
                merged[i] = True
            i += 1
        A2 = _np.vstack(new_rows)
        self.weights = A2.tolist()
        # if we track active_features/features, truncate to new size
        k = A2.shape[0]
        if hasattr(self, 'active_features') and isinstance(self.active_features, list):
            self.active_features = self.active_features[:k]
        if hasattr(self, 'features') and isinstance(self.features, list):
            self.features = self.features[:k]
        # aligned arrays
        for name in ('utilization', 'entropy', 'mi'):
            if hasattr(self, name):
                arr = getattr(self, name)
                try:
                    a = _np.asarray(arr)
                    if a.ndim == 1 and len(a) >= k:
                        setattr(self, name, a[:k].tolist())
                except Exception:
                    pass

class SkillsManager:
    def __init__(self, window: int = 256, bias: float = 0.2):
        self.window = int(window)
        self.bias = float(bias)
        self._hist: deque = deque(maxlen=self.window)
        self._counts: Dict[Tuple[str, ...], int] = {}

    def observe(self, keys: List[str]) -> None:
        try:
            keys = [str(k) for k in keys if isinstance(k, str)]
            if not keys:
                return
            self._hist.append(tuple(sorted(set(keys))))
            # update counts for pairs and triples to capture co-activations
            ks = list(sorted(set(keys)))
            combos = []
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    combos.append((ks[i], ks[j]))
            if len(ks) >= 3:
                combos.append((ks[0], ks[1], ks[2]))
            for c in combos:
                self._counts[c] = self._counts.get(c, 0) + 1
        except Exception:
            pass

    def get_gate_biases(self, active: List[str]) -> Dict[str, float]:
        try:
            if not active:
                return {}
            s = set(active)
            # find strongest matching combo in counts
            best_c = None
            best_v = 0
            for c, v in self._counts.items():
                if set(c).issubset(s) and v > best_v:
                    best_v = v
                    best_c = c
            if not best_c:
                return {}
            # assign a small positive bias to specs in the best combo
            return {k: self.bias for k in best_c if k in s}
        except Exception:
            return {}


# ---- Shared representation (whitening + light linear) used by Policy/SDM ----
class SharedRepresentation:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.mu = np.zeros((self.dim,), dtype=np.float32)
        self.s2 = np.zeros((self.dim,), dtype=np.float32)
        self.n = 0
        # small linear residual (identity init)
        self.W = np.eye(self.dim, dtype=np.float32)
        self.b = np.zeros((self.dim,), dtype=np.float32)

    def update_stats(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size != self.dim:
            return
        self.n += 1
        delta = x - self.mu
        self.mu += delta / max(1, self.n)
        self.s2 += delta * (x - self.mu)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.size != self.dim:
            # pad/truncate deterministically
            out = np.zeros((self.dim,), dtype=np.float32)
            take = min(x.size, self.dim)
            if take > 0:
                out[:take] = x[:take]
            x = out
        var = self.s2 / max(1, self.n - 1)
        s = np.sqrt(np.maximum(var, 1e-6))
        z = (x - self.mu) / s
        y = (self.W @ z) + self.b
        return y.astype(np.float32)


    def auto_tune_embed(self, X_window: np.ndarray, target_ev: float = 0.98, cap: int = 512) -> int:
        """Adjust embed_dim to the smallest m that explains target_ev of variance.

        X_window is [T, max_dim] raw (pre-normalization) feature matrix.
        Returns the new embed_dim actually set (clamped to [4, cap]).
        """
        try:
            X = np.asarray(X_window, dtype=np.float32)
            if X.ndim != 2 or X.shape[1] <= 0:
                return int(self.embed_dim)
            C = np.cov(X, rowvar=False)
            s = np.sort(np.maximum(np.linalg.eigvalsh(C), 0.0))[::-1]
            if s.size == 0 or float(np.sum(s)) <= 0:
                return int(self.embed_dim)
            cs = np.cumsum(s)
            thresh = float(target_ev) * float(cs[-1])
            m = int(np.searchsorted(cs, thresh) + 1)
            m = max(4, min(int(cap), int(m)))
            self.embed_dim = int(m)
            return int(m)
        except Exception:
            return int(self.embed_dim)

    # --------- Advanced autodiscovery / pruning / logging ---------
    def set_log_paths(self, outdir: str) -> None:
        try:
            os.makedirs(outdir, exist_ok=True)
        except Exception:
            pass
        self._provenance_path = os.path.join(outdir, 'feature_provenance.jsonl')
        self._metrics_path = os.path.join(outdir, 'metrics.jsonl')

    def _log_jsonl(self, path: Optional[str], rec: Dict[str, Any]) -> None:
        if not path:
            return
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def autodiscover(self, core: Any, controller: Any, trials: int = 8, dim_budget: int = 32,
                     score_thresh: float = 0.02, lambda_size: float = 1e-5) -> List[FeatureSpec]:
        accepted: List[FeatureSpec] = []
        try:
            base = self.build(core)
        except Exception:
            base = np.zeros((self.max_dim,), dtype=np.float32)
        # ask controller for candidates
        try:
            cands = controller.decide(base, k=trials)
        except Exception:
            cands = []
        added = 0
        for spec in cands:
            score, meta = self._score_spec(core, base, spec)
            cost = float(max(1, getattr(spec, 'produced_dim', 1)))
            score_adj = float(score - (lambda_size * cost))
            if score_adj >= float(score_thresh) and (added + cost) <= float(dim_budget):
                try:
                    self.register_spec(spec)
                    accepted.append(spec)
                    added += int(cost)
                    spec_id = hashlib.blake2b(json.dumps({'source':spec.source, 'packer':spec.packer, 'params':spec.params}, default=str).encode('utf-8'), digest_size=8).hexdigest()
                    # begin AB trial for this spec
                    try:
                        self.start_ab_trial(spec.name, epsilon=0.3, trial_steps=1000)
                    except Exception:
                        pass
                    self._log_jsonl(self._provenance_path, {
                        'ts': int(getattr(core, 't', 0)), 'event': 'spec_adopt', 'spec_id': spec_id,
                        'name': spec.name, 'source': str(spec.source), 'packer': spec.packer,
                        'params': spec.params, 'produced_dim': int(spec.produced_dim),
                        'score': float(score), 'mi': float(meta.get('mi', 0.0)),
                        'fisher': float(meta.get('fi', 0.0)), 'red': float(meta.get('red', 0.0)),
                        'cost': float(cost), 'reason': 'autodiscover'} )
                except Exception:
                    pass
            # feedback to controller
            try:
                controller.observe(reward=float(max(0.0, score_adj)), action_meta=meta)
            except Exception:
                pass
        return accepted

    def _score_spec(self, core: Any, base_vec: np.ndarray, spec: "FeatureSpec") -> Tuple[float, Dict[str, Any]]:
        # 1) extract and pack
        try:
            val = None
            if isinstance(spec.source, str):
                # use ctx-like accessor if available
                val = getattr(core, 'get_ctx', lambda k: None)(spec.source) if hasattr(core, 'get_ctx') else None
                if val is None:
                    val = getattr(core, spec.source.replace('.', '_'), None)
            elif callable(spec.source):
                val = spec.source(core)
        except Exception:
            val = None
        outp = np.asarray(self._pack_spec(spec, val), dtype=np.float32).ravel()
        if outp.size == 0:
            return 0.0, {'reason': 'empty'}
        # 2) assemble with extra
        aug = self._assemble_with_extra(base_vec, outp)
        # 3) compute proxies
        mi = self._mi_with_advantage(core, outp)
        fi = self._fisher_wrt_policy(core, base_vec, aug)
        red = self._redundancy_vs_existing(base_vec, outp)
        score = float(0.7 * mi + 0.3 * fi - 0.5 * red)
        return score, {'mi': float(mi), 'fi': float(fi), 'red': float(red), 'spec': spec.name}

    def _pack_spec(self, spec: "FeatureSpec", val: Any) -> np.ndarray:
        try:
            if spec.packer == 'scalar':        return pack_scalar(val, spec.params)
            if spec.packer == 'stats_sample':  return pack_stats_sample(val, spec.params)
            if spec.packer == 'spatial_pool':  return pack_spatial_pool(val, spec.params)
            if spec.packer == 'learned_proj':  return pack_learned_proj(val, spec.params)
        except Exception:
            pass
        return np.zeros((max(1, getattr(spec, 'produced_dim', 1)),), dtype=np.float32)

    def _assemble_with_extra(self, base: np.ndarray, extra: np.ndarray) -> np.ndarray:
        out = np.zeros_like(base, dtype=np.float32)
        L = min(base.size, out.size)
        if L > 0:
            out[:L] = base[:L].astype(np.float32)
        eL = min(extra.size, out.size - L)
        if eL > 0:
            out[L:L+eL] = extra[:eL].astype(np.float32)
        z = self._norm(out) * self.input_mask
        return np.clip(z, -3.0, 3.0)

    def _mi_with_advantage(self, core: Any, x: np.ndarray) -> float:
        # Advantage buffer from policy end_batch (normalized)
        try:
            adv = getattr(getattr(core, 'policy', None), '_last_adv_buf', None)
            if adv is None or len(adv) < 8:
                return 0.0
            a = np.asarray(adv, dtype=np.float32).ravel()
            x = np.asarray(x, dtype=np.float32).ravel()
            L = min(a.size, x.size)
            if L < 8:
                return 0.0
            x = (x[:L] - float(x[:L].mean())) / (float(x[:L].std()) + 1e-6)
            a = (a[:L] - float(a[:L].mean())) / (float(a[:L].std()) + 1e-6)
            rho = float(np.dot(x, a) / (L + 1e-6))
            rho = float(np.clip(rho, -0.999, 0.999))
            return float(-0.5 * np.log(1.0 - rho * rho))
        except Exception:
            return 0.0

    def _fisher_wrt_policy(self, core: Any, base: np.ndarray, aug: np.ndarray) -> float:
        try:
            p = getattr(core, 'policy', None)
            if p is None:
                return 0.0
            mu_b, _ = p.forward(base)
            mu_a, _ = p.forward(aug)
            d = np.asarray(mu_a, np.float32) - np.asarray(mu_b, np.float32)
            return float(np.sqrt((d * d).mean()))
        except Exception:
            return 0.0

    def _redundancy_vs_existing(self, base: np.ndarray, extra: np.ndarray) -> float:
        try:
            X = base.astype(np.float32).reshape(1, -1)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            axes = Vt[:8]
            e = extra.astype(np.float32)
            n = float(np.linalg.norm(e) + 1e-6)
            e = e / n
            sims = [abs(float(np.dot(e, ax) / (np.linalg.norm(ax) + 1e-6))) for ax in axes]
            return float(max(sims) if sims else 0.0)
        except Exception:
            return 0.5

    def start_ab_trial(self, spec_name: str, epsilon: float = 0.3, trial_steps: int = 1000) -> None:
        self._ab_trials[spec_name] = {'active': True, 'epsilon': float(epsilon), 'steps_left': int(trial_steps),
                                      'treat_rewards': [], 'ctrl_rewards': []}

    def ab_update(self, reward: float, applied_names: List[str], kl: float = 0.0, clipped: bool = False) -> None:
        # Update running AB metrics and conclude trials when steps exhausted
        for name, tr in list(self._ab_trials.items()):
            if not tr.get('active', False):
                continue
            is_treat = name in applied_names
            if is_treat:
                tr['treat_rewards'].append(float(reward))
            else:
                tr['ctrl_rewards'].append(float(reward))
            tr['steps_left'] = int(tr.get('steps_left', 0)) - 1
            if tr['steps_left'] <= 0:
                # decide adoption
                try:
                    t = np.asarray(tr['treat_rewards'], dtype=np.float32)
                    c = np.asarray(tr['ctrl_rewards'], dtype=np.float32)
                    if t.size >= 8 and c.size >= 8:
                        d = float(t.mean() - c.mean())
                        noise = float(np.percentile(np.abs(c - c.mean()), 95) + 1e-6)
                        approve = (d / noise) > 0.5 and (not clipped) and (kl <= 0.02 + 1e-6)
                    else:
                        approve = False
                except Exception:
                    approve = False
                # finalize
                self._ab_trials[name]['active'] = False
                if not approve:
                    # retire spec immediately
                    for s in self.specs:
                        if s.name == name:
                            s.active = False
                    # mask range
                    rng = self._ranges.get(name)
                    if rng is not None:
                        i0, i1 = rng
                        self.input_mask[i0:i1] = 0.0
                    self._log_jsonl(self._provenance_path, {'ts': int(time.time()), 'event': 'spec_retire', 'spec_id': name, 'name': name, 'reason': 'ab_reject'})

    def prune(self, k_steps: int = 1024, mi_floor: float = 0.005, red_ceiling: float = 0.9, psi_thresh: float = 0.2) -> List[str]:
        dead: List[str] = []
        # single-step approximation using last built vector and current adv
        base = self._last_built_z if self._last_built_z is not None else np.zeros((self.max_dim,), dtype=np.float32)
        for name, (i0, i1) in list(self._ranges.items()):
            seg = base[i0:i1] if i1 > i0 else np.zeros((1,), dtype=np.float32)
            mi = self._mi_with_advantage(getattr(self, '_core_for_prune', None) or None, seg)
            red = self._redundancy_vs_existing(base, seg)
            psi = 0.0  # placeholder drift metric; requires distribution buffers
            if (mi < mi_floor) or (red > red_ceiling) or (psi > psi_thresh):
                dead.append(name)
        for name in dead:
            for s in self.specs:
                if s.name == name:
                    s.active = False
            i0, i1 = self._ranges.get(name, (0, 0))
            if i1 > i0:
                self.input_mask[i0:i1] = 0.0
            self._log_jsonl(self._provenance_path, {'ts': int(time.time()), 'event': 'spec_retire', 'spec_id': name, 'name': name, 'reason': 'prune'})
        return dead

    def grow(self, new_max_dim: int, new_embed_dim: int | None = None) -> None:
        """Grow the internal buffers to accommodate a larger feature vector.

        Pads mu, s2, and input_mask preserving existing statistics. If new_embed_dim
        is provided, update embed_dim (but ensure embed_dim <= max_dim).
        """
        try:
            new_max_dim = int(new_max_dim)
            if new_max_dim <= self.max_dim:
                return
            old = int(self.max_dim)
            add = new_max_dim - old
            self.mu = np.concatenate([self.mu, np.zeros(add, dtype=np.float32)])
            self.s2 = np.concatenate([self.s2, np.ones(add, dtype=np.float32)])
            self.input_mask = np.concatenate([self.input_mask, np.ones(add, dtype=np.float32)])
            # embed_dim must remain sensible
            if new_embed_dim is not None:
                self.embed_dim = int(min(new_embed_dim, new_max_dim))
            # update max_dim last
            self.max_dim = int(new_max_dim)
        except Exception:
            # best-effort fallback: re-init to zeros
            self.max_dim = int(new_max_dim)
            self.input_mask = np.ones(self.max_dim, dtype=np.float32)
            self.mu = np.zeros(self.max_dim, dtype=np.float32)
            self.s2 = np.ones(self.max_dim, dtype=np.float32)
    
    # --- FeatureSpec registry helpers ---
    def _register_default_specs(self) -> None:
        """Register default specs that mirror the legacy feature ordering.

        Each spec.source can be a context key (string) that build() computes once per call.
        """
        self.specs = []
        # vision scalars
        self.register_spec(FeatureSpec(name='vision.r_mean', source='vision.r_mean', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='vision.contrast', source='vision.contrast', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='vision.entropy', source='vision.entropy', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='vision.edge_density', source='vision.edge_density', packer='scalar', produced_dim=1))
        # depth cue (may be array) - use stats+samples
        self.register_spec(FeatureSpec(name='vision.depth_cue', source='vision.depth_cue', packer='stats_sample', params={'samples':8}, produced_dim=12))
        # retina spatial summary
        self.register_spec(FeatureSpec(name='vision.retina', source='vision.retina', packer='spatial_pool', params={'grid':4, 'pool_type':'mean'}, produced_dim=4 + 4))
        # phi trends
        self.register_spec(FeatureSpec(name='phi.last', source='phi.last', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='phi.mean10', source='phi.mean10', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='phi.delta', source='phi.delta', packer='scalar', produced_dim=1))
        # internal scalars
        self.register_spec(FeatureSpec(name='self.meta_awareness', source='self.meta_awareness', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='self.meta_confidence', source='self.meta_confidence', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='energy.activation', source='energy.activation', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='energy.ratio', source='energy.ratio', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='unity', source='unity', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='stability', source='stability', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='delta_hat', source='delta_hat', packer='scalar', produced_dim=1))
        # qualia
        self.register_spec(FeatureSpec(name='qualia.arousal', source='qualia.arousal', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='qualia.valence', source='qualia.valence', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='qualia.entropy', source='qualia.entropy', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='qualia.engagement', source='qualia.engagement', packer='scalar', produced_dim=1))
        self.register_spec(FeatureSpec(name='qualia.frustration', source='qualia.frustration', packer='scalar', produced_dim=1))

    def register_spec(self, spec: FeatureSpec) -> None:
        try:
            # Ensure deterministic ordering: append at end
            self.specs.append(spec)
        except Exception:
            pass

    def unregister_spec(self, name: str) -> bool:
        try:
            keep = [s for s in self.specs if s.name != name]
            if len(keep) != len(self.specs):
                self.specs = keep
                return True
        except Exception:
            pass
        return False

    def compute_total_produced_dim(self) -> int:
        try:
            return int(sum(getattr(s, 'produced_dim', 1) for s in self.specs if s.active))
        except Exception:
            return 0
    
    # === AUTONOMOUS GROWTH METHODS ===
    
    def measure_complexity(self, core: Any) -> Dict[str, float]:
        """현재 시스템 복잡도 측정"""
        metrics = {
            'feature_utilization': 0.0,
            'variance_diversity': 0.0,
            'information_density': 0.0,
            'processing_load': 0.0
        }
        
        try:
            # Feature utilization: how many dimensions are actually used
            if self._last_built_z is not None:
                active_dims = np.sum(np.abs(self._last_built_z) > 0.01)
                metrics['feature_utilization'] = active_dims / max(1, self.max_dim)
            
            # Variance diversity: how varied are the features
            if len(self.complexity_metrics) > 10:
                recent_variance = [m.get('variance_diversity', 0) for m in list(self.complexity_metrics)[-10:]]
                metrics['variance_diversity'] = float(np.mean(recent_variance))
            else:
                var_per_dim = self.s2 / max(1, self._count - 1)
                metrics['variance_diversity'] = float(np.mean(var_per_dim[:min(50, len(var_per_dim))]))
            
            # Information density: entropy of feature distribution
            if self._last_built_z is not None:
                abs_vals = np.abs(self._last_built_z) + 1e-8
                probs = abs_vals / np.sum(abs_vals)
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                max_entropy = np.log(len(probs))
                metrics['information_density'] = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            
            # Processing load: ratio of active specs to capacity
            if hasattr(self, 'specs'):
                active_specs = sum(1 for s in self.specs if s.active)
                metrics['processing_load'] = active_specs / max(1, len(self.specs))
        
        except Exception as e:
            logging.debug(f"Complexity measurement error: {e}")
        
        self.complexity_metrics.append(metrics)
        return metrics
    
    def should_grow(self, current_time: float, core: Any) -> Tuple[bool, str]:
        """성장 타이밍 자동 결정"""
        
        # Cooldown check
        if current_time - self.last_growth_time < self.growth_cooldown:
            return False, "cooldown"
        
        # Measure current complexity
        complexity = self.measure_complexity(core)
        
        # Growth trigger conditions
        reasons = []
        
        # 1. High feature utilization → need more capacity
        if complexity['feature_utilization'] > self.adaptive_growth_threshold:
            reasons.append(f"high_utilization({complexity['feature_utilization']:.2f})")
        
        # 2. High information density → complex enough to expand
        if complexity['information_density'] > 0.8:
            reasons.append(f"high_density({complexity['information_density']:.2f})")
        
        # 3. Sustained high processing load
        if complexity['processing_load'] > 0.85:
            reasons.append(f"high_load({complexity['processing_load']:.2f})")
        
        # 4. Stability check - don't grow if system is unstable
        if hasattr(core, 'self_model') and hasattr(core.self_model, 'belief_stability'):
            if core.self_model.belief_stability < 0.4:
                return False, "unstable_system"
        
        # 5. Energy check - don't grow if low energy
        if hasattr(core, 'energy_ctrl') and hasattr(core.energy_ctrl, 'activation_level'):
            if core.energy_ctrl.activation_level < 0.3:
                return False, "low_energy"
        
        # Decide
        if len(reasons) >= 2:
            return True, " | ".join(reasons)
        
        return False, "insufficient_triggers"
    
    def adaptive_grow(self, current_time: float, core: Any, apply: bool = True) -> Optional[Dict[str, Any]]:
        """조건이 충족되면 자동으로 성장"""
        should, reason = self.should_grow(current_time, core)
        
        if not should:
            return None
        
        # Determine growth amount based on complexity
        complexity = self.measure_complexity(core)
        
        # Base growth: 25% increase
        growth_factor = 1.25
        
        # Adaptive adjustment
        if complexity['feature_utilization'] > 0.9:
            growth_factor = 1.5  # Aggressive growth
        elif complexity['feature_utilization'] < 0.7:
            growth_factor = 1.1  # Conservative growth
        
        new_max_dim = int(self.max_dim * growth_factor)
        new_embed_dim = int(self.embed_dim * 1.2)
        
        # Safety limits
        new_max_dim = min(new_max_dim, 2048)
        new_embed_dim = min(new_embed_dim, 256)
        
        # Execute growth
        old_max_dim = self.max_dim
        old_embed_dim = self.embed_dim
        if bool(apply):
            self.grow(new_max_dim, new_embed_dim)
        
        growth_report = {
            'timestamp': current_time,
            'old_max_dim': old_max_dim,
            'new_max_dim': int(self.max_dim if bool(apply) else new_max_dim),
            'old_embed_dim': old_embed_dim,
            'new_embed_dim': int(self.embed_dim if bool(apply) else new_embed_dim),
            'reason': reason,
            'complexity': complexity,
            'growth_factor': growth_factor,
            'applied': bool(apply),
        }
        
        if bool(apply):
            self.growth_history.append(growth_report)
            self.last_growth_time = current_time
        
        logging.info(f"FeatureBank adaptive growth: {old_max_dim} → {self.max_dim} | Reason: {reason}")
        
        return growth_report





class MetaFeatureController:
    """Learned controller that proposes FeatureSpec templates.

    This class uses a small MLP to produce logits for categorical choices
    (packer type and source index) plus a few continuous parameters which
    are interpreted per-packer. It implements decide(features,k) -> List[FeatureSpec]
    and observe(reward, action_meta) to update the internal policy via
    PolicyMLP.record/end_batch.
    """
    def __init__(self, in_dim: int = 128, hidden: int = 128, rng=None, packer_candidates: Optional[List[str]] = None, source_candidates: Optional[List[str]] = None, lr: float = 1e-3, sigma: float = 0.6):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.rng = rng or np.random.default_rng()
        self.packer_candidates = packer_candidates or ['scalar', 'stats_sample', 'spatial_pool', 'learned_proj']
        # default sources (will be filtered by core's ctx keys)
        self.source_candidates = source_candidates or ['vision.retina', 'vision.depth_cue', 'self.meta_awareness', 'qualia.entropy']
        # policy network outputs a vector we will split: packer logits, source logits, continuous params
        self.num_cat = len(self.packer_candidates) + len(self.source_candidates)
        # continuous params: [samples_norm, grid_norm, produced_scale]
        self.num_cont = 3
        self.out_dim = self.num_cat + self.num_cont
        self.policy = PolicyMLP(in_dim=self.in_dim, out_dim=self.out_dim, hidden=self.hidden, sigma=sigma, rng=self.rng)
        self.lr = float(lr)
        # autosize bandit state
        self.width_candidates = [0.25, 0.5, 0.75, 1.0]
        self.width_stats = {w: {'n': 0, 'mean': 0.0} for w in self.width_candidates}
        self.current_width = 1.0
        self.H_max = int(hidden)
        self.target_policy = None  # will be set by core to point to main PolicyMLP

    def decide(self, features: np.ndarray, k: int = 1) -> List[FeatureSpec]:
        feats = np.asarray(features, dtype=np.float32).reshape(-1)
        proposals: List[FeatureSpec] = []
        for _ in range(k):
            mu, _ = self.policy.forward(feats)
            mu = np.asarray(mu, dtype=np.float32).ravel()
            # softmax for categorical parts
            cats = mu[:self.num_cat].astype(np.float32)
            # split
            p_logits = cats[:len(self.packer_candidates)]
            s_logits = cats[len(self.packer_candidates):len(self.packer_candidates) + len(self.source_candidates)]
            p_idx = int(np.argmax(p_logits)) if p_logits.size else 0
            s_idx = int(np.argmax(s_logits)) if s_logits.size else 0
            packer = self.packer_candidates[p_idx]
            source = self.source_candidates[s_idx]
            cont = mu[self.num_cat:self.num_cat + self.num_cont]
            # interpret continuous params
            samples = max(1, int(np.clip(8 + (cont[0] * 8), 1, 64)))
            grid = max(1, int(np.clip(4 + (cont[1] * 4), 1, 16)))
            produced_scale = float(np.clip(1.0 + cont[2], 0.1, 8.0))
            params: Dict[str, Any] = {}
            if packer == 'scalar':
                produced_dim = 1
            elif packer == 'stats_sample':
                params = {'samples': samples}
                produced_dim = 4 + samples
            elif packer == 'spatial_pool':
                params = {'grid': grid, 'pool_type': 'mean'}
                produced_dim = 4 + (grid if isinstance(grid, int) else int(np.prod(grid)))
            else:
                params = {'proj_scale': produced_scale}
                produced_dim = int(max(2, round(8 * produced_scale)))
            name = f'auto.learned.{packer}.{source.replace(".","_")}.{int(time.time()*1000)%100000}'
            spec = FeatureSpec(name=name, source=source, packer=packer, params=params, produced_dim=int(produced_dim), active=True)
            proposals.append(spec)
            # record last action vector for learning
            try:
                self.last_mu = mu.copy()
                self.last_act_vec = mu.copy()
            except Exception:
                self.last_mu = None
                self.last_act_vec = None
        return proposals

    def observe(self, reward: float, action_meta: Dict[str, Any]) -> None:
        # simple scalar reward to update policy: record (features->action) pairs
        try:
            feats = np.asarray(action_meta.get('features', np.zeros((self.in_dim,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            mu = action_meta.get('mu')
            act_vec = action_meta.get('act_vec')
            if mu is None or act_vec is None:
                return
            # record a synthetic sample using policy.sample semantics
            act = np.asarray(act_vec, dtype=np.float32).reshape(-1)
            # compute logp via policy.sample is somewhat involved; here we record and call end_batch with reward
            self.policy.record(feats, act, float(0.0), mu, float(reward))
            # small batch update
            self.policy.end_batch(gamma=0.99, kl_coeff=0.01, lr=self.lr)
        except Exception:
            pass

    # ----- autosize (bandit over width multipliers) -----
    def _pick_width(self, t: int) -> float:
        vals = []
        for w, st in self.width_stats.items():
            bonus = 0.0 if st['n'] == 0 else 0.5 * np.sqrt(np.log(max(2, t + 1)) / st['n'])
            vals.append((st['mean'] + bonus, w))
        vals.sort(key=lambda x: x[0], reverse=True)
        return float(vals[0][1])

    def autosize_step(self, perf: float, lambda_size: float = 1e-5) -> None:
        try:
            pol = self.target_policy
            if pol is None:
                return
            params = int(pol.W1.size + pol.W2.size)
            score = float(perf) - float(lambda_size) * float(params)
            w = float(self.current_width)
            st = self.width_stats[w]
            st['n'] += 1
            st['mean'] += (score - st['mean']) / st['n']
            # next width
            self.current_width = self._pick_width(st['n'])
            new_h = max(8, int(round(self.current_width * max(8, self.H_max))))
            pol.resize_hidden(new_h)
        except Exception:
            pass


class GrowthTrigger:
    """Binary trigger policy that learns when to request meta-proposals.

    Uses a small PolicyMLP producing a 1D Gaussian; action sampled and converted
    to binary (act>0 -> trigger). Trained via REINFORCE-style record/end_batch.
    """
    def __init__(self, in_dim: int = 64, hidden: int = 64, sigma: float = 1.0, rng=None, lr: float = 1e-3):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.rng = rng or np.random.default_rng()
        self.policy = PolicyMLP(in_dim=self.in_dim, out_dim=1, hidden=self.hidden, sigma=float(sigma), rng=self.rng)
        self.lr = float(lr)

    def decide(self, features: np.ndarray) -> tuple[bool, dict]:
        obs = np.asarray(features, dtype=np.float32).reshape(-1)
        try:
            act, logp, mu, v = self.policy.sample(obs)
            act = np.asarray(act).ravel()
            mu = np.asarray(mu).ravel()
            # binary decision
            trigger = bool(act[0] > 0.0)
            # record for learning (reward will be provided to end_batch later)
            try:
                self.policy.record(obs, act, float(logp), mu, 0.0)
            except Exception:
                pass
            meta = {'mu': mu.copy() if hasattr(mu, 'copy') else mu, 'act_vec': act.copy() if hasattr(act, 'copy') else act}
            return trigger, meta
        except Exception:
            return False, {}

    def observe(self, reward: float) -> None:
        try:
            # end_batch will update weights using recorded trajectory
            self.policy.end_batch(gamma=0.99, kl_coeff=0.01, lr=self.lr)
        except Exception:
            pass


class SelfDynamicsModel(nn.Module):
    """Tiny MLP predictor for next-step internal dynamics with multi-head outputs.

    PyTorch implementation.
    """
    def __init__(self, in_dim: int, hidden: int = 128, lr: float = 1e-3, rng=None, z_dim: int | None = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.lr = float(lr) # Handled by optimizer, but kept for interface
        self.rng = rng or np.random.default_rng()
        self.z_dim = int(z_dim) if z_dim is not None else None
        
        # Heads: delta_hat(1), stability(1), meta(1), reward(1) -> Total 4
        # Plus optional z_dim
        
        self.base = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.Tanh()
        )
        
        # Output Heads
        self.output_head = nn.Linear(self.hidden, 4) # 4 basics
        
        if self.z_dim is not None:
            self.z_head = nn.Linear(self.hidden, self.z_dim)
        else:
            self.z_head = None
            
        # Running Normalization (Manual using buffer since BatchNorm is tricky with single samples)
        self.register_buffer('mu', torch.zeros(self.in_dim))
        self.register_buffer('s2', torch.ones(self.in_dim))
        self.register_buffer('_n', torch.tensor(1.0))
        
        self.to(_default_torch_device())
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _update_stats(self, x: torch.Tensor):
        # Welford's online algorithm for batch
        # x: (B, dim)
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(0)
                batch_var = x.var(0, unbiased=False)
                batch_count = x.shape[0]
                
                delta = batch_mean - self.mu
                tot_count = self._n + batch_count
                
                new_mean = self.mu + delta * batch_count / tot_count
                m_a = self.s2 * self._n
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta**2 * self._n * batch_count / tot_count
                new_var = M2 / tot_count
                
                self.mu.copy_(new_mean)
                self.s2.copy_(new_var)
                self._n += batch_count

    def forward(self, x: np.ndarray | torch.Tensor) -> Dict[str, Any]:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(_default_torch_device())
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Normalize
        x_norm = (x - self.mu) / torch.sqrt(self.s2 + 1e-6)
        
        h = self.base(x_norm)
        y = self.output_head(h)
        
        out = {
            'delta_hat': y[0, 0].item(),
            'stability': y[0, 1].item(),
            'meta': y[0, 2].item(),
            'reward': y[0, 3].item(),
        }
        
        if self.z_head is not None:
             z_next = self.z_head(h)
             out['z_next'] = z_next.detach().cpu().numpy().squeeze(0)
             
        return out

    def train_batch(self, X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None) -> float:
        """X shape (B, in), Y shape (B, 4 + z_dim) targets; w optional head weights (4,)"""
        self.train()
        
        X_t = torch.from_numpy(X).float().to(_default_torch_device())
        Y_t = torch.from_numpy(Y).float().to(_default_torch_device())
        
        if w is None:
            w_t = torch.tensor([1.0, 0.5, 0.5, 1.0], device=_default_torch_device())
        else:
            w_t = torch.from_numpy(w).float().to(_default_torch_device())
            
        # Update Stats
        self._update_stats(X_t)
        
        # Normalize Input
        X_norm = (X_t - self.mu) / torch.sqrt(self.s2 + 1e-6)
        
        # Forward
        h = self.base(X_norm)
        y_pred = self.output_head(h)
        
        # Basic Loss (first 4 dims)
        y_true = Y_t[:, :4]
        diff = (y_pred - y_true) * w_t
        loss = 0.5 * (diff ** 2).sum(1).mean()
        
        # Latent Loss
        if self.z_head is not None:
            z_pred = self.z_head(h)
            z_true = Y_t[:, 4:]
            if z_true.shape[1] == self.z_dim:
                z_loss = 0.25 * nn.MSELoss()(z_pred, z_true)
                loss = loss + z_loss
                
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path + ".pth")

    def resize(self, new_in_dim: int | None = None, new_z_dim: int | None = None) -> None:
        """Resize internal matrices to support a different input dim or latent z_dim."""
        
        # Input resize
        if new_in_dim is not None and new_in_dim != self.in_dim:
            old_base = self.base[0]
            new_base = nn.Linear(new_in_dim, self.hidden).to(_default_torch_device())
            
            with torch.no_grad():
                min_dim = min(self.in_dim, new_in_dim)
                new_base.weight[:, :min_dim] = old_base.weight[:, :min_dim]
                new_base.bias.copy_(old_base.bias)
                
            self.base[0] = new_base
            
            # Resize buffers
            old_mu = self.mu
            old_s2 = self.s2
            self.mu = torch.zeros(new_in_dim).to(_default_torch_device())
            self.s2 = torch.ones(new_in_dim).to(_default_torch_device())
            self.mu[:min_dim] = old_mu[:min_dim]
            self.s2[:min_dim] = old_s2[:min_dim]
            
            self.in_dim = new_in_dim

        # Output resize (z_dim)
        if new_z_dim is not None:
            if self.z_head is None:
                 self.z_head = nn.Linear(self.hidden, new_z_dim).to(_default_torch_device())
            elif new_z_dim != self.z_dim:
                old_z = self.z_head
                new_z = nn.Linear(self.hidden, new_z_dim).to(_default_torch_device())
                with torch.no_grad():
                    min_z = min(self.z_dim, new_z_dim)
                    new_z.weight[:min_z, :] = old_z.weight[:min_z, :]
                    new_z.bias[:min_z] = old_z.bias[:min_z]
                self.z_head = new_z
            self.z_dim = new_z_dim
            
        # Re-create optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def load(self, path: str) -> bool:
        try:
            # Try loading pytorch model
            self.load_state_dict(torch.load(path + ".pth"))
            return True
        except:
            return False

    def sample(self, obs: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        # Legacy stub
        return np.zeros(1), 0.0, np.zeros(1)

    def update(self, obs: np.ndarray, action: np.ndarray, reward: float, lr: float = 0.01):
        # Legacy single-step update, can just call train_batch with batch 1
        x = obs.reshape(1, -1)
        # We don't have target Y here easily if it's just RL update, skipping or implementing dummy
        pass

    def record(self, *args):
        pass

    def end_batch(self, *args, **kwargs):
        pass


class SimpleBanditEnv:
    """Minimal environment wrapper (extendable to grid) that exposes observation and accepts an action vector.

    The reward is shaped from internal signals by the core (passed in), keeping the API explicit for future swap.
    """
    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0

    def step(self, action_vec: np.ndarray, shaped_reward: float) -> float:
        # Here we simply return shaped reward; hook kept for future task wiring.
        self.t += 1
        return float(shaped_reward)



class IITPhiCalculator:

    def __init__(self, n_elements: int, rng_registry: Optional[RNGRegistry]=None, message_bus: Optional[MessageBus]=None):
        self.n_elements = n_elements
        # USHTS-based sparse CES (no dense TPM/CPT)
        self.ces = CauseEffectStructure(n_elements)
        self.current_mip = None
        self.mip_history: deque = deque(maxlen=256)
        self.phi_history: deque = deque(maxlen=256)
        self.mics_cache: Dict[str, float] = {}
        self.requires_structural_revision = False
        self.revision_reason = ''
        # RNG management: prefer injected RNGRegistry, fallback to a default
        self.rng_registry = rng_registry if rng_registry is not None else RNGRegistry(None)
        # Use a named child generator for phi computations
        try:
            self.rng = self.rng_registry.get('phi')
        except Exception:
            # Fallback to numpy's global RNG if registry fails
            self.rng = np.random.default_rng()
        # Internal streaming state for sparse transitions
        self._last_state_idx: Optional[int] = None
        self._updates: int = 0
        
        # MULTI-LOOP INFRASTRUCTURE
        self.message_bus = message_bus
        if self.message_bus:
            self.message_bus.register_module('phi_calculator')

    # --- Bridge methods delegating to CauseEffectStructure ---
    # Provide a stable IITPhiCalculator API while using the underlying sparse CES methods.
    def _repertoire_entropy(self, repertoire: np.ndarray) -> float:
        return self.ces._repertoire_entropy(repertoire)

    def _compute_joint_entropy(self, cause_rep: np.ndarray, effect_rep: np.ndarray) -> float:
        return self.ces._compute_joint_entropy(cause_rep, effect_rep)

    def _compute_integrated_information(self, cause_rep: np.ndarray, effect_rep: np.ndarray, elements: set, excluded: set) -> float:
        return self.ces._compute_integrated_information(cause_rep, effect_rep, elements, excluded)

    def _compute_partitioned_information(self, cause_rep: np.ndarray, effect_rep: np.ndarray, subset1: set, subset2: set) -> float:
        return self.ces._compute_partitioned_information(cause_rep, effect_rep, subset1, subset2)

    def _compute_partition_loss(self, cause_rep: np.ndarray, effect_rep: np.ndarray, subset1: set, subset2: set) -> float:
        return self.ces._compute_partition_loss(cause_rep, effect_rep, subset1, subset2)

    def _normalize_phi(self, phi: float, cause_rep: np.ndarray) -> float:
        max_possible_phi = np.log2(len(cause_rep)) * 2
        phi_normalized = phi / max(max_possible_phi, 1e-10)
        return float(np.clip(phi_normalized, 0.0, 1.0))

    def update_state(self, state: np.ndarray):
        """Consume a real-valued state vector, binarize, and record sparse transition.
        Converts state into a binary vector via median threshold, encodes to an integer index,
        and adds transition from the previous state index if available."""
        if state is None or len(state) == 0:
            return
        state = np.asarray(state)
        med = float(np.median(state))
        binary = (state > med).astype(np.int32)
        idx = self._state_to_index(binary)
        if self._last_state_idx is not None:
            self.ces.add(int(self._last_state_idx), int(idx), 1)
            self._updates += 1
            # opportunistic maintenance to keep WAL bounded
            if (self._updates % max(1, self.ces.cfg.cooldown_updates)) == 0:
                self.ces.compact()
        self._last_state_idx = int(idx)
        # health placeholder (no dense metrics)
        self._check_structural_health()

    def _check_structural_health(self):
        """Sparse health proxy: flag revision only if no transitions observed."""
        has_data = any(self.ces._rowsum.values()) if hasattr(self.ces, '_rowsum') else False
        self.requires_structural_revision = not has_data
        self.revision_reason = '' if has_data else 'no transitions observed'

    def compute_phi(self, cause_repertoire: Optional[np.ndarray]=None, effect_repertoire: Optional[np.ndarray]=None, state: Optional[np.ndarray]=None, method: str='simple') -> float:
        """Phi (integrated information) computation. If state is provided, it updates the internal state.
        If repertoires are not given, it derives them from USHTS at the current index."""
        if state is not None:
            self.update_state(state)
        # If repertoires not given, derive from USHTS at current index
        if (cause_repertoire is None or effect_repertoire is None) and self._last_state_idx is not None:
            src = int(self._last_state_idx)
            # effect: outgoing from src
            eff_ids, eff_counts = self._effect_from_ces(src)
            # cause: incoming into src
            cau_ids, cau_counts = self._cause_into_ces(src)
            # unify support set
            support = sorted(set(eff_ids).union(cau_ids))
            if not support:
                return 0.0
            id_to_pos = {sid: i for i, sid in enumerate(support)}
            eff_vec = np.zeros(len(support), dtype=np.float64)
            for sid, c in zip(eff_ids, eff_counts):
                eff_vec[id_to_pos[sid]] = c
            cau_vec = np.zeros(len(support), dtype=np.float64)
            for sid, c in zip(cau_ids, cau_counts):
                cau_vec[id_to_pos[sid]] = c
            # Dirichlet smoothing  
            alpha = float(getattr(self.ces.cfg, 'alpha', 0.3))
            V = max(1, len(support))
            effect_repertoire = (eff_vec + alpha) / (max(1e-12, eff_vec.sum() + alpha * V))
            cause_repertoire = (cau_vec + alpha) / (max(1e-12, cau_vec.sum() + alpha * V))
        if cause_repertoire is None or effect_repertoire is None:
            return 0.0
        if len(cause_repertoire) != len(effect_repertoire):
            # pad to equal length if needed
            L = max(len(cause_repertoire), len(effect_repertoire))
            cause_repertoire = np.pad(cause_repertoire, (0, L - len(cause_repertoire)))
            effect_repertoire = np.pad(effect_repertoire, (0, L - len(effect_repertoire)))
        phi = self.ces.compute_phi(
            cause_repertoire,
            effect_repertoire,
            method=method,
            state=state,
        )
        if np.isnan(phi) or np.isinf(phi):
            phi = 0.0
        self.phi_history.append(phi)
        
        # MULTI-LOOP: Broadcast phi update to all modules
        self._broadcast_phi_update(phi)
        
        return phi
    
    def _broadcast_phi_update(self, phi: float):
        """Broadcast phi value to all modules"""
        if not self.message_bus:
            return
        
        # Compute trend
        trend = 'stable'
        if len(self.phi_history) >= 10:
            recent_states= list(self.phi_history)[-10:]
            slope = np.mean(np.diff(recent_states))
            if slope > 0.001:
                trend = 'rising'
            elif slope < -0.001:
                trend = 'falling'
        
        msg = Message(
            source='phi_calculator',
            target='broadcast',
            type='phi_update',
            payload={
                'phi': float(phi),
                'trend': trend,
                'history_len': len(self.phi_history)
            },
            priority=0.6
        )
        self.message_bus.send(msg)

    def _state_to_index(self, binary_state: np.ndarray) -> int:
        # Map an arbitrary-length binary state to a stable 31-bit positive ID.
        # This avoids overflows like "int too big to convert" in downstream np.int32/ctypes paths.
        st = np.asarray(binary_state, dtype=np.uint8)
        if st.size < self.n_elements:
            pad = np.zeros(self.n_elements, dtype=np.uint8)
            pad[:st.size] = st
            st = pad
        elif st.size > self.n_elements:
            st = st[:self.n_elements]
        try:
            h = hashlib.blake2b(st.tobytes(), digest_size=8, person=b'M3_IIT_idx')
            val = int.from_bytes(h.digest()[:4], 'little') & 0x7FFFFFFF  # 31-bit positive
            return int(val)
        except Exception:
            # Fallback: modulo-2**31 of bit-vector dot-product
            vec = st.astype(np.int64)
            weights = (1 << np.arange(vec.size, dtype=np.int64))[::-1]
            return int(int(np.dot(vec, weights)) & 0x7FFFFFFF)

    def _effect_from_ces(self, src: int) -> tuple[list[int], list[int]]:
        items = self.ces.iter_row(int(src), mode="exact")
        ids = [int(d) for d, _ in items]
        counts = [int(c) for _, c in items]
        return ids, counts

    def _cause_into_ces(self, dst: int) -> tuple[list[int], list[int]]:
        """Compute incoming distribution into state dst by scanning CSR/Top with Top overriding CSR if overlap."""
        dst = int(dst)
        acc: Dict[int, int] = {}
        topk_rows = getattr(self.ces.topk, 'rows', {})
        csr_rows = getattr(self.ces.csr, 'rows', {})
        # First collect Top contributors (override)
        for src, rowmap in topk_rows.items():
            if dst in rowmap:
                acc[int(src)] = int(rowmap[dst])
        # Now add CSR where Top did not override
        for src, (idx, data) in csr_rows.items():
            if src in acc and acc[src] > 0:
                # already covered by Top 
                continue
            if idx.size == 0:
                continue
            pos = int(np.searchsorted(idx, np.int32(dst)))
            if pos < idx.size and int(idx[pos]) == dst:
                acc[int(src)] = int(data[pos])
        if not acc:
            return [], []
        ids = sorted(acc.keys())
        counts = [acc[i] for i in ids]
        return ids, counts

    def compute_mics(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Compute Maximally Irreducible Cause-Effect Structure (MICS) using USHTS data.
        Returns Phi (integrated information) and detailed repertoire analysis.
        """
        state_key = str(np.asarray(state).tobytes())
        
        # Check cache first
        if state_key in self.mics_cache:
            return {'phi': self.mics_cache[state_key], 'cached': True}
        
        # Update state in USHTS system
        self.update_state(state)
        if self._last_state_idx is None:
            return {'phi': 0.0, 'error': 'no previous state'}
        
        src = int(self._last_state_idx)
        
        # Extract cause and effect transitions from USHTS
        eff_ids, eff_counts = self._effect_from_ces(src)
        cau_ids, cau_counts = self._cause_into_ces(src)
        
        # Build support set (union of all involved states)
        support = sorted(set(eff_ids).union(cau_ids))
        
        # Construct repertoires
        if not support:
            # No transitions observed: default to maximum uncertainty
            cause_rep = np.array([0.5, 0.5], dtype=np.float64)
            effect_rep = np.array([0.5, 0.5], dtype=np.float64)
        else:
            # Map state IDs to repertoire positions
            id_to_pos = {sid: i for i, sid in enumerate(support)}
            
            # Build effect repertoire (what this state causes)
            eff_vec = np.zeros(len(support), dtype=np.float64)
            for sid, c in zip(eff_ids, eff_counts):
                eff_vec[id_to_pos[sid]] = float(c)
            
            # Build cause repertoire (what causes this state)
            cau_vec = np.zeros(len(support), dtype=np.float64)
            for sid, c in zip(cau_ids, cau_counts):
                cau_vec[id_to_pos[sid]] = float(c)
            
            # Dirichlet smoothing: add pseudocounts to prevent zero probabilities
            alpha = float(getattr(self.ces.cfg, 'alpha', 0.3))
            V = max(1, len(support))
            
            effect_rep = (eff_vec + alpha) / max(1e-12, eff_vec.sum() + alpha * V)
            cause_rep = (cau_vec + alpha) / max(1e-12, cau_vec.sum() + alpha * V)
        
        # Compute Phi using repertoire analysis
        phi = self.compute_phi(cause_rep, effect_rep, method='simple')
        
        # Analyze repertoire information content
        cause_entropy = float(self._repertoire_entropy(cause_rep))
        effect_entropy = float(self._repertoire_entropy(effect_rep))
        
        # Compute mutual information if possible
        mutual_info = 0.0
        if len(cause_rep) == len(effect_rep):
            joint_entropy = float(self._compute_joint_entropy(cause_rep, effect_rep))
            mutual_info = max(0.0, cause_entropy + effect_entropy - joint_entropy)
        
        # Build comprehensive MICS result
        mics = {
            'phi': float(phi),
            'cause_repertoire': cause_rep.tolist(),
            'effect_repertoire': effect_rep.tolist(),
            'cause_info': cause_entropy,
            'effect_info': effect_entropy,
            'mutual_info': mutual_info,
            'support_size': len(support),
            'n_cause_transitions': len(cau_ids),
            'n_effect_transitions': len(eff_ids),
            'state': np.asarray(state).tolist(),
            'mip': self.current_mip,
            'cached': False
        }
        
        # Cache result
        self.mics_cache[state_key] = float(phi)
        
        # Maintain cache size limit
        if len(self.mics_cache) > 5000:
            oldest_keys = list(self.mics_cache.keys())[:250]
            for k in oldest_keys:
                del self.mics_cache[k]
        
        return mics

    def _compute_phi_simple(self, cause_rep: np.ndarray, effect_rep: np.ndarray) -> float:
        return self.ces._compute_phi_simple(cause_rep, effect_rep)

    def _compute_phi_full(self, cause_rep: np.ndarray, effect_rep: np.ndarray, state: Optional[np.ndarray]=None) -> float:
        return self.ces._compute_phi_full(cause_rep, effect_rep, state)

    def _compute_phi_full_exhaustive(self, cause_rep: np.ndarray, effect_rep: np.ndarray, state: Optional[np.ndarray]=None) -> float:
        return self.ces._compute_phi_full_exhaustive(cause_rep, effect_rep, state)

# =============================
# USHTS (Unified Sparse Heavy-Hitter Transition Store)
# New CauseEffectStructure replacement (redefines the class name)
# =============================

from typing import Literal  # for API typing
import heapq
from collections import OrderedDict



class _WalLog:
    def __init__(self, cfg: _CESConfig):
        self.cfg = cfg
        if cfg.wal_use_memmap:
            raise RuntimeError("wal_use_memmap=True not allowed: external files are forbidden in this environment")
        cap = max(1, cfg.wal_segment_bytes // 12)
        self._cap = int(cap)
        self._n = 0
        self._s = np.zeros(self._cap, dtype=np.int32)
        self._d = np.zeros(self._cap, dtype=np.int32)
        self._w = np.zeros(self._cap, dtype=np.uint32)

    def append(self, src: int, dst: int, w: int) -> bool:
        if self._n >= self._cap:
            return False
        i = self._n
        # ensure indices stay in int32 positive range
        self._s[i] = np.int32(int(src) & 0x7FFFFFFF)
        self._d[i] = np.int32(int(dst) & 0x7FFFFFFF)
        self._w[i] = np.uint32(w)
        self._n += 1
        return True

    @property
    def bytes(self) -> int:
        return int(self._n) * 12

    def cut_and_aggregate_sorted(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self._n
        if n == 0:
            return (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.uint32))
        s = self._s[:n]
        d = self._d[:n]
        w = self._w[:n]
        order = np.lexsort((d, s))
        s_sorted = s[order]
        d_sorted = d[order]
        w_sorted = w[order].astype(np.uint64)
        # identify unique (s,d) runs
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = (s_sorted[1:] != s_sorted[:-1]) | (d_sorted[1:] != d_sorted[:-1])
        idx = np.nonzero(change)[0]
        # reduce sums per run
        sums = np.add.reduceat(w_sorted, idx)
        s_u = s_sorted[idx]
        d_u = d_sorted[idx]
        w_u = sums.astype(np.uint32)
        # reset
        self._n = 0
        return (s_u.copy(), d_u.copy(), w_u.copy())


class _CmsSketch:
    def __init__(self, cfg: _CESConfig):
        self.cfg = cfg
        self.depth = int(cfg.cms_depth)
        self.width = int(cfg.cms_width)
        self.table = np.zeros((self.depth, self.width), dtype=np.uint32)
        self._seed = int(cfg.seed)
        # initialize as arrays so static analyzers know these are arrays (not None)
        self._salts_a = np.zeros(self.depth, dtype=np.uint64)
        self._salts_b = np.zeros(self.depth, dtype=np.uint64)
        self._salts_c = np.zeros(self.depth, dtype=np.uint64)
        self._init_hash_salts()

    def _init_hash_salts(self):
        rng = np.random.default_rng(self._seed)
        # 64-bit salts as odd numbers
        self._salts_a = rng.integers(1, (1 << 63) - 1, size=self.depth, dtype=np.uint64) | 1
        self._salts_b = rng.integers(1, (1 << 63) - 1, size=self.depth, dtype=np.uint64) | 1
        self._salts_c = rng.integers(1, (1 << 63) - 1, size=self.depth, dtype=np.uint64) | 1

    def set_seed(self, seed: int):
        self._seed = int(seed)
        self._init_hash_salts()

    def _hashes(self, src: int, dst: int) -> np.ndarray:
        # Tolerate arbitrarily large Python ints by masking to 64 bits in pure Python first
        mask64 = (1 << 64) - 1
        x = np.uint64(int(src) & mask64)
        y = np.uint64(int(dst) & mask64)
        # simple mix
        # coerce salts to arrays and use explicit bitwise_xor for clearer typing
        salts_a = np.asarray(self._salts_a, dtype=np.uint64)
        salts_b = np.asarray(self._salts_b, dtype=np.uint64)
        salts_c = np.asarray(self._salts_c, dtype=np.uint64)
        
        # Suppress expected overflow warnings in hash computation (intentional for randomization)
        with np.errstate(over='ignore'):
            base = (x * np.uint64(0x9E3779B185EBCA87) ^ y * np.uint64(0xC2B2AE3D27D4EB4F)) + np.uint64(0x165667B19E3779F9)
            h = (base * salts_a + np.bitwise_xor(x, salts_b) + np.bitwise_xor(y, salts_c))
        
        return (h % np.uint64(self.width)).astype(np.int64)

    def estimate(self, src: int, dst: int) -> int:
        idx = self._hashes(src, dst)
        vals = self.table[np.arange(self.depth), idx]
        return int(vals.min())

    def add(self, src: int, dst: int, w: int):
        if w <= 0:
            return
        idx = self._hashes(src, dst)
        arr = self.table
        current = arr[np.arange(self.depth), idx]
        m = current.min()
        # conservative update: only raise minima
        mask = (current == m)
        # add with clipping to uint32 max
        add_val = np.uint64(w)
        upd = current.astype(np.uint64)
        upd[mask] = np.minimum(np.uint64(0xFFFFFFFF), upd[mask] + add_val)
        arr[np.arange(self.depth), idx] = upd.astype(np.uint32)

    def scale_down_half(self):
        self.table >>= 1

    def maybe_rehash(self):
        # Optional: clear table and reseed (approximate aging)
        self.table >>= 1
        self.table.fill(0)
        self._seed = (self._seed * 1103515245 + 12345) & 0x7FFFFFFF
        self._init_hash_salts()


class _TopKIndex:
    def __init__(self, cfg: _CESConfig):
        self.cfg = cfg
        self.rows: Dict[int, Dict[int, int]] = {}
        self.heaps: Dict[int, list[tuple[int, int]]] = {}

    def _target_k(self, rowsum: int) -> int:
        k = int(math.ceil(self.cfg.topk_c * math.log1p(max(0, int(rowsum)))))
        return max(self.cfg.topk_min, min(self.cfg.topk_max, k))

    def _ensure_row(self, row: int):
        if row not in self.rows:
            self.rows[row] = {}
            self.heaps[row] = []

    def _rebuild_heap(self, row: int, k: int):
        d = self.rows.get(row)
        if not d:
            self.heaps[row] = []
            return
        if len(d) <= k:
            self.heaps[row] = [(c, dst) for dst, c in d.items()]
            heapq.heapify(self.heaps[row])
            return
        # keep only top-k
        items = sorted(((c, dst) for dst, c in d.items()), reverse=True)
        keep = list(reversed(items[:k]))  # min-heap
        heapq.heapify(keep)
        self.heaps[row] = keep

    def inc(self, row: int, dst: int, w: int, rowsum: int):
        self._ensure_row(row)
        d = self.rows[row]
        d[dst] = d.get(dst, 0) + int(w)
        k = self._target_k(rowsum)
        self._rebuild_heap(row, k)

    def maybe_promote(self, row: int, dst: int, est: int, rowsum: int) -> bool:
        self._ensure_row(row)
        d = self.rows[row]
        k = self._target_k(rowsum)
        if dst in d:
            return False
        if len(d) < k:
            d[dst] = int(est)
            self._rebuild_heap(row, k)
            return True
        hp = self.heaps.get(row) or []
        if not hp:
            # build heap if missing
            self._rebuild_heap(row, k)
            hp = self.heaps[row]
        margin = max(self.cfg.promote_margin_min, int(self.cfg.promote_margin_frac * max(1, rowsum)))
        min_in_topk = hp[0][0] if hp else 0
        if int(est) >= int(min_in_topk) + int(margin):
            # evict min and insert
            # ensure heap reflects dict
            self._rebuild_heap(row, k)
            hp = self.heaps[row]
            if hp and len(hp) >= k:
                evicted = heapq.heappop(hp)
                if evicted[1] in d:
                    del d[evicted[1]]
            d[dst] = int(est)
            self._rebuild_heap(row, k)
            return True
        return False

    def items(self, row: int) -> list[tuple[int, int]]:
        d = self.rows.get(row)
        if not d:
            return []
        # return sorted by count desc
        return sorted(((dst, c) for dst, c in d.items()), key=lambda x: (-x[1], x[0]))

    def scale_down_half(self):
        to_del_rows = []
        for r, d in self.rows.items():
            to_del = []
            for dst, c in d.items():
                nc = c >> 1
                if nc <= 0:
                    to_del.append(dst)
                else:
                    d[dst] = nc
            for dst in to_del:
                del d[dst]
            if not d:
                to_del_rows.append(r)
        for r in to_del_rows:
            self.rows.pop(r, None)
            self.heaps.pop(r, None)
        # rebuild heaps
        for r, d in self.rows.items():
            k = max(self.cfg.topk_min, min(self.cfg.topk_max, len(d)))
            self._rebuild_heap(r, k)


class _CsrStore:
    def __init__(self):
        self.rows: Dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get(self, row: int, dst: int) -> int:
        if row not in self.rows:
            return 0
        idx, data = self.rows[row]
        if idx.size == 0:
            return 0
        pos = int(np.searchsorted(idx, np.int32(dst)))
        if pos < idx.size and int(idx[pos]) == int(dst):
            return int(data[pos])
        return 0

    def items(self, row: int) -> list[tuple[int, int]]:
        if row not in self.rows:
            return []
        idx, data = self.rows[row]
        return [(int(idx[i]), int(data[i])) for i in range(idx.size)]

    def merge_delta_from_sorted_arrays(self, s_u: np.ndarray, d_u: np.ndarray, w_u: np.ndarray):
        if s_u.size == 0:
            return
        # iterate per row block
        # find run starts
        change = np.empty(s_u.size, dtype=bool)
        change[0] = True
        change[1:] = (s_u[1:] != s_u[:-1])
        starts = np.nonzero(change)[0]
        ends = np.append(starts[1:], s_u.size)
        for st, en in zip(starts, ends):
            row = int(s_u[st])
            di = d_u[st:en]
            dw = w_u[st:en]
            if row in self.rows:
                base_idx, base_data = self.rows[row]
                # 2-way merge (sorted unique in di)
                i = j = 0
                new_idx = []
                new_data = []
                while i < base_idx.size and j < di.size:
                    a = int(base_idx[i])
                    b = int(di[j])
                    if a == b:
                        new_idx.append(a)
                        new_data.append(int(base_data[i]) + int(dw[j]))
                        i += 1
                        j += 1
                    elif a < b:
                        new_idx.append(a)
                        new_data.append(int(base_data[i]))
                        i += 1
                    else:
                        new_idx.append(b)
                        new_data.append(int(dw[j]))
                        j += 1
                while i < base_idx.size:
                    new_idx.append(int(base_idx[i]))
                    new_data.append(int(base_data[i]))
                    i += 1
                while j < di.size:
                    new_idx.append(int(di[j]))
                    new_data.append(int(dw[j]))
                    j += 1
                self.rows[row] = (np.asarray(new_idx, dtype=np.int32), np.asarray(new_data, dtype=np.uint32))
            else:
                self.rows[row] = (di.astype(np.int32).copy(), dw.astype(np.uint32).copy())

    def scale_down_half(self):
        to_del = []
        for r, (idx, data) in self.rows.items():
            nd = (data >> 1).astype(np.uint32)
            mask = nd > 0
            self.rows[r] = (idx[mask].copy(), nd[mask].copy())
            if self.rows[r][0].size == 0:
                to_del.append(r)
        for r in to_del:
            self.rows.pop(r, None)


class _RowCache:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.od: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()

    def get(self, row: int) -> Optional[list[tuple[int, int]]]:
        if row in self.od:
            v = self.od.pop(row)
            self.od[row] = v
            return v
        return None

    def put(self, row: int, items: list[tuple[int, int]]):
        if row in self.od:
            self.od.pop(row)
        self.od[row] = items
        if len(self.od) > self.capacity:
            self.od.popitem(last=False)

    def invalidate(self, row: int):
        if row in self.od:
            self.od.pop(row)

    def clear(self):
        self.od.clear()


class CauseEffectStructure:
    """USHTS-based sparse cause-effect storage.

    Public API:
    - add(src:int, dst:int, w:int=1) -> None
    - get(src:int, dst:int) -> int
    - topk_items(src:int, k:int|None=None) -> list[(dst:int,count:int)]
    - iter_row(src:int, mode:Literal["exact","mix"]="exact") -> list[(dst:int,count:int)]
    - rowsum(src:int) -> int
    - prob(src:int, dst:int) -> float
    - compact() -> None
    - decay() -> None
    - set_seed(seed:int) -> None
    """

    def __init__(self, n_elements: int | None = None, cfg: Optional[_CESConfig] = None):
        self.cfg = cfg or _CESConfig()
        self.cms = _CmsSketch(self.cfg)
        self.topk = _TopKIndex(self.cfg)
        self.csr = _CsrStore()
        self.wal = _WalLog(self.cfg)
        self.row_cache = _RowCache(self.cfg.row_cache_capacity)
        self._rowsum: Dict[int, int] = {}
        self._updates_total: int = 0
        self._last_decay_at: int = 0
        self._last_cms_rehash_at: int = 0
        self.n_elements = n_elements or 0
        self.current_mip = None
        self.mip_history: deque = deque(maxlen=50)
        self.phi_history: deque = deque(maxlen=256)

    def add(self, src: int, dst: int, w: int = 1) -> None:
        """Record transition count w from src to dst. May trigger compaction/decay."""
        w = int(w)
        # Normalize keys defensively to 31-bit to avoid overflow in numpy/ctypes paths
        src = int(src) & 0x7FFFFFFF
        dst = int(dst) & 0x7FFFFFFF
        if w <= 0:
            return
        # rowsum
        self._rowsum[src] = self._rowsum.get(src, 0) + w
        # WAL append; compact if full
        if not self.wal.append(src, dst, w):
            self.compact()
            if not self.wal.append(src, dst, w):
                raise RuntimeError("WAL append failed after compaction; segment too small")
        # Top-K / CMS
        if src in self.topk.rows and dst in self.topk.rows[src]:
            self.topk.inc(src, dst, w, self._rowsum.get(src, 0))
            self.row_cache.invalidate(src)
        else:
            est = self.cms.estimate(src, dst) + w
            promoted = self.topk.maybe_promote(src, dst, est, self._rowsum.get(src, 0))
            if promoted:
                # initialize with best available absolute count
                base = self.csr.get(src, dst)
                init_c = max(int(est), int(base))
                self.topk.rows[src][dst] = init_c
                self.topk._rebuild_heap(src, self.topk._target_k(self._rowsum.get(src, 0)))
                self.row_cache.invalidate(src)
            else:
                self.cms.add(src, dst, w)
        # periodic maintenance
        self._updates_total += 1
        if self._updates_total - self._last_decay_at >= self.cfg.decay_half_life_updates:
            self.decay()
            self._last_decay_at = self._updates_total
        if self._updates_total - self._last_cms_rehash_at >= self.cfg.cms_rehash_window_updates:
            self.cms.maybe_rehash()
            self._last_cms_rehash_at = self._updates_total

    def get(self, src: int, dst: int) -> int:
        """Return count for (src,dst) using Top-K, else CSR, else CMS estimate."""
        if src in self.topk.rows and dst in self.topk.rows[src]:
            return int(self.topk.rows[src][dst])
        c = self.csr.get(src, dst)
        if c > 0:
            return int(c)
        return int(self.cms.estimate(src, dst))

    def topk_items(self, src: int, k: Optional[int] = None) -> list[tuple[int, int]]:
        """Return Top-K items for row src, optionally truncating to k."""
        items = self.topk.items(src)
        if k is not None:
            return items[: int(k)]
        return items

    def iter_row(self, src: int, mode: Literal["exact", "mix"] = "exact") -> list[tuple[int, int]]:
        """Iterate a row.
        - exact: CSR overlaid by Top-K absolute counts (no CMS).
        - mix: Top-K list plus tail as (-1, tail_mass).
        """
        if mode == "mix":
            head = self.topk.items(src)
            head_sum = sum(c for _, c in head)
            tail = max(0, self._rowsum.get(src, 0) - head_sum)
            if tail > 0:
                return head + [(-1, tail)]
            return head
        # exact with caching
        cached = self.row_cache.get(src)
        if cached is not None:
            return cached
        merged: Dict[int, int] = {}
        for dst, c in self.csr.items(src):
            merged[dst] = c
        # overlay topk absolute counts
        for dst, c in self.topk.items(src):
            merged[dst] = c
        out = sorted(merged.items(), key=lambda x: x[0])
        self.row_cache.put(src, out)
        return out

    def rowsum(self, src: int) -> int:
        """Return total outgoing mass from src."""
        return int(self._rowsum.get(src, 0))

    def prob(self, src: int, dst: int) -> float:
        """Return smoothed probability p = (count+alpha)/(rowsum+alpha*Veff)."""
        rs = self._rowsum.get(src, 0)
        if rs <= 0:
            Veff = max(1, len(self.iter_row(src, mode="exact")))
            return float((0.0 + self.cfg.alpha) / (0.0 + self.cfg.alpha * Veff))
        Veff = max(1, len(self.iter_row(src, mode="exact")))
        cnt = self.get(src, dst)
        return float((cnt + self.cfg.alpha) / (rs + self.cfg.alpha * Veff))

    def compact(self) -> None:
        """Compact WAL into CSR; invalidate exact row cache."""
        s_u, d_u, w_u = self.wal.cut_and_aggregate_sorted()
        if s_u.size > 0:
            self.csr.merge_delta_from_sorted_arrays(s_u, d_u, w_u)
            # any rows touched should invalidate cache
            for r in np.unique(s_u):
                self.row_cache.invalidate(int(r))

    def decay(self) -> None:
        """Global half-down of all structures; invalidates cache."""
        self.cms.scale_down_half()
        self.topk.scale_down_half()
        self.csr.scale_down_half()
        # rowsum
        keys = list(self._rowsum.keys())
        for k in keys:
            v = self._rowsum[k] >> 1
            if v <= 0:
                del self._rowsum[k]
            else:
                self._rowsum[k] = v
        self.row_cache.clear()

    def set_seed(self, seed: int) -> None:
        """Reset seeds to ensure determinism."""
        self.cfg.seed = int(seed)
        self.cms.set_seed(int(seed))

    def _normalize_phi(self, phi: float, cause_rep: np.ndarray) -> float:
        max_possible_phi = np.log2(len(cause_rep)) * 2
        phi_normalized = phi / max(max_possible_phi, 1e-10)
        return float(np.clip(phi_normalized, 0.0, 1.0))


    def compute_phi(self, cause_rep: np.ndarray, effect_rep: np.ndarray, method: str = "simple", state: Optional[np.ndarray]=None) -> float:
        """Compute phi using method-specific CES approximations."""
        method_key = str(method or "simple").strip().lower()
        if method_key in {"simple", "approx", "fast"}:
            return float(self._compute_phi_simple(cause_rep, effect_rep))
        if method_key in {"full", "integrated"}:
            return float(self._compute_phi_full(cause_rep, effect_rep, state))
        if method_key in {"exhaustive"}:
            return float(self._compute_phi_full_exhaustive(cause_rep, effect_rep, state))
        if method_key in {"cutset", "cutset_sampling"}:
            return float(self._compute_phi_cutset_sampling(cause_rep, effect_rep, state))
        if method_key in {"community", "community_cluster"}:
            return float(self._compute_phi_community_cluster(cause_rep, effect_rep, state))
        return float(self._compute_phi_simple(cause_rep, effect_rep))


    def _compute_phi_cutset_sampling(self, cause_rep: np.ndarray, effect_rep: np.ndarray, state: Optional[np.ndarray]=None) -> float:
        n = self.n_elements
        full_integrated_info = self._compute_integrated_information(cause_rep, effect_rep, set(range(n)), set())
        edge_strengths = []
        edge_strengths.sort(reverse=True)
        candidate_partitions = []
        for i in range(min(10, n - 1)):
            partition1 = set(range(i + 1))
            partition2 = set(range(i + 1, n))
            if len(partition1) > 0 and len(partition2) > 0:
                candidate_partitions.append((partition1, partition2))
        for _ in range(min(10, 2 ** (n - 1) - 10)):
            size = np.random.randint(1, n)
            indices = np.random.choice(n, size, replace=False)
            partition1 = set(indices)
            partition2 = set(range(n)) - partition1
            if len(partition1) > 0 and len(partition2) > 0:
                candidate_partitions.append((partition1, partition2))
        min_partitioned_info = float('inf')
        best_mip = None
        for subset, complement in candidate_partitions:
            partitioned_info = self._compute_partitioned_information(cause_rep, effect_rep, subset, complement)
            if partitioned_info < min_partitioned_info:
                min_partitioned_info = partitioned_info
                best_mip = (subset, complement)
        phi = max(0.0, full_integrated_info - min_partitioned_info)
        self.current_mip = best_mip
        self.mip_history.append({'mip': best_mip, 'phi': phi, 'full_info': full_integrated_info, 'partitioned_info': min_partitioned_info, 'method': 'cutset_sampling', 'sampled_partitions': len(candidate_partitions)})
        norm_phi = self._normalize_phi(phi, cause_rep)
        try:
            self.phi_history.append(float(norm_phi))
        except Exception:
            pass
        return norm_phi

    def _find_connected_component(self, start: int, excluded: Set[int], n: int) -> Set[int]:
        return {start}

    def _compute_phi_community_cluster(self, cause_rep: np.ndarray, effect_rep: np.ndarray, state: Optional[np.ndarray]=None) -> float:
        n = self.n_elements
        communities = self._detect_communities_greedy(n)
        if len(communities) <= 1:
            print(f'WARNING: Community detection failed, fallback to cutset sampling')
            return self._compute_phi_cutset_sampling(cause_rep, effect_rep, state)
        full_integrated_info = self._compute_integrated_information(cause_rep, effect_rep, set(range(n)), set())
        num_communities = len(communities)
        min_partitioned_info = float('inf')
        best_mip = None
        if num_communities <= 6:
            for partition_size in range(1, num_communities):
                for subset_indices in combinations(range(num_communities), partition_size):
                    partition1 = set()
                    for idx in subset_indices:
                        partition1.update(communities[idx])
                    partition2 = set(range(n)) - partition1
                    if len(partition1) > 0 and len(partition2) > 0:
                        partitioned_info = self._compute_partitioned_information(cause_rep, effect_rep, partition1, partition2)
                        if partitioned_info < min_partitioned_info:
                            min_partitioned_info = partitioned_info
                            best_mip = (partition1, partition2)
        else:
            max_samples = min(50, 2 ** (num_communities - 1))
            for _ in range(max_samples):
                num_in_group1 = np.random.randint(1, num_communities)
                indices_group1 = np.random.choice(num_communities, num_in_group1, replace=False)
                partition1 = set()
                for idx in indices_group1:
                    partition1.update(communities[idx])
                partition2 = set(range(n)) - partition1
                if len(partition1) > 0 and len(partition2) > 0:
                    partitioned_info = self._compute_partitioned_information(cause_rep, effect_rep, partition1, partition2)
                    if partitioned_info < min_partitioned_info:
                        min_partitioned_info = partitioned_info
                        best_mip = (partition1, partition2)
        phi = max(0.0, full_integrated_info - min_partitioned_info)
        self.current_mip = best_mip
        self.mip_history.append({'mip': best_mip, 'phi': phi, 'full_info': full_integrated_info, 'partitioned_info': min_partitioned_info, 'method': 'community_cluster', 'num_communities': len(communities), 'community_sizes': [len(c) for c in communities]})
        norm_phi = self._normalize_phi(phi, cause_rep)
        try:
            self.phi_history.append(float(norm_phi))
        except Exception:
            pass
        return norm_phi

    def _detect_communities_greedy(self, n: int) -> List[Set[int]]:
        return [set(range(n))]

    def _modularity_gain(self, comm1: Set[int], comm2: Set[int], total_strength: float) -> float:
        return 0.0

    def _inter_community_strength(self, comm1: Set[int], comm2: Set[int]) -> float:
        return 0.0

    def _compute_integrated_information(self, cause_rep: np.ndarray, effect_rep: np.ndarray, elements: set, excluded: set) -> float:
        """
        Compute integrated information using causal-effect repertoire analysis.
        
        Integrated information measures:
        1. Marginal information (cause + effect entropy)
        2. Mutual information (statistical dependency)
        3. Causal specificity (how constraining the repertoires are)
        """
        cause_entropy = self._repertoire_entropy(cause_rep)
        effect_entropy = self._repertoire_entropy(effect_rep)
        
        # 1. MUTUAL INFORMATION: Statistical dependency between cause and effect
        mutual_info = 0.0
        if len(cause_rep) == len(effect_rep):
            joint_entropy = self._compute_joint_entropy(cause_rep, effect_rep)
            mutual_info = max(0.0, cause_entropy + effect_entropy - joint_entropy)

        # 2. CAUSAL SPECIFICITY: How much do repertoires constrain possibilities
        # Measured as deviation from maximum entropy (uniform distribution)
        n_states = len(cause_rep)
        max_entropy = np.log2(n_states) if n_states > 1 else 0.0
        
        # Specificity = how far from uniform (0=uniform, 1=maximally specific)
        cause_specificity = 0.0 if max_entropy == 0 else 1.0 - (cause_entropy / max_entropy)
        effect_specificity = 0.0 if max_entropy == 0 else 1.0 - (effect_entropy / max_entropy)
        specificity_score = (cause_specificity + effect_specificity) / 2.0
        
        # 3. ELEMENT COMPLEXITY: Systems with more elements can have higher integration
        element_factor = 1.0
        n_elements = len(elements) if elements else 1
        if n_elements > 1:
            # Logarithmic scaling: doubling elements doesn't double integration
            element_factor = 1.0 + np.log2(n_elements) * 0.2
        
        # 4. INTEGRATION FORMULA:
        # Base information (entropy) weighted by mutual dependency and specificity
        base_info = (cause_entropy + effect_entropy) / 2.0
        integration_multiplier = 1.0 + mutual_info * 0.8 + specificity_score * 0.4
        
        integrated_info = base_info * integration_multiplier * element_factor
        
        # 5. MINIMUM THRESHOLD: adapt floor to observed phi scale.
        if (n_elements > 1 and 
            np.any(np.abs(cause_rep) > 1e-08) and 
            np.any(np.abs(effect_rep) > 1e-08)):
            try:
                pol = _compute_phi_policy_from_history(list(self.phi_history), cfg=None)
                phi_floor = float(pol.get("floor", 0.01))
            except Exception:
                phi_floor = 0.01
            integrated_info = max(integrated_info, float(phi_floor))
        
        return float(np.clip(integrated_info, 0.0, 20.0))

    def _compute_partitioned_information(self, cause_rep: np.ndarray, effect_rep: np.ndarray, subset1: set, subset2: set) -> float:
        n1, n2 = (len(subset1), len(subset2))
        subset1_entropy = np.log2(2 ** n1) if n1 > 0 else 0.0
        subset2_entropy = np.log2(2 ** n2) if n2 > 0 else 0.0
        cross_causal_loss = 0.0
        # If a causal graph is attached to this structure, incorporate cross-links
        if hasattr(self, 'causal_graph'):
            for i in subset1:
                for j in subset2:
                    if i < self.n_elements and j < self.n_elements:
                        cross_causal_loss += abs(self.causal_graph[i, j])
            for i in subset2:
                for j in subset1:
                    if i < self.n_elements and j < self.n_elements:
                        cross_causal_loss += abs(self.causal_graph[i, j])
        partitioned_info = subset1_entropy + subset2_entropy - cross_causal_loss * 0.5
        partitioned_info = float(partitioned_info)
        return max(0.0, float(partitioned_info))

    def _compute_partition_loss(self, cause_rep: np.ndarray, effect_rep: np.ndarray, subset1: set, subset2: set) -> float:
        full_entropy = self._repertoire_entropy(cause_rep) + self._repertoire_entropy(effect_rep)
        n1, n2 = (len(subset1), len(subset2))
        partition_entropy = np.log2(2 ** n1) + np.log2(2 ** n2)
        loss = abs(full_entropy - partition_entropy)
        return loss

    def _repertoire_entropy(self, repertoire: np.ndarray) -> float:
        """
        Compute Shannon entropy with precision corrections for numerical stability.
        Returns information content in bits.
        """
        epsilon = 1e-12
        repertoire = np.abs(repertoire) + epsilon
        total = repertoire.sum()
        
        if total < epsilon:
            return 0.0
        
        # Normalize to probability distribution
        p = repertoire / total
        
        # Shannon entropy: H = -Σ p_i log2(p_i)
        # Only include non-negligible probabilities
        mask = p > epsilon
        if not np.any(mask):
            return 0.0
        
        entropy = -np.sum(p[mask] * np.log2(p[mask]))
        
        # Validate: entropy must be between 0 and log2(n_states)
        n_states = len(repertoire)
        max_entropy = np.log2(n_states)
        entropy = float(np.clip(entropy, 0.0, max_entropy))
        
        return entropy

    def _compute_joint_entropy(self, cause_rep: np.ndarray, effect_rep: np.ndarray) -> float:
        """
        Compute joint entropy of cause and effect repertoires.
        Uses empirical transition data when available, otherwise makes informed estimate.
        """
        if len(cause_rep) != len(effect_rep):
            # Different dimensionality: assume independence
            return self._repertoire_entropy(cause_rep) + self._repertoire_entropy(effect_rep)
        
        n = len(cause_rep)
        joint_prob = None
        
        # STRATEGY 1: Use empirical transition counts (most accurate)
        if hasattr(self, 'transition_counts') and getattr(self, 'transition_counts') is not None:
            tc = np.asarray(self.transition_counts, dtype=float)
            if tc.shape == (n, n) and tc.sum() > 0:
                joint_prob = tc / tc.sum()
        
        # STRATEGY 2: Use TPM with empirical state prior from history
        if joint_prob is None and hasattr(self, 'tpm') and self.tpm is not None:
            tpm = np.asarray(self.tpm, dtype=float)
            if tpm.shape[0] == n and tpm.shape[1] == n:
                # Estimate prior distribution from state history
                prior = None
                if hasattr(self, 'state_history') and len(self.state_history) > 0:
                    counts = np.zeros(n, dtype=float)
                    for past_state in self.state_history:
                        try:
                            idx = self._state_to_index(past_state)
                            if 0 <= idx < n:
                                counts[idx] += 1.0
                        except Exception:
                            continue
                    
                    if counts.sum() > 0:
                        prior = counts / counts.sum()
                
                # Uniform prior if no history
                if prior is None:
                    prior = np.ones(n, dtype=float) / float(n)
                
                # Joint probability: P(cause, effect) = P(cause) * P(effect|cause)
                joint_prob = prior.reshape(-1, 1) * tpm
                s = joint_prob.sum()
                if s > 0:
                    joint_prob = joint_prob / s
        
        # STRATEGY 3: Use repertoires to estimate joint distribution
        if joint_prob is None:
            # Estimate coupling strength from repertoire similarity
            # More similar repertoires → stronger coupling → lower joint entropy
            cause_norm = cause_rep / (np.sum(cause_rep) + 1e-12)
            effect_norm = effect_rep / (np.sum(effect_rep) + 1e-12)
            
            # Compute similarity (1 - normalized L1 distance)
            l1_dist = np.sum(np.abs(cause_norm - effect_norm))
            similarity = 1.0 - (l1_dist / 2.0)  # L1 distance ranges [0, 2]
            
            # Joint entropy interpolates between independence and perfect correlation
            independent_entropy = self._repertoire_entropy(cause_rep) + self._repertoire_entropy(effect_rep)
            max_marginal_entropy = max(self._repertoire_entropy(cause_rep), self._repertoire_entropy(effect_rep))
            
            # High similarity → lower joint entropy (more mutual info)
            # similarity=0 → independent_entropy
            # similarity=1 → max_marginal_entropy
            joint_entropy = independent_entropy * (1 - similarity) + max_marginal_entropy * similarity
            
            return float(np.clip(joint_entropy, 0.0, independent_entropy))
        
        # Compute Shannon entropy from joint probability matrix
        epsilon = 1e-12
        joint_prob = joint_prob + epsilon
        joint_prob = joint_prob / joint_prob.sum()
        
        mask = joint_prob > epsilon
        if not np.any(mask):
            return 0.0
        
        joint_entropy = -np.sum(joint_prob[mask] * np.log2(joint_prob[mask]))
        
        # Validate: joint entropy cannot exceed sum of marginals
        max_joint = self._repertoire_entropy(cause_rep) + self._repertoire_entropy(effect_rep)
        
        return float(np.clip(joint_entropy, 0.0, max_joint))

    def get_phi_trend(self) -> str:
        if len(self.phi_history) < 10:
            return 'insufficient_data'
        recent = list(self.phi_history)[-10:]
        trend = np.mean(np.diff(recent))
        if trend > 0.01:
            return 'increasing'
        elif trend < -0.01:
            return 'decreasing'
        else:
            return 'stable'

    def get_consciousness_level(self, phi: float) -> str:
        try:
            pol = _compute_phi_policy_from_history(list(self.phi_history), cfg=None)
        except Exception:
            pol = _normalize_phi_policy(None)
        if phi < float(pol.get("floor", 0.01)):
            return '(unconscious)'
        elif phi < float(pol.get("low", 0.1)):
            return '(minimal)'
        elif phi < float(pol.get("mid", 0.3)):
            return '(low)'
        elif phi < float(pol.get("high", 0.5)):
            return '(moderate)'
        elif phi < float(pol.get("very_high", 0.7)):
            return '(high)'
        else:
            return '(very high)'

class EvolutionVisualizer:

    __old_update__: Optional[Callable] = None  # For method overriding in subclasses
    _scope_wired: bool = False
    _scope_gui_wired: bool = False

    def __init__(self):
        self.neural_map = None
        self.generation = 0
        self.total_connections = 0
        self.consciousness_level = 0
        self.phi_value = 0.0
        self.major_events = deque(maxlen=5)
        self.energy_history = deque(maxlen=50)
        self.growth_stage = 'embryo'
        self.qualia_data = None
        self.current_experience = 'unknown'
        self.memory_count = 0
        self.unity_score = 0.5
        self.neuron_count = 4
        self.connection_count = 4
        self.growth_events = 0
        self.scale_factor = 1.0
        self.max_intensity = 0.1
        self.connection_history = deque(maxlen=100)
        self.scope_image = None  # SCOPE image
        # Reuse a single Scope encoder instance for performance
        try:
            self.scope_encoder = Scope()
        except Exception:
            self.scope_encoder = None

    def update(self, system_state: Dict[str, Any]):
        self.generation += 1
        u_matrix = system_state.get('u_matrix', None)
        if u_matrix is not None:
            self._update_neural_map(u_matrix)
        else:
            # Ensure total_connections is reset if no u_matrix is provided
            self.total_connections = 0
        self.qualia_data = system_state.get('qualia', None)
        self.current_experience = system_state.get('current_experience', 'unknown')
        self.memory_count = system_state.get('memories', 0)
        self.unity_score = system_state.get('unity', 0.5)
        self.neuron_count = system_state.get('neuron_count', 4)
        self.connection_count = system_state.get('connection_count', 4)
        self.growth_events = system_state.get('growth_events', 0)
        strange_loop = system_state.get('strange_loop', False)
        meta_awareness = system_state.get('meta_awareness', 0.0)
        self.phi_value = system_state.get('phi', 0.0)
        old_level = self.consciousness_level
        # SCOPE image update (prefer real retina when available)
        try:
            import cv2
            # Prefer externally built retina passed via system_state
            if isinstance(system_state.get('retina', None), np.ndarray):
                rr = np.asarray(system_state['retina'], dtype=np.float32)
                # ensure 0..1
                if rr.max() > 1.0:
                    rr = rr / max(1.0, float(rr.max()))
                frame = cv2.resize(rr, (256, 256), interpolation=cv2.INTER_CUBIC)
            elif self.neural_map is not None:
                norm_map = (self.neural_map - np.min(self.neural_map)) / (np.ptp(self.neural_map) + 1e-8)
                frame = cv2.resize(norm_map.astype(np.float32), (256,256), interpolation=cv2.INTER_CUBIC)
            else:
                frame = np.zeros((256,256), dtype=np.float32)
            # Optional debug
            if os.environ.get('M3_DEBUG_SCOPE', '0') in ('1', 'true', 'TRUE'):
                print(f"[DEBUG] SCOPE input frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
            arousal = float(system_state.get('arousal', 0.5))
            drivers = {'arousal': arousal}
            try:
                if 'scope_one_bit' in system_state:
                    drivers['one_bit'] = bool(system_state['scope_one_bit'])
                # If vision source is external (camera/folder/push), bypass glitch effect for realism
                if system_state.get('vision_mode', 'internal') in ('camera', 'folder', 'push'):
                    drivers['bypass_glitch'] = True
            except Exception:
                pass
            if getattr(self, 'scope_encoder', None) is None:
                self.scope_encoder = Scope()
            image, meta = self.scope_encoder.encode(frame, drivers)
            if os.environ.get('M3_DEBUG_SCOPE', '0') in ('1', 'true', 'TRUE'):
                print(f"[DEBUG] SCOPE output image shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
            self.scope_image = image
        except Exception as e:
            print(f"[DEBUG] SCOPE encoding error: {e}")
            # Fallback: avoid cv2 dependency. Try PIL or numpy-only upsample, then Scope encode.
            try:
                if self.neural_map is not None:
                    base = (self.neural_map - np.min(self.neural_map)) / (np.ptp(self.neural_map) + 1e-8)
                    base = base.astype(np.float32)
                else:
                    base = np.zeros((30,80), dtype=np.float32)
                try:
                    from PIL import Image as _PIL_Image
                    from PIL import Image as _PIL
                    resample = getattr(_PIL, 'Resampling', _PIL)
                    im = _PIL_Image.fromarray((base * 255).astype(np.uint8)).resize((256, 256), resample.BICUBIC)
                    frame = (np.asarray(im).astype(np.float32) / 255.0)
                except Exception:
                    sh, sw = base.shape[:2]
                    ry = max(1, int(np.ceil(256 / max(1, sh))))
                    rx = max(1, int(np.ceil(256 / max(1, sw))))
                    frame = np.kron(base, np.ones((ry, rx), dtype=np.float32))[:256, :256]
                drivers = {'arousal': float(system_state.get('arousal', 0.5))}
                if getattr(self, 'scope_encoder', None) is None:
                    self.scope_encoder = Scope()
                image, _ = self.scope_encoder.encode(frame, drivers)
                self.scope_image = image
            except Exception:
                self.scope_image = np.zeros((256,256), dtype=np.uint8)
        phi_policy = _normalize_phi_policy(system_state.get("phi_policy") if isinstance(system_state, dict) else None)
        if self.phi_value > float(phi_policy.get("high", 0.5)):
            self.consciousness_level = 4
            self.growth_stage = 'transcendent'
        elif self.phi_value > float(phi_policy.get("mid", 0.3)):
            self.consciousness_level = 3
            self.growth_stage = 'adult'
        elif meta_awareness > 0.3:
            self.consciousness_level = 2
            self.growth_stage = 'child'
        elif strange_loop:
            self.consciousness_level = 1
            self.growth_stage = 'infant'
        if self.consciousness_level > old_level:
            self.add_major_event(f'EVOLVED to {self.growth_stage.upper()}!')
        energy = system_state.get('energy', 0.0)
        self.energy_history.append(energy)
        self.connection_history.append(self.total_connections)
        if len(self.connection_history) > 10:
            recent_avg = np.mean(list(self.connection_history)[-10:])
            if recent_avg > 1000:
                target_scale = 1000 / recent_avg
                self.scale_factor = 0.9 * self.scale_factor + 0.1 * target_scale
                self.scale_factor = max(0.1, min(1.0, self.scale_factor))

    def _update_neural_map(self, u_matrix):
        if u_matrix is None:
            return
        n, k = u_matrix.shape
        if n <= 0 or k <= 0:
            return
        self.total_connections = int(np.sum(np.abs(u_matrix) > 0.1))
        current_max = np.max(np.abs(u_matrix))
        self.max_intensity = 0.95 * self.max_intensity + 0.05 * current_max
        if self.neural_map is None:
            self.neural_map = np.zeros((30, 80))
        for i in range(30):
            for j in range(80):
                u_i = int(i / 30 * n)
                u_j = int(j / 80 * k)
                # Clamp indices to valid range to avoid negative/out-of-bounds (e.g., due to rounding edge cases)
                if u_i < 0:
                    u_i = 0
                elif u_i >= n:
                    u_i = n - 1
                if u_j < 0:
                    u_j = 0
                elif u_j >= k:
                    u_j = k - 1
                val = u_matrix[u_i, u_j]
                if val is None:
                    continue
                val = abs(val)
                normalized_val = val / max(0.01, self.max_intensity)
                self.neural_map[i, j] = 0.9 * self.neural_map[i, j] + 0.1 * normalized_val

    def add_major_event(self, description: str):
        self.major_events.append(f'Gen {self.generation:,}: {description}')

    def render_brain_growth(self) -> str:
        if self.neural_map is None:
            return 'Initializing neural network...'
        lines = []
        center_x, center_y = (40, 15)
        color_map = {'tension': '\x1b[91m', 'harmony': '\x1b[92m', 'uncertainty': '\x1b[93m', 'flow': '\x1b[96m', 'resistance': '\x1b[95m'}
        reset = '\x1b[0m'
        dominant_qualia = 'flow'
        if self.qualia_data:
            max_val = 0
            for q_name, q_val in self.qualia_data.items():
                if q_val > max_val:
                    max_val = q_val
                    dominant_qualia = q_name
        base_color = color_map.get(dominant_qualia, reset)
        memory_boost = min(1.0, self.memory_count / 100)
        unity_boost = self.unity_score
        for i, row in enumerate(self.neural_map):
            line = ''
            for j, val in enumerate(row):
                dx = (j - center_x) / 2.0
                dy = (i - center_y) * 1.5
                dist = np.sqrt(dx ** 2 + dy ** 2)
                brain_shape = np.exp(-dist ** 2 / 400.0)
                effective_val = val * brain_shape * self.scale_factor
                consciousness_boost = 1.0 + self.consciousness_level * 0.5
                phi_boost = 1.0 + self.phi_value * 2.0
                effective_val *= consciousness_boost * phi_boost
                effective_val += memory_boost * 0.3 + unity_boost * 0.2
                if effective_val > 3.0:
                    char = '@'
                elif effective_val > 2.0:
                    char = '#'
                elif effective_val > 1.5:
                    char = '*'
                elif effective_val > 1.0:
                    char = '+'
                elif effective_val > 0.5:
                    char = '.'
                elif effective_val > 0.2:
                    char = '-'
                else:
                    char = ' '
                if char != ' ':
                    line += base_color + char + reset
                else:
                    line += char
            lines.append(line)
        if self.current_experience != 'unknown' and len(lines) > 15:
            exp_label = f'[{self.current_experience[:10]}]'
            center_line = lines[15]
            insert_pos = 35
            lines[15] = center_line[:insert_pos] + f'\x1b[97;40m{exp_label}\x1b[0m' + center_line[insert_pos + len(exp_label):]
        return '\n'.join(lines)

    def render_network_growth(self, width: int=60, height: int=30) -> str:
        if self.neuron_count == 0:
            return 'Initializing network...'
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        for i in range(min(self.neuron_count, 200)):
            x = (i * 37 + 13) % width
            y = (i * 23 + 7) % height
            age_factor = min(1.0, i / max(1, self.neuron_count))
            if age_factor > 0.8:
                char = '*'
            elif age_factor > 0.5:
                char = '+'
            elif age_factor > 0.2:
                char = '-'
            else:
                char = '.'
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = char
                char = '.'
            if 0 <= y < height and 0 <= x < width:
                grid[y][x] = char
            # define endpoints for a simple connection to the next neuron index
            i1 = i
            cap = max(1, min(self.neuron_count, 200))
            i2 = (i + 1) % cap
            x1 = (i1 * 37 + 13) % width
            y1 = (i1 * 23 + 7) % height
            x2 = (i2 * 37 + 13) % width
            y2 = (i2 * 23 + 7) % height
            if abs(x2 - x1) > abs(y2 - y1):
                for step in range(abs(x2 - x1)):
                    t = step / max(1, abs(x2 - x1))
                    x = int(x1 + (x2 - x1) * t)
                    y = int(y1 + (y2 - y1) * t)
                    if 0 <= y < height and 0 <= x < width and (grid[y][x] == ' '):
                        grid[y][x] = '\x1b[90m-\x1b[0m'
            else:
                for step in range(abs(y2 - y1)):
                    t = step / max(1, abs(y2 - y1))
                    x = int(x1 + (x2 - x1) * t)
                    y = int(y1 + (y2 - y1) * t)
                    if 0 <= y < height and 0 <= x < width and (grid[y][x] == ' '):
                        grid[y][x] = '\x1b[90m|\x1b[0m'
        lines = []
        for row in grid:
            lines.append(''.join(row))
        return '\n'.join(lines)

    def render_full_display(self) -> str:
        width = 100
        border = '+' + '-' * (width - 2) + '+'
        output = [border]
        title = 'M3 CONSCIOUSNESS SYSTEM (AUTONOMOUS)'
        output.append('|' + title.center(width - 2) + '|')
        output.append('|' + (('Generation: ' + str(self.generation)).center(width - 2)) + '|')
        output.append('|' + (('Stage: ' + self.growth_stage.upper()).center(width - 2)) + '|')
        output.append(border)
        brain = self.render_brain_growth()
        for line in brain.split('\n'):
            output.append('| ' + line[:width - 4].ljust(width - 4) + ' |')
        output.append(border)
        conn_bar = self._make_bar(float(min(1.0, self.total_connections * self.scale_factor / 5000)), 40)
        output.append(('| Connections: ' + conn_bar + f' {self.total_connections:,} (scale: {self.scale_factor:.2f})').ljust(width - 1) + '|')
        if self.energy_history:
            energy = self.energy_history[-1]
            energy_bar = self._make_bar(energy / 100, 40)
            output.append(('| Energy:      ' + energy_bar + f' {energy:.0f}%').ljust(width - 1) + '|')
        level_bar = '#' * (self.consciousness_level * 10) + '-' * ((4 - self.consciousness_level) * 10)
        output.append(('| Level:       ' + level_bar + f' {self.consciousness_level}/4').ljust(width - 1) + '|')
        if self.phi_value > 0.001:
            phi_bar = self._make_bar(min(1.0, self.phi_value * 2), 40)
            output.append(('| Phi:         ' + phi_bar + f' {self.phi_value:.4f}').ljust(width - 1) + '|')
        output.append(border)
        if self.major_events:
            for event in self.major_events:
                output.append('| ' + ('* ' + event)[:width - 4].ljust(width - 4) + ' |')
        else:
            output.append('| ' + '(system initializing...)'.ljust(width - 4) + ' |')
        output.append(border)
        return '\n'.join(output)

    def _make_bar(self, ratio: float, width: int=40) -> str:
        filled = int(ratio * width)
        return '#' * filled + '-' * (width - filled)
    # Helper to provide an image-like numpy array for external GUIs
    def get_scope_image(self) -> Optional['np.ndarray']:
        """Return a uint8 image (H,W) suitable for display, preferring SCOPE output if available.

        Returns None if no image data is available.
        """
        try:
            if getattr(self, 'scope_image', None) is not None:
                img = self.scope_image
                # ensure uint8
                if img.dtype != getattr(__import__('numpy'), 'uint8'):
                    import numpy as _np
                    arr = _np.asarray(img)
                    arr = _np.clip(arr * 255.0, 0, 255).astype(_np.uint8) if arr.dtype in (_np.float32, _np.float64) else arr.astype(_np.uint8)
                    return arr
                return img
            if getattr(self, 'neural_map', None) is not None:
                import numpy as _np
                nm = _np.asarray(self.neural_map, dtype=_np.float32)
                if nm.size == 0:
                    return None
                amin = float(nm.min())
                amax = float(nm.max())
                if amax - amin > 1e-12:
                    norm = (nm - amin) / (amax - amin)
                else:
                    norm = _np.clip(nm, 0.0, 1.0)
                img = (_np.clip(norm * 255.0, 0, 255)).astype(_np.uint8)
                return img
        except Exception:
            return None
        return None

    def get_ascii(self, width: int = 100) -> str:
        """Return an ASCII/text visualization string. Defaults to render_full_display()."""
        try:
            if hasattr(self, 'render_full_display') and callable(self.render_full_display):
                return self.render_full_display()
            # fallback to brain growth
            if hasattr(self, 'render_brain_growth') and callable(self.render_brain_growth):
                return self.render_brain_growth()
        except Exception:
            return '[visualizer error]'
        return ''

@dataclass
class ConceptPrototype:
    name: str
    coordinates: np.ndarray
    experiences_count: int = 0
    discovered_at: int = 0
    confidence: float = 0.0

class ConceptualSpace:

    def __init__(self):
        self.prototypes: Dict[str, ConceptPrototype] = {}
        self.experience_points: List[np.ndarray] = []
        self.experience_contexts: List[Dict] = []
        self.min_cluster_size = 10
        self.similarity_threshold = 0.3
        self.discovery_interval = 50
        self.experience_count = 0
        self.last_clustering_at = 0
        self.prototype_id_counter = 0
        self.neural_map = None
        self._initialize_basic_dimensions()

    def _initialize_basic_dimensions(self):
        # Initialize basic qualia dimensions for experience grounding
        basic_dimensions = [
            {'name': 'prediction_error_magnitude', 'range': (0.0, 1.0)},
            {'name': 'stability_confidence_product', 'range': (0.0, 1.0)},
            {'name': 'inverse_meta_confidence', 'range': (0.0, 1.0)},
            {'name': 'stability_modulated_error', 'range': (0.0, 1.0)},
            {'name': 'barrier_instability_sum', 'range': (0.0, 1.0)}
        ]
        for dim in basic_dimensions:
            prototype = ConceptPrototype(
                name=dim['name'],
                coordinates=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),  # Initial neutral coordinates
                experiences_count=0,
                discovered_at=0,
                confidence=0.5
            )
            self.prototypes[dim['name']] = prototype

    def add_experience(self, qualia: 'QualiaState', context: Optional[Dict]=None):
        experience_vector = np.array([qualia.arousal, qualia.valence, qualia.entropy, qualia.engagement, qualia.frustration])
        self.experience_points.append(experience_vector)
        self.experience_contexts.append(context or {})
        self.experience_count += 1
        
        # Update nearest prototype
        distances = {}
        for key, proto in self.prototypes.items():
            dist = np.linalg.norm(experience_vector - proto.coordinates)
            distances[key] = dist
        if distances:
            nearest_key = min(distances, key=lambda k: distances[k])
            nearest_proto = self.prototypes[nearest_key]
            # Update prototype coordinates with moving average
            nearest_proto.coordinates = (nearest_proto.coordinates + experience_vector) / 2
            nearest_proto.experiences_count += 1
        
        if self.experience_count - self.last_clustering_at >= self.discovery_interval:
            self._discover_new_prototypes()
            self.last_clustering_at = self.experience_count

    def _discover_new_prototypes(self):
        # Get the latest experience vector and system state
        experience_vector = self.experience_points[-1] if self.experience_points else np.zeros(5)
        system_state = {'arousal': 0.5}  # Placeholder for actual system state
        if len(self.experience_points) < self.min_cluster_size:
            try:
                # Deferred imports for optional SCOPE visualization
                import cv2

                if hasattr(self, 'neural_map') and self.neural_map is not None:
                    norm_map = (self.neural_map - np.min(self.neural_map)) / (np.ptp(self.neural_map) + 1e-8)
                    frame = cv2.resize(norm_map.astype(np.float32), (256, 256), interpolation=cv2.INTER_CUBIC)
                else:
                    frame = np.zeros((256, 256), dtype=np.float32)

                print(f"[DEBUG] SCOPE input frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
                arousal = float(system_state.get('arousal', 0.5))
                drivers = {'arousal': arousal}
                scope_encoder = Scope()
                image, meta = scope_encoder.encode(frame, drivers)
                print(f"[DEBUG] SCOPE output image shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
                self.scope_image = image
            except Exception as e:
                # Keep exception handler aligned with the try block
                print(f"[DEBUG] SCOPE encoding error: {e}")
                self.scope_image = np.zeros((256, 256), dtype=np.uint8)
        current_coords = experience_vector
        distances = {}
        for key, proto in self.prototypes.items():
            dist = np.linalg.norm(current_coords - proto.coordinates)
            distances[key] = dist
        if not distances:
            return {'nearest_concept': 'unknown', 'nearest_distance': float('inf'), 'concept_blend': {}, 'coordinates': current_coords, 'semantic_meaning': 'Experiencing unknown qualia state'}
        nearest_key = min(distances, key=lambda k: distances[k])
        nearest_proto = self.prototypes[nearest_key]
        nearest_dist = distances[nearest_key]
        similarities = {k: 1.0 / (1.0 + d) for k, d in distances.items()}
        total_sim = sum(similarities.values())
        normalized_sim = {k: v / total_sim for k, v in similarities.items()}
        return {'nearest_concept': nearest_proto.name, 'nearest_distance': nearest_dist, 'concept_blend': normalized_sim, 'coordinates': current_coords, 'semantic_meaning': self._generate_meaning(nearest_proto, nearest_dist, normalized_sim)}

    def ground_experience(self, qualia: 'QualiaState') -> Dict[str, Any]:
        """Map a QualiaState to the nearest prototype (ground the experience).

        Returns a dict with nearest_concept, nearest_distance, concept_blend, coordinates and semantic_meaning.
        """
        current_coords = np.array([qualia.arousal, qualia.valence, qualia.entropy, qualia.engagement, qualia.frustration])
        distances = {}
        for key, proto in self.prototypes.items():
            dist = np.linalg.norm(current_coords - proto.coordinates)
            distances[key] = dist
        if not distances:
            return {'nearest_concept': 'unknown', 'nearest_distance': float('inf'), 'concept_blend': {}, 'coordinates': current_coords, 'semantic_meaning': 'Experiencing unknown qualia state'}
        nearest_key = min(distances, key=lambda k: distances[k])
        nearest_proto = self.prototypes[nearest_key]
        nearest_dist = distances[nearest_key]
        similarities = {k: 1.0 / (1.0 + d) for k, d in distances.items()}
        total_sim = sum(similarities.values())
        normalized_sim = {k: v / total_sim for k, v in similarities.items()}
        return {'nearest_concept': nearest_proto.name, 'nearest_distance': nearest_dist, 'concept_blend': normalized_sim, 'coordinates': current_coords, 'semantic_meaning': self._generate_meaning(nearest_proto, nearest_dist, normalized_sim)}

    def _generate_meaning(self, nearest: ConceptPrototype, distance: float, blend: Dict) -> str:
        if distance < 0.3:
            return f'Clear concept: {nearest.name}'
        elif distance < 0.6:
            top2 = sorted(blend.items(), key=lambda x: x[1], reverse=True)[:2]
            concept1 = self.prototypes[top2[0][0]].name
            concept2 = self.prototypes[top2[1][0]].name
            return f'Blend of {concept1} + {concept2}'
        else:
            return f'Ambiguous concept (nearest: {nearest.name})'

class EventType(Enum):
    HIGH_UNCERTAINTY = 'high_uncertainty'
    PREDICTION_ERROR = 'prediction_error'
    GOAL_ACHIEVED = 'goal_achieved'
    GOAL_FAILED = 'goal_failed'
    MODEL_CONFIDENCE_LOW = 'model_confidence_low'
    MODEL_REVISION_NEEDED = 'model_revision_needed'
    SELF_CONTRADICTION = 'self_contradiction'
    TENSION_SPIKE = 'tension_spike'
    HARMONY_ACHIEVED = 'harmony_achieved'
    FLOW_STATE_ENTERED = 'flow_state_entered'
    STATE_INSTABILITY = 'state_instability'
    BARRIER_VIOLATION = 'barrier_violation'
    CONVERGENCE_DETECTED = 'convergence_detected'
    META_MODEL_UPDATED = 'meta_model_updated'
    ATTENTION_EXHAUSTED = 'attention_exhausted'
    WAKE_UP_TRIGGER = 'wake_up_trigger'

@dataclass
class Event:
    type: EventType
    timestamp: int
    importance: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        return self.importance < other.importance

class GrowingSOM(EvolutionVisualizer):
    def _adaptive_sigma(self, iteration: int = 0):
        # Use the same logic as sigma_decay
        initial_sigma = 2.0
        final_sigma = 0.5
        decay_rate = 0.001
        import numpy as _np
        return final_sigma + (initial_sigma - final_sigma) * _np.exp(-decay_rate * iteration)
    def _neighborhood_function(self, distance: float, sigma: float) -> float:
        # Standard Gaussian neighborhood function
        import numpy as _np
        return _np.exp(- (distance ** 2) / (2 * (sigma ** 2)))

    def __init__(self, input_dim: int=5, initial_size: int=2, rng=None):
        super().__init__()
        self.rng = rng or default_rng()
        self.input_dim = input_dim
        self.neurons = []
        self.connections = []
        self.input_history = deque(maxlen=1000)
        self.total_activations = 0
        self.generation = 0
        self.consciousness_level = 0
        self.growth_stage = 'infant'
        self.energy_history = deque(maxlen=1000)
        self.connection_history = deque(maxlen=1000)
        self.total_connections = 0
        self.scale_factor = 1.0
        self.neural_map = None
        self.scope_image = None
        self.phi_value = 0.0
        self.qualia_data = None
        self.current_experience = 'unknown'
        self.memory_count = 0
        self.unity_score = 0.5
        self.neuron_count = initial_size * initial_size
        self.connection_count = 0
        self.growth_events = 0
        self.pruning_events = 0
        self.connection_threshold = 0.01
        self.min_error_variance = 0.001  # Adaptive threshold for error variance
        for i in range(initial_size):
            for j in range(initial_size):
                neuron = {
                    'weights': self.rng.standard_normal(self.input_dim) * 0.1,
                    'position': (i, j),
                    'age': 0,
                    'error': 0.0,
                    'activation_count': 0,
                    'recent_errors': deque(maxlen=20),
                    'specialization': 0.0,
                    'utility': 1.0
                }
                self.neurons.append(neuron)
        for i in range(len(self.neurons)):
            for j in range(i + 1, len(self.neurons)):
                pos1 = self.neurons[i]['position']
                pos2 = self.neurons[j]['position']
                dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                if dist == 1:
                    self.connections.append({'from': i, 'to': j, 'strength': 1.0, 'age': 0, 'usage': 0})
        self.base_neighbor_rate = 0.02
        # SCOPE image (updated each generation)
    def update(self, system_state: Dict[str, Any]):
        self.generation += 1
        u_matrix = system_state.get('u_matrix', None)
        if u_matrix is not None:
            self._update_neural_map(u_matrix)
        self.qualia_data = system_state.get('qualia', None)
        self.current_experience = system_state.get('current_experience', 'unknown')
        self.memory_count = system_state.get('memories', 0)
        self.unity_score = system_state.get('unity', 0.5)
        self.neuron_count = system_state.get('neuron_count', 4)
        self.connection_count = system_state.get('connection_count', 4)
        self.growth_events = system_state.get('growth_events', 0)
        strange_loop = system_state.get('strange_loop', False)
        meta_awareness = system_state.get('meta_awareness', 0.0)
        self.phi_value = system_state.get('phi', 0.0)
        old_level = self.consciousness_level
        # SCOPE image generation
        import numpy as _np
        try:
            frame = None
            if self.neural_map is not None:
                norm_map = (self.neural_map - _np.min(self.neural_map)) / (_np.ptp(self.neural_map) + 1e-8)
                frame = norm_map.astype(_np.float32)
            else:
                frame = _np.zeros((30,80), dtype=_np.float32)
            arousal = float(system_state.get('arousal', 0.5))
            drivers = {'arousal': arousal}
            scope_encoder = Scope()
            image, meta = scope_encoder.encode(frame, drivers)
            self.scope_image = image
        except Exception as e:
            self.scope_image = _np.zeros((256,256), dtype=_np.uint8)
        # consciousness/growth stage update
        phi_policy = _normalize_phi_policy(system_state.get("phi_policy") if isinstance(system_state, dict) else None)
        if self.phi_value > float(phi_policy.get("high", 0.5)):
            self.consciousness_level = 4
            self.growth_stage = 'transcendent'
        elif self.phi_value > float(phi_policy.get("mid", 0.3)):
            self.consciousness_level = 3
            self.growth_stage = 'adult'
        elif meta_awareness > 0.5:
            self.consciousness_level = 2
            self.growth_stage = 'child'
        elif strange_loop:
            self.consciousness_level = 1
            self.growth_stage = 'infant'
        if self.consciousness_level > old_level:
            self.add_major_event(f'EVOLVED to {self.growth_stage.upper()}!')
        energy = system_state.get('energy', 0.0)
        self.energy_history.append(energy)
        self.connection_history.append(self.total_connections)
        if len(self.connection_history) > 10:
            recent_avg = np.mean(list(self.connection_history)[-10:])
            if recent_avg > 1000:
                target_scale = 1000 / recent_avg
                self.scale_factor = 0.9 * self.scale_factor + 0.1 * target_scale
                self.scale_factor = max(0.1, min(1.0, self.scale_factor))
    # sigma decay function
    def sigma_decay(self, iteration: int = 0):
        import numpy as _np
        initial_sigma = 2.0
        final_sigma = 0.5
        decay_rate = 0.001
        return final_sigma + (initial_sigma - final_sigma) * _np.exp(-decay_rate * iteration)

    def find_bmu(self, input_vec: np.ndarray) -> Tuple[int, float]:
        min_dist = float('inf')
        bmu_idx = 0
        for i, neuron in enumerate(self.neurons):
            base_dist = np.linalg.norm(input_vec - neuron['weights'])
            specialization_bonus = neuron['specialization'] * 0.1
            adjusted_dist = base_dist - specialization_bonus
            if adjusted_dist < min_dist:
                min_dist = adjusted_dist
                bmu_idx = i
        return (bmu_idx, max(0.0, min_dist))

    def learn(self, input_vec: np.ndarray) -> Dict[str, Any]:
        self.input_history.append(input_vec.copy())
        bmu_idx, error = self.find_bmu(input_vec)
        bmu = self.neurons[bmu_idx]
        bmu['activation_count'] += 1
        bmu['error'] += error
        bmu['age'] += 1
        bmu['recent_errors'].append(error)
        self.total_activations += 1
        learning_rate = self._learning_gate(bmu)
        sigma = self._adaptive_sigma(self.total_activations)
        delta = learning_rate * (input_vec - bmu['weights'])
        bmu['weights'] += delta
        if len(bmu['recent_errors']) >= 5:
            avg_recent_error = np.mean(list(bmu['recent_errors']))
            bmu['specialization'] = 1.0 / (1.0 + avg_recent_error)
        bmu_pos = np.array(bmu['position'])
        for i, neuron in enumerate(self.neurons):
            if i == bmu_idx:
                continue
            pos = np.array(neuron['position'])
            distance = np.linalg.norm(pos - bmu_pos)
            neighbor_influence = self._neighborhood_function(float(distance), float(sigma))
            if neighbor_influence > 0.01:
                neighbor_rate = self.base_neighbor_rate * neighbor_influence
                neighbor_delta = neighbor_rate * (input_vec - neuron['weights'])
                neuron['weights'] += neighbor_delta
                neuron['age'] += 1
                neuron['utility'] *= 1.001
        grew = False
        # Check if we should grow a new neuron
        should_grow = self._should_grow(bmu_idx, error)
        if should_grow:
            grew = self._grow_neuron_intelligent(bmu_idx, input_vec)
        self._update_connections(bmu_idx)
        pruned_count = self._prune_weak_connections()
        for i, neuron in enumerate(self.neurons):
            if i != bmu_idx:
                neuron['utility'] *= 0.9999
        return {'bmu': bmu_idx, 'error': error, 'grew': grew, 'pruned': pruned_count, 'neuron_count': len(self.neurons), 'learning_rate': learning_rate, 'sigma': sigma}

    def _should_grow(self, bmu_idx: int, current_error: float) -> bool:
        bmu = self.neurons[bmu_idx]
        if len(bmu['recent_errors']) < 10:
            return False
        recent_errors = list(bmu['recent_errors'])
        error_variance = np.var(recent_errors)
        error_mean = np.mean(recent_errors)
        if error_variance < self.min_error_variance or error_mean < 0.15:
            return False
        if len(self.input_history) >= 20:
            recent_inputs = list(self.input_history)[-20:]
            input_variance = np.mean(np.var(recent_inputs, axis=0))
            if input_variance < 0.02:
                return False
        if self.total_activations - getattr(self, '_last_growth', 0) < 15:
            return False
        return True

    def _grow_neuron_intelligent(self, bmu_idx: int, input_vec: np.ndarray) -> bool:
        bmu_pos = self.neurons[bmu_idx]['position']
        best_pos = None
        best_score = -1
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            candidate = (bmu_pos[0] + dx, bmu_pos[1] + dy)
            position_occupied = False
            for n in self.neurons:
                n_pos = tuple(n['position']) if isinstance(n['position'], np.ndarray) else n['position']
                if n_pos == candidate:
                    position_occupied = True
                    break
            if position_occupied:
                continue
            score = self._evaluate_growth_position(candidate, bmu_idx)
            if score > best_score:
                best_score = score
                best_pos = candidate
        if best_pos is None:
            return False
        bmu_weights = self.neurons[bmu_idx]['weights']
        direction = input_vec - bmu_weights
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
            new_weights = bmu_weights + 0.3 * direction_norm * direction
        else:
            new_weights = bmu_weights + self.rng.standard_normal(self.input_dim) * 0.05
        new_neuron = {'weights': new_weights, 'position': best_pos, 'age': 0, 'error': 0.0, 'activation_count': 0, 'recent_errors': deque(maxlen=20), 'specialization': 0.0, 'utility': 1.0}
        new_idx = len(self.neurons)
        self.neurons.append(new_neuron)
        self._create_intelligent_connections(new_idx, bmu_idx)
        self.growth_events += 1
        self._last_growth = self.total_activations
        return True

    def _evaluate_growth_position(self, position: Tuple[int, int], bmu_idx: int) -> float:
        score = 0.0
        for i, neuron in enumerate(self.neurons):
            pos = neuron['position']
            dist = abs(pos[0] - position[0]) + abs(pos[1] - position[1])
            if dist == 1:
                score += neuron['utility'] * 0.5
            elif dist == 2:
                score += neuron['utility'] * 0.2
        return score

    def _create_intelligent_connections(self, new_idx: int, bmu_idx: int):
        new_pos = self.neurons[new_idx]['position']
        self.connections.append({'from': min(bmu_idx, new_idx), 'to': max(bmu_idx, new_idx), 'strength': 1.5, 'age': 0, 'usage': 0})
        for i, neuron in enumerate(self.neurons):
            if i == new_idx:
                continue
            pos = neuron['position']
            dist = abs(pos[0] - new_pos[0]) + abs(pos[1] - new_pos[1])
            if dist == 1:
                strength = 1.0 * neuron['utility']
                self.connections.append({'from': min(i, new_idx), 'to': max(i, new_idx), 'strength': strength, 'age': 0, 'usage': 0})

    def _update_connections(self, bmu_idx: int):
        for conn in self.connections:
            conn['age'] += 1
            if conn['from'] == bmu_idx or conn['to'] == bmu_idx:
                conn['strength'] = min(3.0, conn['strength'] * 1.02)
                conn['usage'] += 1
            else:
                conn['strength'] *= 0.999

    def _prune_weak_connections(self) -> int:
        initial_count = len(self.connections)
        self.connections = [conn for conn in self.connections if conn['strength'] > self.connection_threshold and conn['usage'] > 0]
        max_age = max(1000, self.total_activations // 10)
        self.connections = [conn for conn in self.connections if conn['age'] < max_age or conn['usage'] > 10]
        pruned = initial_count - len(self.connections)
        if pruned > 0:
            self.pruning_events += 1
        return pruned
        return pruned

    def get_statistics(self) -> Dict[str, Any]:
        if not self.neurons:
            return {}
        neuron_count = len(self.neurons)
        connection_count = len(self.connections)
        ages = [n['age'] for n in self.neurons]
        activations = [n['activation_count'] for n in self.neurons]
        specializations = [n['specialization'] for n in self.neurons]
        utilities = [n['utility'] for n in self.neurons]
        strengths = [c['strength'] for c in self.connections] if self.connections else [0]
        connection_ages = [c['age'] for c in self.connections] if self.connections else [0]
        if neuron_count > 1:
            weights_matrix = np.array([n['weights'] for n in self.neurons])
            weight_diversity = np.mean(np.std(weights_matrix, axis=0))
        else:
            weight_diversity = 0.0
        return {'neuron_count': neuron_count, 'connection_count': connection_count, 'total_activations': self.total_activations, 'growth_events': self.growth_events, 'pruning_events': self.pruning_events, 'avg_age': np.mean(ages), 'avg_activations': np.mean(activations), 'avg_specialization': np.mean(specializations), 'avg_utility': np.mean(utilities), 'max_strength': np.max(strengths), 'avg_strength': np.mean(strengths), 'avg_connection_age': np.mean(connection_ages), 'weight_diversity': weight_diversity, 'network_efficiency': connection_count / max(1, neuron_count ** 2), 'specialization_ratio': len([s for s in specializations if s > 0.7]) / neuron_count}

    def get_network_state(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.neurons:
            return (np.array([]), np.array([]))
        positions = np.array([n['position'] for n in self.neurons])
        activities = np.array([n['activation_count'] * (1 + n['specialization']) for n in self.neurons])
        return (positions, activities)

    def get_topology_health(self) -> Dict[str, float]:
        if len(self.neurons) < 2:
            return {'health': 1.0, 'connectivity': 1.0, 'balance': 1.0}
        total_possible_connections = len(self.neurons) * (len(self.neurons) - 1) // 2
        connectivity = len(self.connections) / max(1, total_possible_connections)
        activations = [n['activation_count'] for n in self.neurons]
        if max(activations) > 0:
            activation_balance = 1.0 - np.std(activations) / np.mean(activations)
        else:
            activation_balance = 1.0
        health = (connectivity + activation_balance) / 2.0
        # Ensure returned values are built-in float (not numpy types) to satisfy typing Dict[str, float]
        return {'health': float(health), 'connectivity': float(connectivity), 'balance': float(activation_balance)}
    def _learning_gate(self, bmu: dict) -> float:
        """
        Compute an adaptive learning rate for the BMU (best matching unit).
        Uses recent error and activation count to modulate the base learning rate.
        Returns a plain Python float in range [0.01, 0.3].
        """

        # recent_errors may be a deque or list of numeric values
        recent_errors = list(bmu.get('recent_errors', [])) if bmu.get('recent_errors', None) is not None else []
        if recent_errors:
            try:
                recent_error = float(sum(recent_errors) / len(recent_errors))
            except Exception:
                # Fallback if elements are numpy types etc.
                recent_error = float(sum(float(x) for x in recent_errors) / len(recent_errors))
        else:
            recent_error = 0.1

        activation_count = int(bmu.get('activation_count', 1))

        base = 0.05
        error_factor = min(1.0, recent_error / 0.5)  # normalize error contribution
        # reduce learning rate as activation_count grows (logarithmic damping)
        activation_factor = 1.0 / (1.0 + math.log1p(max(1, activation_count)))
        rate = base + 0.2 * error_factor * activation_factor

       
        rate = float(max(0.01, min(0.3, rate)))
        return rate

class QualiaState:

    def __init__(self, arousal: float=0.0, valence: float=0.0, entropy: float=0.0, engagement: float=0.0, frustration: float=0.0, history_size: Optional[int]=None, message_bus: Optional[MessageBus]=None):
        self.arousal = arousal
        self.valence = valence
        self.entropy = entropy
        self.engagement = engagement
        self.frustration = frustration
        if history_size is None:
            available_mb = 0 // (1024 * 1024)
            if available_mb > 8000:
                history_size = 512
            elif available_mb > 4000:
                history_size = 512
            else:
                history_size = 512
        self.history = deque(maxlen=history_size)
        self.max_history_size = 512
        
        # MULTI-LOOP INFRASTRUCTURE
        self.message_bus = message_bus
        if self.message_bus:
            self.message_bus.register_module('qualia')
        # Influences from other modules
        self.workspace_fullness = 0.0
        self.phi_level = 0.0
        self.energy_level = 100.0

    def compute(self, delta_hat: float, m: float, gap: float, barrier_viols: int, meta_conf: float, stability: float) -> 'QualiaState':
        # MULTI-LOOP: Process incoming messages before computing
        self._process_incoming_messages()
        
        arousal_input = delta_hat * 3.0
        # Workspace fullness affects arousal
        arousal_input += self.workspace_fullness * 0.5
        self.arousal = np.tanh(arousal_input)
        
        valence_raw = stability * meta_conf * np.exp(-barrier_viols * 0.2)
        # Phi level affects integration feeling (valence)
        valence_raw *= (0.8 + 0.2 * self.phi_level)
        self.valence = np.clip(valence_raw, 0.0, 1.0)
        
        entropy_input = (1.0 - meta_conf) * 2.0
        self.entropy = np.tanh(entropy_input)
        stability_factor = 1.0 - abs(stability - 0.65) * 1.5
        stability_factor = max(0.1, stability_factor)
        engagement_input = delta_hat * stability_factor * 4.0
        # Phi affects engagement
        engagement_input *= (0.5 + 0.5 * self.phi_level)
        self.engagement = np.tanh(engagement_input)
        
        frustration_input = barrier_viols * 0.5 + (1.0 - stability) * 2.0
        # Low energy increases frustration
        if self.energy_level < 30:
            frustration_input += 0.3
        self.frustration = np.tanh(frustration_input)
        
        self.history.append({'arousal': self.arousal, 'valence': self.valence, 'entropy': self.entropy, 'engagement': self.engagement, 'frustration': self.frustration})
        
        # MULTI-LOOP: Broadcast qualia state to all modules
        self._broadcast_qualia_state()
        
        return self
    
    def _process_incoming_messages(self):
        """Process messages from other modules"""
        if not self.message_bus:
            return
        
        msgs = self.message_bus.receive_all('qualia', max_msgs=30)
        for msg in msgs:
            try:
                if msg.type == 'workspace_update':
                    self.workspace_fullness = msg.payload.get('fullness', 0.0)
                    
                elif msg.type == 'phi_update':
                    self.phi_level = msg.payload.get('phi', 0.0)
                    # Normalize to 0-1
                    self.phi_level = float(np.clip(self.phi_level * 2.0, 0.0, 1.0))
                    
                elif msg.type == 'energy_state':
                    self.energy_level = msg.payload.get('energy', 100.0)
                    
            except Exception:
                pass
    
    def _broadcast_qualia_state(self):
        """Broadcast qualia to all modules"""
        if not self.message_bus:
            return
        
        msg = Message(
            source='qualia',
            target='broadcast',
            type='qualia_state',
            payload={
                'arousal': float(self.arousal),
                'valence': float(self.valence),
                'entropy': float(self.entropy),
                'engagement': float(self.engagement),
                'frustration': float(self.frustration)
            },
            priority=0.7
        )
        self.message_bus.send(msg)

    def dominant_feeling(self) -> str:
        feelings = {'arousal': self.arousal, 'valence': self.valence, 'entropy': self.entropy, 'engagement': self.engagement, 'frustration': self.frustration}
        return max(feelings, key=lambda k: feelings[k])

    def to_dict(self) -> Dict[str, float]:
        return {'q_arousal': float(self.arousal), 'q_valence': float(self.valence), 'q_entropy': float(self.entropy), 'q_engagement': float(self.engagement), 'q_frustration': float(self.frustration)}

@dataclass
class ConsciousContent:
    timestamp: int
    content_type: str
    content: Any
    salience: float
    semantic_meaning: str

    def __lt__(self, other):
        return self.salience < other.salience

class GlobalWorkspace:
    def __init__(self, capacity: int=3, concept_space: Optional[ConceptualSpace]=None, message_bus: Optional[MessageBus]=None):
        self.capacity = capacity
        self.current_contents: List[ConsciousContent] = []
        self.competitors: List[ConsciousContent] = []
        self.broadcast_history: deque = deque(maxlen=100)
        self.attention_focus: Optional[ConsciousContent] = None
        self.attention_switching_threshold = 0.45
        self.attention_persistence = 0.0
        self.attention_history: deque = deque(maxlen=20)
        self.content_associations: Dict[str, List[str]] = {}
        self.semantic_clusters: Dict[str, List[ConsciousContent]] = {}
        self.integration_threshold = 0.35
        self.exploration_bias = 0.75
        self.policy_state_history: deque = deque(maxlen=200)
        self.error_detection_buffer: deque = deque(maxlen=50)
        self.policy_adjustments: deque = deque(maxlen=100)
        self.meta_learning_rate = 0.12
        self.policy_momentum = {}
        self.policy_variance_tracking = {}
        self.policy_params = {
            'exploration_bias': 0.75,
            'stability_bias': 0.3,
            'confidence_threshold': 0.25,
            'error_sensitivity': 1.8,
            'learning_rate': 0.15,
            'attention_focus_strength': 0.85,
            'integration_eagerness': 0.8,
            'prediction_confidence': 0.4,
            'connection_prune_threshold': 0.02,
        }
        for param in self.policy_params:
            self.policy_momentum[param] = 0.0
            self.policy_variance_tracking[param] = deque(maxlen=50)
        self.cumulative_reward = 0.0
        self.reward_history: deque = deque(maxlen=100)
        self.performance_metrics = {'prediction_accuracy': deque(maxlen=100), 'response_time': deque(maxlen=100), 'integration_success': deque(maxlen=100)}
        self.predictions: Dict[str, Any] = {}
        self.prediction_confidence = 0.5
        self.broadcast_hooks: List[Callable] = []
        self.last_broadcast_state: Optional[Dict] = None
        self.last_content: Optional[Dict[str, Any]] = None
        self._last_selection_mode = False
        # Optional conceptual space that can influence policy/attention
        self.concept_space: Optional[ConceptualSpace] = concept_space
        # Minimize direct phi influence; rate-limit already applied in apply_system_health
        self.phi_weight = 0.2
        # Track policy adjustments and simple outcome signals for offline regression/bandit
        self.policy_effects: deque = deque(maxlen=500)
        
        # MULTI-LOOP INFRASTRUCTURE
        self.message_bus = message_bus
        if self.message_bus:
            self.message_bus.register_module('workspace')
        # Track influence from other modules
        self.qualia_influence = 0.5  # From Qualia module
        self.phi_influence = 0.5  # From Phi calculator
        self.energy_influence = 1.0  # From Energy controller
        self.policy_influence = 0.5  # From Policy module
        self._policy_param_allowlist = {'exploration_bias'}
        self._policy_param_shadow_writes = deque(maxlen=200)

    def _write_policy_param(self, param_name: str, value: float, source: str = 'unknown') -> bool:
        current_value = self.policy_params.get(param_name)
        if param_name in self._policy_param_allowlist:
            self.policy_params[param_name] = value
            if param_name == 'exploration_bias':
                try:
                    self.exploration_bias = float(value)
                except Exception:
                    pass
            logging.debug(
                '[GlobalWorkspace] policy param write: %s=%s (prev=%s, source=%s)',
                param_name,
                value,
                current_value,
                source,
            )
            return True
        if current_value == value:
            return False

        self._policy_param_shadow_writes.append(
            {
                'timestamp': self._get_current_timestamp(),
                'source': source,
                'param': param_name,
                'requested_value': value,
                'current_value': self.policy_params.get(param_name),
            }
        )
        logging.info(
            '[GlobalWorkspace] policy param write blocked by allowlist (shadow mode): %s %s=%s',
            source,
            param_name,
            value,
        )
        return False

    def get_policy_param_shadow_writes(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        if n is None:
            return list(self._policy_param_shadow_writes)
        try:
            n = int(n)
            if n <= 0:
                return []
            return list(self._policy_param_shadow_writes)[-n:]
        except Exception:
            return list(self._policy_param_shadow_writes)

    def _offline_tune_policy(self):
        """Use a simple ridge regression over recent policy_effects to recommend tiny nudges.

        Recommends small, rate-limited tweaks to exploration_bias and learning_rate only.
        """
        try:
            import numpy as _np
            data = list(self.policy_effects)
            if len(data) < 50:
                return
            X = []
            y = []
            for r in data:
                X.append([r.get('d_lr', 0.0), r.get('d_eb', 0.0), r.get('d_att', 0.0), r.get('d_prune', 0.0), r.get('phi', 0.0), r.get('overall', 0.5)])
                # target: reward_delta with a small boost for -td_error
                y.append(float(r.get('reward_delta', 0.0)) + 0.2 * float(r.get('neg_td_error', 0.0)))
            X = _np.asarray(X, dtype=_np.float64)
            y = _np.asarray(y, dtype=_np.float64)
            # ridge: (X^T X +  I)w = X^T y
            lam = 1e-3
            XT = X.T
            A = XT @ X + lam * _np.eye(X.shape[1])
            b = XT @ y
            w = _np.linalg.solve(A, b)
            # interpret signs for d_lr, d_eb
            coef_lr = float(w[0])
            coef_eb = float(w[1])
            # propose tiny nudges ( 0.02)
            eb0 = float(self.policy_params.get('exploration_bias', 0.75))
            lr0 = float(self.policy_params.get('learning_rate', 0.15))
            def _rl(old, delta, lim):
                return float(_np.clip(old + delta, old - lim, old + lim))
            # scale deltas conservatively
            new_eb = _rl(eb0, _np.clip(coef_eb, -0.02, 0.02), 0.02)
            new_lr = _rl(lr0, _np.clip(coef_lr, -0.02, 0.02), 0.02)
            adjustments = {}
            if abs(new_eb - eb0) > 1e-6:
                if self._write_policy_param('exploration_bias', new_eb, source='offline_tune'):
                    adjustments['exploration_bias'] = new_eb - eb0
            if abs(new_lr - lr0) > 1e-6:
                if self._write_policy_param('learning_rate', new_lr, source='offline_tune'):
                    adjustments['learning_rate'] = new_lr - lr0
            if adjustments:
                try:
                    self.policy_adjustments.append({'timestamp': self._get_current_timestamp(), 'adjustments': adjustments, 'reason': 'offline_tune', 'coef_lr': coef_lr, 'coef_eb': coef_eb})
                except Exception:
                    pass
        except Exception:
            pass

    def apply_system_health(self, health: Optional[Dict[str, float]], phi: float):
        """Adjust GlobalWorkspace policy parameters from health and phi.

        This is intentionally conservative: small adjustments are recorded and
        can be examined via self.policy_adjustments.
        """
        try:
            overall = float(health.get('overall_health', 0.5)) if health else 0.5
        except Exception:
            overall = 0.5
        try:
            phi_val = float(phi)
        except Exception:
            phi_val = 0.0

        base_lr = float(self.policy_params.get('learning_rate', 0.15))
        # Response to higher phi: accelerate learning, but down-weighted and rate-limited per step
        lr_scale = 1.0 + self.phi_weight * (phi_val - 0.5) * 1.2
        if overall < 0.3:
            lr_scale *= 0.8
        target_lr = float(np.clip(base_lr * lr_scale, 0.01, 1.2))

        eb0 = float(self.policy_params.get('exploration_bias', 0.75))
        # Reduce exploration when phi is high (favor exploitation) down-weighted by phi_weight
        target_eb = float(np.clip(eb0 * (1.0 - self.phi_weight * (phi_val - 0.5) * 0.9), 0.05, 0.95))

        att0 = float(self.policy_params.get('attention_focus_strength', 0.85))
        target_att = float(np.clip(att0 * (1.0 + self.phi_weight * phi_val * 0.6) * (0.8 + overall * 0.5), 0.1, 2.0))

        prune0 = float(self.policy_params.get('connection_prune_threshold', 0.02))
        target_prune = float(np.clip(prune0 * (1.0 + self.phi_weight * (0.5 - phi_val) * 2.0) * (1.0 + (0.5 - overall) * 1.0), 1e-4, 1.0))

        # Rate limit per-step change
        def _rl(old, new, lim):
            return float(np.clip(new, old - lim, old + lim))
        new_lr = _rl(base_lr, target_lr, 0.05)
        eb = _rl(eb0, target_eb, 0.05)
        att = _rl(att0, target_att, 0.1)
        prune = _rl(prune0, target_prune, 0.01)

        adjustments = {}
        if self._write_policy_param('learning_rate', new_lr, source='system_health') and abs(new_lr - base_lr) > 1e-6:
            adjustments['learning_rate'] = new_lr - base_lr
        if self._write_policy_param('exploration_bias', eb, source='system_health') and abs(eb - eb0) > 1e-6:
            adjustments['exploration_bias'] = eb - eb0
        if self._write_policy_param('attention_focus_strength', att, source='system_health') and abs(att - att0) > 1e-6:
            adjustments['attention_focus_strength'] = att - att0
        if self._write_policy_param('connection_prune_threshold', prune, source='system_health') and abs(prune - prune0) > 1e-8:
            adjustments['connection_prune_threshold'] = prune - prune0

        if adjustments:
            record = {
                'timestamp': self._get_current_timestamp(),
                'adjustments': adjustments,
                'health_overall': overall,
                'phi': phi_val,
                'pre': {'learning_rate': base_lr, 'exploration_bias': eb0, 'attention_focus_strength': att0, 'connection_prune_threshold': prune0},
                'post': {'learning_rate': new_lr, 'exploration_bias': eb, 'attention_focus_strength': att, 'connection_prune_threshold': prune},
            }
            try:
                self.policy_adjustments.append(record)
            except Exception:
                pass
            # Log coarse outcome signals (reward delta, -td_error) for offline estimation
            try:
                rew_hist = list(self.reward_history)
                reward_delta = float(rew_hist[-1] - rew_hist[-2]) if len(rew_hist) >= 2 else 0.0
                td_err = 0.0
                if isinstance(self.last_broadcast_state, dict):
                    td_err = float(self.last_broadcast_state.get('td_error', 0.0))
                self.policy_effects.append({
                    'd_lr': new_lr - base_lr,
                    'd_eb': eb - eb0,
                    'd_att': att - att0,
                    'd_prune': prune - prune0,
                    'reward_delta': reward_delta,
                    'neg_td_error': -td_err,
                    'phi': phi_val,
                    'overall': overall,
                })
            except Exception:
                pass
            # Periodically run offline tuner
            try:
                if len(self.policy_effects) % 25 == 0:
                    self._offline_tune_policy()
            except Exception:
                pass

    def _process_incoming_messages(self):
        """
        MULTI-LOOP: Process messages from other modules to update workspace state.
        This enables bidirectional influence between workspace and other systems.
        """
        if not self.message_bus:
            return
        
        msgs = self.message_bus.receive_all('workspace', max_msgs=50)
        
        for msg in msgs:
            try:
                if msg.type == 'qualia_state':
                    # Qualia influences salience weighting
                    arousal = msg.payload.get('arousal', 0.5)
                    self.qualia_influence = float(arousal)
                    # High arousal -> increase exploration
                    if arousal > 0.7:
                        self._write_policy_param(
                            'exploration_bias',
                            min(0.95, self.exploration_bias * 1.1),
                            source='incoming_message_qualia',
                        )
                    
                elif msg.type == 'phi_update':
                    # Phi influences integration threshold
                    phi = msg.payload.get('phi', 0.0)
                    self.phi_influence = float(np.clip(phi * 2.0, 0.0, 1.0))
                    # High phi -> easier integration
                    self.integration_threshold = 0.35 * (1.0 - self.phi_influence * 0.3)
                    
                elif msg.type == 'energy_state':
                    # Energy influences capacity
                    energy_ratio = msg.payload.get('ratio', 1.0)
                    self.energy_influence = float(energy_ratio)
                    # Low energy -> reduce capacity
                    if energy_ratio < 0.3:
                        self.capacity = max(1, int(3 * energy_ratio))
                    else:
                        self.capacity = 3
                    
                elif msg.type == 'policy_request':
                    # Policy module requesting current focus
                    self._send_focus_to_policy()
                    
                elif msg.type == 'self_model_state':
                    # Self-model state can be submitted as content
                    meta_awareness = msg.payload.get('belief', {}).get('stability', 0.5).get('meta_awareness', 0.5)
                    if meta_awareness > 0.6:
                        # High meta-awareness increases attention persistence
                        self.attention_persistence = min(0.9, self.attention_persistence + 0.1)
                        
            except Exception as e:
                pass  # Silently ignore malformed messages
    
    def _send_focus_to_policy(self):
        """Send current attention focus to policy module"""
        if not self.message_bus or not self.attention_focus:
            return
        
        msg = Message(
            source='workspace',
            target='policy',
            type='focus_response',
            payload={'focus': {
                'type': self.attention_focus.content_type,
                'salience': self.attention_focus.salience,
                'content': str(self.attention_focus.content)[:100]  # Truncate
            }},
            priority=0.9
        )
        self.message_bus.send(msg)
    
    def _broadcast_workspace_state(self):
        """
        MULTI-LOOP: Broadcast workspace state to all other modules.
        Called after competition to inform modules of consciousness contents.
        """
        if not self.message_bus:
            return
        
        focus_data = None
        if self.attention_focus:
            focus_data = {
                'type': self.attention_focus.content_type,
                'salience': self.attention_focus.salience,
                'timestamp': self.attention_focus.timestamp
            }
        
        msg = Message(
            source='workspace',
            target='broadcast',
            type='workspace_update',
            payload={
                'focus': focus_data,
                'contents': [{
                    'type': c.content_type,
                    'salience': c.salience
                } for c in self.current_contents],
                'fullness': len(self.current_contents) / max(1, self.capacity),
                'exploration_bias': self.exploration_bias
            },
            priority=0.8
        )
        self.message_bus.send(msg)

    def submit_for_competition(self, content: ConsciousContent):
        integration_threshold = 0.7 - self.exploration_bias * 0.3
        integrated = self._try_integrate_content(content, integration_threshold)
        if integrated:
            return
        adjusted_content = self._adjust_salience_with_exploration(content)
        self.competitors.append(adjusted_content)
        self._update_semantic_clusters(adjusted_content)

    def _try_integrate_content(self, new_content: ConsciousContent, threshold: float=0.7) -> bool:
        for existing in self.current_contents + self.competitors:
            similarity = self._calculate_semantic_similarity(new_content, existing)
            if similarity > threshold:
                if self.exploration_bias > 0.5:
                    existing.salience = max(existing.salience, new_content.salience * 1.2)
                    existing.content = f'{existing.content} + {new_content.content}'
                else:
                    existing.salience *= 1.1
                return True
        return False

    def _calculate_semantic_similarity(self, content1: ConsciousContent, content2: ConsciousContent) -> float:
        type_similarity = 0.8 if content1.content_type == content2.content_type else 0.0
        if hasattr(content1, 'semantic_meaning') and hasattr(content2, 'semantic_meaning'):
            meaning1_str = str(content1.semantic_meaning) if content1.semantic_meaning else ''
            meaning2_str = str(content2.semantic_meaning) if content2.semantic_meaning else ''
            meaning1_words = set(meaning1_str.lower().split())
            meaning2_words = set(meaning2_str.lower().split())
        else:
            return type_similarity
        if len(meaning1_words) == 0 or len(meaning2_words) == 0:
            semantic_similarity = 0.0
        else:
            intersection = len(meaning1_words & meaning2_words)
            union = len(meaning1_words | meaning2_words)
            semantic_similarity = intersection / union if union > 0 else 0.0
        time_diff = abs(content1.timestamp - content2.timestamp)
        time_similarity = max(0.0, 1.0 - time_diff / 100.0)
        return type_similarity * 0.4 + semantic_similarity * 0.4 + time_similarity * 0.2

    def _adjust_salience_contextually(self, content: ConsciousContent) -> ConsciousContent:
        adjusted_salience = content.salience
        if len(self.attention_history) > 0:
            recent_types = [att.content_type for att in list(self.attention_history)[-5:]]
            same_type_count = recent_types.count(content.content_type)
            if same_type_count >= 3:
                adjusted_salience *= 0.7
            elif content.content_type not in recent_types:
                adjusted_salience *= 1.2
        associations = self.content_associations.get(content.content_type, [])
        if self.attention_focus and self.attention_focus.content_type in associations:
            adjusted_salience *= 1.15
        try:
            if hasattr(content, 'semantic_meaning') and 'uncertainty' in content.semantic_meaning.lower():
                adjusted_salience *= 1.0 + self.policy_params['error_sensitivity'] * 0.3
        except (AttributeError, TypeError):
            pass
        if 'goal' in content.content_type:
            adjusted_salience *= 1.0 + self.policy_params['stability_bias'] * 0.2
        return ConsciousContent(timestamp=content.timestamp, content_type=content.content_type, content=content.content, salience=max(0.0, min(1.0, adjusted_salience)), semantic_meaning=content.semantic_meaning)

    def _adjust_salience_with_exploration(self, content: ConsciousContent) -> ConsciousContent:
        adjusted_content = self._adjust_salience_contextually(content)
        base_salience = adjusted_content.salience
        if self.exploration_bias > 0.6:
            novelty_bonus = 0.0
            if len(self.attention_history) > 0:
                recent_types = [att.content_type for att in list(self.attention_history)[-3:]]
                if content.content_type not in recent_types:
                    novelty_bonus += 0.3 * self.exploration_bias
            content_str = str(content.content) if hasattr(content, 'content') and content.content else ''
            if any((word in content_str.lower() for word in ['uncertain', 'error', 'new', 'unknown'])):
                novelty_bonus += 0.2 * self.exploration_bias
                pass
            base_salience *= 1.0 + novelty_bonus
        elif self.exploration_bias < 0.4:
            stability_bonus = 0.0
            if len(self.attention_history) > 0:
                recent_types = [att.content_type for att in list(self.attention_history)[-5:]]
                if content.content_type in recent_types:
                    stability_bonus += 0.2 * (1.0 - self.exploration_bias)
            content_str = str(content.content) if hasattr(content, 'content') and content.content else ''
            if any((word in content_str.lower() for word in ['confident', 'goal', 'stable', 'known'])):
                stability_bonus += 0.15 * (1.0 - self.exploration_bias)
            base_salience *= 1.0 + stability_bonus
        random_factor = 1.0 + np.random.normal(0, 0.05 + 0.1 * self.exploration_bias)
        base_salience *= random_factor
        return ConsciousContent(timestamp=adjusted_content.timestamp, content_type=adjusted_content.content_type, content=adjusted_content.content, salience=max(0.0, min(1.0, base_salience)), semantic_meaning=adjusted_content.semantic_meaning)

    def _record_association(self, type1: str, type2: str):
        if type1 not in self.content_associations:
            self.content_associations[type1] = []
        if type2 not in self.content_associations:
            self.content_associations[type2] = []
        if type2 not in self.content_associations[type1]:
            self.content_associations[type1].append(type2)
        if type1 not in self.content_associations[type2]:
            self.content_associations[type2].append(type1)

    def _update_semantic_clusters(self, content: ConsciousContent):
        if hasattr(content, 'semantic_meaning'):
            meaning_str = str(content.semantic_meaning) if content.semantic_meaning else ''
            words = meaning_str.lower().split()
        else:
            words = []
        if words:
            main_word = words[0]
            if main_word not in self.semantic_clusters:
                self.semantic_clusters[main_word] = []
            self.semantic_clusters[main_word].append(content)
            if len(self.semantic_clusters[main_word]) > 10:
                self.semantic_clusters[main_word] = self.semantic_clusters[main_word][-10:]

    def compete_for_consciousness(self) -> List[ConsciousContent]:
        # MULTI-LOOP: Process incoming messages from other modules
        self._process_incoming_messages()
        
        # If an external system state (last_broadcast_state) exists, use its health/phi
        # to adapt policy parameters automatically before selection.
        if self.last_broadcast_state:
            lbs = self.last_broadcast_state
            health = None
            if isinstance(lbs, dict):
                health = lbs.get('health') or lbs.get('structural_health') or lbs.get('ces_health')
            phi_val = 0.0
            if isinstance(lbs, dict):
                phi_val = float(lbs.get('phi', lbs.get('phi_value', 0.0) or 0.0))
            if health or phi_val:
                try:
                    self.apply_system_health(health or {}, float(phi_val))
                except Exception:
                    pass

        current_policy_bias = float(self.policy_params.get('exploration_bias', 0.5))
        # Avoid hard resets; nudge upward with a small, rate-limited step when too low
        if current_policy_bias < 0.4:
            new_eb = float(min(0.45, current_policy_bias + 0.05))
            if abs(new_eb - current_policy_bias) > 1e-6 and self._write_policy_param('exploration_bias', new_eb, source='maintain_exploration_soft_nudge'):
                try:
                    self.policy_adjustments.append({'timestamp': self._get_current_timestamp(), 'adjustments': {'exploration_bias': new_eb - current_policy_bias}, 'reason': 'maintain_exploration_soft_nudge'})
                except Exception:
                    pass
        self.exploration_bias = float(self.policy_params.get('exploration_bias', 0.5))
        if not self.competitors:
            return self.current_contents
        # If a ConceptualSpace is connected and a recent qualia vector exists in
        # last_broadcast_state, use grounding to bias exploration vs. exploitation.
        if self.concept_space and isinstance(self.last_broadcast_state, dict) and 'qualia' in self.last_broadcast_state:
            try:
                qual = self.last_broadcast_state.get('qualia')
                grounded = None
                if qual is not None:
                    grounded = self.concept_space.ground_experience(qual)
                if grounded:
                    nearest_distance = float(grounded.get('nearest_distance', 1.0))
                    # If experience maps close to a known prototype, bias exploitation
                    if nearest_distance < 0.35:
                        self._write_policy_param(
                            'exploration_bias',
                            max(0.05, self.policy_params.get('exploration_bias', 0.75) * 0.8),
                            source='concept_space_grounding',
                        )
                    else:
                        # Novel experience -> encourage exploration
                        self._write_policy_param(
                            'exploration_bias',
                            min(0.95, self.policy_params.get('exploration_bias', 0.75) * 1.15),
                            source='concept_space_grounding',
                        )
                    # Also slightly modulate attention strength based on confidence
                    concept_blend = grounded.get('concept_blend', {})
                    top_confidence = max(concept_blend.values()) if concept_blend else 0.0
                    self._write_policy_param(
                        'attention_focus_strength',
                        float(np.clip(self.policy_params.get('attention_focus_strength', 0.85) * (1.0 + top_confidence * 0.2), 0.1, 2.0)),
                        source='concept_space_grounding',
                    )
            except Exception:
                pass
        exploration_factor = self.exploration_bias + np.random.normal(0, 0.05)
        exploration_mode = exploration_factor > 0.5
        base_threshold = self.policy_params.get('confidence_threshold', 0.4)
        if exploration_mode:
            dynamic_threshold = base_threshold * (0.3 + 0.4 * (1.0 - self.exploration_bias))
            min_candidates = max(1, len(self.competitors) // 2)
        else:
            dynamic_threshold = base_threshold * (0.8 + 0.4 * self.exploration_bias)
            min_candidates = min(3, len(self.competitors))
        candidates = [c for c in self.competitors if c.salience > dynamic_threshold]
        if len(candidates) < min_candidates:
            sorted_competitors = sorted(self.competitors, key=lambda x: x.salience, reverse=True)
            candidates = sorted_competitors[:min_candidates]
        if exploration_mode:
            selected = self._explore_b_selection(candidates, aggressive=True)
        else:
            selected = self._conservative_selection_enhanced(candidates)
        self.current_contents = selected[:self.capacity]
        self._update_attention_explore_b(selected, exploration_mode)
        for content in selected:
            if content:
                self._update_semantic_clusters(content)
                for other in selected:
                    if other != content:
                        self._record_association(content.content_type, other.content_type)
        baseline_reward = 0.03 + 0.01 * len(selected)
        if exploration_mode:
            baseline_reward += 0.02
        self.cumulative_reward += baseline_reward
        self.reward_history.append(baseline_reward)
        self.competitors.clear()
        
        # MULTI-LOOP: Broadcast final workspace state to all modules
        self._broadcast_workspace_state()
        
        return self.current_contents

    def _explore_b_selection(self, candidates: List[ConsciousContent], aggressive: bool=False) -> List[ConsciousContent]:
        if not candidates:
            return []
        selected = []
        remaining = candidates.copy()
        if aggressive and self.exploration_bias > 0.7:
            first_pick = remaining[np.random.choice(len(remaining))]
        else:
            top_half_size = max(1, len(remaining) // 2)
            sorted_remaining = sorted(remaining, key=lambda x: x.salience, reverse=True)
            top_half = sorted_remaining[:top_half_size]
            first_pick = top_half[np.random.choice(len(top_half))]
        selected.append(first_pick)
        remaining.remove(first_pick)
        while len(selected) < self.capacity and remaining:
            best_candidate = None
            best_score = -float('inf')
            for candidate in remaining:
                quality_score = candidate.salience * 0.3
                diversity_score = self._calculate_exploration_diversity(candidate, selected) * 0.5
                random_bonus = np.random.random() * 0.2
                exploration_boost = self.exploration_bias * 0.1
                total_score = quality_score + diversity_score + random_bonus + exploration_boost
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        return selected

    def _calculate_exploration_diversity(self, candidate: ConsciousContent, selected: List[ConsciousContent]) -> float:
        if not selected:
            return 1.0
        diversity_scores = []
        for existing in selected:
            type_diversity = 1.0 if candidate.content_type != existing.content_type else -0.2
            semantic_similarity = self._calculate_semantic_similarity(candidate, existing)
            semantic_diversity = (1.0 - semantic_similarity) * 1.5
            time_gap = abs(candidate.timestamp - existing.timestamp)
            time_diversity = min(1.0, time_gap / 50.0)
            combined_diversity = (type_diversity + semantic_diversity + time_diversity) / 3.0
            diversity_scores.append(combined_diversity)
        return float(np.mean(diversity_scores))

    def _conservative_selection_enhanced(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        if not candidates:
            return []
        selected = []
        sorted_candidates = sorted(candidates, key=lambda x: x.salience, reverse=True)
        priority_count = min(2, len(sorted_candidates), self.capacity)
        for i in range(priority_count):
            selected.append(sorted_candidates[i])
        remaining = [c for c in candidates if c not in selected]
        while len(selected) < self.capacity and remaining:
            best_candidate = None
            best_score = -float('inf')
            for candidate in remaining:
                quality_score = candidate.salience * 0.7
                diversity_score = self._calculate_diversity_score(candidate, selected) * 0.25
                random_factor = np.random.random() * 0.05
                total_score = quality_score + diversity_score + random_factor
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = candidate
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        return selected

    def _update_attention_explore_b(self, selected: List[ConsciousContent], exploration_mode: bool):
        if not selected:
            return
        if exploration_mode:
            self.attention_switching_threshold *= 0.7
            if np.random.random() < 0.3:
                new_focus = selected[np.random.choice(len(selected))]
            elif self.attention_focus and self.attention_focus in selected:
                most_different = None
                max_difference = -1.0
                for content in selected:
                    if content != self.attention_focus:
                        difference = 1.0 - self._calculate_semantic_similarity(content, self.attention_focus)
                        if difference > max_difference:
                            max_difference = difference
                            most_different = content
                new_focus = most_different if most_different else max(selected, key=lambda x: x.salience)
            else:
                new_focus = max(selected, key=lambda x: x.salience)
        else:
            self.attention_switching_threshold *= 1.1
            new_focus = max(selected, key=lambda x: x.salience)
        should_switch = self.attention_focus is None or new_focus != self.attention_focus or np.random.random() < 1.0 - self.attention_switching_threshold
        if should_switch:
            self.attention_focus = new_focus
            self.attention_persistence = 0.0
            self.attention_history.append(new_focus)
        else:
            self.attention_persistence += 0.1
        self.attention_switching_threshold = np.clip(self.attention_switching_threshold, 0.2, 0.95)

    def _exploratory_selection(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        selected = []
        if not candidates:
            return selected
        random_picks = min(2, len(candidates), self.capacity)
        if random_picks > 0 and len(candidates) > 0:
            actual_picks = min(random_picks, len(candidates))
            random_indices = np.random.choice(len(candidates), size=actual_picks, replace=False)
            random_selection = [candidates[i] for i in random_indices]
            selected.extend(random_selection)
        remaining_candidates = [c for c in candidates if c not in selected]
        while len(selected) < self.capacity and remaining_candidates:
            best_candidate = None
            max_diversity_score = -1
            for candidate in remaining_candidates:
                diversity_score = self._calculate_diversity_score(candidate, selected)
                diversity_score += np.random.normal(0, 0.2)
                if diversity_score > max_diversity_score:
                    max_diversity_score = diversity_score
                    best_candidate = candidate
            if best_candidate:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
            else:
                break
        return selected[:self.capacity]

    def _conservative_selection(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        selected = []
        if not candidates:
            return selected
        if len(candidates) > 0:
            best = max(candidates, key=lambda x: x.salience)
            selected.append(best)
        else:
            return selected
        remaining = [c for c in candidates if c != best]
        while len(selected) < self.capacity and remaining:
            best_candidate = None
            max_score = -1
            for candidate in remaining:
                quality_score = candidate.salience * 0.6
                diversity_score = self._calculate_diversity_score(candidate, selected) * 0.3
                random_bonus = np.random.random() * 0.1
                total_score = quality_score + diversity_score + random_bonus
                if total_score > max_score:
                    max_score = total_score
                    best_candidate = candidate
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        return selected

    def _calculate_diversity_score(self, candidate: ConsciousContent, selected: List[ConsciousContent]) -> float:
        if not selected:
            return 1.0
        diversity_scores = []
        for existing in selected:
            type_diversity = 0.8 if candidate.content_type != existing.content_type else 0.0
            semantic_diversity = 1.0 - self._calculate_semantic_similarity(candidate, existing)
            diversity_scores.append((type_diversity + semantic_diversity) / 2.0)
        return float(np.mean(diversity_scores))

    def _update_attention_dynamically(self, selected: List[ConsciousContent], exploration_mode: bool):
        if not selected:
            return
        if exploration_mode:
            self.attention_switching_threshold *= 0.8
            if np.random.random() < 0.3:
                new_focus = selected[np.random.choice(len(selected))]
            else:
                new_focus = max(selected, key=lambda x: x.salience)
        else:
            self.attention_switching_threshold *= 1.05
            new_focus = max(selected, key=lambda x: x.salience)
        should_switch = self.attention_focus is None or new_focus != self.attention_focus or np.random.random() < 1.0 - self.attention_switching_threshold
        if should_switch:
            self.attention_focus = new_focus
            self.attention_persistence = 0.0
            self.attention_history.append(new_focus)
        else:
            self.attention_persistence += 0.1
        self.attention_switching_threshold = np.clip(self.attention_switching_threshold, 0.3, 0.9)

    def _select_cluster_representatives(self) -> List[ConsciousContent]:
        representatives = []
        temp_clusters = {}
        for competitor in self.competitors:
            try:
                if hasattr(competitor, 'semantic_meaning'):
                    meaning_str = str(competitor.semantic_meaning) if competitor.semantic_meaning else ''
                    words = meaning_str.lower().split()
                    cluster_key = words[0] if words else 'misc'
                else:
                    cluster_key = 'misc'
            except (AttributeError, TypeError):
                cluster_key = 'misc'
            if cluster_key not in temp_clusters:
                temp_clusters[cluster_key] = []
            temp_clusters[cluster_key].append(competitor)
        for cluster_contents in temp_clusters.values():
            if cluster_contents:
                best = max(cluster_contents, key=lambda x: x.salience)
                representatives.append(best)
        return representatives

    def _balanced_selection(self, candidates: List[ConsciousContent]) -> List[ConsciousContent]:
        if len(candidates) <= self.capacity:
            return candidates
        selected = []
        remaining = candidates.copy()
        best = max(remaining, key=lambda x: x.salience)
        selected.append(best)
        remaining.remove(best)
        while len(selected) < self.capacity and remaining:
            diversity_scores = []
            for candidate in remaining:
                score = candidate.salience
                diversity_bonus = 0.0
                for selected_content in selected:
                    similarity = self._calculate_semantic_similarity(candidate, selected_content)
                    diversity_bonus += 1.0 - similarity
                diversity_bonus /= len(selected)
                final_score = score * 0.7 + diversity_bonus * 0.3
                diversity_scores.append((candidate, final_score))
            best_candidate, _ = max(diversity_scores, key=lambda x: x[1])
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        return selected

    def _manage_attention_focus(self, winners: List[ConsciousContent]):
        if not winners:
            return
        current_best = winners[0]
        should_switch = False
        if self.attention_focus is None:
            should_switch = True
        else:
            salience_gap = current_best.salience - self.attention_focus.salience
            if salience_gap > self.attention_switching_threshold:
                should_switch = True
            self.attention_persistence *= 0.9
            if self.attention_persistence < 0.3:
                should_switch = True
            if current_best.content_type != self.attention_focus.content_type:
                type_novelty = current_best.content_type not in [att.content_type for att in list(self.attention_history)[-3:]]
                if type_novelty and salience_gap > 0.1:
                    should_switch = True
        if should_switch:
            if self.attention_focus:
                self.attention_history.append(self.attention_focus)
            self.attention_focus = current_best
            self.attention_persistence = 1.0
            focus_strength = self.policy_params['attention_focus_strength']
            self.attention_switching_threshold = 0.9 - focus_strength * 0.3

    def broadcast(self, world_state: Optional[Dict]=None) -> Dict[str, Any]:
        start_time = time.perf_counter()
        current_timestamp = int(self._get_current_timestamp() * 1000)
        primary_focus = self._create_safe_focus_description()
        conscious_contents = self._create_safe_content_descriptions()
        workspace_metrics = self._calculate_safe_workspace_metrics()
        explore_b_state = {'exploration_bias': self.policy_params.get('exploration_bias', 0.5), 'exploration_mode': self.exploration_bias > 0.5, 'last_selection_type': 'exploratory' if hasattr(self, '_last_selection_mode') and self._last_selection_mode else 'conservative', 'diversity_score': self._calculate_current_diversity(), 'novelty_score': self._calculate_current_novelty()}
        broadcast_msg = {'timestamp': current_timestamp, 'version': '2.0', 'primary_focus': primary_focus, 'conscious_contents': conscious_contents, 'workspace_fullness': len(self.current_contents) / self.capacity, 'explore_b': explore_b_state, 'policy_state': {'exploration_bias': self.policy_params.get('exploration_bias', 0.5), 'stability_bias': self.policy_params.get('stability_bias', 0.5), 'confidence_threshold': self.policy_params.get('confidence_threshold', 0.4), 'error_sensitivity': self.policy_params.get('error_sensitivity', 1.0), 'attention_focus_strength': self.policy_params.get('attention_focus_strength', 0.7), 'recent_reward': list(self.reward_history)[-1] if self.reward_history else 0.0, 'cumulative_reward': self.cumulative_reward}, 'workspace_metrics': workspace_metrics, 'attention_context': {'switching_threshold': self.attention_switching_threshold, 'persistence': self.attention_persistence, 'recent_switches': len(self.attention_history), 'focus_stability': self._calculate_focus_stability()}, 'world_state': world_state if world_state else {}, 'performance': {'response_time': time.perf_counter() - start_time, 'error_count': len(self.error_detection_buffer), 'adjustment_count': len(self.policy_adjustments), 'success_rate': self._calculate_success_rate()}}
        predictions = self._generate_safe_predictions(world_state)
        if predictions:
            broadcast_msg['predictions'] = predictions
        if self.semantic_clusters:
            broadcast_msg['semantic_clusters'] = {k: len(v) for k, v in self.semantic_clusters.items()}
        self.broadcast_history.append(broadcast_msg)
        self.last_broadcast_state = broadcast_msg.copy()
        if hasattr(self, 'broadcast_hooks') and self.broadcast_hooks:
            hook_results = []
            for i, hook in enumerate(self.broadcast_hooks):
                try:
                    result = hook(broadcast_msg)
                    if result:
                        hook_results.append({'hook_id': i, 'result': result, 'timestamp': self._get_current_timestamp()})
                except Exception as e:
                    pass
            if hook_results:
                broadcast_msg['hook_responses'] = hook_results
        total_response_time = time.perf_counter() - start_time
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics['response_time'].append(total_response_time)
        return broadcast_msg

    def _extract_safe_content_attrs(self, obj) -> Dict[str, Any]:
        return {'type': getattr(obj, 'content_type', 'unknown'), 'salience': getattr(obj, 'salience', 0.0), 'meaning': str(getattr(obj, 'semantic_meaning', ''))[:100], 'timestamp': getattr(obj, 'timestamp', 0)}

    def _get_current_timestamp(self) -> float:
        return time.perf_counter()

    def _calculate_age(self, obj) -> float:
        timestamp = getattr(obj, 'timestamp', 0)
        if timestamp == 0:
            return 0.0
        return time.perf_counter() - timestamp

    @contextmanager
    def _measure_performance(self, metric_name: Optional[str] = None):
        start_time = time.perf_counter()
        try:
            yield start_time
        finally:
            elapsed = time.perf_counter() - start_time
            if metric_name and hasattr(self, 'performance_metrics'):
                if metric_name not in self.performance_metrics:
                    self.performance_metrics[metric_name] = deque(maxlen=100)
                self.performance_metrics[metric_name].append(elapsed)

    def _create_safe_focus_description(self) -> Dict[str, Any]:
        if not self.attention_focus:
            return {'active': False, 'type': 'none', 'salience': 0.0, 'meaning': '', 'content': None}
        attrs = self._extract_safe_content_attrs(self.attention_focus)
        return {'active': True, **attrs, 'content': str(getattr(self.attention_focus, 'content', ''))[:100], 'persistence': self.attention_persistence}

    def _create_safe_content_descriptions(self) -> List[Dict[str, Any]]:
        descriptions = []
        for i, content in enumerate(self.current_contents):
            attrs = self._extract_safe_content_attrs(content)
            desc = {'index': i, **attrs, 'age': self._calculate_age(content)}
            descriptions.append(desc)
        return descriptions

    def _calculate_safe_workspace_metrics(self) -> Dict[str, float]:
        try:
            metrics = {'fullness': len(self.current_contents) / self.capacity, 'avg_salience': 0.0, 'diversity': 0.0, 'stability': 0.0}
            if self.current_contents:
                saliences = [self._extract_safe_content_attrs(content)['salience'] for content in self.current_contents]
                metrics['avg_salience'] = float(np.mean(saliences)) if saliences else 0.0
                if len(self.current_contents) > 1:
                    types = [self._extract_safe_content_attrs(c)['type'] for c in self.current_contents]
                    unique_types = len(set(types))
                    metrics['diversity'] = unique_types / len(types)
                if len(self.broadcast_history) > 0:
                    prev_types = set()
                    prev_contents = self.broadcast_history[-1].get('conscious_contents', [])
                    prev_types = {c.get('type', 'unknown') for c in prev_contents}
                    current_types = {self._extract_safe_content_attrs(c)['type'] for c in self.current_contents}
                    if len(prev_types) > 0 and len(current_types) > 0:
                        intersection = len(prev_types & current_types)
                        union = len(prev_types | current_types)
                        metrics['stability'] = intersection / union if union > 0 else 0.0
            return metrics
        except Exception:
            return {'fullness': 0.0, 'avg_salience': 0.0, 'diversity': 0.0, 'stability': 0.0}

    def _calculate_current_diversity(self) -> float:
        if len(self.current_contents) <= 1:
            return 0.0
        types = [getattr(c, 'content_type', 'unknown') for c in self.current_contents]
        unique_types = len(set(types))
        return unique_types / len(types)

    def _calculate_current_novelty(self) -> float:
        if not self.current_contents or len(self.attention_history) < 5:
            return 0.5
        current_types = {getattr(c, 'content_type', 'unknown') for c in self.current_contents}
        recent_types = {getattr(att, 'content_type', 'unknown') for att in list(self.attention_history)[-5:]}
        new_types = current_types - recent_types
        novelty = len(new_types) / len(current_types) if current_types else 0.0
        return min(1.0, novelty)

    def _calculate_focus_stability(self) -> float:
        if len(self.attention_history) < 3:
            return 1.0
        try:
            recent_focuses = list(self.attention_history)[-3:]
            if all((getattr(f, 'content_type', '') == getattr(recent_focuses[0], 'content_type', '') for f in recent_focuses)):
                return 1.0
            else:
                return 0.3
        except:
            return 0.5

    def _calculate_success_rate(self) -> float:
        if len(self.reward_history) < 5:
            return 0.5
        recent_rewards = list(self.reward_history)[-10:]
        positive_count = sum((1 for r in recent_rewards if r > 0))
        return positive_count / len(recent_rewards)

    def _generate_safe_predictions(self, world_state: Optional[Dict]) -> Optional[Dict]:
        if not world_state:
            return None
        current_stability = world_state.get('stability', 0.5)
        current_energy = world_state.get('energy_level', 0.5)
        if len(self.broadcast_history) >= 3:
            recent_states = []
            for msg in list(self.broadcast_history)[-3:]:
                ws = msg.get('world_state', {})
                if ws:
                    recent_states.append(ws)
            if len(recent_states) >= 2:
                stabilities = [s.get('stability', 0.5) for s in recent_states]
                stability_trend = np.mean(np.diff(stabilities)) if len(stabilities) > 1 else 0.0
                energies = [s.get('energy_level', 0.5) for s in recent_states]
                energy_trend = np.mean(np.diff(energies)) if len(energies) > 1 else 0.0
                return {'stability': np.clip(current_stability + stability_trend, 0.0, 1.0), 'energy': np.clip(current_energy + energy_trend, 0.0, 1.0), 'confidence': 0.6, 'method': 'linear_trend'}
            return {'stability': current_stability, 'energy': current_energy, 'confidence': 0.3, 'method': 'baseline'}

    def _generate_predictions(self, world_state: Optional[Dict]):
        if not world_state:
            return
        current_stability = world_state.get('stability', 0.5)
        current_energy = world_state.get('energy_level', 0.5)
        if len(self.broadcast_history) >= 5:
            recent_states = []
            for msg in list(self.broadcast_history)[-5:]:
                if msg.get('world_state'):
                    recent_states.append(msg['world_state'])
            if recent_states:
                stability_trend = np.diff([s.get('stability', 0.5) for s in recent_states[-3:]])
                predicted_stability = current_stability + np.mean(stability_trend) if len(stability_trend) > 0 else current_stability
                energy_trend = np.diff([s.get('energy_level', 0.5) for s in recent_states[-3:]])
                predicted_energy = current_energy + np.mean(energy_trend) if len(energy_trend) > 0 else current_energy
                self.predictions = {'stability': np.clip(predicted_stability, 0.0, 1.0), 'energy': np.clip(predicted_energy, 0.0, 1.0), 'confidence': self.prediction_confidence, 'horizon': 1}
        if len(self.performance_metrics['prediction_accuracy']) > 0:
            recent_accuracy = list(self.performance_metrics['prediction_accuracy'])[-10:]
            self.prediction_confidence = np.mean(recent_accuracy)

    def _create_focus_description(self) -> Optional[Dict]:
        if not self.attention_focus:
            return None
        try:
            return {'type': getattr(self.attention_focus, 'content_type', 'unknown'), 'meaning': getattr(self.attention_focus, 'semantic_meaning', ''), 'salience': getattr(self.attention_focus, 'salience', 0.0), 'content': getattr(self.attention_focus, 'content', None), 'persistence': self.attention_persistence, 'switch_count': len(self.attention_history), 'associations': self.content_associations.get(getattr(self.attention_focus, 'content_type', 'unknown'), [])}
        except Exception as e:
            logging.error(f'Error in _create_focus_description: {str(e)[:50]}')
            return {'type': 'error', 'meaning': 'Failed to create focus description', 'salience': 0.0, 'content': None, 'persistence': 0.0, 'switch_count': 0, 'associations': []}

    def _create_content_descriptions(self) -> List[Dict]:
        descriptions = []
        for c in self.current_contents:
            try:
                attrs = self._extract_safe_content_attrs(c)
                description = {**attrs, 'age': self._calculate_age(c), 'cluster': self._find_content_cluster(c)}
                descriptions.append(description)
            except Exception as e:
                print(f'Error in content description: {str(e)[:50]}')
                descriptions.append({'type': 'error', 'meaning': 'Failed to describe content', 'salience': 0.0, 'age': 0.0, 'cluster': 'error'})
        return descriptions

    def _find_content_cluster(self, content: ConsciousContent) -> str:
        try:
            semantic_meaning = getattr(content, 'semantic_meaning', '')
            if not semantic_meaning:
                return 'empty'
            meaning_str = str(semantic_meaning) if semantic_meaning else ''
            words = meaning_str.lower().split()
            if words and words[0] in self.semantic_clusters:
                return words[0]
            return 'unclustered'
        except Exception:
            return 'error'

    def _calculate_workspace_metrics(self) -> Dict[str, float]:
        metrics = {'fullness': len(self.current_contents) / self.capacity, 'avg_salience': float(np.mean([c.salience for c in self.current_contents])) if self.current_contents else 0.0, 'diversity': self._calculate_content_diversity(), 'stability': self._calculate_content_stability(), 'integration_rate': len(self.content_associations) / max(1, len(self.semantic_clusters))}
        return metrics

    def _calculate_content_diversity(self) -> float:
        if len(self.current_contents) <= 1:
            return 0.0
        total_similarity = 0.0
        pair_count = 0
        for i in range(len(self.current_contents)):
            for j in range(i + 1, len(self.current_contents)):
                similarity = self._calculate_semantic_similarity(self.current_contents[i], self.current_contents[j])
                total_similarity += similarity
                pair_count += 1
        avg_similarity = total_similarity / pair_count if pair_count > 0 else 0.0
        return 1.0 - avg_similarity

    def _calculate_content_stability(self) -> float:
        if len(self.broadcast_history) < 2:
            return 1.0
        prev_msg = self.broadcast_history[-2]
        prev_types = [c['type'] for c in prev_msg.get('conscious_contents', [])]
        current_types = [c.content_type for c in self.current_contents]
        prev_set = set(prev_types)
        current_set = set(current_types)
        intersection = len(prev_set & current_set)
        union = len(prev_set | current_set)
        return intersection / union if union > 0 else 0.0

    def _create_attention_context(self) -> Dict[str, Any]:
        return {'switching_threshold': self.attention_switching_threshold, 'persistence': self.attention_persistence, 'history_length': len(self.attention_history), 'recent_switches': self._count_recent_switches(), 'focus_coherence': self._calculate_focus_coherence()}

    def _count_recent_switches(self) -> int:
        if len(self.attention_history) < 2:
            return 0
        recent_types = [att.content_type for att in list(self.attention_history)[-5:]]
        switches = 0
        for i in range(1, len(recent_types)):
            if recent_types[i] != recent_types[i - 1]:
                switches += 1
        return switches

    def _calculate_focus_coherence(self) -> float:
        if not self.attention_focus or not self.current_contents:
            return 1.0
        coherence_scores = []
        for content in self.current_contents:
            if content != self.attention_focus:
                similarity = self._calculate_semantic_similarity(self.attention_focus, content)
                coherence_scores.append(similarity)
        return float(np.mean(coherence_scores)) if coherence_scores else 1.0

    def _calculate_performance_trend(self) -> str:
        if len(self.reward_history) < 5:
            return 'insufficient_data'
        recent_rewards = list(self.reward_history)[-5:]
        if len(recent_rewards) == 0:
            return 'insufficient_data'
        trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        if trend > 0.1:
            return 'improving'
        elif trend < -0.1:
            return 'declining'
        else:
            return 'stable'

    def detect_errors_from_broadcast(self, broadcast_msg: Dict, outcome: Dict) -> List[Dict]:
        errors = []
        try:
            if broadcast_msg.get('primary_focus'):
                focus_type = broadcast_msg['primary_focus'].get('type', '<unknown>')
                if 'goal' in focus_type:
                    expected_improvement = True
                    actual_improvement = outcome.get('improvement', False)
                    if expected_improvement and (not actual_improvement):
                        errors.append({'type': 'goal_prediction_error', 'severity': 0.7, 'description': 'Goal prediction error', 'timestamp': broadcast_msg.get('timestamp', 0)})
                if 'qualia' in focus_type and broadcast_msg['primary_focus'].get('meaning'):
                    focus_meaning = broadcast_msg['primary_focus']['meaning']
                    if '   ' in focus_meaning or '   ' in focus_meaning:
                        if outcome.get('tension_increased', False):
                            errors.append({'type': 'qualia_mismatch', 'severity': 0.6, 'description': 'Qualia mismatch', 'timestamp': broadcast_msg.get('timestamp', 0)})
            workspace_fullness = 0.0
            if 'workspace_fullness' in broadcast_msg:
                workspace_fullness = broadcast_msg['workspace_fullness']
            elif 'workspace_metrics' in broadcast_msg:
                workspace_metrics = broadcast_msg['workspace_metrics']
                if isinstance(workspace_metrics, dict):
                    workspace_fullness = workspace_metrics.get('fullness', 0.0)
            if workspace_fullness > 0.9:
                if outcome.get('performance_degraded', False):
                    errors.append({'type': 'workspace_overload', 'severity': 0.8, 'description': 'workspace overloaded; performance degraded', 'timestamp': broadcast_msg.get('timestamp', 0)})
                world_urgency = broadcast_msg['world_state'].get('urgency', 0.5)
                attention_salience = 0.0
                if broadcast_msg.get('primary_focus') and 'salience' in broadcast_msg['primary_focus']:
                    attention_salience = broadcast_msg['primary_focus']['salience']
                if world_urgency > 0.7 and attention_salience < 0.4:
                    errors.append({'type': 'attention_mismatch', 'severity': world_urgency, 'description': 'high world urgency but low attention salience', 'timestamp': broadcast_msg.get('timestamp', 0)})
            for error in errors:
                self.error_detection_buffer.append({'error': error, 'timestamp': self._get_current_timestamp()})
        except Exception as e:
            logging.error(f'Error in detect_errors_from_broadcast: {str(e)[:100]}')
            errors = []
        return errors

    def update_policy_from_errors(self, errors: List[Dict]) -> Dict[str, float]:
        if not errors:
            success_reward = 0.25 + 0.1 * self._calculate_performance_bonus()
            self.cumulative_reward += success_reward
            self.reward_history.append(success_reward)
            self._reinforce_current_policy()
            return {}
        minor_errors = [e for e in errors if e['severity'] < 0.3]
        major_errors = [e for e in errors if e['severity'] >= 0.3]
        if not major_errors and len(minor_errors) <= 2:
            small_reward = 0.05 + 0.02 * self._calculate_performance_bonus()
            self.cumulative_reward += small_reward
            self.reward_history.append(small_reward)
            return {}
        adjustments = {}
        total_severity = sum((e['severity'] for e in major_errors))
        consecutive_failures = self._count_consecutive_failures()
        penalty_multiplier = 0.5 + consecutive_failures * 0.1
        reward = -total_severity * penalty_multiplier
        self.cumulative_reward += reward
        self.reward_history.append(reward)
        for error in errors:
            self.error_detection_buffer.append({'error': error, 'timestamp': self._get_current_timestamp(), 'policy_state': self.policy_params.copy()})
        adaptive_lr = self._calculate_adaptive_learning_rate()
        for error in errors:
            error_type = error['type']
            severity = error['severity']
            lr = adaptive_lr * severity
            if error_type == 'goal_prediction_error':
                self._update_param_with_momentum('stability_bias', lr * 0.3, adjustments, bounds=(0.0, 1.0))
                self._update_param_with_momentum('exploration_bias', -lr * 0.1, adjustments, bounds=(0.3, 1.0))
                self._update_param_with_momentum('prediction_confidence', -lr * 0.2, adjustments, bounds=(0.1, 1.0))
            elif error_type == 'qualia_mismatch':
                self._update_param_with_momentum('error_sensitivity', lr * 0.4, adjustments, bounds=(0.5, 3.0))
                self._update_param_with_momentum('integration_eagerness', lr * 0.3, adjustments, bounds=(0.0, 1.0))
            elif error_type == 'workspace_overload':
                self._update_param_with_momentum('confidence_threshold', lr * 0.2, adjustments, bounds=(0.2, 0.9))
                self._update_param_with_momentum('attention_focus_strength', lr * 0.25, adjustments, bounds=(0.3, 1.0))
            elif error_type == 'attention_mismatch':
                self._update_param_with_momentum('attention_focus_strength', -lr * 0.3, adjustments, bounds=(0.3, 1.0))
                self.attention_switching_threshold *= 1.0 - lr * 0.2
                self.attention_switching_threshold = np.clip(self.attention_switching_threshold, 0.3, 0.9)
        if adjustments:
            self.policy_adjustments.append({'timestamp': self._get_current_timestamp(), 'adjustments': adjustments.copy(), 'trigger_errors': [e['type'] for e in errors], 'total_severity': total_severity})
        self._evaluate_policy_effectiveness()
        return adjustments

    def _calculate_performance_bonus(self) -> float:
        if len(self.reward_history) < 10:
            return 0.0
        recent_rewards = list(self.reward_history)[-10:]
        positive_ratio = sum((1 for r in recent_rewards if r > 0)) / len(recent_rewards)
        return positive_ratio - 0.5

    def _count_consecutive_failures(self) -> int:
        count = 0
        for reward in reversed(list(self.reward_history)[-10:]):
            if reward < 0:
                count += 1
            else:
                break
        return count

    def _reinforce_current_policy(self):
        for param in self.policy_params:
            variance_history = self.policy_variance_tracking[param]
            if len(variance_history) > 0:
                current_variance = np.var(list(variance_history)[-10:]) if len(variance_history) >= 10 else 0.1
                reduced_variance = current_variance * 0.9

    def _calculate_adaptive_learning_rate(self) -> float:
        base_lr = self.policy_params['learning_rate']
        if len(self.reward_history) >= 5:
            recent_performance = np.mean(list(self.reward_history)[-5:])
            if recent_performance < -0.3:
                return base_lr * 1.5
            elif recent_performance > 0.2:
                return base_lr * 0.7
        if len(self.policy_adjustments) >= 3:
            recent_adjustments = list(self.policy_adjustments)[-3:]
            adjustment_magnitudes = []
            for adj_record in recent_adjustments:
                total_magnitude = sum((abs(v) for v in adj_record['adjustments'].values()))
                adjustment_magnitudes.append(total_magnitude)
            avg_magnitude = np.mean(adjustment_magnitudes)
            if avg_magnitude > 0.3:
                return base_lr * 0.8
        return base_lr

    def _update_param_with_momentum(self, param_name: str, delta: float, adjustments: Dict, bounds: Tuple[float, float]):
        self.policy_momentum[param_name] = 0.9 * self.policy_momentum[param_name] + 0.1 * delta
        old_val = self.policy_params[param_name]
        update = self.policy_momentum[param_name]
        new_val = np.clip(old_val + update, bounds[0], bounds[1])
        actual_change = new_val - old_val
        if abs(actual_change) > 1e-06 and self._write_policy_param(param_name, new_val, source='momentum_adjustment'):
            adjustments[param_name] = actual_change
            self.policy_variance_tracking[param_name].append(new_val)

    def _evaluate_policy_effectiveness(self):
        if len(self.policy_adjustments) < 5:
            return
        recent_adjustments = list(self.policy_adjustments)[-5:]
        recent_rewards = list(self.reward_history)[-5:]
        param_effectiveness = {}
        for adj_record, reward in zip(recent_adjustments, recent_rewards):
            for param, change in adj_record['adjustments'].items():
                if param not in param_effectiveness:
                    param_effectiveness[param] = []
                effectiveness = reward / (abs(change) + 1e-06)
                param_effectiveness[param].append(effectiveness)
        for param, effectiveness_scores in param_effectiveness.items():
            avg_effectiveness = np.mean(effectiveness_scores)
            if avg_effectiveness < -0.5:
                self.policy_momentum[param] *= 0.8
            elif avg_effectiveness > 0.3:
                self.policy_momentum[param] *= 1.1

    def get_policy_recommendations(self) -> Dict[str, Any]:
        recommendations: Dict[str, Any] = {'immediate_actions': [], 'strategic_adjustments': [], 'risk_mitigation': [], 'performance_optimizations': [], 'meta_learning_insights': []}
        immediate = self._generate_immediate_recommendations()
        recommendations['immediate_actions'] = immediate
        strategic = self._generate_strategic_recommendations()
        recommendations['strategic_adjustments'] = strategic
        risk_mitigation = self._generate_risk_mitigation_recommendations()
        recommendations['risk_mitigation'] = risk_mitigation
        optimization = self._generate_optimization_recommendations()
        recommendations['performance_optimizations'] = optimization
        meta_insights = self._generate_meta_learning_insights()
        recommendations['meta_learning_insights'] = meta_insights
        self._prioritize_recommendations(recommendations)
        recommendations['legacy_policy'] = {'exploration_factor': self.policy_params['exploration_bias'], 'stability_preference': self.policy_params['stability_bias'], 'attention_selectivity': self.policy_params['confidence_threshold'], 'error_correction_strength': self.policy_params['error_sensitivity'], 'learning_rate_modifier': 1.0 + (self.policy_params['error_sensitivity'] - 1.0) * 0.5}
        return recommendations

    def _generate_immediate_recommendations(self) -> List[Dict]:
        recommendations = []
        if len(self.reward_history) >= 3:
            recent_performance = np.mean(list(self.reward_history)[-3:])
            if recent_performance < -0.5:
                recommendations.append({'action': 'increase_stability_bias', 'priority': 'high', 'reason': 'recent performance drop detected', 'target_value': min(self.policy_params['stability_bias'] + 0.2, 1.0), 'expected_impact': 'stabilization'})
        if len(self.error_detection_buffer) >= 5:
            recent_errors = list(self.error_detection_buffer)[-5:]
            error_types = [e['error']['type'] for e in recent_errors if 'error' in e]
            error_counts = {}
            for et in error_types:
                error_counts[et] = error_counts.get(et, 0) + 1
            dominant_error = max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
            if dominant_error and error_counts[dominant_error] >= 3:
                recommendations.append({'action': f'adjust_for_{dominant_error}', 'priority': 'high', 'reason': f'{dominant_error}        ', 'specific_adjustments': self._get_error_specific_adjustments(dominant_error), 'expected_impact': 'error_reduction'})
        if hasattr(self, 'last_content') and self.last_content:
            content_density = len(self.last_content.get('content_items', []))
            if content_density > 15:
                recommendations.append({'action': 'increase_selectivity', 'priority': 'medium', 'reason': 'workspace content density high', 'target_value': min(self.policy_params['confidence_threshold'] + 0.1, 0.9), 'expected_impact': 'load_reduction'})
        return recommendations

    def _generate_strategic_recommendations(self) -> List[Dict]:
        recommendations = []
        if len(self.reward_history) >= 10:
            long_term_trend = self._calculate_performance_trend()
            if long_term_trend == 'declining':
                recommendations.append({'strategy': 'exploration_increase', 'timeframe': 'medium_term', 'reason': 'long-term performance declining', 'adjustments': {'exploration_bias': '+0.15', 'learning_rate': '+0.02'}, 'expected_outcome': 'new_strategies_discovery'})
            elif long_term_trend == 'improving':
                recommendations.append({'strategy': 'stability_reinforcement', 'timeframe': 'medium_term', 'reason': 'performance improving - reinforce current strategy', 'adjustments': {'stability_bias': '+0.1', 'exploration_bias': '-0.05'}, 'expected_outcome': 'performance_consolidation'})
        # Ensure policy_diversity is defined via recent adjustment variability
        policy_diversity = self._calculate_policy_diversity()
        if policy_diversity < 0.3:
            recommendations.append({'strategy': 'diversification', 'timeframe': 'long_term', 'reason': 'policy diversity insufficient', 'adjustments': {'exploration_bias': '+0.2', 'attention_focus_strength': '-0.1'}, 'expected_outcome': 'increased_adaptability'})
        return recommendations

    def _generate_risk_mitigation_recommendations(self) -> List[Dict]:
        recommendations = []
        for param, value in self.policy_params.items():
            if param == 'stability_bias' and value > 0.9:
                recommendations.append({'risk_type': 'over_stability', 'mitigation': 'gradual_exploration_increase', 'urgency': 'medium', 'reason': 'excessive stability may reduce adaptability', 'action': {'exploration_bias': '+0.1'}})
            elif param == 'exploration_bias' and value > 0.8:
                recommendations.append({'risk_type': 'over_exploration', 'mitigation': 'stability_reinforcement', 'urgency': 'high', 'reason': 'excessive exploration may harm consistency', 'action': {'stability_bias': '+0.15'}})
        consecutive_failures = self._count_consecutive_failures()
        if consecutive_failures > 5:
            recommendations.append({'risk_type': 'performance_collapse', 'mitigation': 'reset_to_baseline', 'urgency': 'high', 'reason': f'{consecutive_failures} consecutive failures', 'action': 'restore_default_policy'})
        return recommendations

    def _generate_optimization_recommendations(self) -> List[Dict]:
        recommendations = []
        if len(self.reward_history) >= 20:
            rewards = list(self.reward_history)
            high_performance_indices = [i for i, r in enumerate(rewards) if r > np.percentile(rewards, 75)]
            if high_performance_indices and len(self.policy_adjustments) > 0:
                optimal_patterns = self._analyze_optimal_policy_patterns(high_performance_indices)
                if optimal_patterns:
                    recommendations.append({'optimization': 'policy_pattern_replication', 'confidence': 0.8, 'reason': 'Replicate successful policy patterns', 'patterns': optimal_patterns, 'expected_improvement': '15-25%'})
        meta_optimization = self._suggest_meta_parameter_optimization()
        if meta_optimization:
            recommendations.append(meta_optimization)
        return recommendations

    def _generate_meta_learning_insights(self) -> List[Dict]:
        insights = []
        if len(self.policy_adjustments) >= 10:
            lr_effectiveness = self._analyze_learning_rate_effectiveness()
            if lr_effectiveness['significant']:
                insights.append({'insight_type': 'learning_rate_optimization', 'finding': lr_effectiveness['finding'], 'confidence': lr_effectiveness['confidence'], 'recommendation': lr_effectiveness['recommendation']})
        error_patterns = self._analyze_error_patterns()
        if error_patterns:
            insights.append({'insight_type': 'error_pattern_recognition', 'patterns': error_patterns, 'predictive_value': 'high', 'actionable_insights': self._extract_actionable_insights(error_patterns)})
        param_interactions = self._analyze_parameter_interactions()
        if param_interactions:
            insights.append({'insight_type': 'parameter_synergies', 'interactions': param_interactions, 'optimization_potential': 'medium'})
        return insights

    def _prioritize_recommendations(self, recommendations: Dict[str, List]):
        priority_weights = {'immediate_actions': 1.0, 'risk_mitigation': 0.9, 'strategic_adjustments': 0.7, 'performance_optimizations': 0.6, 'meta_learning_insights': 0.5}
        for category, items in recommendations.items():
            if category == 'legacy_policy':
                continue
            weight = priority_weights.get(category, 0.5)
            items_list = items if isinstance(items, list) else []
            for item in items_list:
                urgency_bonus = 0.0
                if item.get('urgency') == 'high':
                    urgency_bonus = 0.3
                elif item.get('urgency') == 'medium':
                    urgency_bonus = 0.15
                impact_bonus = 0.0
                if item.get('expected_impact') in ['error_reduction', 'stabilization']:
                    impact_bonus = 0.2
                item['priority_score'] = weight + urgency_bonus + impact_bonus
            if isinstance(items, list):
                items.sort(key=lambda x: x.get('priority_score', 0), reverse=True)

    def _get_error_specific_adjustments(self, error_type: str) -> Dict[str, float]:
        adjustments = {'goal_prediction_error': {'stability_bias': 0.15, 'prediction_confidence': -0.1}, 'qualia_mismatch': {'error_sensitivity': 0.2, 'integration_eagerness': 0.1}, 'workspace_overload': {'confidence_threshold': 0.1, 'attention_focus_strength': 0.15}, 'attention_mismatch': {'attention_focus_strength': -0.1, 'exploration_bias': 0.05}}
        return adjustments.get(error_type, {})

    def _calculate_policy_diversity(self) -> float:
        if len(self.policy_adjustments) < 5:
            return 0.5
        recent_adjustments = list(self.policy_adjustments)[-10:]
        param_variations = {}
        for adj_record in recent_adjustments:
            for param, change in adj_record['adjustments'].items():
                if param not in param_variations:
                    param_variations[param] = []
                param_variations[param].append(change)
        diversity_scores = []
        for param, changes in param_variations.items():
            if len(changes) > 1:
                diversity = np.std(changes)
                diversity_scores.append(diversity)
        return float(np.mean(diversity_scores)) if diversity_scores else 0.3

    def _analyze_optimal_policy_patterns(self, high_performance_indices: List[int]) -> Dict:
        if not high_performance_indices or len(self.policy_adjustments) == 0:
            return {}
        optimal_states = []
        for idx in high_performance_indices:
            if idx < len(self.policy_adjustments):
                optimal_states.append(self.policy_adjustments[idx]['adjustments'])
        if not optimal_states:
            return {}
        common_patterns = {}
        for param in self.policy_params.keys():
            param_values = [state.get(param, 0) for state in optimal_states]
            if param_values:
                common_patterns[param] = {'mean': np.mean(param_values), 'std': np.std(param_values), 'frequency': len([v for v in param_values if abs(v) > 0.01])}
        return common_patterns

    def _suggest_meta_parameter_optimization(self) -> Dict:
        if len(self.reward_history) < 20:
            return {}
        lr_values = []
        corresponding_rewards = []
        for i, adj_record in enumerate(list(self.policy_adjustments)[-10:]):
            if 'learning_rate' in adj_record.get('adjustments', {}):
                lr_change = adj_record['adjustments']['learning_rate']
                reward_after = list(self.reward_history)[-(10 - i)] if i < len(self.reward_history) else 0
                lr_values.append(lr_change)
                corresponding_rewards.append(reward_after)
        if len(lr_values) >= 3:
            correlation = np.corrcoef(lr_values, corresponding_rewards)[0, 1]
            if abs(correlation) > 0.5:
                return {'meta_param': 'learning_rate', 'correlation': correlation, 'recommendation': 'increase' if correlation > 0 else 'decrease', 'confidence': min(abs(correlation), 0.9)}
        return {}

    def _analyze_learning_rate_effectiveness(self) -> Dict:
        if len(self.policy_adjustments) < 5:
            return {'significant': False}
        lr_impacts = []
        for i, adj_record in enumerate(list(self.policy_adjustments)[-5:]):
            if 'learning_rate' in adj_record.get('adjustments', {}):
                lr_change = adj_record['adjustments']['learning_rate']
                start_idx = max(0, len(self.reward_history) - (5 - i) - 2)
                end_idx = min(len(self.reward_history), start_idx + 3)
                if end_idx > start_idx:
                    subsequent_rewards = list(self.reward_history)[start_idx:end_idx]
                    avg_reward = np.mean(subsequent_rewards)
                    lr_impacts.append((lr_change, avg_reward))
        if len(lr_impacts) >= 3:
            lr_changes, rewards = zip(*lr_impacts)
            correlation = np.corrcoef(lr_changes, rewards)[0, 1]
            return {'significant': abs(correlation) > 0.4, 'finding': f'Learning rate impact analysis: {correlation:.3f}', 'confidence': min(abs(correlation), 0.9), 'recommendation': f"Adjust learning rate: {('Increase' if correlation > 0 else 'Decrease')}"}
        return {'significant': False}

    def _analyze_error_patterns(self) -> List[Dict]:
        if len(self.error_detection_buffer) < 10:
            return []
        recent_errors = list(self.error_detection_buffer)[-20:]
        error_sequences = []
        for i in range(len(recent_errors) - 2):
            sequence = [recent_errors[i]['error']['type'], recent_errors[i + 1]['error']['type'], recent_errors[i + 2]['error']['type']]
            error_sequences.append(tuple(sequence))
        pattern_counts = {}
        for seq in error_sequences:
            pattern_counts[seq] = pattern_counts.get(seq, 0) + 1
        significant_patterns = []
        for pattern, count in pattern_counts.items():
            if count >= 2:
                significant_patterns.append({'pattern': pattern, 'frequency': count, 'total_sequences': len(error_sequences), 'significance': count / len(error_sequences)})
        return significant_patterns

    def _extract_actionable_insights(self, error_patterns: List[Dict]) -> List[str]:
        insights = []
        for pattern_info in error_patterns:
            pattern = pattern_info['pattern']
            if pattern[0] == pattern[1] == pattern[2]:
                insights.append(f'{pattern[0]}   ')
            elif 'workspace_overload' in pattern and 'attention_mismatch' in pattern:
                insights.append(' ')
            elif 'goal_prediction_error' in pattern and 'qualia_mismatch' in pattern:
                insights.append('      ')
        return insights

    def _analyze_parameter_interactions(self) -> List[Dict]:
        if len(self.policy_adjustments) < 10:
            return []
        interactions = []
        recent_adjustments = list(self.policy_adjustments)[-10:]
        param_names = list(self.policy_params.keys())
        for i, param1 in enumerate(param_names):
            for param2 in param_names[i + 1:]:
                values1 = []
                values2 = []
                for adj_record in recent_adjustments:
                    adjustments = adj_record.get('adjustments', {})
                    if param1 in adjustments and param2 in adjustments:
                        values1.append(adjustments[param1])
                        values2.append(adjustments[param2])
                if len(values1) >= 3:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    if abs(correlation) > 0.6:
                        interactions.append({'param1': param1, 'param2': param2, 'correlation': correlation, 'relationship': 'synergistic' if correlation > 0 else 'antagonistic', 'strength': 'strong' if abs(correlation) > 0.8 else 'moderate'})
        return interactions

    def get_conscious_summary(self) -> str:
        if not self.current_contents:
            return '(empty consciousness)'
        focus = self.attention_focus
        others = [c for c in self.current_contents if c != focus]
        try:
            if focus is not None:
                summary = f"   : {(focus.semantic_meaning if hasattr(focus, 'semantic_meaning') else '')[:20]:<20} )"
            else:
                summary = '  :   '
        except (AttributeError, TypeError):
            summary = '   :              )'
        if others:
            valid_others = [c for c in others[:2] if hasattr(c, 'semantic_meaning')]
            if valid_others:
                summary += f" |    : {', '.join([c.semantic_meaning[:20] for c in valid_others])}"
        return summary

    def get_policy_status(self) -> Dict[str, Any]:
        return {'policy_params': self.policy_params.copy(), 'cumulative_reward': self.cumulative_reward, 'reward_history': list(self.reward_history), 'recent_avg_reward': float(np.mean(list(self.reward_history)[-20:])) if self.reward_history else 0.0, 'error_count': len(self.error_detection_buffer), 'adjustment_count': len(self.policy_adjustments), 'recent_errors': list(self.error_detection_buffer)[-5:] if self.error_detection_buffer else []}

    def metrics_window(self, W: int = 512):
        """
        Return (feature_names, array[T,F]) over the last W steps for selected metrics.
        feature_names order matches columns of the returned 2D array.
        """
        import numpy as _np
        from collections import deque
        if not hasattr(self, "_reward_hist"): self._reward_hist = deque(maxlen=4096)
        if not hasattr(self, "_success_hist"): self._success_hist = deque(maxlen=4096)
        if not hasattr(self, "_kl_hist"): self._kl_hist = deque(maxlen=1024)
        if not hasattr(self, "_ploss_hist"): self._ploss_hist = deque(maxlen=1024)
        def _tail(lst, w):
            x = _np.asarray(list(lst)[-w:], dtype=float)
            if x.size == 0: return _np.zeros((0,), dtype=float)
            return x
        names, cols = [], []
        r = _tail(self._reward_hist, W); names.append("reward"); cols.append(r)
        s = _tail(self._success_hist, W); names.append("success"); cols.append(s)
        k = _tail(self._kl_hist, W);     names.append("kl");     cols.append(k)
        p = _tail(self._ploss_hist, W);  names.append("policy_loss"); cols.append(p)
        L = max((c.shape[0] for c in cols), default=0)
        if L == 0:
            return names, _np.zeros((0, len(cols)), dtype=float)
        mats = []
        for c in cols:
            if c.shape[0] == 0:
                mats.append(_np.zeros((L,), dtype=float))
            elif c.shape[0] < L:
                pad = _np.ones((L - c.shape[0],), dtype=float) * float(c[-1])
                mats.append(_np.concatenate([c, pad], axis=0))
            else:
                mats.append(c[-L:])
        X = _np.stack(mats, axis=1)
        return names, X

class GoalType(Enum):
    STABILIZE = 'stabilize'
    EXPLORE = 'explore'
    UNDERSTAND_SELF = 'understand_self'
    ADAPT = 'adapt'
    OPTIMIZE = 'optimize'
    REST = 'rest'
    META_OPTIMIZE = 'meta_optimize'

@dataclass
class Goal:
    type: GoalType
    priority: float
    created_at: float
    target_delta: Optional[float] = None
    target_stability: Optional[float] = None
    achieved: bool = False

    def evaluate_achievement(self, current_state: Dict[str, float], state_history: deque) -> bool:
        if len(state_history) < 10:
            return bool(np.random.random() < 0.5)
        recent_states = list(state_history)[-10:]
        if self.type == GoalType.STABILIZE:
            avg_delta = np.mean([s.get('delta_hat', 0.5) for s in recent_states])
            avg_stability = np.mean([s.get('stability', 0.5) for s in recent_states])
            current_delta = current_state.get('delta_hat', 1.0)
            current_stability = current_state.get('stability', 0.0)
            delta_improved = current_delta < avg_delta * 0.9
            stability_improved = current_stability > avg_stability * 1.05
            return bool(delta_improved or stability_improved)
        elif self.type == GoalType.EXPLORE:
            avg_delta = np.mean([s.get('delta_hat', 0.5) for s in recent_states])
            return bool(current_state.get('delta_hat', 0.0) > avg_delta * 1.05)
        elif self.type == GoalType.UNDERSTAND_SELF:
            avg_conf = np.mean([s.get('meta_confidence', 0.5) for s in recent_states])
            return bool(current_state.get('meta_confidence', 0.0) > avg_conf * 1.03)
        elif self.type == GoalType.REST:
            avg_valence = np.mean([s.get('qualia_valence', 0.5) for s in recent_states])
            return bool(current_state.get('qualia_valence', 0.0) > avg_valence * 1.05)
        return bool(np.random.random() < 0.5)

class GoalGenerator:

    def __init__(self):
        self.current_goal: Optional[Goal] = None
        self.goal_history: deque = deque(maxlen=100)
        self.goal_stack: List[Goal] = []
        self.qualia_stats = {'arousal': {'mean': 0.5, 'std': 0.2}, 'valence': {'mean': 0.5, 'std': 0.2}, 'entropy': {'mean': 0.5, 'std': 0.2}, 'engagement': {'mean': 0.5, 'std': 0.2}, 'frustration': {'mean': 0.5, 'std': 0.2}}
        
        # === DYNAMIC GOAL GENERATION ===
        self.discovered_goals: Dict[str, Dict[str, Any]] = {}
        self.goal_patterns: deque = deque(maxlen=100)
        self.goal_composition_history: List[Tuple[List[GoalType], str]] = []

    def update_qualia_statistics(self, qualia: QualiaState):
        if len(qualia.history) > 50:
            recent = list(qualia.history)[-50:]
            for key in self.qualia_stats.keys():
                    values = [h[key] for h in recent]
                    self.qualia_stats[key]['mean'] = float(np.mean(values))
                    self.qualia_stats[key]['std'] = float(max(0.05, np.std(values)))

    def compute_goal_urgency(self, goal_type: GoalType, qualia: QualiaState, self_model, world_state: Dict) -> float:
        urgency = 0.0
        if goal_type == GoalType.UNDERSTAND_SELF:
            entropy_z = (qualia.entropy - self.qualia_stats['entropy']['mean']) / self.qualia_stats['entropy']['std']
            if hasattr(self_model, 'meta_confidence'):
                conf_deficit = max(0, 0.5 - self_model.meta_confidence)
            else:
                conf_deficit = max(0, 0.5 - self_model.get('meta_confidence', 0.5)) if isinstance(self_model, dict) else 0.0
            urgency = np.tanh(entropy_z * 0.5 + conf_deficit * 2.0)
        elif goal_type == GoalType.STABILIZE:
            arousal_z = (qualia.arousal - self.qualia_stats['arousal']['mean']) / self.qualia_stats['arousal']['std']
            urgency = np.tanh(arousal_z * 0.7)
        elif goal_type == GoalType.EXPLORE:
            valence_z = (qualia.valence - self.qualia_stats['valence']['mean']) / self.qualia_stats['valence']['std']
            if hasattr(self_model, 'belief_stability'):
                stability_boost = self_model.belief_stability
            else:
                stability_boost = self_model.get('belief_stability', 0.5) if isinstance(self_model, dict) else 0.5
            urgency = np.tanh((valence_z * 0.5 + stability_boost - 0.5) * 0.8)
        elif goal_type == GoalType.OPTIMIZE:
            engagement_z = (qualia.engagement - self.qualia_stats['engagement']['mean']) / self.qualia_stats['engagement']['std']
            urgency = np.tanh(engagement_z * 0.6)
        elif goal_type == GoalType.REST:
            energy_level = world_state.get('energy_level', 0.5)
            urgency = np.tanh((0.3 - energy_level) * 3.0)
        elif goal_type == GoalType.ADAPT:
            urgency = 0.3
        return max(0.0, urgency)

    def generate_goal(self, self_model, qualia: QualiaState, world_state: Dict, state_history: deque) -> Goal:
        self.update_qualia_statistics(qualia)
        if self.current_goal and self.current_goal.evaluate_achievement(world_state, state_history):
            self.current_goal.achieved = True
            self.goal_history.append(self.current_goal)
            self.current_goal = None
        if self.current_goal is None:
            urgencies = {}
            for gtype in GoalType:
                urgencies[gtype] = self.compute_goal_urgency(gtype, qualia, self_model, world_state)
            selected_type = max(urgencies, key=lambda k: urgencies[k])
            selected_urgency = urgencies[selected_type]
            new_goal = Goal(type=selected_type, priority=selected_urgency, created_at=world_state.get('t', 0))
            self.current_goal = new_goal
            self.goal_history.append(new_goal)
        assert self.current_goal is not None
        return self.current_goal

    def derive_meta_goal(self) -> Goal:
        recent = list(self.goal_history)[-10:]
        success_rate = sum((1 for g in recent if g.achieved)) / max(len(recent), 1)
        if success_rate < 0.3:
            return Goal(type=GoalType.META_OPTIMIZE, priority=1.0, created_at=0)
        return self.current_goal or Goal(type=GoalType.STABILIZE, priority=0.5, created_at=time.time())
    
    # === DYNAMIC GOAL DISCOVERY METHODS ===
    
    def discover_goal_from_pattern(self, qualia: QualiaState, self_model, world_state: Dict) -> Optional[Dict[str, Any]]:
        """내부 패턴에서 새로운 목표 발견"""
        
        # Pattern 1: Frustration + Low Energy → Efficiency Goal
        if qualia.frustration > 0.6 and world_state.get('energy_level', 0.5) < 0.3:
            goal_id = f"efficiency_{int(time.time())}"
            if goal_id not in self.discovered_goals:
                self.discovered_goals[goal_id] = {
                    'id': goal_id,
                    'description': 'Minimize frustration while conserving energy',
                    'reward_fn': lambda q, w: -(q.frustration ** 2) - 0.5 * (1.0 - w.get('energy_level', 0.5)),
                    'success_criteria': {'frustration': 0.3, 'energy_efficiency': 0.7},
                    'created_at': time.time(),
                    'attempts': 0,
                    'successes': 0
                }
                return self.discovered_goals[goal_id]
        
        # Pattern 2: High Entropy + Low Engagement → Focus Goal
        if qualia.entropy > 0.7 and qualia.engagement < 0.4:
            goal_id = f"focus_{int(time.time())}"
            if goal_id not in self.discovered_goals:
                self.discovered_goals[goal_id] = {
                    'id': goal_id,
                    'description': 'Reduce entropy and increase engagement',
                    'reward_fn': lambda q, w: -q.entropy + q.engagement,
                    'success_criteria': {'entropy': 0.4, 'engagement': 0.6},
                    'created_at': time.time(),
                    'attempts': 0,
                    'successes': 0
                }
                return self.discovered_goals[goal_id]
        
        # Pattern 3: Oscillating phi → Stabilization Goal
        if hasattr(self_model, 'state_history') and len(self_model.state_history) > 20:
            recent = list(self_model.state_history)[-20:]
            if recent and 'phi' in recent[0]:
                phi_values = [s.get('phi', 0.5) for s in recent]
                phi_variance = np.var(phi_values)
                if phi_variance > 0.05:
                    goal_id = f"stabilize_phi_{int(time.time())}"
                    if goal_id not in self.discovered_goals:
                        self.discovered_goals[goal_id] = {
                            'id': goal_id,
                            'description': 'Stabilize integration (phi)',
                            'reward_fn': lambda q, w: -phi_variance,
                            'success_criteria': {'phi_variance': 0.02},
                            'created_at': time.time(),
                            'attempts': 0,
                            'successes': 0
                        }
                        return self.discovered_goals[goal_id]
        
        return None
    
    def compose_hierarchical_goal(self, sub_goal_types: List[GoalType]) -> Dict[str, Any]:
        """기존 목표들을 조합하여 상위 목표 생성"""
        goal_id = f"composite_{'_'.join([g.value for g in sub_goal_types])}_{int(time.time())}"
        
        description_parts = [g.value for g in sub_goal_types]
        
        composite_goal = {
            'id': goal_id,
            'description': f"Achieve: {' AND '.join(description_parts)}",
            'sub_goals': sub_goal_types,
            'created_at': time.time(),
            'attempts': 0,
            'successes': 0,
            'is_composite': True
        }
        
        self.discovered_goals[goal_id] = composite_goal
        self.goal_composition_history.append((sub_goal_types, goal_id))
        
        return composite_goal
    
    def evaluate_discovered_goal(self, goal_id: str, qualia: QualiaState, world_state: Dict) -> bool:
        """발견된 목표의 달성 여부 평가"""
        if goal_id not in self.discovered_goals:
            return False
        
        goal = self.discovered_goals[goal_id]
        goal['attempts'] += 1
        
        criteria = goal.get('success_criteria', {})
        achieved = True
        
        for key, target in criteria.items():
            if key == 'frustration':
                if qualia.frustration > target:
                    achieved = False
            elif key == 'energy_efficiency':
                if world_state.get('energy_level', 0) < target:
                    achieved = False
            elif key == 'entropy':
                if qualia.entropy > target:
                    achieved = False
            elif key == 'engagement':
                if qualia.engagement < target:
                    achieved = False
            elif key == 'phi_variance':
                # Would need actual phi variance calculation
                pass
        
        if achieved:
            goal['successes'] += 1
        
        return achieved

class MetaCognitiveNetwork:
    """Neural network for meta-cognitive learning - learns to monitor and adjust beliefs"""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Network weights
        scale1 = np.sqrt(2.0 / input_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        
        # Output heads
        self.W_belief = np.random.randn(output_dim, 3).astype(np.float32) * 0.01
        self.W_meta = np.random.randn(output_dim, 1).astype(np.float32) * 0.01
        self.W_confidence = np.random.randn(output_dim, 1).astype(np.float32) * 0.01
        
        self.meta_buffer: deque = deque(maxlen=1000)
        self.update_count = 0
        self.meta_loss_history: deque = deque(maxlen=100)
    
    def forward(self, experience: np.ndarray) -> Dict[str, Any]:
        h = np.maximum(0, experience @ self.W1 + self.b1)
        core_repr = np.tanh(h @ self.W2 + self.b2)
        
        belief_adj = np.tanh(core_repr @ self.W_belief)
        meta_delta = np.tanh(core_repr @ self.W_meta).item()
        confidence = 0.5 + 0.5 * np.tanh(core_repr @ self.W_confidence).item()
        
        return {
            'belief_adjustments': belief_adj.flatten(),
            'meta_awareness_delta': meta_delta,
            'meta_confidence': confidence,
            'core_representation': core_repr
        }
    
    def learn(self, lr: float = 1e-4, batch_size: int = 32):
        """Train meta-cognitive network on buffered experience"""
        if len(self.meta_buffer) < batch_size:
            return
        
        # Bias-aware sampling: KL-budgeted toward targets using global qualia bias
        
        N = len(self.meta_buffer)
        
        try:
        
            import numpy as _np
        
            b = float(globals().get('_GLOBAL_QUALIA_REPLAY_BIAS', 0.0))
        
            targets = _np.array([float(x.get('target', 0.0)) for x in self.meta_buffer], dtype=_np.float32)
        
            # z-normalize (stable); if std=0, fall back to zeros
        
            t_mean = float(targets.mean()) if targets.size else 0.0
        
            t_std = float(targets.std()) if targets.size else 0.0
        
            if t_std > 1e-8:
        
                z = (targets - t_mean) / t_std
        
            else:
        
                z = _np.zeros_like(targets)
        
            # ---- KL-budget calibration ----
        
            U = _np.ones(N, dtype=_np.float64) / float(N)
        
            kl_budget = float(QUALIA_CFG.get('replay_kl_budget', 0.20))
        
            kl_cap = float(QUALIA_CFG.get('replay_kl_max', 0.80))
        
            kl_target = float(min(kl_cap, max(0.0, abs(b)) * kl_budget))
        
            sign = 1.0 if b >= 0.0 else -1.0
        
            def _softmax(alpha):
        
                if not z.size:
        
                    return U
        
                lo = alpha * sign * z
        
                lo = lo - float(lo.max())
        
                w = _np.exp(lo).astype(_np.float64)
        
                s = w.sum()
        
                if s <= 0 or not _np.isfinite(s):
        
                    return U
        
                return w / s
        
            def _kl(p, q):
        
                eps = 1e-12
        
                p = _np.clip(p, eps, 1.0)
        
                q = _np.clip(q, eps, 1.0)
        
                return float(_np.sum(p * _np.log(p / q)))
        
            alpha_lo, alpha_hi = 0.0, 20.0
        
            best_alpha, best_diff = 0.0, float('inf')
        
            for _ in range(25):
        
                alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        
                p_mid = _softmax(alpha_mid)
        
                d_mid = _kl(p_mid, U) - kl_target
        
                if abs(d_mid) < best_diff:
        
                    best_alpha, best_diff = alpha_mid, abs(d_mid)
        
                if d_mid > 0:
        
                    alpha_hi = alpha_mid
        
                else:
        
                    alpha_lo = alpha_mid
        
            p = _softmax(best_alpha)
        
            indices = np.random.choice(N, size=batch_size, replace=False, p=p)
        
        except Exception:
        
            indices = np.random.choice(len(self.meta_buffer), size=batch_size, replace=False)
        
        batch = [self.meta_buffer[i] for i in indices]
        total_loss = 0.0
        grad_W1 = np.zeros_like(self.W1)
        grad_b1 = np.zeros_like(self.b1)
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)
        grad_W_meta = np.zeros_like(self.W_meta)
        
        for item in batch:
            x = item['experience']
            target = item['target']
            
            h = np.maximum(0, x @ self.W1 + self.b1)
            core_repr = np.tanh(h @ self.W2 + self.b2)
            meta_quality_pred = np.tanh(core_repr @ self.W_meta).item()
            
            loss = (meta_quality_pred - target) ** 2
            total_loss += loss
            
            d_loss = 2.0 * (meta_quality_pred - target)
            d_meta = d_loss * (1.0 - meta_quality_pred ** 2)
            grad_W_meta += np.outer(core_repr, d_meta)
            
            d_core = d_meta * self.W_meta.flatten()
            d_core = d_core * (1.0 - core_repr ** 2)
            
            grad_W2 += np.outer(h, d_core)
            grad_b2 += d_core
            
            d_h = d_core @ self.W2.T
            d_h = d_h * (h > 0)
            
            grad_W1 += np.outer(x, d_h)
            grad_b1 += d_h
        
        grad_W1 /= batch_size
        grad_b1 /= batch_size
        grad_W2 /= batch_size
        grad_b2 /= batch_size
        grad_W_meta /= batch_size
        
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W_meta -= lr * grad_W_meta
        
        avg_loss = total_loss / batch_size
        self.meta_loss_history.append(avg_loss)
        self.update_count += 1


class RecursiveSelfModel:

    def __init__(self, n: int, K: int):
        self.n_nodes = n
        self.K_categories = K
        self.belief_stability = 0.5
        self.belief_adaptation = 0.5
        self.belief_prediction = 0.5
        self.aware_of_beliefs = True
        self.belief_about_beliefs = {'confidence_in_stability_belief': 0.5, 'confidence_in_adaptation_belief': 0.5, 'confidence_in_prediction_belief': 0.5}
        self.meta_awareness = 0.5
        self.knows_it_knows = False
        
        # Neural meta-cognitive network (replaces heuristics)
        self.meta_network = MetaCognitiveNetwork()
        self.use_neural_metacog = True  # Toggle between neural and heuristic
        self.current_introspection_depth = 0
        self.max_safe_depth = 10
        self.state_history: deque = deque(maxlen=200)
        self.prediction_errors: deque = deque(maxlen=100)
        self.action_outcomes: deque = deque(maxlen=100)
        self.introspection_history: deque = deque(maxlen=100)
        self.meta_confidence = 0.5
        self.model_revision_count = 0
        self.prediction_window = 5
        
        # === AUTONOMOUS EXPANSION SYSTEM ===
        self.capability_gaps: deque = deque(maxlen=56)
        self.expansion_history: deque = deque(maxlen=128)
        self.last_expansion_time: float = 0.0
        self.expansion_cooldown: float = 100.0
        self.current_capabilities: Set[str] = {'perception', 'memory', 'reasoning'}
        self.attempted_capabilities: Dict[str, Dict[str, Any]] = {}
        self.self_improvement_hypotheses: deque = deque(maxlen=30)
        self.hypothesis_test_results: Dict[str, Dict[str, Any]] = {}
        self.modular_skills: Dict[str, Callable] = {}
        self.skill_composition_history: List[Tuple[List[str], str, float]] = []
    def introspect(self, depth: int=1) -> Dict[str, Any]:
        """Slim, structured self-introspection (no narratives).
        Returns only machine-usable signals with stable keys.
        """
        # Ensure rolling histories exist
        if not hasattr(self, "_hist_stability"):
            from collections import deque
            self._hist_stability = deque(maxlen=128)
            self._hist_adaptation = deque(maxlen=128)
            self._hist_prediction = deque(maxlen=128)
            self._hist_meta = deque(maxlen=128)

        # Current scalars with robust defaults
        stability = float(getattr(self, "belief_stability", 0.5))
        adaptation = float(getattr(self, "belief_adaptation", 0.5))
        prediction = float(getattr(self, "belief_prediction", 0.5))
        # meta confidence: average over belief_about_beliefs if present, else meta_awareness, else 0.5
        meta_conf = 0.5
        try:
            bab = getattr(self, "belief_about_beliefs", None)
            if isinstance(bab, dict) and len(bab) > 0:
                vals = [float(v) for v in bab.values() if isinstance(v, (int, float))]
                if vals:
                    meta_conf = float(sum(vals) / len(vals))
            else:
                meta_conf = float(getattr(self, "meta_awareness", 0.5))
        except Exception:
            meta_conf = 0.5

        # Update histories
        self._hist_stability.append(stability)
        self._hist_adaptation.append(adaptation)
        self._hist_prediction.append(prediction)
        self._hist_meta.append(meta_conf)

        import numpy as _np
        def _ema(seq, alpha=0.2):
            x = _np.asarray(seq, dtype=_np.float32)
            if x.size == 0:
                return 0.0
            ema = 0.0
            a = float(alpha)
            for v in x[::-1]:  # recent first
                ema = a * v + (1.0 - a) * ema
            return float(ema)

        # Short vs long EMAs for drift (recent - baseline)
        def _drift(hist):
            if len(hist) < 3:
                return 0.0
            short = _ema(list(hist)[-16:], alpha=0.35)
            long = _ema(hist, alpha=0.05)
            return float(_np.clip(short - long, -1.0, 1.0))

        drift = {
            "stability": _drift(self._hist_stability),
            "adaptation": _drift(self._hist_adaptation),
            "prediction": _drift(self._hist_prediction),
            "meta_confidence": _drift(self._hist_meta),
        }

        # Optional hooks (graceful fallbacks)
        try:
            agency = float(getattr(self, "agency", 0.5))
        except Exception:
            agency = 0.5
        try:
            unity = float(getattr(self, "unity_score", 0.5))
        except Exception:
            unity = 0.5

        report = {
            "version": "slim-1",
            "timestamp": float(__import__("time").time()),
            "belief": {
                "stability": stability,
                "adaptation": adaptation,
                "prediction": prediction,
                "meta_confidence": meta_conf,
            },
            "drift": drift,
            "agency": agency,
            "unity": unity,
        }
        return report



    def _verbalize_level1(self) -> str:
        """Concise description of current first-order beliefs."""
        return (
            f"L1: stability={self.belief_stability:.2f}, "
            f"adaptation={self.belief_adaptation:.2f}, "
            f"prediction={self.belief_prediction:.2f}"
        )

    def _verbalize_level2(self) -> str:
        """Summarize confidence about beliefs (second-order)."""
        try:
            vals = list(self.belief_about_beliefs.values())
            conf_avg = sum(vals) / len(vals) if vals else 0.5
        except Exception:
            conf_avg = 0.5
        return f"L2: confidence_in_beliefs={conf_avg:.2f}"

    def _verbalize_level3(self, loop_active: bool) -> str:
        if loop_active:
            return f'L3: strange_loop=ON meta={self.meta_awareness:.2f}'
        else:
            return f'L3: strange_loop=OFF'

    def _verbalize_level4(self) -> str:
        return f'L4: meta-meta_awareness'

    def _verbalize_level5(self) -> str:
        return f'L5: thought_observation'

    def _verbalize_level6(self) -> str:
        return f'L6: observation^2'

    def _verbalize_level7(self) -> str:
        return f'L7: pure_being'

    def _detect_strange_loop(self) -> bool:
        meta_sufficient = self.meta_awareness > 0.6
        belief_values = list(self.belief_about_beliefs.values())
        belief_stable = np.std(belief_values) < 0.4 if belief_values else False
        recent_active = len(self.introspection_history) > 5
        if meta_sufficient and belief_stable and recent_active:
            activation_prob = min(0.9, self.meta_awareness)
            loop = np.random.random() < activation_prob
        else:
            loop = np.random.random() < 0.1
        self.knows_it_knows = loop
        return loop

    def update_meta_awareness(self, conscious_contents: List):
        """Update meta-awareness based on self-referential content in consciousness
        
        Meta-awareness represents the system's awareness of its own processes
        Driven by: self-referential content, introspection depth, belief coherence
        
        More precise dynamics:
        - Self-referential content increases awareness
        - Coherent beliefs stabilize awareness
        - Lack of self-reference causes awareness decay
        - Asymmetric response: harder to gain than to lose (realistic)
        """
        if not conscious_contents:
            # Decay when no conscious content
            self.meta_awareness *= 0.93
            self.meta_awareness = max(0.05, self.meta_awareness)
            return
        
        # Count self-referential elements
        self_referential_count = sum(
            1 for c in conscious_contents 
            if any(keyword in c.content_type.lower() for keyword in ['self', 'belief', 'meta', 'aware', 'know'])
        )
        
        total_count = len(conscious_contents)
        self_ref_ratio = float(self_referential_count) / max(1, total_count)
        
        # Measure belief coherence
        beliefs = [
            self.belief_stability,
            self.belief_adaptation,
            self.belief_prediction
        ]
        belief_variance = float(np.var(beliefs))
        belief_coherence = 1.0 - min(1.0, belief_variance * 3.0)  # High variance = low coherence
        
        # Compute target awareness
        # High self-reference + high coherence -> high awareness
        base_target = self_ref_ratio ** 0.8  # Sublinear response
        coherence_bonus = belief_coherence * 0.3
        target_awareness = min(0.95, base_target + coherence_bonus)
        
        # Asymmetric update: easier to lose than to gain
        current = self.meta_awareness
        
        if target_awareness > current:
            # Gaining awareness is slow
            gain_rate = 0.15
            delta = (target_awareness - current) * gain_rate
        else:
            # Losing awareness is faster
            decay_rate = 0.25
            delta = (target_awareness - current) * decay_rate
        
        self.meta_awareness += delta
        self.meta_awareness = np.clip(self.meta_awareness, 0.05, 0.95)
        
        # Update confidence based on consistency
        # If awareness is stable, confidence increases
        if len(self.introspection_history) > 5:
            recent_awareness = [h['meta_awareness'] for h in list(self.introspection_history)[-5:]]
            awareness_stability = 1.0 - float(np.std(recent_awareness))
            
            # Confidence tracks stability
            target_confidence = awareness_stability * 0.7 + self.meta_awareness * 0.3
            
            # 더 빠른 업데이트
            self.meta_confidence = 0.7 * self.meta_confidence + 0.3 * target_confidence
            self.meta_confidence = np.clip(self.meta_confidence, 0.1, 0.95)
        
        # Record introspection state
        self.introspection_history.append({
            'meta_awareness': float(self.meta_awareness),
            'meta_confidence': float(self.meta_confidence),
            'self_ref_ratio': float(self_ref_ratio),
            'belief_coherence': float(belief_coherence),
            'loop_active': bool(self.knows_it_knows)
        })

    def downward_causation(self) -> Dict[str, float]:
        if not self.knows_it_knows:
            return {}
        adjustments = {}
        if self.belief_about_beliefs['confidence_in_stability_belief'] < 0.4:
            adjustments['exploration_mult'] = 0.3
            adjustments['stability_boost'] = 0.5
        elif self.belief_about_beliefs['confidence_in_stability_belief'] > 0.8:
            adjustments['exploration_mult'] = 3.0
        if self.belief_about_beliefs['confidence_in_prediction_belief'] < 0.4:
            adjustments['learning_rate_mult'] = 5.0
            adjustments['gate_adjust_mult'] = 2.0
        elif self.belief_about_beliefs['confidence_in_prediction_belief'] > 0.8:
            adjustments['learning_rate_mult'] = 0.5
        if self.belief_about_beliefs['confidence_in_adaptation_belief'] > 0.7:
            adjustments['exploration_boost'] = 1.0
            adjustments['complexity_mult'] = 2.0
        elif self.belief_about_beliefs['confidence_in_adaptation_belief'] < 0.3:
            adjustments['exploration_mult'] = 0.1
        if self.meta_awareness > 0.8:
            adjustments['deliberation_factor'] = 2.0
            adjustments['action_precision'] = 1.5
        if self.meta_awareness < 0.3:
            adjustments['automatic_mode'] = True
            adjustments['deliberation_factor'] = 0.5
        if self.meta_confidence < 0.4:
            adjustments['sensitivity_mult'] = 3.0
            adjustments['gate_openness'] = 2.0
        return adjustments

    def predict_next_state(self) -> Optional[Dict[str, float]]:
        if len(self.state_history) < self.prediction_window:
            return None
        recent = list(self.state_history)[-self.prediction_window:]
        predictions = {}
        for key in ['delta_hat', 'm', 'stability']:
            values = [s.get(key, 0.0) for s in recent]
            trend = np.mean(np.diff(values)) if len(values) > 1 else 0.0
            base_pred = values[-1] + trend
            noise = np.random.normal(0, 0.1)
            predictions[key] = float(base_pred + noise)
        return predictions

    def evaluate_prediction(self, actual_state: Dict[str, float]) -> float:
        pred = self.predict_next_state()
        if pred is None:
            return 0.0
        errors = []
        for key in ['delta_hat', 'm', 'stability']:
            if key in pred and key in actual_state:
                err = abs(pred[key] - actual_state[key])
                errors.append(err)
        return float(np.mean(errors)) if errors else 0.0

    def update_beliefs(self, prediction_error: float, outcome_success: bool, current_stability: float):
        """Update beliefs - uses neural network if enabled, otherwise heuristic"""
        if self.use_neural_metacog and self.meta_network:
            self._neural_update_beliefs(prediction_error, outcome_success, current_stability)
        else:
            self._heuristic_update_beliefs(prediction_error, outcome_success, current_stability)
    
    def _neural_update_beliefs(self, prediction_error: float, outcome_success: bool, current_stability: float):
        """Neural network-based belief updating"""
        # Encode experience
        features = []
        features.append(float(prediction_error))
        features.append(1.0 if outcome_success else 0.0)
        features.append(float(current_stability))
        features.append(float(self.meta_awareness))
        features.append(self.belief_stability)
        features.append(self.belief_adaptation)
        features.append(self.belief_prediction)
        
        # Recent error stats
        if len(self.prediction_errors) > 0:
            features.append(float(np.mean(list(self.prediction_errors)[-10:])))
            features.append(float(np.std(list(self.prediction_errors)[-10:]) if len(self.prediction_errors) > 1 else 0.0))
            features.append(float(np.mean(np.diff(list(self.prediction_errors)[-10:])) if len(self.prediction_errors) > 1 else 0.0))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Success rate
        if len(self.action_outcomes) > 0:
            features.append(sum(list(self.action_outcomes)[-10:]) / min(10, len(self.action_outcomes)))
        else:
            features.append(0.5)
        
        # Qualia (placeholder - will be filled by core)
        features.extend([0.5, 0.5])
        
        # Belief consistency
        belief_vals = [self.belief_stability, self.belief_adaptation, self.belief_prediction]
        features.append(float(np.std(np.array(belief_vals))))
        
        # Alignment
        if len(self.prediction_errors) > 0:
            avg_error = np.mean(list(self.prediction_errors)[-10:])
            alignment = 1.0 - abs((1.0 - avg_error) - self.belief_prediction)
        else:
            alignment = 0.5
        features.append(alignment)
        
        # Pad
        while len(features) < self.meta_network.input_dim:
            features.append(0.0)
        
        experience_vec = np.array(features[:self.meta_network.input_dim], dtype=np.float32)
        
        # Get neural outputs
        outputs = self.meta_network.forward(experience_vec)
        
        # Apply learned adjustments
        adj = outputs['belief_adjustments']
        self.belief_stability = np.clip(
            self.belief_stability * (1.0 + 0.2 * adj[0]),
            0.1, 0.99
        )
        self.belief_adaptation = np.clip(
            self.belief_adaptation * (1.0 + 0.2 * adj[1]),
            0.1, 0.99
        )
        self.belief_prediction = np.clip(
            self.belief_prediction * (1.0 + 0.2 * adj[2]),
            0.1, 0.99
        )
        
        # Meta-awareness from network
        self.meta_awareness = np.clip(
            self.meta_awareness + 0.1 * outputs['meta_awareness_delta'],
            0.0, 1.0
        )
        
        # Meta-confidence from network
        self.meta_confidence = outputs['meta_confidence']
        
        # Store for history
        self.prediction_errors.append(prediction_error)
        self.action_outcomes.append(outcome_success)
        
        # Store for meta-learning
        self._last_meta_experience = experience_vec
        self._last_meta_outputs = outputs
        
        # Periodic meta-learning update
        if len(self.prediction_errors) > 50 and len(self.prediction_errors) % 50 == 0:
            self._do_meta_learning()
    
    def _do_meta_learning(self):
        """Trigger meta-learning update"""
        if not hasattr(self, '_last_meta_experience'):
            return
        
        # Compute target based on recent performance
        if len(self.prediction_errors) > 10:
            next_error = np.mean(list(self.prediction_errors)[-5:])
        else:
            next_error = 0.5
        
        if len(self.action_outcomes) > 10:
            next_success = sum(list(self.action_outcomes)[-5:]) / 5.0
        else:
            next_success = 0.5
        
        # Meta target: did our meta-cognitive decisions improve performance
        error_reduction = max(0.0, 0.3 - next_error)
        success_reward = next_success - 0.5
        meta_reward = self.meta_awareness - 0.5
        target = 0.4 * error_reduction + 0.3 * success_reward + 0.3 * meta_reward
        target = float(np.clip(target, -1.0, 1.0))
        
        # Record experience
        self.meta_network.meta_buffer.append({
            'experience': self._last_meta_experience.copy(),
            'outputs': {k: v.copy() if isinstance(v, np.ndarray) else v 
                       for k, v in self._last_meta_outputs.items()},
            'target': target
        })
        
        # Learn
        self.meta_network.learn(lr=1e-4, batch_size=min(32, len(self.meta_network.meta_buffer)))
    
    def _heuristic_update_beliefs(self, prediction_error: float, outcome_success: bool, current_stability: float):
        """Original heuristic belief updating (fallback)"""
        if prediction_error < 0.05:
            self.belief_prediction *= 1.1
        elif prediction_error < 0.15:
            self.belief_prediction *= 1.03
        elif prediction_error < 0.3:
            self.belief_prediction *= 0.97
        else:
            self.belief_prediction *= 0.85
        self.belief_prediction = np.clip(self.belief_prediction, 0.1, 0.99)
        self.prediction_errors.append(prediction_error)
        stability_gap = current_stability - self.belief_stability
        self.belief_stability += stability_gap * 0.5
        if current_stability < 0.3:
            self.belief_stability *= 0.85
        elif current_stability > 0.95:
            self.belief_stability *= 1.1
        self.belief_stability = np.clip(self.belief_stability, 0.1, 0.99)
        if outcome_success:
            self.belief_adaptation *= 1.15
        else:
            self.belief_adaptation *= 0.92
        self.belief_adaptation = np.clip(self.belief_adaptation, 0.1, 0.99)
        self.action_outcomes.append(outcome_success)
        self._update_meta_confidence()

    def _update_meta_confidence(self):
        if len(self.prediction_errors) > 10:
            recent_errors = list(self.prediction_errors)[-10:]
            avg_accuracy = 1.0 - min(1.0, np.mean(recent_errors))
        else:
            avg_accuracy = 0.5
        beliefs = [self.belief_stability, self.belief_adaptation, self.belief_prediction]
        consistency = 1.0 - min(1.0, np.std(np.array(beliefs)))
        if len(self.action_outcomes) > 10:
            recent_success = sum(list(self.action_outcomes)[-10:]) / 10.0
        else:
            recent_success = 0.5
        self.meta_confidence = 0.4 * avg_accuracy + 0.3 * consistency + 0.3 * recent_success

    def revise_self_model(self):
        self.model_revision_count += 1
        if len(self.state_history) > 20:
            recent_states = list(self.state_history)[-50:]
            actual_stabilities = [s.get('stability', 0.5) for s in recent_states]
            self.belief_stability = np.mean(actual_stabilities) * 0.9
            stability_changes = np.abs(np.diff(actual_stabilities))
            self.belief_adaptation = 1.0 - min(1.0, np.mean(stability_changes) * 2.0)
            if len(self.prediction_errors) > 10:
                recent_errors = list(self.prediction_errors)[-20:]
                avg_error = np.mean(recent_errors)
                self.belief_prediction = max(0.2, 1.0 - avg_error)
            else:
                self.belief_prediction = 0.5
        else:
            self.belief_stability = 0.5
            self.belief_adaptation = 0.5
            self.belief_prediction = 0.5
        if len(self.action_outcomes) > 10:
            success_rate = sum(list(self.action_outcomes)[-20:]) / min(20, len(self.action_outcomes))
            self.belief_about_beliefs['confidence_in_stability_belief'] = float(0.5 + (self.belief_stability - 0.5) * success_rate)
            self.belief_about_beliefs['confidence_in_adaptation_belief'] = float(0.5 + (self.belief_adaptation - 0.5) * success_rate)
            self.belief_about_beliefs['confidence_in_prediction_belief'] = float(self.belief_prediction * success_rate)
        else:
            for key in self.belief_about_beliefs:
                self.belief_about_beliefs[key] = 0.5
        if len(self.prediction_errors) > 5:
            recent_errors = list(self.prediction_errors)[-10:]
            avg_error = np.mean(recent_errors)
            if avg_error > 0.3:
                self.prediction_window = max(3, self.prediction_window - 1)
            elif avg_error < 0.1:
                self.prediction_window = min(10, self.prediction_window + 1)
        if self.meta_awareness > 0.7:
            self.max_safe_depth = min(10, self.max_safe_depth + 1)
        elif self.meta_awareness < 0.3:
            self.max_safe_depth = max(5, self.max_safe_depth - 1)
        if len(self.prediction_errors) > 10 and len(self.action_outcomes) > 10:
            recent_errors = list(self.prediction_errors)[-20:]
            recent_success = sum(list(self.action_outcomes)[-20:]) / min(20, len(self.action_outcomes))
            avg_accuracy = 1.0 - min(1.0, np.mean(recent_errors))
            beliefs = [self.belief_stability, self.belief_adaptation, self.belief_prediction]
            consistency = 1.0 - min(1.0, np.std(np.array(beliefs)))
            self.meta_confidence = 0.35 * avg_accuracy + 0.35 * consistency + 0.3 * recent_success
        else:
            self.meta_confidence = 0.4
        if len(self.state_history) > 100:
            recent = list(self.state_history)[-50:]
            historical = list(self.state_history)[:-50]
            if historical:
                scored = []
                for s in historical:
                    importance = abs(s.get('stability', 0.5) - 0.5) + abs(s.get('delta_hat', 0.5) - 0.5)
                    scored.append((importance, s))
                scored.sort(reverse=True)
                important_past = [s for _, s in scored[:50]]
                combined = important_past + recent
                self.state_history = deque(combined, maxlen=200)
            else:
                self.state_history = deque(recent, maxlen=200)
        logging.info(f'Self-model revision #{self.model_revision_count} (Level 1)')
        logging.info(f'   New beliefs: stability={self.belief_stability:.2f}, adaptation={self.belief_adaptation:.2f}, prediction={self.belief_prediction:.2f}')
        logging.info(f'   Meta-confidence: {self.meta_confidence:.2f}, Introspection depth: {self.max_safe_depth}')

    def revise_self_model_level2(self):
        """Level 2: meta-parameter adjustments building on revise_self_model."""
        self.revise_self_model()
        logging.info('Level 2 Revision - Meta-Parameter Adjustment')
        success_rate = 0.5
        if len(self.action_outcomes) > 20:
            success_rate = sum(list(self.action_outcomes)[-20:]) / 20.0
        if success_rate < 0.3:
            self.max_safe_depth = min(10, self.max_safe_depth + 2)
        elif success_rate > 0.7:
            self.max_safe_depth = max(5, self.max_safe_depth - 1)
        if len(self.prediction_errors) > 30:
            recent_errors = list(self.prediction_errors)[-30:]
            error_volatility = np.std(recent_errors)
            if error_volatility > 0.3:
                self.prediction_window = 3
            else:
                self.prediction_window = 8
        for key in self.belief_about_beliefs:
            self.belief_about_beliefs[key] *= 0.7
            self.belief_about_beliefs[key] = max(0.2, self.belief_about_beliefs[key])
        if self.meta_awareness > 0.7:
            self.meta_awareness *= 0.85
        else:
            self.meta_awareness = min(0.9, self.meta_awareness * 1.2)
        logging.info(f'   Adjusted introspection depth: {self.max_safe_depth}')
        logging.info(f'   Adjusted prediction window: {self.prediction_window}')
        logging.info(f'   Adjusted meta-awareness: {self.meta_awareness:.2f}')

    def revise_self_model_level3(self):
        """Level 3 Revision - Structural Reset"""
        logging.info('Level 3 Revision - Structural Reset')
        self.belief_stability = 0.5
        self.belief_adaptation = 0.5
        self.belief_prediction = 0.5
        for key in self.belief_about_beliefs:
            self.belief_about_beliefs[key] = 0.45
        if len(self.state_history) > 30:
            recent = list(self.state_history)[-30:]
            self.state_history = deque(recent, maxlen=200)
        if len(self.prediction_errors) > 20:
            recent_errors = list(self.prediction_errors)[-20:]
            self.prediction_errors = deque(recent_errors, maxlen=100)
        if len(self.action_outcomes) > 20:
            recent_outcomes = list(self.action_outcomes)[-20:]
            self.action_outcomes = deque(recent_outcomes, maxlen=100)
        self.max_safe_depth = 7
        self.current_introspection_depth = 0
        self.introspection_history.clear()
        self.meta_awareness = 0.5
        self.knows_it_knows = False
        self.meta_confidence = 0.4
        self.prediction_window = 5
        self.model_revision_count += 1
        logging.info('   Structure reset complete')
        logging.info('   All beliefs reset to baseline')
        logging.info('   History truncated to recent 30 entries')
        logging.info('   Meta-systems reinitialized')

    def revise_self_model_level4_emergency(self):
        logging.warning('EMERGENCY MODE - Level 4 Revision')
        self.belief_stability = 0.6
        self.belief_adaptation = 0.3
        self.belief_prediction = 0.4
        for key in self.belief_about_beliefs:
            self.belief_about_beliefs[key] = 0.3
        if len(self.state_history) > 15:
            recent = list(self.state_history)[-15:]
            self.state_history = deque(recent, maxlen=200)
        else:
            self.state_history = deque(maxlen=200)
        self.prediction_errors.clear()
        self.action_outcomes.clear()
        self.max_safe_depth = 5
        self.current_introspection_depth = 0
        self.introspection_history.clear()
        self.meta_awareness = 0.3
        self.knows_it_knows = False
        self.meta_confidence = 0.25
        self.prediction_window = 3
        self.model_revision_count += 1
        print('   EMERGENCY: All systems set to safe minimum')
        print('   Exploration suppressed, stability prioritized')
        logging.warning('   EMERGENCY: All systems set to safe minimum')
        logging.warning('   Exploration suppressed, stability prioritized')
        logging.warning('   History cleared, fresh start initiated')

    def log_state(self, state: Dict[str, float]):
        self.state_history.append(state.copy())

    def to_dict(self) -> Dict[str, Any]:
        return {'belief_stability': float(self.belief_stability), 'belief_adaptation': float(self.belief_adaptation), 'belief_prediction': float(self.belief_prediction), 'meta_confidence': float(self.meta_confidence), 'revision_count': int(self.model_revision_count)}

    # === AUTONOMOUS EXPANSION METHODS ===
    
    def detect_capability_gaps(self, internal_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """내부 상태를 분석하여 능력 부족을 자동 감지"""
        gaps = []
        
        # 1. High meta-awareness but low action success
        if self.meta_awareness > 0.7 and len(self.action_outcomes) > 0:
            recent_success = sum(list(self.action_outcomes)[-20:]) / min(20, len(self.action_outcomes))
            if recent_success < 0.4:
                gaps.append({
                    'type': 'action_execution',
                    'severity': 0.8,
                    'description': 'High awareness but low action success',
                    'suggested_capability': 'motor_planning',
                    'evidence': {'meta_awareness': self.meta_awareness, 'success_rate': recent_success}
                })
        
        # 2. High prediction error → need better world model
        if len(self.prediction_errors) >= 10:
            recent_errors = list(self.prediction_errors)[-10:]
            avg_error = sum(recent_errors) / len(recent_errors)
            if avg_error > 0.6:
                gaps.append({
                    'type': 'prediction',
                    'severity': 0.7,
                    'description': 'Consistently high prediction errors',
                    'suggested_capability': 'causal_modeling',
                    'evidence': {'avg_prediction_error': avg_error}
                })
        
        # 3. Low belief stability → need stabilization mechanism
        if self.belief_stability < 0.3 and self.meta_confidence < 0.4:
            gaps.append({
                'type': 'stability',
                'severity': 0.6,
                'description': 'Unstable belief system',
                'suggested_capability': 'belief_consolidation',
                'evidence': {'stability': self.belief_stability, 'confidence': self.meta_confidence}
            })
        
        # 4. Detect oscillations in state history
        if len(self.state_history) > 30:
            recent_states = list(self.state_history)[-30:]
            if 'phi' in recent_states[-1]:
                phi_values = [s.get('phi', 0.5) for s in recent_states]
                variance = np.var(phi_values)
                if variance > 0.05:
                    gaps.append({
                        'type': 'integration',
                        'severity': 0.5,
                        'description': 'High variance in integration (phi)',
                        'suggested_capability': 'integration_stabilizer',
                        'evidence': {'phi_variance': variance}
                    })
        
        self.capability_gaps.extend(gaps)
        return gaps
    
    def generate_self_improvement_hypothesis(self, gap: Dict[str, Any]) -> Dict[str, Any]:
        """능력 갭을 기반으로 자기 개선 가설 생성"""
        hypothesis = {
            'id': f"hyp_{int(time.time() * 1000)}",
            'timestamp': time.time(),
            'gap': gap,
            'proposed_action': None,
            'expected_improvement': {},
            'test_duration': 50,  # steps
            'status': 'proposed'
        }
        
        capability = gap.get('suggested_capability', '')
        
        if capability == 'motor_planning':
            hypothesis['proposed_action'] = {
                'type': 'add_planning_layer',
                'details': 'Increase prediction_window and add action buffering'
            }
            hypothesis['expected_improvement'] = {
                'action_success_rate': 0.6,
                'meta_confidence': 0.6
            }
        
        elif capability == 'causal_modeling':
            hypothesis['proposed_action'] = {
                'type': 'enhance_prediction',
                'details': 'Extend state_history window and improve temporal modeling'
            }
            hypothesis['expected_improvement'] = {
                'prediction_error': 0.3,
                'belief_prediction': 0.7
            }
        
        elif capability == 'belief_consolidation':
            hypothesis['proposed_action'] = {
                'type': 'stabilize_beliefs',
                'details': 'Slow down belief updates, increase consolidation'
            }
            hypothesis['expected_improvement'] = {
                'belief_stability': 0.7,
                'meta_confidence': 0.6
            }
        
        elif capability == 'integration_stabilizer':
            hypothesis['proposed_action'] = {
                'type': 'dampen_oscillations',
                'details': 'Add smoothing to phi calculations'
            }
            hypothesis['expected_improvement'] = {
                'phi_variance': 0.02,
                'belief_stability': 0.6
            }
        
        self.self_improvement_hypotheses.append(hypothesis)
        return hypothesis
    
    def test_hypothesis(self, hypothesis: Dict[str, Any], test_env: Any) -> Dict[str, Any]:
        """가설을 실제로 테스트하고 결과 기록"""
        hypothesis_id = hypothesis['id']
        action = hypothesis['proposed_action']
        
        # Baseline measurement
        baseline_metrics = {
            'belief_stability': self.belief_stability,
            'meta_confidence': self.meta_confidence,
            'action_success': sum(list(self.action_outcomes)[-10:]) / max(1, min(10, len(self.action_outcomes))) if self.action_outcomes else 0.5,
            'prediction_error': sum(list(self.prediction_errors)[-10:]) / max(1, min(10, len(self.prediction_errors))) if self.prediction_errors else 0.5
        }
        
        # Apply proposed change temporarily
        old_state = self._save_state()
        self._apply_hypothesis_action(action)
        
        # Run test episodes
        test_metrics = []
        for _ in range(hypothesis.get('test_duration', 50)):
            # Simulate one step (would integrate with actual loop)
            current_metric = {
                'belief_stability': self.belief_stability,
                'meta_confidence': self.meta_confidence,
                'action_success': sum(list(self.action_outcomes)[-5:]) / max(1, min(5, len(self.action_outcomes))) if self.action_outcomes else 0.5,
                'prediction_error': sum(list(self.prediction_errors)[-5:]) / max(1, min(5, len(self.prediction_errors))) if self.prediction_errors else 0.5
            }
            test_metrics.append(current_metric)
        
        # Compute improvement
        avg_test_metrics = {
            key: np.mean([m[key] for m in test_metrics]) 
            for key in baseline_metrics.keys()
        }
        
        improvement = {
            key: avg_test_metrics[key] - baseline_metrics[key]
            for key in baseline_metrics.keys()
        }
        
        # Decision: adopt or revert
        expected = hypothesis['expected_improvement']
        success = True
        for key, target_val in expected.items():
            if key in avg_test_metrics:
                if key.endswith('_error') or key.endswith('_variance'):
                    # Lower is better
                    if avg_test_metrics[key] > target_val:
                        success = False
                else:
                    # Higher is better
                    if avg_test_metrics[key] < target_val:
                        success = False
        
        result = {
            'hypothesis_id': hypothesis_id,
            'action': action,
            'baseline': baseline_metrics,
            'test_results': avg_test_metrics,
            'improvement': improvement,
            'success': success,
            'timestamp': time.time()
        }
        
        if not success:
            # Revert changes
            self._restore_state(old_state)
            result['decision'] = 'rejected'
        else:
            # Keep changes
            result['decision'] = 'adopted'
            self.expansion_history.append({
                'type': 'hypothesis_adoption',
                'hypothesis': hypothesis,
                'result': result
            })
        
        self.hypothesis_test_results[hypothesis_id] = result
        return result
    
    def _save_state(self) -> Dict[str, Any]:
        """현재 상태 백업"""
        return {
            'belief_stability': self.belief_stability,
            'belief_adaptation': self.belief_adaptation,
            'belief_prediction': self.belief_prediction,
            'meta_awareness': self.meta_awareness,
            'meta_confidence': self.meta_confidence,
            'prediction_window': self.prediction_window,
            'max_safe_depth': self.max_safe_depth
        }
    
    def _restore_state(self, state: Dict[str, Any]):
        """백업된 상태로 복원"""
        self.belief_stability = state['belief_stability']
        self.belief_adaptation = state['belief_adaptation']
        self.belief_prediction = state['belief_prediction']
        self.meta_awareness = state['meta_awareness']
        self.meta_confidence = state['meta_confidence']
        self.prediction_window = state['prediction_window']
        self.max_safe_depth = state['max_safe_depth']
    
    def _apply_hypothesis_action(self, action: Dict[str, Any]):
        """가설의 제안된 변경사항 적용"""
        action_type = action.get('type', '')
        
        if action_type == 'add_planning_layer':
            self.prediction_window = min(10, self.prediction_window + 2)
            self.meta_confidence = min(1.0, self.meta_confidence + 0.1)
        
        elif action_type == 'enhance_prediction':
            if len(self.state_history) < 300:
                self.state_history = deque(self.state_history, maxlen=300)
            self.belief_prediction = min(1.0, self.belief_prediction + 0.1)
        
        elif action_type == 'stabilize_beliefs':
            self.belief_stability = min(1.0, self.belief_stability + 0.2)
            self.belief_adaptation = max(0.1, self.belief_adaptation - 0.1)
        
        elif action_type == 'dampen_oscillations':
            self.belief_stability = min(1.0, self.belief_stability + 0.15)
            self.max_safe_depth = min(12, self.max_safe_depth + 1)
    
    def compose_skills(self, skill_names: List[str]) -> Callable:
        """작은 스킬들을 조합하여 새로운 복합 스킬 생성"""
        def composed_skill(*args, **kwargs):
            results = []
            for skill_name in skill_names:
                if skill_name in self.modular_skills:
                    result = self.modular_skills[skill_name](*args, **kwargs)
                    results.append(result)
            return results
        
        return composed_skill
    
    def register_skill(self, name: str, skill_fn: Callable):
        """새로운 스킬을 라이브러리에 등록"""
        self.modular_skills[name] = skill_fn
        logging.info(f"Registered skill: {name}")
    
    def discover_skill_composition(self, task_context: Dict[str, Any]) -> Optional[str]:
        """현재 태스크에 유용한 스킬 조합 탐색"""
        if len(self.modular_skills) < 2:
            return None
        
        # Try random combinations
        skill_list = list(self.modular_skills.keys())
        for combo_size in range(2, min(4, len(skill_list) + 1)):
            for combo in combinations(skill_list, combo_size):
                # Test this composition
                composed = self.compose_skills(list(combo))
                score = self._evaluate_skill_composition(composed, task_context)
                
                if score > 0.7:
                    new_skill_name = f"composed_{'_'.join(combo)}"
                    self.register_skill(new_skill_name, composed)
                    self.skill_composition_history.append((list(combo), new_skill_name, score))
                    return new_skill_name
        
        return None
    
    def _evaluate_skill_composition(self, skill: Callable, context: Dict[str, Any]) -> float:
        """스킬 조합의 유용성 평가 (간단한 휴리스틱)"""
        # Placeholder: actual evaluation would run skill and measure performance
        return np.random.random()  # Replace with actual evaluation
    
    def autonomous_expansion_step(self, current_time: float, internal_state: Dict[str, Any]) -> Dict[str, Any]:
        """자율 확장 메인 루프 - 매 스텝마다 호출"""
        expansion_report = {
            'gaps_detected': [],
            'hypotheses_generated': [],
            'tests_performed': [],
            'expansions_applied': []
        }
        
        # 1. Check cooldown
        if current_time - self.last_expansion_time < self.expansion_cooldown:
            return expansion_report
        
        # 2. Detect capability gaps
        gaps = self.detect_capability_gaps(internal_state)
        expansion_report['gaps_detected'] = gaps
        
        # 3. Generate hypotheses for top gaps
        for gap in sorted(gaps, key=lambda g: g.get('severity', 0), reverse=True)[:2]:
            hypothesis = self.generate_self_improvement_hypothesis(gap)
            expansion_report['hypotheses_generated'].append(hypothesis)
        
        # 4. Test pending hypotheses (limit to 1 per step for stability)
        if len(self.self_improvement_hypotheses) > 0:
            pending = [h for h in self.self_improvement_hypotheses if h.get('status') == 'proposed']
            if pending:
                test_hyp = pending[0]
                test_hyp['status'] = 'testing'
                result = self.test_hypothesis(test_hyp, None)
                expansion_report['tests_performed'].append(result)
                
                if result['decision'] == 'adopted':
                    expansion_report['expansions_applied'].append(result)
                    self.last_expansion_time = current_time
        
        return expansion_report

@dataclass
class EpisodicMemoryTrace:
    timestamp: float
    experience_name: str
    qualia_vector: np.ndarray
    phi_value: float
    emotional_valence: float
    arousal: float
    context: Dict[str, Any]
    narrative: str = ""
    retrieval_count: int = 0
    consolidation_level: float = 0.0
    kind: str = "internal_state"
    content: str = ""
    embedding: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)

class EpisodicMemory:

    def __init__(self, max_memories: Optional[int]=None):
        if max_memories is None:
            available_mb = 0 // (1024 * 1024)
            if available_mb > 16000:
                max_memories = 5000
            elif available_mb > 8000:
                max_memories = 2000
            elif available_mb > 4000:
                max_memories = 1000
            else:
                max_memories = 500
        self.memories: List[EpisodicMemoryTrace] = []
        self.max_memories = max_memories
        self.temporal_index: List[float] = []
        self.semantic_index: Dict[str, List[int]] = {}
        self.emotional_index: Dict[str, List[int]] = {'positive': [], 'negative': [], 'neutral': []}
        self.total_encoded = 0
        self.total_retrieved = 0
        self.consolidation_cycles = 0

    def encode_experience(
        self,
        experience_name: str,
        qualia_vector: np.ndarray,
        phi_value: float,
        context: Dict[str, Any],
        narrative: str='',
        kind: str = "internal_state",
        content: str = "",
        embedding: Optional[np.ndarray] = None,
        tags: Optional[List[str]] = None,
    ):
        timestamp = time.time()
        emotional_valence = qualia_vector[1] - qualia_vector[0]
        arousal = (qualia_vector[0] + qualia_vector[3]) / 2
        emb = None
        if embedding is not None:
            try:
                emb = np.asarray(embedding, dtype=np.float32).ravel()
            except Exception:
                emb = None
        if not isinstance(context, dict):
            context = {'context': context}
        memory_trace = EpisodicMemoryTrace(
            timestamp=timestamp,
            experience_name=experience_name,
            qualia_vector=qualia_vector.copy(),
            phi_value=phi_value,
            emotional_valence=emotional_valence,
            arousal=arousal,
            context=context.copy(),
            narrative=narrative,
            retrieval_count=0,
            consolidation_level=0.1,
            kind=str(kind or "internal_state"),
            content=str(content or ""),
            embedding=emb,
            tags=list(tags) if tags else [],
        )
        self.memories.append(memory_trace)
        idx = len(self.memories) - 1
        self.temporal_index.append(timestamp)
        if experience_name not in self.semantic_index:
            self.semantic_index[experience_name] = []
        self.semantic_index[experience_name].append(idx)
        if emotional_valence > 0.2:
            self.emotional_index['positive'].append(idx)
        elif emotional_valence < -0.2:
            self.emotional_index['negative'].append(idx)
        else:
            self.emotional_index['neutral'].append(idx)
        self.total_encoded += 1
        if len(self.memories) > self.max_memories:
            self._prune_memories()

    def retrieve_by_time(self, time_window: Optional[Tuple[float, float]]=None, recent_n: Optional[int]=None) -> List[EpisodicMemoryTrace]:
        if recent_n is not None:
            results = self.memories[-recent_n:]
        elif time_window is not None:
            start, end = time_window
            results = [m for m in self.memories if start <= m.timestamp <= end]
        else:
            results = self.memories
        for mem in results:
            mem.retrieval_count += 1
        self.total_retrieved += len(results)
        return results

    def retrieve_by_semantic(self, experience_name: str, limit: int=10) -> List[EpisodicMemoryTrace]:
        if experience_name not in self.semantic_index:
            return []
        indices = self.semantic_index[experience_name][-limit:]
        results = [self.memories[i] for i in indices]
        for mem in results:
            mem.retrieval_count += 1
        self.total_retrieved += len(results)
        return results

    def retrieve_by_emotion(self, emotional_category: str, min_arousal: float=0.0, limit: int=10) -> List[EpisodicMemoryTrace]:
        if emotional_category not in self.emotional_index:
            return []
        indices = self.emotional_index[emotional_category]
        candidates = [self.memories[i] for i in indices]
        filtered = [m for m in candidates if m.arousal >= min_arousal]
        results = sorted(filtered, key=lambda m: m.timestamp, reverse=True)[:limit]
        for mem in results:
            mem.retrieval_count += 1
        self.total_retrieved += len(results)
        return results

    def retrieve_similar_qualia(self, target_qualia: np.ndarray, threshold: float=0.3, limit: int=5) -> List[EpisodicMemoryTrace]:
        similarities = []
        for i, mem in enumerate(self.memories):
            distance = np.linalg.norm(target_qualia - mem.qualia_vector)
            if distance < threshold:
                similarities.append((distance, i))
        similarities.sort()
        results = [self.memories[i] for _, i in similarities[:limit]]
        for mem in results:
            mem.retrieval_count += 1
        self.total_retrieved += len(results)
        return results

    def consolidate(self):
        self.consolidation_cycles += 1
        for mem in self.memories:
            retrieval_score = min(1.0, mem.retrieval_count / 10.0)
            phi_score = min(1.0, mem.phi_value / 3.0)
            arousal_score = mem.arousal
            emotion_score = abs(mem.emotional_valence)
            consolidation_boost = 0.3 * retrieval_score + 0.3 * phi_score + 0.2 * arousal_score + 0.2 * emotion_score
            mem.consolidation_level = min(1.0, mem.consolidation_level * 0.9 + consolidation_boost * 0.1)

    def construct_narrative(self, memory_window: int=20) -> str:
        recent = self.retrieve_by_time(recent_n=memory_window)
        if not recent:
            return 'memory=0'
        experience_counts = {}
        for mem in recent:
            exp = mem.experience_name
            experience_counts[exp] = experience_counts.get(exp, 0) + 1
        dominant_exp = max(experience_counts.items(), key=lambda x: x[1])
        avg_valence = np.mean([m.emotional_valence for m in recent])
        avg_phi = np.mean([m.phi_value for m in recent])
        return f'memories={self.total_encoded} recent={dominant_exp[0]}x{dominant_exp[1]} valence={avg_valence:+.2f} phi={avg_phi:.2f}'

    def _prune_memories(self):
        if len(self.memories) <= self.max_memories:
            return
        kind_priority = {
            'internal_state': 0,
            'dialog': 1,
            'research': 2,
            'knowledge': 3,
        }
        self.memories.sort(
            key=lambda m: (
                kind_priority.get(getattr(m, 'kind', 'internal_state'), 0),
                float(getattr(m, 'consolidation_level', 0.0)),
                int(getattr(m, 'retrieval_count', 0)),
                float(getattr(m, 'timestamp', 0.0)),
            )
        )
        while len(self.memories) > self.max_memories:
            self.memories.pop(0)
        self._rebuild_indices()

    def _rebuild_indices(self):
        self.temporal_index = []
        self.semantic_index = {}
        self.emotional_index = {'positive': [], 'negative': [], 'neutral': []}
        for i, mem in enumerate(self.memories):
            self.temporal_index.append(mem.timestamp)
            if mem.experience_name not in self.semantic_index:
                self.semantic_index[mem.experience_name] = []
            self.semantic_index[mem.experience_name].append(i)
            if mem.emotional_valence > 0.2:
                self.emotional_index['positive'].append(i)
            elif mem.emotional_valence < -0.2:
                self.emotional_index['negative'].append(i)
            else:
                self.emotional_index['neutral'].append(i)

    def get_statistics(self) -> Dict[str, Any]:
        return {'total_memories': len(self.memories), 'total_encoded': self.total_encoded, 'total_retrieved': self.total_retrieved, 'consolidation_cycles': self.consolidation_cycles, 'avg_consolidation': np.mean([m.consolidation_level for m in self.memories]) if self.memories else 0.0, 'avg_retrieval_count': np.mean([m.retrieval_count for m in self.memories]) if self.memories else 0.0, 'experience_types': len(self.semantic_index), 'emotional_distribution': {'positive': len(self.emotional_index['positive']), 'negative': len(self.emotional_index['negative']), 'neutral': len(self.emotional_index['neutral'])}}


    def save(self, path: str) -> None:
        try:
            meta = {
                'total_encoded': int(self.total_encoded),
                'total_retrieved': int(self.total_retrieved),
                'consolidation_cycles': int(self.consolidation_cycles),
                'max_memories': int(self.max_memories),
            }
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'_meta': meta}, ensure_ascii=False) + "\n")
                for mem in self.memories:
                    rec = {
                        'timestamp': float(mem.timestamp),
                        'experience_name': mem.experience_name,
                        'qualia_vector': np.asarray(mem.qualia_vector, dtype=np.float32).tolist(),
                        'phi_value': float(mem.phi_value),
                        'emotional_valence': float(mem.emotional_valence),
                        'arousal': float(mem.arousal),
                        'context': mem.context,
                        'narrative': mem.narrative,
                        'retrieval_count': int(getattr(mem, 'retrieval_count', 0)),
                        'consolidation_level': float(getattr(mem, 'consolidation_level', 0.0)),
                        'kind': str(getattr(mem, 'kind', 'internal_state')),
                        'content': str(getattr(mem, 'content', '')),
                        'embedding': np.asarray(getattr(mem, 'embedding', None), dtype=np.float32).tolist() if getattr(mem, 'embedding', None) is not None else None,
                        'tags': list(getattr(mem, 'tags', []) or []),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def load(self, path: str) -> None:
        try:
            if not os.path.exists(path):
                return
            self.memories = []
            self.temporal_index = []
            self.semantic_index = {}
            self.emotional_index = {'positive': [], 'negative': [], 'neutral': []}
            meta = None
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and '_meta' in obj:
                        meta = obj.get('_meta', {})
                        continue
                    try:
                        qv = np.asarray(obj.get('qualia_vector', []), dtype=np.float32)
                        emb = None
                        emb_obj = obj.get('embedding', None)
                        if isinstance(emb_obj, (list, tuple)):
                            try:
                                emb = np.asarray(emb_obj, dtype=np.float32).ravel()
                            except Exception:
                                emb = None
                        tags_obj = obj.get('tags', [])
                        if isinstance(tags_obj, list):
                            tags_val = [str(t) for t in tags_obj]
                        elif tags_obj:
                            tags_val = [str(tags_obj)]
                        else:
                            tags_val = []
                        mem = EpisodicMemoryTrace(
                            timestamp=float(obj.get('timestamp', time.time())),
                            experience_name=str(obj.get('experience_name', 'unknown')),
                            qualia_vector=qv,
                            phi_value=float(obj.get('phi_value', 0.0)),
                            emotional_valence=float(obj.get('emotional_valence', 0.0)),
                            arousal=float(obj.get('arousal', 0.0)),
                            context=obj.get('context', {}) if isinstance(obj.get('context', {}), dict) else {'context': obj.get('context', {})},
                            narrative=str(obj.get('narrative', '')),
                            retrieval_count=int(obj.get('retrieval_count', 0)),
                            consolidation_level=float(obj.get('consolidation_level', 0.0)),
                            kind=str(obj.get('kind', 'internal_state')),
                            content=str(obj.get('content', '')),
                            embedding=emb,
                            tags=tags_val,
                        )
                        self.memories.append(mem)
                    except Exception:
                        continue
            # restore meta
            if meta:
                try:
                    self.total_encoded = int(meta.get('total_encoded', len(self.memories)))
                    self.total_retrieved = int(meta.get('total_retrieved', 0))
                    self.consolidation_cycles = int(meta.get('consolidation_cycles', 0))
                    self.max_memories = int(meta.get('max_memories', self.max_memories))
                except Exception:
                    pass
            else:
                self.total_encoded = len(self.memories)
            self._rebuild_indices()
        except Exception:
            pass

import copy
import hashlib
import copy
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
DIM_INCREASE = 'dimension_increase'
DIM_DECREASE = 'dimension_decrease'
SPARSIFY = 'sparsify'
DENSIFY = 'densify'
PROJECT = 'project'
COMPOSE = 'compose'
RELINK = 'relink'
PRUNE = 'prune'


class StructuralOperator(Enum):
    """Enumeration of high-level structural operators used by the structural
    revision engine. Values are string tokens matched to existing constant names
    elsewhere in the file to preserve backward compatibility.
    """
    DIM_INCREASE = DIM_INCREASE
    DIM_DECREASE = DIM_DECREASE
    SPARSIFY = SPARSIFY
    DENSIFY = DENSIFY
    PROJECT = PROJECT
    COMPOSE = COMPOSE
    RELINK = RELINK
    PRUNE = PRUNE

@dataclass
class StructuralVersion:
    version_id: str
    timestamp: float
    parent_version: Optional[str]
    operator_applied: StructuralOperator
    parameters: Dict[str, Any]
    structure_snapshot: Dict[str, Any]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    hypothesis: str = ''
    experiment_id: Optional[str] = None
    approved: bool = False

    def compute_diff(self) -> Dict[str, Any]:
        return {'operator': self.operator_applied.value, 'params': self.parameters, 'performance_delta': {k: self.performance_after.get(k, 0) - self.performance_before.get(k, 0) for k in set(self.performance_before.keys()) | set(self.performance_after.keys())}}

@dataclass
class ErrorProfile:
    oscillatory: float = 0.0
    divergent: float = 0.0
    stagnant: float = 0.0
    inconsistent: float = 0.0
    error_list: List[float] = field(default_factory=list)

    @property
    def total_errors(self) -> int:
        return len(self.error_list)

    def add_error(self, error_value: float):
        self.error_list.append(error_value)
        if len(self.error_list) >= 10:
            recent = self.error_list[-10:]
            sign_changes = sum((1 for i in range(1, len(recent)) if (recent[i] > 0) != (recent[i - 1] > 0)))
            self.oscillatory = float(min(1.0, sign_changes / 5.0))
            if all((recent[i] >= recent[i - 1] for i in range(1, len(recent)))):
                self.divergent = float(min(1.0, recent[-1] / max(recent[0], 1e-10)))
            variance = np.var(recent)
            self.stagnant = float(1.0 - min(1.0, variance * 10))
            std_dev = np.std(recent)
            self.inconsistent = float(min(1.0, std_dev / (np.mean(np.abs(recent)) + 1e-10)))

    def dominant_pattern(self) -> str:
        patterns = {'oscillatory': self.oscillatory, 'divergent': self.divergent, 'stagnant': self.stagnant, 'inconsistent': self.inconsistent}
        return max(patterns, key=lambda k: patterns[k])

class StructuralOperatorEngine:

    def __init__(self):
        self.version_history: List[StructuralVersion] = []
        self.current_version_id: str = self._generate_version_id('initial')
        self.rollback_stack: List[str] = []
        self.operator_policy = {'oscillatory': [(StructuralOperator.SPARSIFY, {'sparsity': 0.3}), (StructuralOperator.DIM_DECREASE, {'reduction_factor': 0.8})], 'divergent': [(StructuralOperator.PRUNE, {'threshold': 0.1}), (StructuralOperator.PROJECT, {'target_dim': 0.7})], 'stagnant': [(StructuralOperator.DENSIFY, {'density_boost': 0.2}), (StructuralOperator.RELINK, {'reconnection_prob': 0.3})], 'inconsistent': [(StructuralOperator.DIM_INCREASE, {'new_dims': 2}), (StructuralOperator.COMPOSE, {'layer_count': 1})]}

    def _generate_version_id(self, prefix: str='v') -> str:
        timestamp = time.time()
        hash_input = f'{prefix}_{timestamp}_{np.random.randint(10000)}'
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    def analyze_error_profile(self, error_history: deque) -> ErrorProfile:
        if len(error_history) < 20:
            return ErrorProfile()
        errors = np.array(list(error_history)[-50:])
        fft = np.fft.fft(errors)
        power = np.abs(fft) ** 2
        oscillatory_score = np.max(power[1:len(power) // 2]) / (np.mean(power) + 1e-10)
        oscillatory_score = min(1.0, oscillatory_score / 10.0)
        if len(errors) > 10:
            recent_trend = np.polyfit(range(len(errors)), errors, 1)[0]
            divergent_score = max(0.0, min(1.0, recent_trend * 10))
        else:
            divergent_score = 0.0
        error_variance = np.var(errors)
        stagnant_score = 1.0 - min(1.0, error_variance * 5)
        if len(errors) > 5:
            diffs = np.abs(np.diff(errors))
            inconsistent_score = min(1.0, np.std(diffs) * 3)
        else:
            inconsistent_score = 0.0
        return ErrorProfile(oscillatory=float(oscillatory_score), divergent=float(divergent_score), stagnant=float(stagnant_score), inconsistent=float(inconsistent_score))

    def select_operator_sequence(self, error_profile: ErrorProfile) -> List[Tuple[StructuralOperator, Dict]]:
        dominant = error_profile.dominant_pattern()
        if dominant in self.operator_policy:
            return self.operator_policy[dominant].copy()
        else:
            return [(StructuralOperator.SPARSIFY, {'sparsity': 0.2})]

    def apply_operator(self, operator: StructuralOperator, parameters: Dict[str, Any], current_structure: Dict[str, Any], performance_before: Dict[str, float], hypothesis: str='') -> Tuple[Dict[str, Any], StructuralVersion]:
        structure_copy = copy.deepcopy(current_structure)
        if operator == StructuralOperator.DIM_INCREASE:
            modified = self._op_dim_increase(structure_copy, parameters)
        elif operator == StructuralOperator.DIM_DECREASE:
            modified = self._op_dim_decrease(structure_copy, parameters)
        elif operator == StructuralOperator.SPARSIFY:
            modified = self._op_sparsify(structure_copy, parameters)
        elif operator == StructuralOperator.DENSIFY:
            modified = self._op_densify(structure_copy, parameters)
        elif operator == StructuralOperator.PROJECT:
            modified = self._op_project(structure_copy, parameters)
        elif operator == StructuralOperator.COMPOSE:
            modified = self._op_compose(structure_copy, parameters)
        elif operator == StructuralOperator.RELINK:
            modified = self._op_relink(structure_copy, parameters)
        elif operator == StructuralOperator.PRUNE:
            modified = self._op_prune(structure_copy, parameters)
        else:
            modified = structure_copy
        version = StructuralVersion(version_id=self._generate_version_id(), timestamp=time.time(), parent_version=self.current_version_id, operator_applied=operator, parameters=parameters, structure_snapshot=modified, performance_before=performance_before, performance_after={}, hypothesis=hypothesis)
        self.version_history.append(version)
        self.rollback_stack.append(self.current_version_id)
        self.current_version_id = version.version_id
        return (modified, version)

    def _op_dim_increase(self, structure: Dict, params: Dict) -> Dict:
        new_dims = params.get('new_dims', 1)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            old_shape = structure['weights'].shape
            if len(old_shape) == 2:
                n, k = old_shape
                new_weights = np.random.randn(n, k + new_dims) * np.sqrt(2.0 / (n + k + new_dims))
                new_weights[:, :k] = structure['weights']
                structure['weights'] = new_weights
        return structure

    def _op_dim_decrease(self, structure: Dict, params: Dict) -> Dict:
        reduction_factor = params.get('reduction_factor', 0.8)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            old_shape = structure['weights'].shape
            if len(old_shape) == 2:
                n, k = old_shape
                new_k = max(1, int(k * reduction_factor))
                U, S, Vt = np.linalg.svd(structure['weights'], full_matrices=False)
                structure['weights'] = U[:, :new_k] @ np.diag(S[:new_k]) @ Vt[:new_k, :]
        return structure

    def _op_sparsify(self, structure: Dict, params: Dict) -> Dict:
        sparsity = params.get('sparsity', 0.3)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            threshold = np.percentile(np.abs(structure['weights']), sparsity * 100)
            structure['weights'][np.abs(structure['weights']) < threshold] = 0.0
        if 'connections' in structure and isinstance(structure['connections'], list):
            for conn in structure['connections']:
                if 'strength' in conn and conn['strength'] < sparsity:
                    conn['strength'] = 0.0
        return structure

    def _op_densify(self, structure: Dict, params: Dict) -> Dict:
        density_boost = params.get('density_boost', 0.2)
        if 'connections' in structure and isinstance(structure['connections'], list):
            for conn in structure['connections']:
                if 'strength' in conn and conn['strength'] == 0.0:
                    if np.random.random() < density_boost:
                        conn['strength'] = np.random.uniform(0.1, 0.5)
        return structure

    def _op_project(self, structure: Dict, params: Dict) -> Dict:
        target_dim = params.get('target_dim', 0.7)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            old_shape = structure['weights'].shape
            if len(old_shape) == 2:
                n, k = old_shape
                new_k = max(2, int(k * target_dim))
                mean = np.mean(structure['weights'], axis=0)
                centered = structure['weights'] - mean
                U, S, Vt = np.linalg.svd(centered, full_matrices=False)
                projected = U[:, :new_k] @ np.diag(S[:new_k]) @ Vt[:new_k, :]
                structure['weights'] = projected + mean
        return structure

    def _op_compose(self, structure: Dict, params: Dict) -> Dict:
        layer_count = params.get('layer_count', 1)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            original_weights = structure['weights']
            input_dim, output_dim = original_weights.shape
            hidden_dim = params.get('hidden_dim', max(input_dim, output_dim) // 2)
            hidden_dim = max(2, hidden_dim)
            if layer_count == 1:
                W1 = np.random.normal(0, 0.1, (input_dim, hidden_dim))
                W2 = np.random.normal(0, 0.1, (hidden_dim, output_dim))
                try:
                    U, S, Vt = np.linalg.svd(original_weights, full_matrices=False)
                    k = min(hidden_dim, len(S))
                    W1 = U[:, :k] @ np.diag(np.sqrt(S[:k]))
                    W2 = np.diag(np.sqrt(S[:k])) @ Vt[:k, :]
                    if k < hidden_dim:
                        W1_extra = np.random.normal(0, 0.01, (input_dim, hidden_dim - k))
                        W1 = np.hstack([W1, W1_extra])
                        W2_extra = np.random.normal(0, 0.01, (hidden_dim - k, output_dim))
                        W2 = np.vstack([W2, W2_extra])
                except np.linalg.LinAlgError:
                    pass
                structure['weights'] = [W1, W2]
                structure['biases'] = [np.zeros(hidden_dim), np.zeros(output_dim)]
            else:
                layers = []
                biases = []
                dims = [input_dim]
                for i in range(layer_count):
                    next_dim = max(2, int(hidden_dim * 0.8 ** i))
                    dims.append(next_dim)
                dims.append(output_dim)
                for i in range(len(dims) - 1):
                    W = np.random.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                    layers.append(W)
                    biases.append(np.zeros(dims[i + 1]))
                structure['weights'] = layers
                structure['biases'] = biases
            structure['layer_count'] = structure.get('layer_count', 1) + layer_count
            structure['layer_dims'] = [W.shape for W in structure['weights']]
            structure['activation'] = params.get('activation', 'relu')
            structure['batch_norm'] = params.get('batch_norm', False)
            structure['dropout_rate'] = params.get('dropout_rate', 0.0)
        return structure

    def _op_relink(self, structure: Dict, params: Dict) -> Dict:
        reconnection_prob = params.get('reconnection_prob', 0.3)
        if 'connections' in structure and isinstance(structure['connections'], list):
            for conn in structure['connections']:
                if np.random.random() < reconnection_prob:
                    max_idx = max(conn.get('from', 0), conn.get('to', 0), 10)
                    conn['from'] = np.random.randint(0, max_idx)
                    conn['to'] = np.random.randint(0, max_idx)
        return structure

    def _op_prune(self, structure: Dict, params: Dict) -> Dict:
        threshold = params.get('threshold', 0.1)
        if 'weights' in structure and isinstance(structure['weights'], np.ndarray):
            mask = np.abs(structure['weights']) > threshold
            structure['weights'] = structure['weights'] * mask
        if 'neurons' in structure and isinstance(structure['neurons'], list):
            structure['neurons'] = [n for n in structure['neurons'] if n.get('activation_count', 0) > threshold * 100]
        return structure

    def rollback(self) -> bool:
        if not self.rollback_stack:
            return False
        self.current_version_id = self.rollback_stack.pop()
        for version in reversed(self.version_history):
            if version.version_id == self.current_version_id:
                print(f'Rollback to version {self.current_version_id[:8]}')
                return True
        return False

    def get_current_structure(self) -> Optional[Dict]:
        for version in reversed(self.version_history):
            if version.version_id == self.current_version_id:
                return version.structure_snapshot
        return None

@dataclass
class Hypothesis:
    id: str
    description: str
    manipulation_vars: List[str]
    dependent_vars: List[str]
    predicted_effect: str
    created_at: float = field(default_factory=time.time)
    tested: bool = False
    accepted: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0

class AutonomousExperimentDesigner:

    def __init__(self, operator_engine: StructuralOperatorEngine):
        self.operator_engine = operator_engine
        self.hypotheses: List[Hypothesis] = []
        self.experiment_history: List[Dict] = []
        self.significance_level = 0.05
        self.min_effect_size = 0.2
        # Increase samples/trials for better power
        self.bootstrap_samples = 2000
        self.control_trials = 20
        self.treatment_trials = 20
        # metric weights for acceptance (reward_mean upweighted by default)
        self.metric_weights: Dict[str, float] = {
            'reward_mean': 1.0,
            'td_error': 0.8,
            'phi_median': 0.6,
            'stability': 0.5,
        }
        
        # === SELF-EXPERIMENTATION ===
        self.self_hypotheses: deque = deque(maxlen=256)
        self.self_experiment_results: Dict[str, Dict[str, Any]] = {}

    def generate_hypothesis(self, error_profile: ErrorProfile, current_performance: Dict) -> Hypothesis:
        dominant = error_profile.dominant_pattern()
        if dominant == 'oscillatory':
            hyp = Hypothesis(id=f'H{len(self.hypotheses) + 1}', description='Reduce oscillation by 20% via sparsity/dimension adjustment', manipulation_vars=['sparsity', 'dimension'], dependent_vars=['error_variance', 'oscillation_amplitude'], predicted_effect='decrease')
            hyp = Hypothesis(id=f'H{len(self.hypotheses) + 1}', description='Divergence detected; increase pruning threshold', manipulation_vars=['pruning_threshold'], dependent_vars=['error_trend', 'stability'], predicted_effect='stabilize')
            hyp = Hypothesis(id=f'H{len(self.hypotheses) + 1}', description='Stagnation detected; adjust density/reconnection to boost exploration', manipulation_vars=['density', 'reconnection_rate'], dependent_vars=['exploration_score', 'novelty'], predicted_effect='increase')
        self.hypotheses.append(hyp)
        return hyp

    def design_experiment(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        operator_sequence = []
        for var in hypothesis.manipulation_vars:
            if 'sparsity' in var or 'sparse' in var:
                operator_sequence.append((StructuralOperator.SPARSIFY, {'sparsity': 0.3}))
            elif 'dimension' in var and hypothesis.predicted_effect == 'decrease':
                operator_sequence.append((StructuralOperator.DIM_DECREASE, {'reduction_factor': 0.75}))
            elif 'dimension' in var and hypothesis.predicted_effect == 'increase':
                operator_sequence.append((StructuralOperator.DIM_INCREASE, {'new_dims': 3}))
            elif 'prun' in var:
                operator_sequence.append((StructuralOperator.PRUNE, {'threshold': 0.20}))
            elif 'density' in var or 'densif' in var:
                operator_sequence.append((StructuralOperator.DENSIFY, {'density_boost': 0.40}))
            elif 'reconnect' in var or 'relink' in var:
                operator_sequence.append((StructuralOperator.RELINK, {'reconnection_prob': 0.40}))
        experiment_design = {
            'hypothesis_id': hypothesis.id,
            'control_group': {'description': 'control baseline', 'operators': [], 'trials': self.control_trials},
            'treatment_group': {'description': f'{len(operator_sequence)} operators applied', 'operators': operator_sequence, 'trials': self.treatment_trials},
            'measurements': hypothesis.dependent_vars,
            'protocol': 'bootstrap_ci'
        }
        try:
            base_metrics = ['reward_mean', 'td_error', 'phi_median', 'stability']
            cur = list(experiment_design.get('measurements', []))
            experiment_design['measurements'] = list(set(cur) | set(base_metrics))
        except Exception:
            pass
        return experiment_design

    def execute_experiment(self, experiment_design: Dict, performance_evaluator: Callable, structure: Dict) -> Dict[str, Any]:
        # Try to decorrelate trials via lightweight reseeding of core RNG
        core = getattr(performance_evaluator, '__self__', None)
        orig_rng = getattr(core, 'rng', None) if core is not None else None
        control_results = []
        for trial in range(experiment_design['control_group']['trials']):
            try:
                if core is not None and hasattr(core, 'rng'):
                    core.rng = np.random.default_rng((hash(experiment_design['hypothesis_id']) + trial) & 0xFFFFFFFF)
            except Exception:
                pass
            metrics = performance_evaluator(structure)
            control_results.append(metrics)
        treatment_results = []
        for trial in range(experiment_design['treatment_group']['trials']):
            modified_structure = copy.deepcopy(structure)
            for operator, params in experiment_design['treatment_group']['operators']:
                modified_structure, _ = self.operator_engine.apply_operator(operator, params, modified_structure, performance_before={}, hypothesis=experiment_design['hypothesis_id'])
            try:
                if core is not None and hasattr(core, 'rng'):
                    core.rng = np.random.default_rng((hash(experiment_design['hypothesis_id']) + 100000 + trial) & 0xFFFFFFFF)
            except Exception:
                pass
            metrics = performance_evaluator(modified_structure)
            treatment_results.append(metrics)
        # restore RNG
        try:
            if core is not None and (orig_rng is not None):
                core.rng = orig_rng
        except Exception:
            pass
        statistics = self._statistical_analysis(control_results, treatment_results, experiment_design['measurements'])
        results = {'hypothesis_id': experiment_design['hypothesis_id'], 'control_results': control_results, 'treatment_results': treatment_results, 'statistics': statistics, 'operators_applied': experiment_design['treatment_group']['operators']}
        self.experiment_history.append(results)
        return results

    def _statistical_analysis(self, control: List[Dict], treatment: List[Dict], metrics: List[str]) -> Dict[str, Any]:
        analysis = {}
        for metric in metrics:
            control_vals = [c.get(metric, 0.0) for c in control]
            treatment_vals = [t.get(metric, 0.0) for t in treatment]
            if not control_vals or not treatment_vals:
                continue
            mean_diff = np.mean(treatment_vals) - np.mean(control_vals)
            pooled_std = np.sqrt((np.var(control_vals) + np.var(treatment_vals)) / 2)
            effect_size = mean_diff / (pooled_std + 1e-10)
            p_value = self._bootstrap_test(control_vals, treatment_vals)
            ci_lower, ci_upper = self._bootstrap_ci(treatment_vals, confidence=0.95)
            analysis[metric] = {'control_mean': np.mean(control_vals), 'treatment_mean': np.mean(treatment_vals), 'effect_size': effect_size, 'p_value': p_value, 'ci_95': (ci_lower, ci_upper), 'significant': p_value < self.significance_level}

        # Composite score analysis (principled acceptance criterion)
        def _score(d: Dict[str, float]) -> float:
            return (
                1.0 * float(d.get('reward_mean', 0.0))
                - 0.8 * float(d.get('td_error', 0.0))
                + 0.6 * float(d.get('phi_median', 0.0))
                + 0.5 * float(d.get('stability', 0.0))
            )
        s_ctrl = [_score(c) for c in control]
        s_treat = [_score(t) for t in treatment]
        if s_ctrl and s_treat:
            obs_diff = float(np.mean(s_treat) - np.mean(s_ctrl))
            diffs = []
            for _ in range(self.bootstrap_samples):
                bc = np.random.choice(s_ctrl, size=len(s_ctrl), replace=True)
                bt = np.random.choice(s_treat, size=len(s_treat), replace=True)
                diffs.append(float(np.mean(bt) - np.mean(bc)))
            lb = float(np.percentile(diffs, 2.5))
            ub = float(np.percentile(diffs, 97.5))
            analysis['composite'] = {'obs_diff': obs_diff, 'ci_95': (lb, ub), 'improved': lb > 0.0}
        return analysis

    def _bootstrap_test(self, control: List[float], treatment: List[float]) -> float:
        observed_diff = np.mean(treatment) - np.mean(control)
        combined = control + treatment
        n_control = len(control)
        bootstrap_diffs = []
        for _ in range(self.bootstrap_samples):
            resampled = np.random.choice(combined, size=len(combined), replace=True)
            boot_control = resampled[:n_control]
            boot_treatment = resampled[n_control:]
            boot_diff = np.mean(boot_treatment) - np.mean(boot_control)
            bootstrap_diffs.append(boot_diff)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        return p_value

    def _bootstrap_ci(self, data: List[float], confidence: float=0.95) -> Tuple[float, float]:
        bootstrap_means = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
        return (float(lower), float(upper))

    def evaluate_hypothesis(self, experiment_results: Dict) -> bool:
        hypothesis_id = experiment_results['hypothesis_id']
        statistics = experiment_results['statistics']
        hypothesis = None
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                hypothesis = h
                break
        if hypothesis is None:
            return False
        # 1) Primary criterion: composite CI lower bound > 0
        comp = statistics.get('composite') if isinstance(statistics, dict) else None
        if comp and comp.get('improved'):
            accepted = True
            avg_effect_size = float('nan')
            avg_p_value = float('nan')
            hypothesis.tested = True
            hypothesis.accepted = True
            hypothesis.p_value = avg_p_value
            hypothesis.effect_size = avg_effect_size
            print(f"Hypothesis {hypothesis_id} ACCEPTED")
            lb, ub = comp['ci_95']
            print(f"  Composite ΔS CI95: [{lb:.4f}, {ub:.4f}]")
            return True

        # 2) Secondary (legacy) weighted significance rule
        total_w = 0.0
        sig_w = 0.0
        eff_w = 0.0
        p_w = 0.0
        for metric, stats in statistics.items():
            if metric == 'composite':
                continue
            w = float(self.metric_weights.get(metric, 1.0))
            total_w += w
            if stats['significant']:
                sig_w += w
            eff_w += w * abs(stats['effect_size'])
            p_w += w * stats['p_value']
        avg_effect_size = eff_w / total_w if total_w > 0 else 0.0
        avg_p_value = p_w / total_w if total_w > 0 else 1.0
        accepted = (sig_w >= 0.5 * total_w) and (avg_effect_size >= self.min_effect_size)
        hypothesis.tested = True
        hypothesis.accepted = accepted
        hypothesis.p_value = avg_p_value
        hypothesis.effect_size = avg_effect_size
        if accepted:
            print(f'Hypothesis {hypothesis_id} ACCEPTED')
            print(f'  Effect size: {avg_effect_size:.3f}, p-value: {avg_p_value:.4f}')
        else:
            print(f'Hypothesis {hypothesis_id} REJECTED')
            print(f'  Effect size: {avg_effect_size:.3f}, p-value: {avg_p_value:.4f}')
        return accepted

    def apply_if_accepted(self, experiment_results: Dict, structure: Dict) -> Tuple[Dict, bool]:
        accepted = self.evaluate_hypothesis(experiment_results)
        if accepted:
            modified_structure = copy.deepcopy(structure)
            for operator, params in experiment_results['operators_applied']:
                modified_structure, version = self.operator_engine.apply_operator(operator, params, modified_structure, performance_before=experiment_results['control_results'][0], hypothesis=experiment_results['hypothesis_id'])
                version.approved = True
                version.performance_after = experiment_results['treatment_results'][0]
            print(f'Structural changes APPLIED (experiment passed)')
            return (modified_structure, True)
        else:
            self.operator_engine.rollback()
            print(f'Structural changes ROLLED BACK (experiment failed)')
            return (structure, False)
    
    # === SELF-EXPERIMENTATION METHODS ===
    
    def generate_self_hypothesis(self, core: Any) -> Dict[str, Any]:
        """시스템 자신에 대한 가설 생성"""
        hypothesis = {
            'id': f"self_hyp_{int(time.time() * 1000)}",
            'timestamp': time.time(),
            'target': 'self',
            'intervention': None,
            'expected_outcome': {},
            'rationale': ''
        }
        
        # Analyze current state and generate intervention
        try:
            current_phi = core.phi_calculator.phi_history[-1] if core.phi_calculator.phi_history else 0.5
            current_meta = core.self_model.meta_awareness
            
            # Simple hypothesis: if meta-awareness high but phi low, boost integration
            if current_meta > 0.7 and current_phi < 0.4:
                hypothesis['intervention'] = {
                    'type': 'increase_integration',
                    'params': {'boost': 0.2}
                }
                hypothesis['expected_outcome'] = {'phi': 0.6}
                hypothesis['rationale'] = 'High awareness needs better integration'
        
        except Exception as e:
            logging.debug(f"Self-hypothesis generation error: {e}")
        
        if hypothesis['intervention']:
            self.self_hypotheses.append(hypothesis)
        
        return hypothesis

from typing import Optional, List, Dict, Any, Tuple
import time

class PlanningMode(Enum):
    EXPLOITATION = 'exploitation'
    EXPLORATION = 'exploration'
    RECOVERY = 'recovery'
    OPTIMIZATION = 'optimization'

class GoalStatus(Enum):
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    ABANDONED = 'abandoned'

@dataclass
class SubGoal:
    id: str
    description: str
    parent_goal_id: str
    experiments: List[str] = field(default_factory=list)
    revisions: List[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    target_metrics: Dict[str, float] = field(default_factory=dict)
    current_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    deadline: Optional[float] = None
    evaluation_checkpoints: List[float] = field(default_factory=list)
    last_evaluation: Optional[float] = None
    evaluation_results: List[Dict] = field(default_factory=list)

    def is_completed(self) -> bool:
        if self.status == GoalStatus.COMPLETED:
            return True
        if not self.target_metrics:
            return False
        achieved = 0
        for key, target in self.target_metrics.items():
            current = self.current_metrics.get(key, 0.0)
            if current >= target * 0.9:
                achieved += 1
        achievement_rate = achieved / len(self.target_metrics)
        return achievement_rate >= 0.8

    def should_evaluate(self, current_time: float) -> bool:
        if not self.evaluation_checkpoints:
            return False
        if self.last_evaluation is None:
            return True
        for checkpoint in self.evaluation_checkpoints:
            if self.last_evaluation < checkpoint <= current_time:
                return True
        return False

@dataclass
class LongTermGoal:
    id: str
    description: str
    mode: PlanningMode
    subgoals: List[SubGoal] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    overall_progress: float = 0.0
    success_criteria: Dict[str, float] = field(default_factory=dict)
    failure_threshold: int = 5
    consecutive_failures: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    max_duration: Optional[float] = None
    evaluation_history: List[Dict] = field(default_factory=list)

    def update_progress(self, current_performance: Optional[Dict[str, float]]=None):
        if self.subgoals:
            total_progress = sum((sg.progress for sg in self.subgoals))
            self.overall_progress = total_progress / len(self.subgoals)
        elif current_performance and self.success_criteria:
            progress_scores = []
            for criterion, target in self.success_criteria.items():
                current_value = current_performance.get(criterion, 0.0)
                if target > 0:
                    progress = min(1.0, current_value / target)
                    progress_scores.append(progress)
            if progress_scores:
                self.overall_progress = float(np.mean(progress_scores))
            else:
                self.overall_progress = 0.0
        elif self.started_at and self.max_duration:
            elapsed = time.time() - self.started_at
            time_progress = min(1.0, elapsed / self.max_duration)
            self.overall_progress = time_progress * 0.5
        else:
            self.overall_progress = 0.0

    def should_redefine(self) -> bool:
        if self.consecutive_failures >= self.failure_threshold:
            return True
        if self.max_duration and self.started_at:
            elapsed = time.time() - self.started_at
            if elapsed > self.max_duration:
                return True
        if len(self.subgoals) > 0:
            failed_count = sum((1 for sg in self.subgoals if sg.status == GoalStatus.FAILED))
            failure_rate = failed_count / len(self.subgoals)
            if failure_rate > 0.5:
                return True
        return False

    def should_switch_to_exploration(self) -> bool:
        if self.mode == PlanningMode.EXPLOITATION:
            if self.consecutive_failures >= 3:
                return True
        if self.mode == PlanningMode.OPTIMIZATION:
            if len(self.evaluation_history) >= 5:
                recent = self.evaluation_history[-5:]
                progress_changes = [e.get('progress_delta', 0) for e in recent]
                avg_change = np.mean(progress_changes)
                if avg_change < 0.02:
                    return True
        return False

    def is_completed(self, current_performance: Optional[Dict[str, float]]=None) -> bool:
        if self.status == GoalStatus.COMPLETED:
            return True
        if self.overall_progress >= 0.99:
            return True
        if self.subgoals:
            all_completed = all((sg.status == GoalStatus.COMPLETED for sg in self.subgoals))
            if all_completed:
                return True
        if current_performance and self.success_criteria:
            met_criteria = 0
            for metric, target_value in self.success_criteria.items():
                current_value = current_performance.get(metric, 0.0)
                if current_value >= target_value:
                    met_criteria += 1
            if met_criteria >= len(self.success_criteria):
                return True
        return False

class LongTermPlanner:

    def __init__(self, experiment_designer: AutonomousExperimentDesigner, operator_engine: StructuralOperatorEngine):
        self.experiment_designer = experiment_designer
        self.operator_engine = operator_engine
        self.current_goal: Optional[LongTermGoal] = None
        self.goal_history: List[LongTermGoal] = []
        self.experiment_queue: deque = deque()
        self.revision_queue: deque = deque()
        self.total_goals_completed = 0
        self.total_goals_failed = 0
        self.total_goals_redefined = 0
        self.mode_switches = 0

    def create_goal(self, description: str, mode: PlanningMode, success_criteria: Dict[str, float], max_duration: Optional[float]=None) -> LongTermGoal:
        goal = LongTermGoal(id=f'GOAL_{len(self.goal_history) + 1}', description=description, mode=mode, success_criteria=success_criteria, max_duration=max_duration)
        self.current_goal = goal
        self.goal_history.append(goal)
        self._last_goal_created = {'id': goal.id, 'description': description, 'mode': mode.value, 'timestamp': time.time()}
        return goal

    def decompose_goal(self, goal: LongTermGoal, error_profile: ErrorProfile, current_performance: Dict[str, float]) -> List[SubGoal]:
        subgoals = []
        improvement_areas = {}
        for metric, target in goal.success_criteria.items():
            current = current_performance.get(metric, 0.0)
            gap = target - current
            if gap > 0:
                improvement_areas[metric] = gap
        sorted_areas = sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)
        for idx, (metric, gap) in enumerate(sorted_areas[:5]):
            subgoal_desc = f'Improve {metric} by {gap:.2f}'
            subgoal = SubGoal(id=f'{goal.id}_SG{idx + 1}', description=subgoal_desc, parent_goal_id=goal.id, target_metrics={metric: goal.success_criteria[metric]}, current_metrics={metric: current_performance.get(metric, 0.0)})
            if goal.max_duration:
                duration_per_subgoal = goal.max_duration / len(sorted_areas[:5])
                checkpoints = [time.time() + duration_per_subgoal * 0.33, time.time() + duration_per_subgoal * 0.67]
                subgoal.evaluation_checkpoints = checkpoints
            subgoals.append(subgoal)
        for subgoal in subgoals:
            self._schedule_actions(subgoal, error_profile, goal.mode)
        goal.subgoals = subgoals
        print(f'Decomposed into {len(subgoals)} subgoals')
        return subgoals

    def _schedule_actions(self, subgoal: SubGoal, error_profile: ErrorProfile, mode: PlanningMode):
        if mode == PlanningMode.EXPLOITATION:
            num_experiments = 2
            subgoal.experiments = [f'EXP_{subgoal.id}_{i}' for i in range(num_experiments)]
            subgoal.revisions = ['REV_conservative']
        elif mode == PlanningMode.EXPLORATION:
            num_experiments = 5
            subgoal.experiments = [f'EXP_{subgoal.id}_{i}' for i in range(num_experiments)]
            subgoal.revisions = ['REV_aggressive', 'REV_exploratory']
        elif mode == PlanningMode.RECOVERY:
            num_experiments = 1
            subgoal.experiments = [f'EXP_{subgoal.id}_stabilize']
            subgoal.revisions = ['REV_emergency', 'REV_level3']
        elif mode == PlanningMode.OPTIMIZATION:
            num_experiments = 4
            subgoal.experiments = [f'EXP_{subgoal.id}_{i}' for i in range(num_experiments)]
            subgoal.revisions = ['REV_level1', 'REV_level2']

    def execute_plan(self, goal: LongTermGoal, performance_evaluator: Callable, structure: Dict) -> Dict[str, Any]:
        if goal.status == GoalStatus.PENDING:
            goal.status = GoalStatus.IN_PROGRESS
            goal.started_at = time.time()
        execution_log = {'goal_id': goal.id, 'subgoals_executed': [], 'experiments_run': 0, 'revisions_applied': 0, 'evaluations': []}
        for subgoal in goal.subgoals:
            if subgoal.status == GoalStatus.PENDING:
                subgoal.status = GoalStatus.IN_PROGRESS
                subgoal.started_at = time.time()
            for exp_id in subgoal.experiments:
                hypothesis = self.experiment_designer.generate_hypothesis(ErrorProfile(), performance_evaluator(structure))
                exp_design = self.experiment_designer.design_experiment(hypothesis)
                exp_results = self.experiment_designer.execute_experiment(exp_design, performance_evaluator, structure)
                structure, applied = self.experiment_designer.apply_if_accepted(exp_results, structure)
                execution_log['experiments_run'] += 1
                if applied:
                    subgoal.progress += 0.2
                else:
                    goal.consecutive_failures += 1
            if subgoal.should_evaluate(time.time()):
                eval_result = self._evaluate_subgoal(subgoal, performance_evaluator(structure))
                execution_log['evaluations'].append(eval_result)
                if not eval_result['on_track']:
                    goal.consecutive_failures += 1
            if subgoal.is_completed():
                subgoal.status = GoalStatus.COMPLETED
                subgoal.completed_at = time.time()
                print(f'SubGoal {subgoal.id} COMPLETED')
            execution_log['subgoals_executed'].append(subgoal.id)
        goal.update_progress()
        termination_result = self._check_termination(goal)
        execution_log['termination'] = termination_result
        return execution_log

    def _evaluate_subgoal(self, subgoal: SubGoal, current_performance: Dict[str, float]) -> Dict[str, Any]:
        subgoal.last_evaluation = time.time()
        achieved_metrics = 0
        total_metrics = len(subgoal.target_metrics)
        for metric, target in subgoal.target_metrics.items():
            current = current_performance.get(metric, 0.0)
            subgoal.current_metrics[metric] = current
            if current >= target * 0.5:
                achieved_metrics += 1
        progress = achieved_metrics / max(total_metrics, 1)
        progress_delta = 0.0
        if subgoal.evaluation_results:
            last_progress = subgoal.evaluation_results[-1].get('progress', 0.0)
            progress_delta = progress - last_progress
        on_track = progress >= 0.3 and progress_delta >= -0.1
        result = {'timestamp': time.time(), 'progress': progress, 'progress_delta': progress_delta, 'on_track': on_track, 'metrics': subgoal.current_metrics.copy(), 'recommendations': self._generate_recommendations(progress, progress_delta)}
        subgoal.evaluation_results.append(result)
        if on_track:
            print(f'SubGoal {subgoal.id} on track ({progress * 100:.1f}%)')
        else:
            print(f'SubGoal {subgoal.id} off track ({progress * 100:.1f}%)')
        return result

    def _generate_recommendations(self, progress: float, progress_delta: float) -> List[str]:
        recommendations = []
        if progress < 0.2:
            recommendations.append('Consider switching approach')
        if progress_delta < -0.05:
            recommendations.append('Progress declining, review strategy')
        if progress > 0.8:
            recommendations.append('Near completion, maintain current approach')
        if progress_delta > 0.2:
            recommendations.append('Excellent progress, continue current strategy')
        return recommendations

    def _check_termination(self, goal: LongTermGoal) -> Dict[str, Any]:
        should_terminate = False
        reason = ''
        next_action = 'continue'
        completed_subgoals = sum((1 for sg in goal.subgoals if sg.status == GoalStatus.COMPLETED))
        completion_rate = completed_subgoals / max(len(goal.subgoals), 1)
        if completion_rate >= 0.8:
            should_terminate = True
            reason = 'Goal achieved'
            next_action = 'celebrate'
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
            self.total_goals_completed += 1
        elif goal.should_redefine():
            should_terminate = True
            reason = 'Goal redefinition needed'
            next_action = 'redefine_goal'
            goal.status = GoalStatus.FAILED
            self.total_goals_failed += 1
        elif goal.should_switch_to_exploration():
            should_terminate = False
            reason = 'Switching to exploration mode'
            next_action = 'switch_mode'
        return {'should_terminate': should_terminate, 'reason': reason, 'next_action': next_action, 'completion_rate': completion_rate, 'overall_progress': goal.overall_progress}

    def handle_termination(self, termination_result: Dict) -> Optional[LongTermGoal]:
        next_action = termination_result['next_action']
        if next_action == 'celebrate':
            print(f' GOAL COMPLETED ')
            print(f"   Progress: {termination_result['overall_progress'] * 100:.1f}%")
            return None
        elif next_action == 'redefine_goal':
            print(f'REDEFINING GOAL...')
            print(f"   Reason: {termination_result['reason']}")
            if self.current_goal is not None:
                new_goal = self._redefine_goal(self.current_goal)
                self.total_goals_redefined += 1
                return new_goal
            else:
                print("   No current goal to redefine")
                return None
        elif next_action == 'switch_mode':
            print(f'SWITCHING MODE...')
            if self.current_goal is not None:
                self._switch_mode(self.current_goal)
                self.mode_switches += 1
            else:
                print("   No current goal to switch mode")
            return None
        return None

    def _redefine_goal(self, failed_goal: LongTermGoal) -> LongTermGoal:
        failed_subgoals = [sg for sg in failed_goal.subgoals if sg.status == GoalStatus.FAILED]
        new_criteria = {metric: target * 0.8 for metric, target in failed_goal.success_criteria.items()}
        print(f'  Success criteria reduced by 20%')
        print(f'  Mode switched to EXPLORATION')
        # Create a redefined goal with relaxed criteria and exploration mode
        desc = f"Redefine: {failed_goal.description}"
        new_goal = self.create_goal(
            description=desc,
            mode=PlanningMode.EXPLORATION,
            success_criteria=new_criteria,
            max_duration=failed_goal.max_duration
        )
        return new_goal

    def _switch_mode(self, goal: LongTermGoal):
        old_mode = goal.mode
        if goal.mode == PlanningMode.EXPLOITATION:
            goal.mode = PlanningMode.EXPLORATION
        elif goal.mode == PlanningMode.EXPLORATION:
            if goal.consecutive_failures >= 3:
                goal.mode = PlanningMode.RECOVERY
            else:
                goal.mode = PlanningMode.OPTIMIZATION
        elif goal.mode == PlanningMode.OPTIMIZATION:
            goal.mode = PlanningMode.EXPLORATION
        elif goal.mode == PlanningMode.RECOVERY:
            goal.mode = PlanningMode.EXPLOITATION
        goal.consecutive_failures = 0
        print(f'  Mode: {old_mode.value} -> {goal.mode.value}')
        error_profile = ErrorProfile()
        for subgoal in goal.subgoals:
            if subgoal.status != GoalStatus.COMPLETED:
                self._schedule_actions(subgoal, error_profile, goal.mode)

    def get_status_report(self) -> Dict[str, Any]:
        if self.current_goal is None:
            return {'status': 'No active goal'}
        goal = self.current_goal
        return {'goal_id': goal.id, 'description': goal.description, 'mode': goal.mode.value, 'status': goal.status.value, 'overall_progress': f'{goal.overall_progress * 100:.1f}%', 'consecutive_failures': goal.consecutive_failures, 'subgoals': {'total': len(goal.subgoals), 'completed': sum((1 for sg in goal.subgoals if sg.status == GoalStatus.COMPLETED)), 'in_progress': sum((1 for sg in goal.subgoals if sg.status == GoalStatus.IN_PROGRESS)), 'failed': sum((1 for sg in goal.subgoals if sg.status == GoalStatus.FAILED))}, 'statistics': {'total_completed': self.total_goals_completed, 'total_failed': self.total_goals_failed, 'total_redefined': self.total_goals_redefined, 'mode_switches': self.mode_switches}}
from typing import Dict, List, Optional, Any, Tuple

class ResourceType(Enum):
    COMPUTE = 'compute'
    MEMORY = 'memory'
    EXPERIMENTS = 'experiments'
    REVISIONS = 'revisions'
    TIME = 'time'

@dataclass
class Budget:
    resource_type: ResourceType
    total: float
    consumed: float = 0.0
    reserved: float = 0.0

    def available(self) -> float:
        return self.total - self.consumed - self.reserved

    def reserve(self, amount: float) -> bool:
        if self.available() >= amount:
            self.reserved += amount
            return True
        return False

    def consume(self, amount: float) -> bool:
        if amount <= self.reserved:
            self.reserved -= amount
            self.consumed += amount
            return True
        elif self.available() >= amount:
            self.consumed += amount
            return True
        return False

    def release(self, amount: float):
        self.reserved = max(0, self.reserved - amount)

    def utilization(self) -> float:
        return self.consumed / max(self.total, 1e-10)

@dataclass
class RewardSignal:
    source: str
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_positive(self) -> bool:
        return self.value > 0

@dataclass
class BudgetDecision:
    approved: bool
    allocated_resources: Dict[ResourceType, float]
    priority: float
    reason: str

class GlobalRewardBudgetScheduler:

    def __init__(self, initial_budgets: Optional[Dict[ResourceType, float]]=None, reward_decay: float=0.95, budget_realloc_interval: float=100.0):
        if initial_budgets is None:
            initial_budgets = {ResourceType.COMPUTE: 10000.0, ResourceType.MEMORY: 1000.0, ResourceType.EXPERIMENTS: 100.0, ResourceType.REVISIONS: 50.0, ResourceType.TIME: 3600.0}
        self.budgets: Dict[ResourceType, Budget] = {rtype: Budget(resource_type=rtype, total=amount) for rtype, amount in initial_budgets.items()}
        self.reward_history: deque = deque(maxlen=1000)
        self.reward_decay = reward_decay
        self.cumulative_reward = 0.0
        self.activity_roi: Dict[str, Dict[str, Any]] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = []
        self.rejected_tasks: List[Dict[str, Any]] = []
        self.budget_realloc_interval = budget_realloc_interval
        self.last_realloc_time = time.time()
        self.total_rewards_received = 0
        self.total_budget_consumed = 0.0
        self.reallocation_count = 0

    def receive_reward(self, reward: RewardSignal):
        self.reward_history.append(reward)
        self.cumulative_reward = self.cumulative_reward * self.reward_decay + reward.value
        self.total_rewards_received += 1
        if reward.source not in self.activity_roi:
            self.activity_roi[reward.source] = {'total_reward': 0.0, 'total_cost': 0.0, 'count': 0, 'roi': 0.0}
        self.activity_roi[reward.source]['total_reward'] += reward.value
        self.activity_roi[reward.source]['count'] += 1
        if 'cost' in reward.metadata:
            self.activity_roi[reward.source]['total_cost'] += reward.metadata['cost']
        self._update_roi(reward.source)
        if reward.is_positive() and reward.value > 0.5:
            print(f' High Reward: {reward.source} = {reward.value:.3f}')

    def _update_roi(self, activity: str):
        data = self.activity_roi[activity]
        total_cost = data['total_cost']
        total_reward = data['total_reward']
        if total_cost > 0:
            data['roi'] = total_reward / total_cost
        else:
            data['roi'] = total_reward

    def request_budget(self, task_name: str, resource_requirements: Dict[ResourceType, float], priority: float=0.5, metadata: Optional[Dict]=None) -> Tuple[bool, Optional[str]]:
        can_allocate = True
        for rtype, amount in resource_requirements.items():
            if rtype not in self.budgets:
                return (False, f'Unknown resource type: {rtype}')
            if self.budgets[rtype].available() < amount:
                can_allocate = False
                break
        if can_allocate:
            for rtype, amount in resource_requirements.items():
                self.budgets[rtype].reserve(amount)
            task_id = f'TASK_{len(self.task_queue) + len(self.completed_tasks) + 1}'
            task = {'id': task_id, 'name': task_name, 'resources': resource_requirements, 'priority': priority, 'status': 'approved', 'metadata': metadata or {}, 'approved_at': time.time()}
            self.task_queue.append(task)
            self.task_queue.sort(key=lambda t: t['priority'], reverse=True)
            return (True, task_id)
        else:
            if priority > 0.8:
                freed = self._try_defer_low_priority_tasks(resource_requirements)
                if freed:
                    return self.request_budget(task_name, resource_requirements, priority, metadata)
            rejection = {'name': task_name, 'resources': resource_requirements, 'priority': priority, 'reason': 'insufficient_budget', 'rejected_at': time.time()}
            self.rejected_tasks.append(rejection)
            return (False, 'Insufficient budget')

    def _try_defer_low_priority_tasks(self, needed_resources: Dict[ResourceType, float]) -> bool:
        low_priority_tasks = [t for t in self.task_queue if t['priority'] < 0.5]
        for task in low_priority_tasks:
            for rtype, amount in task['resources'].items():
                self.budgets[rtype].release(amount)
            self.task_queue.remove(task)
            task['status'] = 'deferred'
            self.rejected_tasks.append(task)
            can_allocate = all((self.budgets[rtype].available() >= amount for rtype, amount in needed_resources.items() if rtype in self.budgets))
            if can_allocate:
                return True
        return False

    def consume_budget(self, task_id: str) -> bool:
        task = None
        for t in self.task_queue:
            if t['id'] == task_id:
                task = t
                break
        if task is None:
            return False
        success = True
        for rtype, amount in task['resources'].items():
            if not self.budgets[rtype].consume(amount):
                success = False
                break
        if success:
            task['status'] = 'completed'
            task['completed_at'] = time.time()
            self.task_queue.remove(task)
            self.completed_tasks.append(task)
            self.total_budget_consumed += sum(task['resources'].values())
        return success

    def get_priority_recommendation(self, task_type: str, historical_roi: Optional[float]=None) -> float:
        if historical_roi is not None:
            roi = historical_roi
        elif task_type in self.activity_roi:
            roi = self.activity_roi[task_type]['roi']
        else:
            roi = 0.5
        priority = 1.0 / (1.0 + np.exp(-roi))
        return float(np.clip(priority, 0.1, 0.95))

    def reallocate_budgets(self):
        current_time = time.time()
        if current_time - self.last_realloc_time < self.budget_realloc_interval:
            return
        self.last_realloc_time = current_time
        self.reallocation_count += 1
        print(f'\nBudget Reallocation #{self.reallocation_count}')
        activity_consumption = {}
        for task in self.completed_tasks:
            activity = task['name']
            if activity not in activity_consumption:
                activity_consumption[activity] = {rtype: 0.0 for rtype in ResourceType}
            for rtype, amount in task['resources'].items():
                activity_consumption[activity][rtype] += amount
        for activity, roi_data in self.activity_roi.items():
            roi = roi_data['roi']
            if roi > 1.0:
                print(f'  {activity}: ROI={roi:.2f} -> Budget +10%')
            elif roi < 0.5:
                print(f'  {activity}: ROI={roi:.2f} -> Budget -10%')
        print(f'\n  Budget Utilization:')
        for rtype, budget in self.budgets.items():
            util = budget.utilization()
            color = 'green' if util < 0.7 else 'yellow' if util < 0.9 else 'red'
            print(f'    {rtype.value}: {util * 100:.1f}%')

    def get_status(self) -> Dict[str, Any]:
        return {'cumulative_reward': self.cumulative_reward, 'recent_avg_reward': np.mean([r.value for r in list(self.reward_history)[-20:]]) if self.reward_history else 0.0, 'budgets': {rtype.value: {'total': budget.total, 'consumed': budget.consumed, 'reserved': budget.reserved, 'available': budget.available(), 'utilization': budget.utilization()} for rtype, budget in self.budgets.items()}, 'tasks': {'queued': len(self.task_queue), 'completed': len(self.completed_tasks), 'rejected': len(self.rejected_tasks)}, 'top_roi_activities': sorted([{'activity': name, 'roi': data['roi'], 'count': data['count']} for name, data in self.activity_roi.items()], key=lambda x: x['roi'], reverse=True)[:5], 'statistics': {'total_rewards': self.total_rewards_received, 'budget_consumed': self.total_budget_consumed, 'reallocations': self.reallocation_count}}

    def reset_budget(self, resource_type: ResourceType, new_total: float):
        if resource_type in self.budgets:
            old_total = self.budgets[resource_type].total
            self.budgets[resource_type] = Budget(resource_type=resource_type, total=new_total)
            print(f'Budget reset: {resource_type.value} {old_total} -> {new_total}')

    def allocate_budget(self, reward: RewardSignal) -> 'BudgetDecision':
        resource_costs = reward.metadata.get('resource_costs', {})
        if not resource_costs:
            return BudgetDecision(approved=True, allocated_resources={}, priority=0.5, reason='No resource costs specified')
        can_allocate = True
        for rtype, amount in resource_costs.items():
            if rtype not in self.budgets:
                can_allocate = False
                break
            if self.budgets[rtype].available() < amount:
                can_allocate = False
                break
        if can_allocate:
            for rtype, amount in resource_costs.items():
                if rtype in self.budgets:
                    self.budgets[rtype].reserve(amount)
            return BudgetDecision(approved=True, allocated_resources=resource_costs, priority=0.5, reason='Budget allocated successfully')
        else:
            return BudgetDecision(approved=False, allocated_resources={}, priority=0.5, reason='Insufficient budget')

class UnifiedSubject:

    def __init__(self, subject_id: str='I'):
        self.subject_id = subject_id
        self.subjective_experiences: deque = deque(maxlen=500)
        self.identity = {'stable_traits': [], 'life_story': [], 'self_concept': 'undefined'}
        self.agency = 0.5
        self.ownership = 0.5
        self.unity_score = 0.5

    def experience(self, content: Any, content_type: str, is_intentional: bool=False) -> Dict[str, Any]:
        subjective_exp = {'timestamp': len(self.subjective_experiences), 'subject': self.subject_id, 'experience_type': content_type, 'content': str(content)[:100], 'phenomenal_character': self._compute_phenomenal_character(content), 'agency': self.agency if is_intentional else 0.1, 'ownership': self.ownership, 'integrated': True}
        self.subjective_experiences.append(subjective_exp)
        self._update_identity(subjective_exp)
        return subjective_exp

    def _compute_phenomenal_character(self, content: Any) -> str:
        if isinstance(content, dict):
            if 'meaning' in content:
                return "phenomenal character: " + str(content.get("meaning", "unknown"))
        return 'no phenomenal character'
    def _update_identity(self, experience: Dict):
        if len(self.subjective_experiences) > 20:
            recent = list(self.subjective_experiences)[-20:]
            type_counts = {}
            for exp in recent:
                t = exp['experience_type']
                type_counts[t] = type_counts.get(t, 0) + 1
            if type_counts:
                dominant_type = max(type_counts, key=lambda k: type_counts[k])
                if type_counts[dominant_type] > 10:
                    trait = f'{dominant_type}-  oriented'
                    if trait not in self.identity['stable_traits']:
                        self.identity['stable_traits'].append(trait)

    def reflect_on_self(self) -> str:
        if not self.subjective_experiences:
            return 'insufficient evidence to summarize'
        recent = list(self.subjective_experiences)[-10:]
        narrative = f'Subject {self.subject_id}: '
        if self.identity['stable_traits']:
            narrative += f"traits: {', '.join(self.identity['stable_traits'][:3])}; "
        recent_types = [e['experience_type'] for e in recent]
        from collections import Counter
        common = Counter(recent_types).most_common(2)
        narrative += f'most frequent concept: {common[0][0]}. '
        if self.agency > 0.7:
            narrative += 'experiences show notable variance '
        elif self.agency < 0.3:
            narrative += 'experiences appear consistent.'
        return narrative

    def bind_experience(self, qualia, beliefs, goals, workspace_contents, t) -> Dict[str, Any]:
        print(f"[BIND_EXPERIENCE] t={t}, workspace_contents={len(workspace_contents) if workspace_contents else 0}")
        if not isinstance(goals, list):
            goals = [goals]
        unified_exp = {'subject': self.subject_id, 'unified_moment': {'what_i_feel': f'   ={qualia.arousal:.2f},    ={qualia.valence:.2f}', 'what_i_believe': f"   ={(beliefs.get('belief_stability', 0.5) if isinstance(beliefs, dict) else getattr(beliefs, 'belief_stability', 0.5)):.2f}", 'what_i_want': goals[0].type.value if goals and len(goals) > 0 and hasattr(goals[0], 'type') else 'none', 'what_i_attend': workspace_contents[0].semantic_meaning if workspace_contents and len(workspace_contents) > 0 and hasattr(workspace_contents[0], 'semantic_meaning') else 'nothing'}, 'gestalt': self._create_gestalt(qualia, beliefs, goals, workspace_contents), 'unity_achieved': self._check_unity(qualia, workspace_contents, t)}
        return unified_exp

    def _create_gestalt(self, qualia, beliefs, goals, workspace) -> str:
        dominant_qualia = qualia.dominant_feeling()
        if hasattr(beliefs, 'belief_stability'):
            beliefs_list = [beliefs.belief_stability, beliefs.belief_adaptation, beliefs.belief_prediction]
            if max(beliefs_list) == beliefs.belief_stability:
                dominant_belief = '   '
            elif max(beliefs_list) == beliefs.belief_adaptation:
                dominant_belief = '   '
            else:
                dominant_belief = '   '
        elif isinstance(beliefs, dict):
            stability = beliefs.get('belief_stability', 0.5)
            adaptation = beliefs.get('belief_adaptation', 0.5)
            prediction = beliefs.get('belief_prediction', 0.5)
            beliefs_list = [stability, adaptation, prediction]
            if max(beliefs_list) == stability:
                dominant_belief = '   '
            elif max(beliefs_list) == adaptation:
                dominant_belief = '   '
            else:
                dominant_belief = '   '
        elif isinstance(beliefs, dict):
            stability = beliefs.get('belief_stability', 0.5)
            adaptation = beliefs.get('belief_adaptation', 0.5)
            prediction = beliefs.get('belief_prediction', 0.5)
            beliefs_list = [stability, adaptation, prediction]
            if max(beliefs_list) == stability:
                dominant_belief = '   '
            elif max(beliefs_list) == adaptation:
                dominant_belief = '   '
            else:
                dominant_belief = '   '
        else:
            dominant_belief = '   '
        return f'dominant qualia: {dominant_qualia}, dominant belief: {dominant_belief}'

    def _check_unity(self, qualia, workspace_contents, t) -> bool:
        unity_score = 0.5
        
        # Workspace coherence
        if workspace_contents and len(workspace_contents) >= 2:
            try:
                valid_contents = [c for c in workspace_contents if hasattr(c, 'salience')]
                if valid_contents:
                    avg_salience = np.mean([c.salience for c in valid_contents])
                    unity_score += (avg_salience - 0.5) * 0.4
            except (AttributeError, TypeError):
                pass
        
        # Qualia coherence
        arousal_balance = 1.0 - abs(qualia.arousal - 0.5)
        valence_clarity = abs(qualia.valence - 0.5)
        engagement_factor = qualia.engagement
        
        unity_score += arousal_balance * 0.2
        unity_score += valence_clarity * 0.15
        unity_score += engagement_factor * 0.25
        unity_score -= qualia.entropy * 0.2
        unity_score -= qualia.frustration * 0.15
        
        unity_score = np.clip(unity_score, 0.0, 1.0)
        
        # Update instance unity_score with EMA
        self.unity_score = 0.3 * self.unity_score + 0.7 * unity_score
        
        return bool(unity_score > 0.5)

class MetaMetaMonitor:

    def __init__(self):
        self.model_quality_history: deque = deque(maxlen=256)
        self.last_revision_time = -1000
        self.revision_threshold = 0.25
        self.revision_chain: deque = deque(maxlen=10)
        self.consecutive_revisions = 0
        self.quality_before_revision: deque = deque(maxlen=256)
        self.quality_after_revision: deque = deque(maxlen=256)
        self.revision_effectiveness: deque = deque(maxlen=256)
        self.failed_revision_count = 0
        self.emergency_mode = False
        self.emergency_cooldown = 0

    def evaluate_model_quality(self, self_model: RecursiveSelfModel, t: int) -> Tuple[bool, float, int]:
        if hasattr(self_model, 'prediction_errors') and len(self_model.prediction_errors) >= 10:
            recent_errors = list(self_model.prediction_errors)[-10:]
            avg_error = np.mean(recent_errors)
            error_trend = np.mean(np.diff(recent_errors)) if len(recent_errors) > 1 else 0.0
        elif isinstance(self_model, dict) and 'prediction_errors' in self_model and (len(self_model['prediction_errors']) >= 10):
            recent_errors = list(self_model['prediction_errors'])[-10:]
            avg_error = np.mean(recent_errors)
            error_trend = np.mean(np.diff(recent_errors)) if len(recent_errors) > 1 else 0.0
        else:
            avg_error = 0.5
            error_trend = 0.0
        if hasattr(self_model, 'belief_stability'):
            beliefs = [self_model.belief_stability, self_model.belief_adaptation, self_model.belief_prediction]
        elif isinstance(self_model, dict):
            beliefs = [self_model.get('belief_stability', 0.5), self_model.get('belief_adaptation', 0.5), self_model.get('belief_prediction', 0.5)]
        else:
            beliefs = [0.5, 0.5, 0.5]
        belief_variance = np.var(np.array(beliefs))
        if hasattr(self_model, 'action_outcomes') and len(self_model.action_outcomes) >= 10:
            recent_success_rate = sum(list(self_model.action_outcomes)[-10:]) / 10.0
        elif isinstance(self_model, dict) and 'action_outcomes' in self_model and (len(self_model['action_outcomes']) >= 10):
            recent_success_rate = sum(list(self_model['action_outcomes'])[-10:]) / 10.0
        else:
            recent_success_rate = 0.5
        if hasattr(self_model, 'meta_confidence'):
            meta_confidence = self_model.meta_confidence
        elif isinstance(self_model, dict):
            meta_confidence = self_model.get('meta_confidence', 0.5)
        else:
            meta_confidence = 0.5
        quality = 0.35 * (1.0 - min(1.0, avg_error)) + 0.25 * meta_confidence + 0.25 * (1.0 - min(1.0, belief_variance)) + 0.15 * recent_success_rate
        self.model_quality_history.append(quality)
        if self.emergency_cooldown > 0:
            self.emergency_cooldown -= 1
            if self.emergency_cooldown == 0:
                self.emergency_mode = False
        needs_revision = False
        revision_level = 0
        if len(self.model_quality_history) >= 30:
            recent_quality = np.mean(list(self.model_quality_history)[-30:])
            time_since_last = t - self.last_revision_time
            if recent_quality < self.revision_threshold and time_since_last > 500:
                needs_revision = True
                if len(self.revision_effectiveness) >= 3:
                    recent_effectiveness = list(self.revision_effectiveness)[-3:]
                    avg_effectiveness = np.mean(recent_effectiveness)
                else:
                    avg_effectiveness = 0.5
                if time_since_last < 1000:
                    self.consecutive_revisions += 1
                else:
                    self.consecutive_revisions = 1
                if len(self.model_quality_history) >= 60:
                    long_term_trend = np.mean(list(self.model_quality_history)[-60:-30]) - recent_quality
                else:
                    long_term_trend = 0.0
                if self.emergency_mode:
                    revision_level = 4
                elif self.consecutive_revisions >= 4 or self.failed_revision_count >= 3:
                    revision_level = 4
                    self.emergency_mode = True
                    self.emergency_cooldown = 2000
                elif self.consecutive_revisions >= 3 or avg_effectiveness < 0.2:
                    revision_level = 3
                elif self.consecutive_revisions >= 2 or (avg_effectiveness < 0.4 and long_term_trend > 0.1):
                    revision_level = 2
                else:
                    revision_level = 1
                self.quality_before_revision.append(quality)
                self.last_revision_time = t
                self.revision_chain.append({'time': t, 'level': revision_level, 'quality_before': quality, 'consecutive_count': self.consecutive_revisions, 'error_trend': error_trend})
        return (needs_revision, float(quality), revision_level)

    def record_revision_outcome(self, quality_after: float):
        if len(self.quality_before_revision) > 0 and len(self.quality_after_revision) < len(self.quality_before_revision):
            quality_before = self.quality_before_revision[-1]
            quality_after_val = quality_after
            self.quality_after_revision.append(quality_after_val)
            improvement = quality_after_val - quality_before
            effectiveness = np.clip((improvement + 0.5) / 1.0, 0.0, 1.0)
            self.revision_effectiveness.append(effectiveness)
            if improvement < -0.05:
                self.failed_revision_count += 1
            else:
                self.failed_revision_count = max(0, self.failed_revision_count - 1)

    def get_revision_policy_status(self) -> Dict[str, Any]:
        return {'consecutive_revisions': self.consecutive_revisions, 'failed_revision_count': self.failed_revision_count, 'emergency_mode': self.emergency_mode, 'emergency_cooldown': self.emergency_cooldown, 'avg_effectiveness': np.mean(list(self.revision_effectiveness)) if self.revision_effectiveness else 0.5, 'recent_revisions': list(self.revision_chain)[-5:] if self.revision_chain else []}

class EnergyBasedController:
    """Precise energy & activation controller with realistic dynamics
    
    Energy represents metabolic/computational resources (0-100)
    Activation represents arousal/processing intensity (0-1)
    Both interact dynamically with realistic depletion/recovery curves
    """

    def __init__(self, initial_energy: float = 100.0, message_bus: Optional[MessageBus] = None):
        self.internal_clock = 0
        
        # Core state variables
        self.cognitive_energy = float(initial_energy)
        self.energy_capacity = float(initial_energy)
        self.activation_level = 0.5
        
        # History tracking
        self.activation_history = deque(maxlen=1000)
        self.energy_history = deque(maxlen=1000)
        
        # Running statistics for normalization
        self.activation_mean = 0.5
        self.activation_std = 0.2
        self.energy_mean = 0.5
        self.energy_std = 0.2
        
        # === CRITICAL FIX 1: 기초대사율 ===
        self.basal_metabolic_rate = 0.001
        
        # === CRITICAL FIX 2: 회복률 ===
        self.recovery_rate_max = 0.70
        self.recovery_efficiency = 0.95
        
        # === CRITICAL FIX 3: Activation 비용 ===
        self.activation_cost_multiplier = 0.15
        
        # === NEW: 에너지 레벨별 적응형 회복 ===
        self.adaptive_recovery_boost = True
        
        # Activation dynamics parameters
        self.activation_decay_rate = 0.18
        self.activation_equilibrium = 0.35
        self.activation_response_speed = 0.12
        
        # Fatigue accumulation
        self.fatigue_level = 0.0
        self.fatigue_threshold = 0.7
        self.fatigue_accumulation_rate = 0.005
        self.fatigue_recovery_rate = 0.03
        
        # Statistics for monitoring
        self.total_actions_taken = 0
        self.total_energy_consumed = 0.0
        self.low_energy_count = 0
        self.high_activation_count = 0
        self.avg_activation = 0.5
        self.avg_energy = initial_energy
        
        # MULTI-LOOP INFRASTRUCTURE
        self.message_bus = message_bus
        if self.message_bus:
            self.message_bus.register_module('energy')
        # Influences from other modules
        self.qualia_arousal = 0.5
        self.workspace_load = 0.0
        self.policy_cost = 0.0

    def update_internal_statistics(self):
        """Update running statistics for normalization"""
        if len(self.activation_history) > 50:
            arr = np.array(list(self.activation_history))
            self.activation_mean = float(np.mean(arr))
            self.activation_std = float(max(0.05, np.std(arr)))
        
        if len(self.energy_history) > 50:
            arr = np.array(list(self.energy_history))
            self.energy_mean = float(np.mean(arr))
            self.energy_std = float(max(0.05, np.std(arr)))

    def compute_qualia_pressure(self, qualia: 'QualiaState') -> float:
        """Compute how qualia state pressures activation level
        
        High arousal/entropy/frustration -> increase pressure
        High valence/engagement -> reduce pressure (satisfaction)
        Returns: pressure in range [-1, 1]
        """
        # Activating pressures
        arousal_pressure = float(qualia.arousal) ** 1.8 * 0.8
        entropy_pressure = float(qualia.entropy) * 0.6
        frustration_pressure = float(qualia.frustration) * 0.7
        
        # Satisfying/calming influences
        valence_release = -float(qualia.valence) * 0.5
        engagement_satisfaction = -float(qualia.engagement) * 0.3
        
        total = arousal_pressure + entropy_pressure + frustration_pressure + valence_release + engagement_satisfaction
        return float(np.tanh(total))

    def compute_self_model_drive(self, self_model) -> float:
        """Compute drive from self-model state
        
        Low confidence, belief imbalance -> increase drive
        High meta-awareness -> reduce drive (satisfied)
        Returns: drive in range [-1, 1]
        """
        # Extract confidence
        if hasattr(self_model, 'meta_confidence'):
            confidence = float(self_model.meta_confidence)
        elif isinstance(self_model, dict):
            confidence = float(self_model.get('meta_confidence', 0.5))
        else:
            confidence = 0.5
        
        # Low confidence drives activation
        confidence_drive = (1.0 - confidence) ** 1.3 * 0.6
        
        # Extract beliefs
        if isinstance(self_model, dict):
            beliefs = [
                float(self_model.get('belief_stability', 0.5)),
                float(self_model.get('belief_adaptation', 0.5)),
                float(self_model.get('belief_prediction', 0.5))
            ]
        elif hasattr(self_model, 'belief_stability'):
            beliefs = [
                float(self_model.belief_stability),
                float(self_model.belief_adaptation),
                float(self_model.belief_prediction)
            ]
        else:
            beliefs = [0.5, 0.5, 0.5]
        
        # Belief imbalance drives exploration
        belief_variance = float(np.var(beliefs))
        belief_imbalance = belief_variance * 2.0
        
        # High meta-awareness is satisfying
        if hasattr(self_model, 'meta_awareness'):
            meta = float(self_model.meta_awareness)
        elif isinstance(self_model, dict):
            meta = float(self_model.get('meta_awareness', 0.5))
        else:
            meta = 0.5
        
        meta_satisfaction = -meta * 0.4
        
        total = confidence_drive + belief_imbalance + meta_satisfaction
        return float(np.tanh(total))

    def update_activation(self, qualia: 'QualiaState', self_model, goal: Optional['Goal']):
        """활성화 업데이트
        
        Activation: 처리 강도
        구동: qualia, self-model, goal, energy
        제약: fatigue, energy
        """
        qualia_pressure = self.compute_qualia_pressure(qualia)
        self_drive = self.compute_self_model_drive(self_model)
        
        goal_demand = 0.0
        if goal:
            if goal.type == GoalType.REST:
                goal_demand = -0.9
            elif goal.type == GoalType.UNDERSTAND_SELF:
                goal_demand = 0.7
            elif goal.type == GoalType.EXPLORE:
                goal_demand = 0.8
            elif goal.type == GoalType.STABILIZE:
                goal_demand = -0.3
            else:
                goal_demand = 0.0
        
        energy_ratio = self.cognitive_energy / self.energy_capacity
        
        if energy_ratio < 0.15:
            energy_feedback = -3.5 * (1.0 - energy_ratio / 0.15) ** 2
        elif energy_ratio < 0.30:
            energy_feedback = -1.5 * (1.0 - energy_ratio / 0.30)
        elif energy_ratio < 0.50:
            energy_feedback = -0.5 * (1.0 - energy_ratio / 0.50)
        else:
            energy_feedback = 0.2 * np.tanh((energy_ratio - 0.5) * 2.0)
        
        fatigue_pressure = -self.fatigue_level * 0.8
        
        total_drive = (
            qualia_pressure * 0.25 * max(0.3, energy_ratio) +
            self_drive * 0.25 * max(0.3, energy_ratio) +
            goal_demand * 0.15 +
            energy_feedback * 0.30 +
            fatigue_pressure * 0.05
        )
        
        decay_force = -self.activation_decay_rate * (self.activation_level - self.activation_equilibrium)
        
        delta_activation = (total_drive + decay_force) * self.activation_response_speed
        self.activation_level += delta_activation
        self.activation_level = np.clip(self.activation_level, 0.0, 1.0)
        
        if self.activation_level > self.fatigue_threshold:
            self.fatigue_level += self.fatigue_accumulation_rate * (
                self.activation_level - self.fatigue_threshold
            )
        else:
            self.fatigue_level -= self.fatigue_recovery_rate * 1.5
        
        self.fatigue_level = np.clip(self.fatigue_level, 0.0, 1.0)
        self.activation_history.append(float(self.activation_level))

    def compute_cognitive_cost(self, action_plan: Dict[str, Any]) -> float:
        """에너지 비용 계산"""
        base_complexity = float(action_plan.get('complexity', 1.0))
        base_cost = max(0.3, base_complexity * 0.5)
        activation_multiplier = 1.0 + (self.activation_level ** 1.5) * self.activation_cost_multiplier
        fatigue_penalty = 1.0 + (self.fatigue_level * 0.3)
        total_cost = base_cost * activation_multiplier * fatigue_penalty
        return float(total_cost)

    def update_energy(self, cost: float):
        """에너지 업데이트
        
        에너지 소모: Action costs, Basal metabolic rate, Activation overhead
        에너지 회복: Rest factor, Adaptive boost, Sleep bonus
        """
        # === 1. 소모 계산 ===
        basal_drain = self.basal_metabolic_rate
        action_cost = float(cost) * 0.3
        activation_overhead = (self.activation_level ** 2.5) * 0.05
        
        total_expenditure = basal_drain + action_cost + activation_overhead
        self.cognitive_energy -= total_expenditure
        
        # === 2. 회복 시스템 ===
        rest_factor = (1.0 - self.activation_level) ** 1.8
        energy_ratio = self.cognitive_energy / self.energy_capacity
        
        # === 3. 에너지 레벨별 회복 부스트 ===
        if self.adaptive_recovery_boost:
            if energy_ratio < 0.15:
                crisis_boost = 3.5 * (1.0 - energy_ratio / 0.15)
                recovery_efficiency = min(1.0, self.recovery_efficiency * (1.0 + crisis_boost))
            elif energy_ratio < 0.3:
                crisis_boost = 2.5 * (1.0 - energy_ratio / 0.3)
                recovery_efficiency = min(1.0, self.recovery_efficiency * (1.0 + crisis_boost))
            elif energy_ratio < 0.5:
                crisis_boost = 1.0 * (1.0 - energy_ratio / 0.5)
                recovery_efficiency = min(1.0, self.recovery_efficiency * (1.0 + crisis_boost))
            else:
                recovery_efficiency = self.recovery_efficiency
        else:
            recovery_efficiency = self.recovery_efficiency
        
        # === 4. 회복 계산 ===
        base_recovery = self.recovery_rate_max * rest_factor * recovery_efficiency
        
        if self.activation_level < 0.2:
            sleep_bonus = 0.6 * (0.2 - self.activation_level) / 0.2
            base_recovery += sleep_bonus
        
        if self.activation_level < 0.1:
            turbo_bonus = 0.4 * (0.1 - self.activation_level) / 0.1
            base_recovery += turbo_bonus
        
        self.cognitive_energy += base_recovery
        
        # === 5. 범위 제한 ===
        self.cognitive_energy = np.clip(
            self.cognitive_energy, 
            -5.0,
            self.energy_capacity
        )
        
        # === 6. 히스토리 기록 ===
        energy_ratio = self.cognitive_energy / self.energy_capacity
        self.energy_history.append(float(energy_ratio))
        
        # === 7. 긴급 상황 대응 ===
        if energy_ratio < 0.15:
            self.activation_level *= 0.70
        elif energy_ratio < 0.25:
            self.activation_level *= 0.85
        
        # === 8. 통계 업데이트 ===
        self.total_energy_consumed += total_expenditure
        self.total_actions_taken += 1
        if self.cognitive_energy < 30:
            self.low_energy_count += 1
        if self.activation_level > 0.7:
            self.high_activation_count += 1
        
        # 평균 업데이트
        if len(self.energy_history) > 0:
            self.avg_energy = float(np.mean(list(self.energy_history)[-100:])) * 100
        if len(self.activation_history) > 0:
            self.avg_activation = float(np.mean(list(self.activation_history)[-100:]))
        
        # MULTI-LOOP 브로드캐스트
        self._broadcast_energy_state()
    
    def _broadcast_energy_state(self):
        """Broadcast energy state to influence other modules"""
        if not self.message_bus:
            return
        
        energy_ratio = self.cognitive_energy / self.energy_capacity
        
        msg = Message(
            source='energy',
            target='broadcast',
            type='energy_state',
            payload={
                'energy': float(self.cognitive_energy),
                'ratio': float(energy_ratio),
                'activation': float(self.activation_level),
                'fatigue': float(self.fatigue_level),
                'avg_energy': self.avg_energy,
                'avg_activation': self.avg_activation
            },
            priority=0.9  # High priority - critical resource state
        )
        self.message_bus.send(msg)

    def should_continue(self) -> Tuple[bool, float]:
        """Determine if processing should continue and at what intensity
        
        Returns:
            (should_continue, processing_intensity)
        """
        # Critical energy shutdown - 임계점 낮춤 (더 오래 버팀)
        if self.cognitive_energy < -20.0:  # -5.0 -> -20.0
            return (False, 0.0)
        
        # Very low activation - minimal processing - 임계점 낮춤
        if self.activation_level < 0.01:  # 0.05 -> 0.01
            return (True, 0.1)  # 최소 강도도 올림
        
        # Normal operation
        processing_intensity = float(self.activation_level)
        
        # Reduce intensity if energy is critically low - 임계점 낮춤
        energy_ratio = self.cognitive_energy / self.energy_capacity
        if energy_ratio < 0.05:  # 0.1 -> 0.05 (더 낮아도 버팀)
            processing_intensity *= (energy_ratio / 0.05)
        
        return (True, float(np.clip(processing_intensity, 0.0, 1.0)))


class HebbianTrace:
    def __init__(self, dim: int, decay: float = 0.995):
        self.dim = int(dim)
        self.decay = float(decay)
        self.trace = np.zeros((self.dim,), dtype=np.float32)

    def update(self, vec: np.ndarray, scale: float = 1.0) -> None:
        v = np.asarray(vec, dtype=np.float32).ravel()
        if v.size < self.dim:
            v2 = np.zeros((self.dim,), dtype=np.float32)
            v2[:v.size] = v
            v = v2
        elif v.size > self.dim:
            v = v[:self.dim]
        # accumulate with decay
        self.trace = self.trace * self.decay + (v * float(scale))

    def read(self) -> np.ndarray:
        return self.trace.copy()

    def decay_step(self) -> None:
        self.trace *= self.decay


class RunningBaseline:
    def __init__(self, alpha: float = 0.01):
        self.alpha = float(alpha)
        self.value = 0.0
        self.initialized = False

    def update(self, x: float) -> float:
        if not self.initialized:
            self.value = float(x)
            self.initialized = True
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * float(x)
        return float(self.value)

    def get(self) -> float:
        return float(self.value)


class MetaController:
    """Stabilized meta-policy: linear actor + Hebbian regularization + slow/fast weights.

    API:
      - decide(features: np.ndarray) -> int (delta)
      - observe(reward: float, action: Optional[int]=None) -> None
      - save(path), load(path)
      - snapshot() -> dict for logging
    """
    
    def __init__(self,
                 in_dim: int = 6,
                 save_path: Optional[str] = None,
                 lr: float = 1e-3,
                 momentum: float = 0.9,
                 sigma: float = 8.0,
                 min_delta: int = 8,
                 max_delta: int = 2048,
                 wd: float = 1e-6,
                 hebb_dim: int = 128,
                 consolidation_alpha: float = 0.01,
                 target_W_norm: float = 1.0):
        self.in_dim = int(in_dim)
        self.save_path = save_path
        # fast actor weights
        self.W = np.zeros((self.in_dim,), dtype=np.float32)
        self.b = float(4.0)
        # slow weights for consolidation
        self.slow_W = np.zeros_like(self.W)
        # optimizer state
        self.m_W = np.zeros_like(self.W)
        self.m_b = 0.0
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.sigma = float(sigma)
        self.min_delta = int(min_delta)
        self.max_delta = int(max_delta)
        self.wd = float(wd)
        # hebbian
        self.hebb = HebbianTrace(dim=hebb_dim)
        self.consolidation_alpha = float(consolidation_alpha)
        self.target_W_norm = float(target_W_norm)
        # baseline
        self.baseline = RunningBaseline(alpha=0.01)
        # bookkeeping
        self.last_features = None
        self.last_action = None
        self.last_mu = None

    def decide(self, features: np.ndarray) -> int:
        f = np.asarray(features, dtype=np.float32).ravel()
        if f.size != self.in_dim:
            f2 = np.zeros((self.in_dim,), dtype=np.float32)
            f2[:min(f.size, self.in_dim)] = f[:min(f.size, self.in_dim)]
            f = f2
        # compute policy mean (fast weights) -- slow_W is used for consolidation and
        # not added directly to the fast mean here to avoid double-counting.
        mu = float(np.dot(f, self.W) + self.b)
        sampled = float(np.random.normal(loc=mu, scale=self.sigma))
        delta = int(round(sampled))
        delta = max(self.min_delta, min(self.max_delta, delta))
        # store
        self.last_features = f
        self.last_action = int(delta)
        self.last_mu = mu
        return int(delta)
        mu = float(np.dot(f, self.W) + self.b + float(np.dot(self.slow_W, 0.0)))
        sampled = float(np.random.normal(loc=mu, scale=self.sigma))
        delta = int(round(sampled))
        delta = max(self.min_delta, min(self.max_delta, delta))
        # store
        self.last_features = f
        self.last_action = int(delta)
        self.last_mu = mu
        return int(delta)

    def observe(self, reward: float, action: Optional[int] = None) -> None:
        if self.last_features is None:
            return
        a = float(action if action is not None else self.last_action)
        mu = float(self.last_mu)
        f = self.last_features
        base = self.baseline.update(reward)
        adv = float(reward - base)
        # score function grad
        if self.sigma <= 0:
            score = (a - mu)
        else:
            score = (a - mu) / (self.sigma * self.sigma)
        gW = adv * score * f - self.wd * self.W
        gb = adv * score - self.wd * self.b
        # clip grads
        gn = max(1e-12, np.linalg.norm(gW))
        clip = 1.0
        if gn > clip:
            gW = gW * (clip / gn)
        # momentum
        self.m_W = self.momentum * self.m_W + (1.0 - self.momentum) * gW
        self.m_b = self.momentum * self.m_b + (1.0 - self.momentum) * gb
        self.W += self.lr * self.m_W
        self.b += self.lr * self.m_b
        # hebbian update
        try:
            act_scale = float(a) / max(1.0, float(self.max_delta))
            self.hebb.update(f * act_scale, scale=1.0)
        except Exception:
            pass
        # tiny hebbian regularization
        try:
            hb = self.hebb.read()
            if hb is not None and hb.size > 0:
                hb_seg = hb[:self.in_dim].astype(float)
                nrm = np.linalg.norm(hb_seg) + 1e-12
                hb_norm = hb_seg / nrm
                reg_alpha = 1e-4
                self.W = (1.0 - reg_alpha) * self.W + reg_alpha * hb_norm
        except Exception:
            pass
        # synaptic scaling
        wn = np.linalg.norm(self.W) + 1e-12
        if wn > 0:
            self.W = self.W * (self.target_W_norm / wn)
        # adapt sigma
        if adv > 0:
            self.sigma = max(1.0, self.sigma * 0.999)
        else:
            self.sigma = min(float(self.max_delta), self.sigma * 1.001)
        # consolidation into slow weights on positive adv
        if adv > 0:
            hb = self.hebb.read()[:self.in_dim]
            self.slow_W = (1.0 - self.consolidation_alpha) * self.slow_W + self.consolidation_alpha * hb
        # clear
        self.last_features = None
        self.last_action = None
        self.last_mu = None

    def save(self, path: Optional[str] = None) -> None:
        p = path or self.save_path
        if not p:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.savez_compressed(p, W=self.W, b=self.b, slow_W=self.slow_W, sigma=self.sigma, baseline=self.baseline.get())

    def load(self, path: Optional[str] = None) -> None:
        p = path or self.save_path
        if not p or not os.path.exists(p):
            return
        d = np.load(p)
        try:
            self.W = d['W'].astype(np.float32)
        except Exception:
            pass
        try:
            self.b = float(d['b'])
        except Exception:
            pass
        try:
            if 'slow_W' in d.files:
                self.slow_W = d['slow_W'].astype(np.float32)
        except Exception:
            pass
        try:
            if 'sigma' in d.files:
                self.sigma = float(d['sigma'])
        except Exception:
            pass
        try:
            if 'baseline' in d.files:
                self.baseline.value = float(d['baseline'])
                self.baseline.initialized = True
        except Exception:
            pass

    def snapshot(self) -> Dict[str, Any]:
        return {
            'W_norm': float(np.linalg.norm(self.W)),
            'slow_W_norm': float(np.linalg.norm(self.slow_W)),
            'sigma': float(self.sigma),
            'baseline': float(self.baseline.get()),
        }


class ConsciousnessBus:
    """Lightweight broadcast bus for salient tokens.

    Tokens are dicts: {source,key,vector_len,salience,confidence,ttl}.
    Only top-K by salience are considered active for gating.
    """
    def __init__(
        self,
        top_k: int = 4,
        outdir: Optional[str] = None,
        async_dispatch: bool = False,
        max_queue: int = 4096,
        drop_policy: str = "drop_low_priority",
        min_dispatch_interval_ms: int = 0,
        default_topic: str = "bus",
    ):
        self.top_k = int(top_k)
        self.tokens: List[Dict[str, Any]] = []
        self._subs: Dict[int, Dict[str, Any]] = {}
        self._sub_seq: int = 0
        self._event_heap: List[Tuple[float, float, int, Dict[str, Any]]] = []
        self._event_seq: int = 0
        self._lock = threading.Lock()
        self._async_dispatch = bool(async_dispatch)
        self._worker: Optional[threading.Thread] = None
        self._worker_running: bool = False
        self._max_queue = int(max(1, max_queue))
        self._drop_policy = str(drop_policy or "drop_low_priority").strip().lower()
        self._min_dispatch_interval = float(max(0.0, min_dispatch_interval_ms)) / 1000.0
        self._last_dispatch_ts: float = 0.0
        self._default_topic = str(default_topic or "bus")
        self._stats: Dict[str, int] = {
            "published": 0,
            "dispatched": 0,
            "dropped": 0,
            "handler_errors": 0,
        }
        self._log_path = None
        if outdir is not None:
            try:
                os.makedirs(outdir, exist_ok=True)
                self._log_path = os.path.join(outdir, 'bus.jsonl')
            except Exception:
                self._log_path = None

    def _log(self, obj: Dict[str, Any]) -> None:
        if not self._log_path:
            return
        try:
            with open(self._log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _enqueue(self, event: Dict[str, Any]) -> bool:
        with self._lock:
            if len(self._event_heap) >= self._max_queue:
                if self._drop_policy == "drop_oldest":
                    oldest_idx = min(
                        range(len(self._event_heap)),
                        key=lambda i: float(self._event_heap[i][1]),
                    )
                    self._event_heap.pop(oldest_idx)
                    heapq.heapify(self._event_heap)
                    self._stats["dropped"] += 1
                else:
                    # drop_low_priority: replace one lowest-priority pending item only if incoming is higher priority.
                    lowest_idx = max(
                        range(len(self._event_heap)),
                        key=lambda i: float(self._event_heap[i][0]),
                    )
                    lowest_pri = -float(self._event_heap[lowest_idx][0])
                    if float(event.get("priority", 0.0)) <= lowest_pri:
                        self._stats["dropped"] += 1
                        return False
                    self._event_heap.pop(lowest_idx)
                    heapq.heapify(self._event_heap)
                    self._stats["dropped"] += 1
            self._event_seq += 1
            pri = float(event.get("priority", 0.0))
            ts = float(event.get("ts", time.time()))
            heapq.heappush(self._event_heap, (-pri, ts, int(self._event_seq), event))
        return True

    def _dispatch_event(self, event: Dict[str, Any]) -> None:
        if self._min_dispatch_interval > 0:
            dt = float(time.time()) - float(self._last_dispatch_ts)
            if dt < self._min_dispatch_interval:
                try:
                    time.sleep(self._min_dispatch_interval - dt)
                except Exception:
                    pass
        self._last_dispatch_ts = float(time.time())
        topic = str(event.get("topic", self._default_topic))
        pri = float(event.get("priority", 0.0))
        payload = event.get("payload", {})
        for _, sub in list(self._subs.items()):
            try:
                st = sub.get("topic")
                if st and str(st) != topic:
                    continue
                if pri < float(sub.get("min_priority", 0.0)):
                    continue
                pred = sub.get("predicate")
                if callable(pred) and (not bool(pred(event))):
                    continue
                handler = sub.get("handler")
                if callable(handler):
                    handler(payload)
            except Exception:
                self._stats["handler_errors"] += 1
                self._log(
                    {
                        "kind": "consciousness_bus_dispatch",
                        "event": "handler_error",
                        "topic": topic,
                        "priority": pri,
                        "ts": time.time(),
                    }
                )
        self._stats["dispatched"] += 1
        self._log(
            {
                "kind": "consciousness_bus_dispatch",
                "event": "dispatch",
                "topic": topic,
                "priority": pri,
                "ts": time.time(),
            }
        )

    def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: float = 0.0,
        async_dispatch: Optional[bool] = None,
    ) -> bool:
        event = {
            "topic": str(topic or self._default_topic),
            "payload": dict(payload or {}),
            "priority": float(priority),
            "ts": float(time.time()),
        }
        self._stats["published"] += 1
        use_async = self._async_dispatch if async_dispatch is None else bool(async_dispatch)
        if use_async:
            ok = self._enqueue(event)
            if ok and self._async_dispatch and (self._worker is None or not self._worker.is_alive()):
                self.start_async_worker()
            return bool(ok)
        self._dispatch_event(event)
        return True

    def subscribe(
        self,
        handler: Callable[[Dict[str, Any]], None],
        topic: Optional[str] = None,
        min_priority: float = 0.0,
        predicate: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> int:
        with self._lock:
            self._sub_seq += 1
            sid = int(self._sub_seq)
            self._subs[sid] = {
                "handler": handler,
                "topic": None if topic is None else str(topic),
                "min_priority": float(min_priority),
                "predicate": predicate,
            }
            return sid

    def unsubscribe(self, sub_id: int) -> bool:
        with self._lock:
            return self._subs.pop(int(sub_id), None) is not None

    def start_async_worker(self) -> None:
        if self._worker_running:
            return
        self._worker_running = True

        def _run():
            while self._worker_running:
                evt = None
                with self._lock:
                    if self._event_heap:
                        _, _, _, evt = heapq.heappop(self._event_heap)
                if evt is None:
                    time.sleep(0.002)
                    continue
                self._dispatch_event(evt)

        self._worker = threading.Thread(target=_run, daemon=True)
        self._worker.start()

    def stop_async_worker(self) -> None:
        self._worker_running = False
        if self._worker is not None:
            try:
                self._worker.join(timeout=1.0)
            except Exception:
                pass

    def drain(self, max_items: Optional[int] = None) -> int:
        dispatched = 0
        limit = int(max_items) if max_items is not None else 1_000_000
        while dispatched < max(0, limit):
            evt = None
            with self._lock:
                if self._event_heap:
                    _, _, _, evt = heapq.heappop(self._event_heap)
            if evt is None:
                break
            self._dispatch_event(evt)
            dispatched += 1
        return int(dispatched)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            q_len = int(len(self._event_heap))
            n_sub = int(len(self._subs))
        out = dict(self._stats)
        out.update(
            {
                "queue_len": q_len,
                "subscribers": n_sub,
                "top_k": int(self.top_k),
                "async_dispatch": bool(self._async_dispatch),
                "drop_policy": str(self._drop_policy),
            }
        )
        return out

    def push(self, source: str, key: str, vector: np.ndarray, salience: float, confidence: float = 1.0, ttl: int = 5) -> None:
        try:
            tok = {
                'ts': time.time(),
                'source': str(source),
                'key': str(key),
                'salience': float(salience),
                'confidence': float(confidence),
                'ttl': int(ttl),
                'vector_len': int(np.asarray(vector).size),
            }
            self.tokens.append(tok)
            if len(self.tokens) > self._max_queue:
                self.tokens = self.tokens[-self._max_queue:]
            self.publish(topic="token", payload=tok, priority=float(salience), async_dispatch=None)
            self._log(tok)
        except Exception:
            pass

    def step(self) -> None:
        nxt = []
        for t in self.tokens:
            t['ttl'] = int(t.get('ttl', 0) - 1)
            if t['ttl'] > 0:
                nxt.append(t)
        self.tokens = nxt
        if not self._async_dispatch:
            self.drain(max_items=256)

    def top(self) -> List[Dict[str, Any]]:
        return sorted(self.tokens, key=lambda d: d.get('salience', 0.0), reverse=True)[: self.top_k]

    def top_keys(self) -> List[str]:
        return [t.get('key', '') for t in self.top()]


class AutoNamer:
    """Deterministic English naming for specs/tokens with collision-free short hashes.

    - spec_name(source, packer, params, produced_dim) -> 'spec.<src>.<packer>.<hash>'
    - token_name_policy(ctx) -> e.g., 'policy.entropy_shift', 'policy.action_var_high'
    """
    def __init__(self):
        self._used: Set[str] = set()

    @staticmethod
    def _slug(text: str) -> str:
        try:
            t = unicodedata.normalize('NFKD', str(text))
            t = t.encode('ascii', 'ignore').decode('ascii')
        except Exception:
            t = str(text)
        t = re.sub(r'[^A-Za-z0-9]+', '_', t).strip('_').lower()
        return t or 'x'

    @staticmethod
    def _hash(obj: Dict[str, Any], n: int = 10) -> str:
        try:
            payload = json.dumps(obj, sort_keys=True, default=str).encode('utf-8')
            h = hashlib.blake2b(payload, digest_size=8).hexdigest()
            return h[:max(6, int(n))]
        except Exception:
            return '000000'

    def spec_name(self, spec: 'FeatureSpec') -> str:
        src = self._slug(getattr(spec, 'source', 'src'))
        pack = self._slug(getattr(spec, 'packer', 'p'))
        dim = int(getattr(spec, 'produced_dim', 1) or 1)
        h = self._hash({'source': str(getattr(spec, 'source', 'src')),
                        'packer': str(getattr(spec, 'packer', 'p')),
                        'params': getattr(spec, 'params', {}),
                        'dim': dim})
        base = f"spec.{src}.{pack}.{h}"
        name = base
        i = 1
        while name in self._used:
            name = f"{base}.{i}"
            i += 1
        self._used.add(name)
        return name

    def token_name_policy(self, delta_hat: float, var_a: float, dH: float, kl: float, clipped: bool) -> str:
        # Priority: entropy shift > action variance > delta spike > stable
        try:
            if abs(dH) > 0.5:
                return 'policy.entropy_shift'
            if var_a > 0.5:
                return 'policy.action_var_high'
            if abs(delta_hat) > 0.5:
                return 'policy.delta_spike'
            if clipped or kl > 0.02:
                return 'policy.update_constrained'
        except Exception:
            pass
        return 'policy.state'


# ---- Torch-based SDM (Transformer/MoE-MLP) implemented inline ----
try:
    import torch as _t
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    import torch.distributed as _dist
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP  # type: ignore
        from torch.distributed.fsdp.wrap import default_auto_wrap_policy as _fsdp_auto_wrap  # type: ignore
        _FSDP_OK = True
    except Exception:
        _FSDP_OK = False
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False
    _FSDP_OK = False


def _m3_torch_device() -> str:
    return resolve_torch_device_string(require_cuda=False)


if _TORCH_OK:
    def _moe_switch_aux_terms(logits: _t.Tensor, probs: _t.Tensor, top1_idx: _t.Tensor, n_experts: int) -> tuple[_t.Tensor, _t.Tensor, _t.Tensor, _t.Tensor]:
        p = probs.reshape(-1, int(n_experts))
        mean_prob = p.mean(dim=0)
        assign = _F.one_hot(top1_idx.reshape(-1), num_classes=int(n_experts)).float()
        mean_assign = assign.mean(dim=0)
        aux_lb = float(n_experts) * _t.sum(mean_prob * mean_assign)
        z_loss = _t.mean(logits.reshape(-1, int(n_experts)) ** 2)
        entropy = -_t.sum(mean_assign * _t.log(mean_assign + 1e-8))
        max_load = _t.max(mean_assign)
        return aux_lb, z_loss, entropy, max_load

    class _MoEExpert(_nn.Module):
        def __init__(self, d: int, d_ff: int):
            super().__init__()
            self.fc1 = _nn.Linear(d, d_ff)
            self.fc2 = _nn.Linear(d_ff, d)

        def forward(self, x: _t.Tensor) -> _t.Tensor:
            return self.fc2(_F.gelu(self.fc1(x)))

    class _MoEFFN(_nn.Module):
        def __init__(self, d: int, d_ff: int, n_experts: int = 8, top_k: int = 2):
            super().__init__()
            self.n_experts = int(n_experts)
            self.top_k = int(max(1, top_k))
            self.gate = _nn.Linear(d, n_experts)
            self.experts = _nn.ModuleList([_MoEExpert(d, d_ff) for _ in range(n_experts)])
            self.router_noise_std = float(os.environ.get("M3_MOE_ROUTER_NOISE_STD", "0.02"))
            self.router_z_loss_coef = float(os.environ.get("M3_MOE_ROUTER_Z_LOSS_COEF", "0.001"))

        def forward(self, x: _t.Tensor) -> tuple[_t.Tensor, _t.Tensor, dict]:
            logits = self.gate(x)
            if self.training and self.router_noise_std > 0:
                logits = logits + _t.randn_like(logits) * float(self.router_noise_std)
            probs = _F.softmax(logits, dim=-1)
            topv, topi = _t.topk(probs, k=min(self.top_k, self.n_experts), dim=-1)
            y = _t.zeros_like(x)
            B = x.size(0)
            for k in range(topv.size(-1)):
                idx = topi[:, k]
                w = topv[:, k].unsqueeze(-1)
                for e in range(self.n_experts):
                    m = (idx == e)
                    if m.any():
                        xe = x[m]
                        ye = self.experts[e](xe)
                        y[m] = y[m] + w[m] * ye
            aux_lb, z_loss, entropy, max_load = _moe_switch_aux_terms(logits, probs, topi[:, 0], self.n_experts)
            aux = aux_lb + float(self.router_z_loss_coef) * z_loss
            stats = {
                'expert_usage_entropy': entropy.detach(),
                'max_expert_load': max_load.detach(),
                'aux_lb': aux_lb.detach(),
                'router_z_loss': z_loss.detach(),
            }
            return y, aux, stats

    class _SDMModel(_nn.Module):
        def __init__(self, in_dim: int, z_dim: int, d_model: int = 2048, n_layers: int = 16,
                     d_ff: int = 8192, n_experts: int = 0, top_k: int = 2, a_dim: int = 5,
                     shared_repr: Optional[object] = None):
            super().__init__()
            self.in_dim = int(in_dim)
            self.z_dim = int(z_dim)
            self.a_dim = int(a_dim)
            self.shared_repr = shared_repr  # optional TorchSharedRepr
            self.d_model = int(d_model)
            self.proj_z = _nn.Linear(self.z_dim, self.d_model)
            self.proj_a = _nn.Linear(self.a_dim, self.d_model)
            self.layers = _nn.ModuleList([
                (_nn.Sequential(_nn.LayerNorm(self.d_model),
                                _nn.Linear(self.d_model, d_ff), _nn.GELU(), _nn.Linear(d_ff, self.d_model))
                 if n_experts <= 0 else _MoEFFN(self.d_model, d_ff, n_experts=n_experts, top_k=top_k))
                for _ in range(int(n_layers))
            ])
            self.norm = _nn.LayerNorm(self.d_model)
            # heads: 4 scalars + z_next
            self.h_delta = _nn.Linear(self.d_model, 1)
            self.h_stab = _nn.Linear(self.d_model, 1)
            self.h_meta = _nn.Linear(self.d_model, 1)
            self.h_rew = _nn.Linear(self.d_model, 1)
            self.h_znxt = _nn.Linear(self.d_model, self.z_dim)

        def forward(self, x: _t.Tensor) -> tuple[dict, _t.Tensor]:
            # x expected shape [B, z_dim + a_dim]; split
            z = x[:, :self.z_dim]
            a = x[:, self.z_dim:self.z_dim + self.a_dim]
            # apply shared repr if present
            if self.shared_repr is not None:
                try:
                    z = self.shared_repr(z)
                except Exception:
                    pass
            h = self.proj_z(z) + self.proj_a(a)
            lb = h.new_tensor(0.0)
            ent_sum = h.new_tensor(0.0)
            max_load = h.new_tensor(0.0)
            n_moe = 0
            for f in self.layers:
                if isinstance(f, _MoEFFN):
                    h_in = h
                    hf, lbi, stats = f(h)
                    h = h_in + hf
                    lb = lb + lbi
                    if isinstance(stats, dict):
                        if 'expert_usage_entropy' in stats:
                            ent_sum = ent_sum + stats['expert_usage_entropy']
                        if 'max_expert_load' in stats:
                            max_load = _t.maximum(max_load, stats['max_expert_load'])
                        n_moe += 1
                else:
                    h = h + f(h)
            h = self.norm(h)
            if n_moe > 0:
                self.last_expert_usage_entropy = ent_sum / float(n_moe)
                self.last_max_expert_load = max_load
            else:
                self.last_expert_usage_entropy = h.new_tensor(0.0)
                self.last_max_expert_load = h.new_tensor(0.0)
            out = {
                'delta_hat': self.h_delta(h).squeeze(-1),
                'stability': self.h_stab(h).squeeze(-1),
                'meta': self.h_meta(h).squeeze(-1),
                'reward': self.h_rew(h).squeeze(-1),
                'z_next': self.h_znxt(h)
            }
            return out, lb

    class TorchSDM:
        def __init__(self, in_dim: int, z_dim: int, d_model: int = 2048, n_layers: int = 16,
                      d_ff: int = 8192, n_experts: int = 64, top_k: int = 2, fsdp: bool = True,
                      device: Optional[str] = None, a_dim: int = 5, shared_repr: Optional[object] = None):
            if not _TORCH_OK:
                raise RuntimeError('PyTorch not available')
            self.in_dim = int(in_dim)
            self.z_dim = int(z_dim)
            self.a_dim = int(a_dim)
            dev = _t.device(device or _m3_torch_device())
            self.device = dev
            self.model = _SDMModel(in_dim, z_dim, d_model=d_model, n_layers=n_layers,
                                   d_ff=d_ff, n_experts=n_experts, top_k=top_k, a_dim=self.a_dim,
                                   shared_repr=shared_repr).to(dev)
            self.fsdp = bool(fsdp and _FSDP_OK)
            if self.fsdp and not _dist.is_initialized():
                try:
                    backend = 'nccl' if dev.type == 'cuda' else 'gloo'
                    _dist.init_process_group(backend=backend, init_method='tcp://127.0.0.1:29502', rank=0, world_size=1)
                except Exception:
                    pass
            if self.fsdp and _dist.is_initialized():
                self.model = _FSDP(self.model, auto_wrap_policy=_fsdp_auto_wrap)
            wd = float(os.environ.get("M3_STABILITY_WEIGHT_DECAY", "0.01"))
            self.optim = _t.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=wd)
            self.scaler = _GradScaler(enabled=(dev.type == 'cuda'))
            self._last_expert_usage_entropy = 0.0
            self._last_max_expert_load = 0.0
            self._stability_skip_steps = 0

        def forward(self, x: np.ndarray) -> dict:
            self.model.eval()
            with _t.no_grad(), _autocast(self.device.type == 'cuda'):
                xt = _t.as_tensor(np.asarray(x, np.float32).reshape(1, -1), device=self.device)
                out, _ = self.model(xt)
                return {
                    'delta_hat': float(out['delta_hat'].squeeze(0).item()),
                    'stability': float(out['stability'].squeeze(0).item()),
                    'meta': float(out['meta'].squeeze(0).item()),
                    'reward': float(out['reward'].squeeze(0).item()),
                    'z_next': out['z_next'].squeeze(0).detach().cpu().numpy().astype(np.float32),
                }

        def train_batch(self, X: np.ndarray, Y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
            self.model.train()
            xt = _t.as_tensor(np.asarray(X, np.float32), device=self.device)
            yt = _t.as_tensor(np.asarray(Y, np.float32), device=self.device)
            head_w = None
            if w is not None:
                head_w = _t.as_tensor(np.asarray(w, np.float32), device=self.device).view(1, -1)
            with _autocast(self.device.type == 'cuda'):
                out, lb = self.model(xt)
                B = xt.size(0)
                y_heads = yt[:, :4]
                y_z = yt[:, 4:4 + self.z_dim]
                pred_heads = _t.stack([
                    out['delta_hat'], out['stability'], out['meta'], out['reward']
                ], dim=1)
                mse_h = _F.mse_loss(pred_heads, y_heads)
                mse_z = _F.mse_loss(out['z_next'], y_z)
                loss = mse_h + mse_z + 1e-2 * lb
                if head_w is not None:
                    loss = (head_w * (pred_heads - y_heads) ** 2).mean() + mse_z + 1e-2 * lb
            self.scaler.scale(loss).backward()
            if self.device.type == 'cuda':
                self.scaler.unscale_(self.optim)
            _nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            finite_ok = True
            for p in self.model.parameters():
                if p.grad is not None and not _t.isfinite(p.grad).all():
                    finite_ok = False
                    break
                if not _t.isfinite(p.data).all():
                    finite_ok = False
                    break
            if finite_ok:
                self.scaler.step(self.optim)
            else:
                self._stability_skip_steps += 1
            self.scaler.update()
            self.optim.zero_grad(set_to_none=True)
            if finite_ok:
                max_w = float(os.environ.get("M3_STABILITY_MAX_WEIGHT_NORM", "10.0"))
                with _t.no_grad():
                    for p in self.model.parameters():
                        if p.ndim <= 1:
                            continue
                        n = p.data.norm()
                        if _t.isfinite(n) and float(n.item()) > max_w:
                            p.data.mul_(max_w / (n + 1e-8))
            try:
                self._last_expert_usage_entropy = float(getattr(self.model, 'last_expert_usage_entropy', _t.tensor(0.0)).detach().item())
            except Exception:
                self._last_expert_usage_entropy = 0.0
            try:
                self._last_max_expert_load = float(getattr(self.model, 'last_max_expert_load', _t.tensor(0.0)).detach().item())
            except Exception:
                self._last_max_expert_load = 0.0
            return float(loss.detach().item())

        def save(self, path: str) -> None:
            try:
                p = path
                if p.endswith('.npz'):
                    p = p[:-4] + '.pt'
                _t.save({'in_dim': self.in_dim, 'z_dim': self.z_dim, 'state': self.model.state_dict()}, p)
            except Exception:
                pass

        def load(self, path: str) -> bool:
            try:
                p = path[:-4] + '.pt' if path.endswith('.npz') else path
                sd = _t.load(p, map_location=self.device)
                self.model.load_state_dict(sd['state'])
                return True
            except Exception:
                return False


class M3ConsciousnessCore:


    def __init__(self, n: int=512, K: int=12, seed=None, max_iterations=None, outdir: str='docs&tests&data_sets/tests/logs'):
        self.rngr = RNGRegistry(seed)
        if seed is None:
            seed = int(time.time() * 1000) % 2 ** 32
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n = n
        self.K = K
        self.max_iterations: Optional[int] = max_iterations
        self.outdir = outdir
        self._adaptive_threshold_cfg: Dict[str, Any] = {
            "enabled": True,
            "warmup_steps": 256,
            "quantile_low": 0.50,
            "quantile_mid": 0.75,
            "quantile_high": 0.90,
            "announce_cooldown": 200,
            "announce_hysteresis": 0.02,
            "phi_floor_min": 0.005,
            "phi_floor_max": 0.05,
        }
        self._observation_adapter_cfg: Dict[str, Any] = {
            "enabled": True,
            "target_policy_dim": 0,
            "projection_eps": 1e-6,
            "allow_policy_recreate": False,
        }
        self._consciousness_bus_cfg: Dict[str, Any] = {
            "enabled": True,
            "async_dispatch": True,
            "max_queue": 4096,
            "drop_policy": "drop_low_priority",
            "min_dispatch_interval_ms": 0,
            "default_topic": "bus",
        }
        try:
            from llm_adapter.config import get_global_config as _llm_get_cfg
            _cfg = _llm_get_cfg()
            for _k in list(self._adaptive_threshold_cfg.keys()):
                self._adaptive_threshold_cfg[_k] = getattr(getattr(_cfg, "adaptive_threshold", object()), _k, self._adaptive_threshold_cfg[_k])
            for _k in list(self._observation_adapter_cfg.keys()):
                self._observation_adapter_cfg[_k] = getattr(getattr(_cfg, "observation_adapter", object()), _k, self._observation_adapter_cfg[_k])
            for _k in list(self._consciousness_bus_cfg.keys()):
                self._consciousness_bus_cfg[_k] = getattr(getattr(_cfg, "consciousness_bus", object()), _k, self._consciousness_bus_cfg[_k])
        except Exception:
            pass
        self._phi_policy_last: Optional[Dict[str, float]] = None
        self._phi_policy_last_t: int = -10**9
        self._last_phi_announce_t: int = -10**9
        self._last_phi_announce_value: float = -1e9
        self._obs_adapter_W: Optional[np.ndarray] = None
        self._obs_adapter_in_dim: int = 0
        self._obs_adapter_out_dim: int = 0
        self._obs_adapter_last_sig: Optional[Tuple[int, int]] = None
        
        # MULTI-LOOP INFRASTRUCTURE: Create message bus FIRST
        self.message_bus = MessageBus(capacity=10000)
        print("[M3] Multi-loop message bus initialized - enabling bidirectional module communication")
        
        # Reward System Integration
        self.rewards = RewardSystem(hidden_dim=self.n, affect_dim=5)
        print("[M3] RewardSystem initialized (DriveTier + AffectKernel)")

        self.conceptual_space = ConceptualSpace()
        self.global_workspace = GlobalWorkspace(capacity=3, message_bus=self.message_bus)
        self.self_model = RecursiveSelfModel(n, K)
        
        # META-COGNITIVE NETWORK: Neural-based meta-awareness learning (integrated internally)
        self.meta_network = MetaCognitiveNetwork(input_dim=16, hidden_dim=64, output_dim=8)
        self.self_model.meta_network = self.meta_network
        self.self_model.use_neural_metacog = True
        print("[M3] Meta-cognitive network integrated - learning-based belief updates enabled")
        
        self.unified_subject = UnifiedSubject(subject_id='M3.5_System')
        self.episodic_memory = EpisodicMemory(max_memories=1000)
        self._last_policy_recommendations = {}
        self._last_utterance: str = ''
        self._chat_history = deque(maxlen=256)
        
        # Create phi calculator with message bus
        self.phi_calculator = IITPhiCalculator(n_elements=K, message_bus=self.message_bus)
        self.iit_enabled = True
        self.growing_som = GrowingSOM(input_dim=5, initial_size=2, rng=self.rngr.get('growing_som'))
        
        # Create modules with message bus integration
        self.qualia = QualiaState(message_bus=self.message_bus)
        self.meta_meta = MetaMetaMonitor()
        self.energy_ctrl = EnergyBasedController(initial_energy=100.0, message_bus=self.message_bus)
        self.goal_gen = GoalGenerator()
        self.event_queue: List[Event] = []
        self.operator_engine = StructuralOperatorEngine()
        self.experiment_designer = AutonomousExperimentDesigner(self.operator_engine)
        self.long_term_planner = LongTermPlanner(experiment_designer=self.experiment_designer, operator_engine=self.operator_engine)
        # Policy + Env integration
        self.env = SimpleBanditEnv()
        self._prev_meta = 0.5
        # Consciousness Bus
        try:
            if bool(self._consciousness_bus_cfg.get("enabled", True)):
                self.bus = ConsciousnessBus(
                    top_k=int(os.environ.get("M3_BUS_TOPK", "4")),
                    outdir=outdir,
                    async_dispatch=bool(self._consciousness_bus_cfg.get("async_dispatch", True)),
                    max_queue=int(max(1, int(self._consciousness_bus_cfg.get("max_queue", 4096)))),
                    drop_policy=str(self._consciousness_bus_cfg.get("drop_policy", "drop_low_priority")),
                    min_dispatch_interval_ms=int(max(0, int(self._consciousness_bus_cfg.get("min_dispatch_interval_ms", 0)))),
                    default_topic=str(self._consciousness_bus_cfg.get("default_topic", "bus")),
                )
                if bool(self._consciousness_bus_cfg.get("async_dispatch", True)):
                    self.bus.start_async_worker()
            else:
                self.bus = None
        except Exception:
            self.bus = None
        try:
            use_torch = os.environ.get('M3_TORCH', '0') in ('1', 'true', 'TRUE')
        except Exception:
            use_torch = False
        # Optional shared Torch representation (policy + SDM share)
        self._shared_repr_torch = None
        try:
            if os.environ.get('M3_SHARED_REPR_TORCH', '0') in ('1','true','TRUE'):
                from m3.torch_policy import TorchSharedRepr  # type: ignore
                fb_dim = getattr(self, 'feature_bank', None).max_dim if getattr(self, 'feature_bank', None) else 12
                self._shared_repr_torch = TorchSharedRepr(dim=int(fb_dim),
                                                         d_hidden=int(os.environ.get('M3_SHR_DH','512')),
                                                         n_layers=int(os.environ.get('M3_SHR_LAYERS','2')),
                                                         d_ff=int(os.environ.get('M3_SHR_DFF','2048')),
                                                         n_experts=int(os.environ.get('M3_SHR_EXP','8')),
                                                         top_k=int(os.environ.get('M3_SHR_TOPK','2')))
        except Exception:
            self._shared_repr_torch = None
        try:
            if os.environ.get('M3_TORCH_BR', '0') in ('1', 'true', 'TRUE'):
                try:
                    from m3.torch_policy import BRPolicy, TorchSharedRepr  # type: ignore
                    # Optional shared Torch repr
                    shared_mod = None
                    try:
                        if os.environ.get('M3_SHARED_REPR_TORCH', '0') in ('1','true','TRUE'):
                            shared_mod = TorchSharedRepr(dim=(getattr(self, 'feature_bank', None).max_dim if getattr(self, 'feature_bank', None) else 12), d_hidden=int(os.environ.get('M3_SHR_DH','512')), n_layers=int(os.environ.get('M3_SHR_LAYERS','2')), d_ff=int(os.environ.get('M3_SHR_DFF','2048')), n_experts=int(os.environ.get('M3_SHR_EXP','8')), top_k=int(os.environ.get('M3_SHR_TOPK','2')))
                    except Exception:
                        shared_mod = None
                    in_dim = getattr(self, 'feature_bank', None).max_dim if getattr(self, 'feature_bank', None) else 12
                    self.policy = BRPolicy(in_dim=in_dim, out_dim=12, trunk_dim=int(os.environ.get('M3_BR_TRUNK', '512')), shared_repr=self._shared_repr_torch)
                    use_torch = False  # skip TorchPolicy path
                except Exception:
                    pass
            if use_torch:
                try:
                    from m3.torch_policy import TorchPolicy, TorchSharedRepr  # type: ignore
                    # Optional shared Torch repr
                    shared_mod = None
                    try:
                        if os.environ.get('M3_SHARED_REPR_TORCH', '0') in ('1','true','TRUE'):
                            shared_mod = TorchSharedRepr(dim=(getattr(self, 'feature_bank', None).max_dim if getattr(self, 'feature_bank', None) else 12), d_hidden=int(os.environ.get('M3_SHR_DH','512')), n_layers=int(os.environ.get('M3_SHR_LAYERS','2')), d_ff=int(os.environ.get('M3_SHR_DFF','2048')), n_experts=int(os.environ.get('M3_SHR_EXP','8')), top_k=int(os.environ.get('M3_SHR_TOPK','2')))
                    except Exception:
                        shared_mod = None
                    in_dim = getattr(self, 'feature_bank', None).max_dim if getattr(self, 'feature_bank', None) else 12
                    # Default biggish MoE; tune via env if needed
                    d_model = int(os.environ.get('M3_TORCH_DMODEL', '2048'))
                    n_layers = int(os.environ.get('M3_TORCH_LAYERS', '8'))
                    d_ff = int(os.environ.get('M3_TORCH_DFF', '8192'))
                    n_experts = int(os.environ.get('M3_TORCH_EXPERTS', '16'))
                    top_k = int(os.environ.get('M3_TORCH_TOPK', '2'))
                    self.policy = TorchPolicy(in_dim=in_dim, out_dim=12, d_model=d_model, n_layers=n_layers,                                               d_ff=d_ff, n_experts=n_experts, top_k=top_k,                                              device=None, fsdp=True, shared_repr=self._shared_repr_torch)
                except Exception:
                    use_torch = False
            if not use_torch:
                pol_ckpt = os.path.join(outdir, 'policy.npz')
                if os.path.exists(pol_ckpt):
                    with np.load(pol_ckpt) as data:
                        if 'theta' in data.files:
                            # adopt linear into MLP with out_dim=12
                            in_dim = int(data['theta'].shape[1])
                            self.policy = PolicyMLP(in_dim=in_dim if getattr(self, 'feature_bank', None) is None else self.feature_bank.max_dim, out_dim=12, hidden=max(128, in_dim), rng=self.rngr.get('policy'))
                            self.policy.adopt_linear(data['theta'].astype(np.float32))
                            try:
                                self.policy.spectral_clip(c=2.0, c_v=1.0)
                            except Exception:
                                pass
                        elif 'W1' in data.files and 'W2' in data.files:
                            in_dim = int(data['W1'].shape[1])
                            out_dim = int(data['W2'].shape[0])
                            self.policy = PolicyMLP(in_dim=in_dim, out_dim=out_dim, hidden=int(data['W1'].shape[0]), rng=self.rngr.get('policy'))
                            self.policy.W1 = data['W1'].astype(np.float32); self.policy.b1 = data['b1'].astype(np.float32)
                            self.policy.W2 = data['W2'].astype(np.float32); self.policy.b2 = data['b2'].astype(np.float32)
                            if 'Wv' in data.files and 'bv' in data.files:
                                self.policy.Wv = data['Wv'].astype(np.float32); self.policy.bv = data['bv'].astype(np.float32)
                            try:
                                self.policy.spectral_clip(c=2.0, c_v=1.0)
                            except Exception:
                                pass
                        else:
                            raise RuntimeError('Unknown policy snapshot format')
                else:
                    in_dim = getattr(self, 'feature_bank', None).max_dim if getattr(self, 'feature_bank', None) else 12
                    self.policy = PolicyMLP(in_dim=in_dim, out_dim=12, hidden=max(128, in_dim), rng=self.rngr.get('policy'))
                    try:
                        self.policy.spectral_clip(c=2.0, c_v=1.0)
                    except Exception:
                        pass
        except Exception:
            in_dim = getattr(self, 'feature_bank', None).max_dim if getattr(self, 'feature_bank', None) else 12
            self.policy = PolicyMLP(in_dim=in_dim, out_dim=12, hidden=max(128, in_dim), rng=self.rngr.get('policy'))
            try:
                self.policy.spectral_clip(c=2.0, c_v=1.0)
            except Exception:
                pass
        # Policy snapshots & stats (single best checkpoint policy.npz)
        from collections import deque as _deque
        self.policy_reward_history = _deque(maxlen=1000)
        self._best_policy_score = float('-inf')
        try:
            # Initialize best score from existing policy.npz if present
            pol_last = os.path.join(self.outdir, 'policy.npz')
            if os.path.exists(pol_last):
                try:
                    with np.load(pol_last) as _npz:
                        self._best_policy_score = float(_npz.get('recent_avg', float('-inf')))
                except Exception:
                    pass
            # Remove legacy best file if it exists
            legacy_best = os.path.join(self.outdir, 'policy_best.npz')
            if os.path.exists(legacy_best):
                try:
                    os.remove(legacy_best)
                except Exception:
                    pass
        except Exception:
            pass
        self.reward_scheduler = GlobalRewardBudgetScheduler(initial_budgets={ResourceType.COMPUTE: 1000.0, ResourceType.MEMORY: 500.0, ResourceType.EXPERIMENTS: 100.0, ResourceType.REVISIONS: 50.0, ResourceType.TIME: 3600.0})
        self.visualizer = EvolutionVisualizer()
        self.network_visualizer = None
        # Growth safety defaults
        self._last_growth_time: float = 0.0
        self.growth_cooldown_sec: float = float(os.environ.get('M3_GROWTH_COOLDOWN', '300'))
        self.growth_max_added_dim_per_call: int = int(os.environ.get('M3_GROWTH_MAX_ADD', '256'))
        # hard cap for total feature dims (can be tuned externally)
        self._growth_hard_cap: int = int(os.environ.get('M3_GROWTH_HARD_CAP', '4096'))
        # Enable sandbox evaluation before committing auto-growth
        self.enable_growth_sandbox: bool = True
        # Learned meta-feature proposer (initialized on demand)
        self.meta_feature_controller = None
        # Running EMAs for normalizing sandbox metrics used in meta-reward
        try:
            self._meta_metric_ema = {
                'reward_mean': RunningBaseline(alpha=0.02),
                'td_error': RunningBaseline(alpha=0.02),
                'phi_median': RunningBaseline(alpha=0.02),
                'stability': RunningBaseline(alpha=0.02),
            }
        except Exception:
            self._meta_metric_ema = {}
        # Growth trigger policy (learn when to run meta-proposals)
        try:
            self.growth_trigger = GrowthTrigger(in_dim=64, hidden=64, rng=self.rngr.get('growth_trigger'), lr=1e-3)
        except Exception:
            try:
                self.growth_trigger = GrowthTrigger()
            except Exception:
                self.growth_trigger = None
        # Feature bank for high-dimensional observations
        try:
            self.feature_bank = FeatureBank(max_dim=128, embed_dim=32)
            try:
                self.feature_bank.use_bus_gating = os.environ.get('M3_BUS_GATING', '0') in ('1', 'true', 'TRUE')
            except Exception:
                pass
            try:
                self.feature_bank.set_log_paths(self.outdir)
                self.feature_bank._core_for_prune = self
            except Exception:
                pass
            # Skills manager (bus patterns -> gating priors)
            try:
                self.skills = SkillsManager(window=256, bias=0.2)
            except Exception:
                self.skills = None
            # Shared representation (optional)
            try:
                if os.environ.get('M3_SHARED_REPR', '0') in ('1', 'true', 'TRUE'):
                    self.shared_repr = SharedRepresentation(dim=self.feature_bank.max_dim)
                else:
                    self.shared_repr = None
            except Exception:
                self.shared_repr = None
        except Exception:
            self.feature_bank = None
        # Self-dynamics predictor (TorchSDM when enabled) with snapshot
        try:
            act_dim = 5  # [lr, gate, kd, explore, complexity]
            fb_dim = getattr(self, 'feature_bank').max_dim if getattr(self, 'feature_bank', None) else 12
            self._sdm_in_dim = int(fb_dim + act_dim)
            use_torch_sdm = os.environ.get('M3_TORCH_SDM', '0') in ('1', 'true', 'TRUE')
            if use_torch_sdm and _TORCH_OK:
                try:
                    d_model = int(os.environ.get('M3_SDM_DMODEL', '2048'))
                    n_layers = int(os.environ.get('M3_SDM_LAYERS', '16'))
                    d_ff = int(os.environ.get('M3_SDM_DFF', '8192'))
                    n_experts = int(os.environ.get('M3_SDM_MOE_EXPERTS', '64'))
                    top_k = int(os.environ.get('M3_SDM_MOE_TOPK', '2'))
                    use_fsdp = os.environ.get('M3_SDM_FSDP', '1') in ('1', 'true', 'TRUE')
                    self.sdm = TorchSDM(in_dim=self._sdm_in_dim, z_dim=fb_dim, d_model=d_model, n_layers=n_layers,
                                        d_ff=d_ff, n_experts=n_experts, top_k=top_k, fsdp=use_fsdp,
                                        a_dim=5, shared_repr=self._shared_repr_torch)
                except Exception:
                    self.sdm = None
            if getattr(self, 'sdm', None) is None:
                self.sdm = SelfDynamicsModel(in_dim=self._sdm_in_dim, hidden=max(128, fb_dim), rng=self.rngr.get('sdm'), z_dim=fb_dim)
        except Exception:
            self.sdm = None
        # Simple replay buffer for SDM
        self._sdm_replay: list[tuple[np.ndarray, np.ndarray]] = []
        self._sdm_capacity = 20000
        # Vision options
        # Self-evolving policy hooks (lightweight program synthesis of small rules)
        self._policy_rules: List[Dict[str, Any]] = []
        self._last_synthesis_at: int = 0
        # ARC search config (mutable by self-research)
        self._arc_cfg: Dict[str, Any] = {
            'beam_scale': 1.0,
            'depth_cap': 10,
            'score_weights': {'w_hard': 1.0, 'w_soft': 0.5, 'w_soft_min': 0.25, 'w_consistency': 0.2, 'w_len': 0.05},
            'include_ops': {
                'id': True,
                'rot': True,
                'flip': True,
                'transpose': True,
                'remap_hist': True,
                'keep_largest': True,
                'crop_largest': True,
                'bbox_fill_outmode': True,
                'obj_translate': True,
                'obj_paint': True,
                'macros': True,
            },
        }
        # ARC cache for successful programs / macros
        self._arc_cache: Dict[str, Any] = {'solved': {}, 'macro_counts': {}}
        self._arc_macros: List[List[str]] = []
        # Initialize dynamics attributes explicitly
        self.groups = [np.arange(i * self.n // 4, (i + 1) * self.n // 4) for i in range(4)]
        self.h = self.rng.normal(0, 0.1, size=self.n)
        self.U = self.rng.normal(0, 0.2, size=(self.K, 5))
        self.l_obs = np.zeros(self.K)
        self.l_ctrl = np.zeros(self.K)
        self.gate_mid = 0.7
        self.kd_eff = 5.1
        self.P_obs_history: deque = deque(maxlen=1000)
        self.base_vec = np.full(self.K, 1.0 / self.K)
        self.hi = 0
        self.lo = 1
        self.barrier_viols = 0
        self.stability_window: deque = deque(maxlen=50)
        os.makedirs(outdir, exist_ok=True)
        self.log_path = os.path.join(outdir, f'm3_log_seed{seed}.csv')
        self.event_log_path = os.path.join(outdir, f'm3_events_seed{seed}.csv')
        self.checkpoint_path = os.path.join(outdir, 'checkpoint.json')
        self.episodic_memory_path = os.path.join(outdir, 'episodic_memory.jsonl')
        self.chat_history_path = os.path.join(outdir, 'chat_history.jsonl')
        if os.path.exists(self.checkpoint_path):
            self._load_checkpoint()
        else:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
            if os.path.exists(self.event_log_path):
                os.remove(self.event_log_path)
            self.t = 0

        try:
            if os.path.exists(self.episodic_memory_path):
                self.episodic_memory.load(self.episodic_memory_path)
        except Exception:
            pass
        try:
            if os.path.exists(self.chat_history_path):
                self._load_chat_history(self.chat_history_path)
        except Exception:
            pass
        self.log_buffer: List[Dict] = []
        self.event_log_buffer: List[Dict] = []
        if not hasattr(self, 't'):
            self.t = 0
        self.world_state = self._get_current_world_state()
        # Self-vision (retina) state (Phase 1)
        self._vision_prev = None
        # Reward shaping coefficient (keep small; external real rewards should dominate when present)
        self.reward_shape_lambda = 0.2
        # Visualizer preferences (can be overridden by environment)
        # Overlay (attention crosshair / goal ring) off by default
        try:
            self.show_spatial_overlay = os.environ.get('M3_SHOW_SPATIAL_OVERLAY', '0') in ('1', 'true', 'TRUE')
        except Exception:
            self.show_spatial_overlay = False
        # Scope encoder one-bit mode (chessboard pattern)
        try:
            self.scope_one_bit = os.environ.get('M3_SCOPE_ONE_BIT', '0') in ('1', 'true', 'TRUE')
        except Exception:
            self.scope_one_bit = False
        # External vision source controls
        self.vision_mode = 'internal'  # 'internal'|'folder'|'camera'|'push'
        self._vision_frames: list[str] = []
        self._vision_idx = 0
        self._vision_camera = None
        self._pushed_frame = None
        self._vision_loop = True
        self._vision_shuffle = False
        # Spatial field: normalized attention point and optional goal in [0,1]^3
        self.attn_xy = (0.5, 0.5)
        self.attn_xyz = (0.5, 0.5, 0.5)
        self.spatial_goal = None  # e.g., {'xyz': (0.8,0.2,0.6), 'radius': 0.08}
        self.space_size = 256
        if hasattr(self, 'long_term_planner') and self.long_term_planner.current_goal is None:
            initial_goal = self.long_term_planner.create_goal(description='Achieve Basic Consciousness', mode=PlanningMode.EXPLORATION, success_criteria={'phi': 0.1, 'consciousness': 0.3, 'growth_rate': 0.01, 'memory_count': 10, 'spatial_goal_dist': 0.15}, max_duration=10000.0)
            # assign initial spatial goal to hottest region
            try:
                pem = self._scope_build_pred_err_map()
                gy, gx = divmod(int(np.argmax(pem)), pem.shape[1])
                gx_n = gx / max(1, pem.shape[1]-1)
                gy_n = gy / max(1, pem.shape[0]-1)
                # depth from initial contrast proxy
                retina0 = self._vision_build_retina(size=(64,64), foveate=False)
                _, _, _, depth0 = self._vision_features(retina0)
                gz_n = float(np.mean(depth0)) if depth0 is not None else 0.5
                self.set_spatial_goal3d(gx_n, gy_n, gz_n, radius=0.12)
            except Exception:
                pass
            self.t = 0

        # Auto-attach LLM adapter (자체 모델용 - 대화 학습 데이터 기록)
        try:
            auto_attach = os.getenv('M3_AUTO_ATTACH_LLM', '1').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            auto_attach = True
        if auto_attach:
            try:
                from llm_adapter import attach_llm_to_core
                attach_llm_to_core(self, adapter=None, record=True)
                print("[M3] 자체 LLM 어댑터 연결됨 (대화 학습 데이터 기록 활성화)")
                
                # Register llm_adapter module with MessageBus for credit assignment
                try:
                    self.message_bus.register_module('llm_adapter')
                    print("[M3] LLM adapter registered with MessageBus for temporal credit assignment")
                except Exception as e:
                    print(f"[M3] Failed to register llm_adapter with MessageBus: {e}")
            except Exception as e:
                # non-fatal; core still works without adapter
                print(f"[M3] LLM 어댑터 연결 실패 (비필수): {e}")

    def _get_current_world_state(self) -> Dict[str, Union[int, float, bool]]:
        """Compute a compact world-state summary for controller heuristics."""
        if len(self.P_obs_history) > 0:
            try:
                P_obs = self.P_obs_history[-1]
                delta_hat = float(0.5 * np.sum(np.abs(P_obs - self.base_vec)))
            except Exception:
                delta_hat = 1.0
        else:
            delta_hat = 1.0

        if len(self.stability_window) >= 10:
            try:
                recent_deltas = list(self.stability_window)[-10:]
                variance = np.var(recent_deltas)
                stability = float(1.0 - min(1.0, variance * 20.0))
            except Exception:
                stability = 0.5
        else:
            stability = 0.5

        # spatial metrics
        try:
            ax, ay = getattr(self, 'attn_xy', (0.5, 0.5))
            az = getattr(self, 'attn_xyz', (ax, ay, 0.5))[2]
            if getattr(self, 'spatial_goal', None) and ('xy' in self.spatial_goal or 'xyz' in self.spatial_goal):
                if 'xyz' in self.spatial_goal:
                    gx, gy, gz = self.spatial_goal['xyz']
                else:
                    gx, gy = self.spatial_goal['xy']
                    gz = az
                d = float(np.sqrt((ax - gx) ** 2 + (ay - gy) ** 2 + (az - gz) ** 2))
            else:
                d = None
        except Exception:
            ax, ay, az, d = 0.5, 0.5, 0.5, None

        energy_level = 0.0
        activation_level = 0.0
        meta_confidence = 0.0
        adaptation_success = False
        try:
            energy_level = float(self.energy_ctrl.cognitive_energy / self.energy_ctrl.energy_capacity)
            activation_level = float(self.energy_ctrl.activation_level)
            meta_confidence = float(self.self_model.meta_confidence)
        except Exception:
            pass

        return {
            't': int(self.t),
            'delta_hat': float(delta_hat),
            'm': 0.0,
            'stability': float(stability),
            'energy_level': energy_level,
            'activation_level': activation_level,
            'meta_confidence': float(meta_confidence),
            'qualia_valence': float(getattr(self.qualia, 'valence', 0.0)),
            'adaptation_success': bool(adaptation_success),
            'spatial_attn_x': float(ax),
            'spatial_attn_y': float(ay),
            'spatial_attn_z': float(az),
            'spatial_goal_dist': float(d) if d is not None else None,
        }

    def _collect_affect_state_for_llm(self):
        try:
            if hasattr(self, 'rewards') and getattr(self.rewards, 'last_affect', None) is not None:
                return self.rewards.last_affect
            if hasattr(self, 'affect_kernel'):
                for name in ('get_state', 'get_state_vector'):
                    fn = getattr(self.affect_kernel, name, None)
                    if callable(fn):
                        return fn()
        except Exception:
            pass
        return None

    def _collect_llm_memory(self, adapter, affect_state=None):
        mem = None
        panels_output = None
        if hasattr(self, 'feature_bank') and hasattr(self.feature_bank, 'panels'):
            try:
                panels_output = self.feature_bank.panels(self)
            except Exception:
                panels_output = None

        if panels_output is None:
            return None

        try:
            if hasattr(adapter, 'build_m3_memory'):
                mem_tokens = adapter.build_m3_memory(core=self, panels=panels_output, affect_state=affect_state)
            elif hasattr(adapter, 'model') and hasattr(adapter.model, 'm3_encoder') and adapter.model.m3_encoder is not None:
                mem_tokens = adapter.model.m3_encoder(panels_output, affect_state=affect_state)
            else:
                mem_tokens = None
            if mem_tokens is None:
                return None
            if torch.is_tensor(mem_tokens):  # type: ignore
                return mem_tokens.detach().cpu().numpy()
            return np.asarray(mem_tokens)
        except Exception:
            return None

    def _build_llm_prompt(self, fallback_text: str = "") -> str:
        prompt = ""
        if hasattr(self, '_chat_history'):
            try:
                turns = int(os.getenv('M3_CHAT_HISTORY_TURNS', '3'))
            except Exception:
                turns = 3
            recent = list(self._chat_history)[-max(1, 2 * turns):]
            for msg in recent:
                role = "User" if msg.get('role') == 'user' else "M3"
                text = msg.get('text', '')
                prompt += f"{role}: {text}\n"
        else:
            prompt = f"{fallback_text}\n" if fallback_text else ""
        prompt += "M3:"
        return prompt

    def _postprocess_llm_response(self, text: str) -> str:
        if not text:
            return text
        try:
            norm = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
        except Exception:
            return text
        if not norm:
            return norm

        # Remove consecutive duplicate lines.
        lines = [ln.rstrip() for ln in norm.split("\n")]
        out = []
        prev = None
        for ln in lines:
            if prev is not None and ln == prev:
                continue
            out.append(ln)
            prev = ln

        # Remove repeated paragraphs.
        paras = []
        seen = set()
        for block in "\n".join(out).split("\n\n"):
            blk = block.strip()
            if not blk:
                continue
            if blk in seen:
                continue
            seen.add(blk)
            paras.append(blk)
        cleaned = "\n\n".join(paras).strip()

        # Collapse repeated "M3:" prefixes.
        try:
            cleaned = cleaned.replace("M3: M3:", "M3:")
        except Exception:
            pass
        return cleaned

    def _is_backend_or_error_response(self, text: str) -> bool:
        try:
            s = str(text or "").strip()
        except Exception:
            return True
        if not s:
            return True
        sl = s.lower()
        prefixes = (
            "[HFBackend error:",
            "[HFBackend not enabled",
            "[Error: CUDA",
            "[Error: [HFBackend",
            "[Error:",
            "Local Error:",
            "Generation is in safe mode",
            "현재 생성 경로에 장애가 있어 안전 모드로 전환했습니다.",
            "当前生成路径出现故障，已切换到安全模式。",
            "Обнаружен сбой генерации, включен безопасный режим.",
            "[System: LLM Adapter not connected]",
        )
        if s.startswith(prefixes):
            return True
        markers = (
            "local error:",
            "hfbackend",
            "backend fault",
            "safe mode",
            "llm adapter not connected",
        )
        return any(m in sl for m in markers)

    def _is_disallowed_llm_response(self, text: str) -> bool:
        try:
            s = str(text or "").strip()
        except Exception:
            return True
        if not s:
            return True
        if self._is_backend_or_error_response(s):
            return True
        adapter = getattr(self, 'llm_adapter', None) or getattr(self, 'llm', None)
        try:
            if adapter is not None and hasattr(adapter, '_is_refusal_disclaimer'):
                if bool(adapter._is_refusal_disclaimer(s)):
                    return True
        except Exception:
            pass
        patterns = (
            "i am currently untrained",
            "please use the train button",
            "현재 생성 경로에 장애가 있어",
            "현재 생성 경로가 불안정",
            "training data",
            "현재 생성 경로가",
            "세션을 재시작",
            "train button",
            "i am an ai",
            "i'm an ai",
            "as an ai",
            "do not have feelings",
            "i don't have feelings",
            "language model",
            "cannot feel",
        )
        if any(p in s.lower() for p in patterns):
            return True
        return False

    def _verifier_tokens(self, text: str) -> List[str]:
        try:
            s = str(text or "").lower()
        except Exception:
            return []
        tokens = re.findall(r"[a-z0-9_]+|[가-힣]+", s)
        stop = {
            "the", "a", "an", "is", "are", "to", "and", "or", "of", "in", "on", "for", "with",
            "이", "그", "저", "은", "는", "이야", "그리고", "또는", "에서", "으로", "를", "을",
        }
        return [t for t in tokens if len(t) > 1 and t not in stop]

    def _evaluate_dialog_accuracy(self, user_msg: str, response: str) -> Dict[str, float]:
        if self._is_disallowed_llm_response(response):
            return {"score": 0.0, "overlap": 0.0, "penalty": 1.0}
        if self._is_backend_or_error_response(response):
            return {"score": 0.0, "overlap": 0.0, "penalty": 1.0}
        user_tokens = self._verifier_tokens(user_msg)
        resp_tokens = set(self._verifier_tokens(response))
        if not user_tokens:
            return {"score": 0.5, "overlap": 0.0, "penalty": 0.0}
        uniq_user = set(user_tokens)
        overlap = float(len(uniq_user & resp_tokens)) / max(1.0, float(len(uniq_user)))
        user_len = len(str(user_msg or "").strip())
        resp_len = len(str(response or "").strip())
        length_factor = float(np.clip(resp_len / max(24.0, user_len * 0.8), 0.0, 1.0))
        penalty = 0.0
        low_resp = str(response or "").lower()
        low_user = str(user_msg or "").lower()
        generic_phrases = (
            "ready for the next query",
            "please resend a short request",
            "i am currently untrained",
            "안전 모드",
            "짧게 다시 보내",
        )
        if any(p in low_resp for p in generic_phrases):
            penalty += 0.45
        asks_question = ("?" in str(user_msg or "")) or any(
            k in low_user for k in ("what", "why", "how", "explain", "무엇", "왜", "어떻게", "설명")
        )
        if asks_question and overlap < 0.15:
            penalty += 0.25
        score = float(np.clip(0.25 + 0.55 * overlap + 0.20 * length_factor - penalty, 0.0, 1.0))
        return {"score": score, "overlap": overlap, "penalty": penalty}

    def _apply_dialog_verifier_reward(self, user_msg: str, response: str) -> None:
        try:
            metrics = self._evaluate_dialog_accuracy(user_msg, response)
            score = float(metrics.get("score", 0.0))
            scale = float(os.getenv("M3_DIALOG_VERIFIER_SCALE", "0.3"))
            reward = float((score - 0.5) * 2.0 * scale)
            self._last_dialog_verifier_score = score
            self._last_dialog_verifier_reward = reward
            if hasattr(self, 'reward_history'):
                self.reward_history.append(reward)
            if hasattr(self, 'cumulative_reward'):
                self.cumulative_reward += reward
            if hasattr(self, 'reward_scheduler') and self.reward_scheduler is not None:
                try:
                    self.reward_scheduler.receive_reward(
                        RewardSignal(
                            source='dialog_accuracy_verifier',
                            value=reward,
                            metadata={
                                'score': score,
                                'overlap': float(metrics.get('overlap', 0.0)),
                                'penalty': float(metrics.get('penalty', 0.0)),
                            },
                        )
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _emit_llm_response(self, prompt: str, mem=None, affect_state=None, max_len: int = 100, default_on_error: bool = False):
        adapter = getattr(self, 'llm_adapter', None) or getattr(self, 'llm', None)
        if adapter is None:
            return None, "[System: LLM Adapter not connected]"

        try:
            response = adapter.generate(
                prompt,
                mem=mem,
                affect_state=affect_state,
                max_len=max_len,
            )
            response = self._postprocess_llm_response(response)
        except Exception as e:
            if default_on_error:
                logging.debug(f"LLM adapter error: {e}")
            return None, str(e)

        if not response or not response.strip():
            if default_on_error:
                return None, "EMPTY"
            return None, "EMPTY"

        response = response.strip()
        if self._is_disallowed_llm_response(response):
            return None, response
        self._chat_history.append({'role': 'assistant', 'text': response, 't': int(getattr(self, 't', 0))})
        if getattr(self, 'bus', None) is not None:
            try:
                vec_resp = self.feature_bank._hash_embed(response, self.feature_bank.embed_dim) if getattr(self, 'feature_bank', None) else np.zeros((32,), np.float32)
                self.bus.push('system', 'utter.self', vec_resp.astype(np.float32), salience=0.8, confidence=1.0, ttl=10)
            except Exception:
                pass
        return response, None

    def _memory_semantic_prefix(self) -> str:
        try:
            enabled = os.getenv('M3_EMBED_PERSPECTIVE', '0').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            enabled = False
        if not enabled:
            return ""
        try:
            subj = getattr(self, 'unified_subject', None)
            if subj is None or not hasattr(subj, 'reflect_on_self'):
                return ""
            summary = str(subj.reflect_on_self()).strip()
            if not summary:
                return ""
            return f"Perspective: {summary}\n\n"
        except Exception:
            return ""

    def _semantic_text_for_embedding(self, text: str) -> str:
        return self._memory_semantic_prefix() + str(text or "")

    def _current_memory_qualia_vector(self) -> np.ndarray:
        try:
            return np.asarray([
                float(getattr(self.qualia, 'arousal', 0.0)),
                float(getattr(self.qualia, 'valence', 0.0)),
                float(getattr(self.qualia, 'entropy', 0.0)),
                float(getattr(self.qualia, 'engagement', 0.0)),
                float(getattr(self.qualia, 'frustration', 0.0)),
            ], dtype=np.float32)
        except Exception:
            return np.zeros((5,), dtype=np.float32)

    def _current_memory_phi(self) -> float:
        try:
            phi_hist = getattr(self.phi_calculator, 'phi_history', None)
            if phi_hist:
                return float(phi_hist[-1])
        except Exception:
            pass
        return 0.0

    def _encode_semantic_memory_trace(
        self,
        experience_name: str,
        kind: str,
        content: str,
        embedding: Optional[np.ndarray],
        tags: Optional[List[str]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            if not hasattr(self, 'episodic_memory') or self.episodic_memory is None:
                return
            context = {
                'iteration': int(getattr(self, 't', 0)),
                'source': str(kind),
            }
            if isinstance(extra_context, dict):
                context.update(extra_context)
            qualia_vec = self._current_memory_qualia_vector()
            phi_val = self._current_memory_phi()
            self.episodic_memory.encode_experience(
                experience_name=experience_name,
                qualia_vector=qualia_vec,
                phi_value=phi_val,
                context=context,
                narrative=str(content)[:256],
                kind=kind,
                content=str(content),
                embedding=embedding,
                tags=list(tags) if tags else [],
            )
        except Exception:
            pass

    def _record_dialog_trace(self, adapter, user_msg: str, response: str, prompt_for_embedding: str) -> None:
        if not response:
            return
        if self._is_disallowed_llm_response(response):
            return
        try:
            content = f"User: {user_msg}\nM3: {response}"
            emb = None
            if adapter is not None and hasattr(adapter, 'embed_text'):
                emb_text = self._semantic_text_for_embedding(prompt_for_embedding)
                emb = adapter.embed_text(emb_text, sys_identity="")
            self._encode_semantic_memory_trace(
                experience_name='dialog_turn',
                kind='dialog',
                content=content,
                embedding=emb,
                tags=['dialog'],
                extra_context={'channel': 'handle_user_message'},
            )
        except Exception:
            pass

    def _llm_fallback_text(self, prompt: str = "") -> str:
        lang = "en"
        adapter = getattr(self, 'llm_adapter', None) or getattr(self, 'llm', None)
        try:
            if adapter is not None and hasattr(adapter, '_detect_language'):
                lang = adapter._detect_language(prompt) or "en"
        except Exception:
            lang = "en"

        if lang == "ko":
            return "현재 생성 경로가 일시적으로 안전 모드입니다. 잠시 뒤 다시 질문해 주세요."
        if lang == "zh":
            return "当前生成路径处于安全模式。请稍后再发送请求。"
        if lang == "ru":
            return "Сейчас режим генерации переведен в безопасный режим. Пожалуйста, повторите запрос позже."
        return "Generation is in safe mode. Please retry your request in a moment."

    def handle_user_message(self, text: str) -> str:
        msg = str(text).strip()
        if not msg:
            return ""

        self._chat_history.append({'role': 'user', 'text': msg, 't': int(getattr(self, 't', 0))})
        if getattr(self, 'bus', None) is not None:
            vec = self.feature_bank._hash_embed(msg, self.feature_bank.embed_dim) if getattr(self, 'feature_bank', None) else np.zeros((32,), np.float32)
            self.bus.push('user', 'utter.user', vec.astype(np.float32), salience=0.7, confidence=0.9, ttl=8)

        adapter = getattr(self, 'llm_adapter', None) or getattr(self, 'llm', None)
        prompt = self._build_llm_prompt()
        if adapter is None:
            response = self._generate_utterance()
            self._apply_dialog_verifier_reward(msg, response)
            self._record_dialog_trace(None, msg, response, prompt)
            return response

        affect = self._collect_affect_state_for_llm()
        mem = self._collect_llm_memory(adapter, affect_state=affect)
        response, err = self._emit_llm_response(prompt, mem=mem, affect_state=affect, max_len=100)
        if response is not None:
            self._apply_dialog_verifier_reward(msg, response)
            self._record_dialog_trace(adapter, msg, response, prompt)
            return response

        if err:
            logging.debug(f"LLM generation failed: {err}")
        fallback_response = self._generate_utterance()
        self._apply_dialog_verifier_reward(msg, fallback_response)
        self._record_dialog_trace(adapter, msg, fallback_response, prompt)
        return fallback_response

    def _generate_utterance(self) -> str:
        response, err = self._emit_llm_response(self._build_llm_prompt(), max_len=100)
        if response is not None:
            self._last_utterance = response
            return response

        fallback = self._llm_fallback_text(self._build_llm_prompt())
        if err and err != "EMPTY":
            logging.debug(f"_generate_utterance fallback due to error: {err}")
        self._last_utterance = fallback
        self._chat_history.append({'role': 'assistant', 'text': fallback, 't': int(getattr(self, 't', 0))})
        return fallback

    def _save_checkpoint(self):
        beliefs_dict = {}
        if hasattr(self.self_model, 'belief_stability'):
            beliefs_dict['stability'] = float(self.self_model.belief_stability)
        if hasattr(self.self_model, 'belief_adaptation'):
            beliefs_dict['adaptation'] = float(self.self_model.belief_adaptation)
        if hasattr(self.self_model, 'belief_prediction'):
            beliefs_dict['prediction'] = float(self.self_model.belief_prediction)
        checkpoint = {'t': int(self.t), 'seed': int(self.seed), 'strange_loop_active': int(hasattr(self.self_model, 'knows_it_knows') and self.self_model.knows_it_knows), 'meta_awareness': float(hasattr(self.self_model, 'meta_awareness') and self.self_model.meta_awareness or 0.0), 'unity_score': float(self.unified_subject.unity_score), 'energy': float(self.energy_ctrl.cognitive_energy), 'activation': float(self.energy_ctrl.activation_level), 'beliefs': beliefs_dict, 'qualia': {'arousal': float(self.qualia.arousal), 'valence': float(self.qualia.valence), 'entropy': float(self.qualia.entropy), 'engagement': float(self.qualia.engagement), 'frustration': float(self.qualia.frustration)}}
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        # Also snapshot self-dynamics model
        try:
            if getattr(self, 'sdm', None) is not None:
                self.sdm.save(os.path.join(self.outdir, 'self_model.npz'))
        except Exception:
            pass
        

        try:
            if hasattr(self, 'episodic_memory'):
                try:
                    self.episodic_memory.consolidate()
                except Exception:
                    pass
                if hasattr(self, 'episodic_memory_path'):
                    self.episodic_memory.save(self.episodic_memory_path)
        except Exception:
            pass
        try:
            self._save_chat_history()
        except Exception:
            pass

        # Also save LLM adapter if available
        try:
            if hasattr(self, 'llm_adapter') and self.llm_adapter is not None:
                llm_path = os.path.join(self.outdir, 'llm_checkpoint.pt')
                if hasattr(self.llm_adapter, 'save_model'):
                    if getattr(self.llm_adapter, '_hf_circuit_open', False):
                        print(f"Skipping LLM checkpoint save due to HF circuit breaker: {llm_path}")
                    else:
                        self.llm_adapter.save_model(llm_path)
                        print(f"LLM checkpoint saved to {llm_path}")
        except Exception as e:
            print(f"Failed to save LLM checkpoint: {e}")

    def _load_checkpoint(self):
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            self.t = checkpoint['t']
            print(f'Checkpoint loaded: resuming from t={self.t:,}')
        except Exception as e:
            print(f'Checkpoint load failed: {e}')
            self.t = 0


    def _save_chat_history(self, path: Optional[str] = None) -> None:
        try:
            p = path or getattr(self, 'chat_history_path', None)
            if not p:
                return
            with open(p, 'w', encoding='utf-8') as f:
                for item in list(getattr(self, '_chat_history', [])):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _load_chat_history(self, path: Optional[str] = None) -> None:
        try:
            p = path or getattr(self, 'chat_history_path', None)
            if not p or not os.path.exists(p):
                return
            hist = []
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and 'role' in obj and 'text' in obj:
                            hist.append(obj)
                    except Exception:
                        continue
            if hasattr(self, '_chat_history'):
                self._chat_history.clear()
                for item in hist:
                    self._chat_history.append(item)
        except Exception:
            pass

    def run_autonomous(self):
        start_msg = f'M3 Consciousness System - Infinite Evolution\n'
        start_msg += f'Seed: {self.seed}\n'
        start_msg += f'n: {self.n}   K: {self.K}\n'
        if self.t > 0:
            start_msg += f'Resuming from t={self.t:,}\n'
        start_msg += f'Ctrl+C: Save & Exit\n'
        start_msg += f'Ctrl+S: Save checkpoint (manual)'
        print(start_msg)
        start_time = time.perf_counter()
        last_save = 0
        self.manual_save_requested = False


        try:
            while True:
                    try:
                        self.energy_ctrl.internal_clock += 1
                        self.world_state = self._get_current_world_state()
                        current_goal = self.goal_gen.generate_goal(self.self_model, self.qualia, self.world_state, self.self_model.state_history)
                        self.energy_ctrl.update_activation(self.qualia, self.self_model, current_goal)
                        self.energy_ctrl.update_internal_statistics()

                        # --- Reward System Hook ---
                        # Calculate viability cost (e.g. based on energy depletion or stability)
                        viability_cost = 0.0
                        if hasattr(self.energy_ctrl, 'cognitive_energy') and hasattr(self.energy_ctrl, 'energy_capacity'):
                            # Cost increases as energy drops
                            energy_ratio = self.energy_ctrl.cognitive_energy / max(1.0, self.energy_ctrl.energy_capacity)
                            viability_cost = max(0.0, 1.0 - energy_ratio)

                        reward_ctx = {
                            "viability_cost": viability_cost,
                            "context": self.world_state,
                            "phi": self.phi_calculator.phi_history[-1] if self.phi_calculator.phi_history else 0.0,
                            "stability": self.world_state.get('stability', 0.5),
                            "t": self.t
                        }
                        
                        # Calculate total cost (J_t)
                        total_cost = self.rewards.total_cost(self.h, reward_ctx)
                        
                        # Optional: Log or use total_cost
                        if self.t % 100 == 0:
                             self.message_bus.send(Message(
                                 source='core', 
                                 target='broadcast', 
                                 type='reward_signal', 
                                 payload={'t': self.t, 'total_cost': total_cost, 'viability': viability_cost}
                             ))
                        # --------------------------

                        if self.t % 50 == 0 and hasattr(self, 'long_term_planner'):
                            current_performance = {'phi': self.phi_calculator.phi_history[-1] if self.phi_calculator.phi_history else 0.0, 'consciousness': self.self_model.meta_awareness, 'growth_rate': len(self.growing_som.neurons) / max(1, self.t // 100) if hasattr(self.growing_som, 'neurons') else 0.0, 'memory_count': len(self.episodic_memory.memories)}
                            # Include spatial performance (distance to goal)
                            try:
                                current_performance['spatial_goal_dist'] = float(self.world_state.get('spatial_goal_dist')) if self.world_state.get('spatial_goal_dist') is not None else None
                            except Exception:
                                current_performance['spatial_goal_dist'] = None
                            if self.long_term_planner.current_goal:
                                self.long_term_planner.current_goal.update_progress(current_performance)
                                if self.long_term_planner.current_goal.is_completed(current_performance):
                                    self.long_term_planner.total_goals_completed += 1
                                    # assign new spatial target to next hottest region
                                    try:
                                        pem = self._scope_build_pred_err_map()
                                        gy, gx = divmod(int(np.argmax(pem)), pem.shape[1])
                                        gx_n = gx / max(1, pem.shape[1]-1)
                                        gy_n = gy / max(1, pem.shape[0]-1)
                                        retina0 = self._vision_build_retina(size=(64,64), foveate=False)
                                        _, _, _, depth0 = self._vision_features(retina0)
                                        gz_n = float(np.mean(depth0)) if depth0 is not None else 0.5
                                        self.set_spatial_goal3d(gx_n, gy_n, gz_n, radius=0.1)
                                    except Exception:
                                        pass
                        should_continue, processing_intensity = self.energy_ctrl.should_continue()
                        if not should_continue:
                            # --- Energy deadlock prevention: passive recovery even when halted ---
                            # Without this, energy stays below critical threshold forever
                            # because update_energy is never called in the halted path.
                            try:
                                self.energy_ctrl.update_energy(0.0)  # zero cost → pure recovery
                            except Exception:
                                # Manual minimum recovery as ultimate fallback
                                self.energy_ctrl.cognitive_energy += max(
                                    0.5, self.energy_ctrl.recovery_rate_max * 0.3
                                )
                                self.energy_ctrl.cognitive_energy = min(
                                    self.energy_ctrl.cognitive_energy,
                                    self.energy_ctrl.energy_capacity
                                )
                            self.t += 1
                            continue
                        if processing_intensity < 0.2:
                            if self.t % 5 == 0:
                                self.generate_internal_events()
                            self.t += 1
                            continue
                        if self.t % 100 == 0:
                            perturbation = self.rng.normal(0, 0.5, size=self.n)
                            self.h += perturbation
                            self.base_vec = np.abs(self.rng.normal(0, 0.5, size=self.K))
                            self.base_vec /= self.base_vec.sum()
                            U_noise = self.rng.normal(0, 0.1, size=self.U.shape)
                            self.U += U_noise
                            self.U = np.clip(self.U, -5.0, 5.0)
                            self.event_queue.append(Event(type=EventType.HIGH_UNCERTAINTY, timestamp=self.t, importance=0.5, payload={'reason': 'external_perturbation'}))
                        if self.t % 2 == 0:
                            self.generate_internal_events()
                        if self.event_queue:
                            event = self.select_most_important_event()
                            if event:
                                self.process_event(event, current_goal)
                        action_plan = self._decide_action(current_goal, processing_intensity)
                        if self.self_model.knows_it_knows:
                            downward = self.self_model.downward_causation()
                            if 'exploration_mult' in downward:
                                action_plan['exploration_factor'] *= downward['exploration_mult']
                            if 'exploration_boost' in downward:
                                action_plan['exploration_factor'] += downward['exploration_boost']
                            if 'learning_rate_mult' in downward:
                                action_plan['learning_rate'] = action_plan.get('learning_rate', 1.0) * downward['learning_rate_mult']
                            if 'complexity_mult' in downward:
                                action_plan['complexity'] = action_plan.get('complexity', 1.0) * downward['complexity_mult']
                            if 'gate_adjust_mult' in downward:
                                action_plan['gate_adjust'] = action_plan.get('gate_adjust', 1.0) * downward['gate_adjust_mult']
                            if 'gate_openness' in downward:
                                self.gate_mid *= downward['gate_openness']
                            if 'stability_preference' in downward:
                                action_plan['exploration_factor'] *= 1.0 - downward['stability_preference']
                            if downward.get('automatic_mode', False):
                                action_plan['use_cached'] = True
                            if 'sensitivity_mult' in downward:
                                action_plan['sensitivity'] = downward['sensitivity_mult']
                        delta_hat, P_obs = self._execute_action(action_plan)
                        self._experience_qualia(delta_hat, action_plan)
                        grounded_experience = self.conceptual_space.ground_experience(self.qualia)
                        self._submit_to_workspace(grounded_experience, current_goal)
                        conscious_contents = self.global_workspace.compete_for_consciousness()
                        broadcast = self.global_workspace.broadcast(world_state=self.world_state)
                        action_outcome = self._evaluate_action_outcome(delta_hat, current_goal, action_plan)
                        detected_errors = self.global_workspace.detect_errors_from_broadcast(broadcast, action_outcome)
                        if detected_errors:
                            policy_adjustments = self.global_workspace.update_policy_from_errors(detected_errors)
                            if policy_adjustments:
                                if self.t % 100 == 0 or any((abs(v) > 0.1 for v in policy_adjustments.values())):
                                    print(f' Policy adjusted at t={self.t}:')
                                    for param, change in policy_adjustments.items():
                                        if abs(change) > 0.01:
                                            direction = 'UP' if change > 0 else 'DOWN'
                                            print(f'   {direction} {param}: {change:+.3f}')
                        self._last_policy_recommendations = self.global_workspace.get_policy_recommendations()
                        if self.t % 100 == 0:
                            introspection = self.self_model.introspect(depth=3)
                        
                        # === AUTONOMOUS EXPANSION INTEGRATION ===
                        if self.t % 50 == 0:
                            try:
                                # 1. RecursiveSelfModel autonomous expansion
                                internal_state = {
                                    'phi': self.phi_calculator.phi_history[-1] if self.phi_calculator.phi_history else 0.5,
                                    'meta_awareness': self.self_model.meta_awareness,
                                    'delta_hat': delta_hat,
                                    'qualia': self.qualia,
                                    't': self.t
                                }
                                expansion_report = self.self_model.autonomous_expansion_step(float(self.t), internal_state)
                                
                                if expansion_report.get('expansions_applied'):
                                    print(f"\n[t={self.t}] SELF-MODEL EXPANSION:")
                                    for exp in expansion_report['expansions_applied']:
                                        print(f"  ✓ {exp['action']['type']}: {exp['action']['details']}")
                                        print(f"    Improvement: {exp['improvement']}")
                                
                                # 2. GoalGenerator dynamic goal discovery
                                discovered_goal = self.goal_gen.discover_goal_from_pattern(self.qualia, self.self_model, self.world_state)
                                if discovered_goal and self.t % 200 == 0:
                                    print(f"\n[t={self.t}] NEW GOAL DISCOVERED:")
                                    print(f"  {discovered_goal['description']}")
                                
                                # 3. FeatureBank adaptive growth
                                if hasattr(self, 'feature_bank'):
                                    growth_result = self._maybe_adaptive_feature_bank_growth()
                                    if growth_result:
                                        print(f"\n[t={self.t}] FEATURE BANK GROWTH:")
                                        print(f"  {growth_result.get('old_max_dim')} -> {growth_result.get('new_max_dim')}")
                                        print(f"  Reason: {growth_result.get('reason', 'n/a')}")
                                        if not bool(growth_result.get("applied", False)):
                                            print("  Growth proposal rejected by synchronized grow path.")
                            
                            except Exception as e:
                                logging.debug(f"Autonomous expansion error: {e}")
                        
                        self.self_model.update_meta_awareness(conscious_contents)
                        qualia_vec = np.array([self.qualia.arousal, self.qualia.valence, self.qualia.entropy, self.qualia.engagement, self.qualia.frustration])
                        som_result = self.growing_som.learn(qualia_vec)
                        growth_event = som_result['grew']
                        self._last_growth_event = growth_event
                        if growth_event:
                            self.event_queue.append(Event(type=EventType.GOAL_ACHIEVED, timestamp=self.t, importance=0.7, payload={'reason': 'neuron_growth', 'neuron_count': som_result['neuron_count']}))
                        unified_exp = self.unified_subject.bind_experience(qualia=self.qualia, beliefs=self.self_model.to_dict() if hasattr(self, 'self_model') else {}, goals=[self.long_term_planner.current_goal] if hasattr(self, 'long_term_planner') and self.long_term_planner.current_goal else [], workspace_contents=conscious_contents, t=self.t)
                        if self.t % 200 == 0 and hasattr(self, 'experiment_designer'):
                            error_profile = ErrorProfile()
                            if detected_errors:
                                for error in detected_errors:
                                    error_profile.add_error(error['severity'])
                            if error_profile.total_errors > 0:
                                structure = self._get_structure_snapshot()
                                current_performance = {'phi': self.phi_calculator.phi_history[-1] if self.phi_calculator.phi_history else 0.0, 'consciousness': self.self_model.meta_awareness, 'stability': self.world_state.get('stability', 0.5)}
                                hypothesis = self.experiment_designer.generate_hypothesis(error_profile, current_performance)
                                exp_design = self.experiment_designer.design_experiment(hypothesis)
                                def perf_eval(structure_view: Dict[str, Any]):
                                    return self._evaluate_structure(structure_view, steps=60)
                                exp_results = self.experiment_designer.execute_experiment(exp_design, perf_eval, structure)
                                modified, applied = self.experiment_designer.apply_if_accepted(exp_results, structure)
                                if applied:
                                    self._apply_structure_snapshot(modified)
                                    try:
                                        self._save_checkpoint()
                                    except Exception:
                                        pass
                        if hasattr(self, 'reward_scheduler'):
                            computational_cost = processing_intensity * 10
                            memory_cost = len(self.episodic_memory.memories) * 0.1
                            reward_signal = RewardSignal(source='consciousness_processing', value=self.self_model.meta_awareness * 10, metadata={'resource_costs': {ResourceType.COMPUTE: computational_cost, ResourceType.MEMORY: memory_cost}})
                            budget_decision = self.reward_scheduler.allocate_budget(reward_signal)
                            if not budget_decision.approved:
                                processing_intensity *= 0.5
                        self.unified_subject.experience(unified_exp, 'unified_consciousness', is_intentional=current_goal is not None)
                        self._reflect_and_learn(delta_hat, current_goal)
                        # Policy reward shaping and update (REINFORCE batch)
                        try:
                            shaped_reward = float(-delta_hat + 0.5 * (self.self_model.meta_awareness - self._prev_meta))
                            # Surprise/Agency scheduler update
                            try:
                                sched = self._scheduler_update(delta_hat)
                            except Exception:
                                sched = None
                            self._last_shaped_reward = shaped_reward
                            self._prev_meta = float(self.self_model.meta_awareness)
                            if hasattr(self, '_last_policy_obs') and hasattr(self, '_last_policy_action'):
                                self.env.step(self._last_policy_action, shaped_reward)
                                # Counterfactual advantages via BRPolicy diagnostics (optional)
                                try:
                                    cda_bonus = 0.0
                                    self._last_cda_biases = {}
                                    if hasattr(self.policy, 'diagnose'):
                                        diag = self.policy.diagnose(self._last_policy_obs)
                                        g = np.asarray(diag.get('g'), dtype=np.float32) if diag.get('g') is not None else None
                                        contribs = diag.get('contribs')
                                        order = diag.get('order', [])
                                        if g is not None and contribs is not None:
                                            r_vals = []
                                            for i, c in enumerate(contribs):
                                                norm = float(np.sqrt(np.mean((np.asarray(c, dtype=np.float32) ** 2))))
                                                r_vals.append(float(g[i] * norm))
                                            cda_bonus = float(0.1 * np.sum(r_vals))
                                        try:
                                            if getattr(self, 'sdm', None) is not None and g is not None and order:
                                                H = int(os.environ.get('M3_CDA_H', '3')) if 'os' in globals() else 3
                                                credits = self._cda_rollout_advantages(self._last_policy_obs, action_plan, order, g, H=H, gamma=0.95)
                                                if credits:
                                                    m = max(1e-6, max(credits.values()))
                                                    self._last_cda_biases = {k: 0.2 * (v / m) for k, v in credits.items()}
                                        except Exception:
                                            pass
                                except Exception:
                                    cda_bonus = 0.0
                                total_reward = float(shaped_reward + cda_bonus)
                                self.policy.record(self._last_policy_obs, self._last_policy_action, getattr(self, '_last_policy_logp', 0.0), getattr(self, '_last_policy_mu', self.policy.forward(self._last_policy_obs)), total_reward)
                                # Bus token push (policy state)
                                try:
                                    if getattr(self, 'bus', None) is not None:
                                        mu_vec = getattr(self, '_last_policy_mu', self.policy.forward(self._last_policy_obs)[0])
                                        mu_vec = np.asarray(mu_vec, dtype=np.float32)
                                        H = 0.5 * mu_vec.size * (1.0 + float(np.log(2.0 * np.pi * (self.policy.sigma ** 2))))
                                        target_H = float(1.5) * mu_vec.size
                                        dH = float(abs(H - getattr(self, '_prev_policy_entropy', H)))
                                        self._prev_policy_entropy = H
                                        # bounded variance to avoid initial spikes
                                        var_a = float(np.var(np.tanh(np.asarray(self._last_policy_action, dtype=np.float32)))) if hasattr(self, '_last_policy_action') else 0.0
                                        sal = 0.5 * abs(float(delta_hat)) + 0.3 * var_a + 0.2 * dH
                                        # derive confidence from KL budget adherence and entropy proximity
                                        kl = float(getattr(self.policy, '_last_kl', 0.0))
                                        conf_k = 1.0 / (1.0 + max(0.0, kl))
                                        conf_h = float(np.exp(- (abs(H - target_H) / (target_H + 1e-6))))
                                        conf = 0.5 * conf_k + 0.5 * conf_h
                                        if bool(getattr(self.policy, '_last_clipped', False)):
                                            conf *= 0.8
                                        conf = float(np.clip(conf, 0.2, 1.0))
                                        # push generic policy token
                                        self.bus.push('policy', 'policy', mu_vec, salience=sal, confidence=conf, ttl=5)
                                        # also push a semantically named policy token
                                        try:
                                            name = getattr(self.feature_bank, '_namer', AutoNamer()).token_name_policy(float(delta_hat), float(var_a), float(dH), float(kl), bool(getattr(self.policy, '_last_clipped', False)))
                                            self.bus.push('policy', name, mu_vec, salience=sal, confidence=conf, ttl=5)
                                        except Exception:
                                            pass
                                        self.bus.step()
                                        try:
                                            _ = self._generate_utterance()
                                        except Exception:
                                            pass
                                        try:
                                            if hasattr(self, 'skills') and self.skills is not None:
                                                self.skills.observe(list(self.bus.top_keys()))
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                # Track recent policy rewards for persistence/selection
                                try:
                                    self.policy_reward_history.append(shaped_reward)
                                except Exception:
                                    pass
                                # AB online update for FeatureBank trials
                                try:
                                    if getattr(self, 'feature_bank', None) is not None:
                                        applied = list(getattr(self.feature_bank, '_last_applied_specs', []))
                                        self.feature_bank.ab_update(shaped_reward, applied_names=applied, kl=float(getattr(self.policy, '_last_kl', 0.0)), clipped=bool(getattr(self.policy, '_last_clipped', False)))
                                except Exception:
                                    pass
                                # every 50 steps, run a batch update with KL regularization (scheduler-applied)
                                if self.t % 50 == 0:
                                    try:
                                        target_kl = float(getattr(self, '_sched_target_kl', 0.02))
                                        # PolicyMLP path uses kl_budget; Torch policies use target_kl
                                        try:
                                            self.policy.end_batch(gamma=0.97, kl_coeff=0.02, lr=0.01 * processing_intensity, kl_budget=target_kl)
                                        except TypeError:
                                            self.policy.end_batch(gamma=0.97, kl_coeff=0.02, lr=0.01 * processing_intensity, target_kl=target_kl)
                                    except Exception:
                                        self.policy.end_batch(gamma=0.97, kl_coeff=0.02, lr=0.01 * processing_intensity)
                                if self.t % 200 == 0:
                                    try:
                                        self._save_policy_snapshots()
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        cost = self.energy_ctrl.compute_cognitive_cost(action_plan)
                        self.energy_ctrl.update_energy(cost)
                        if self.t % 10 == 0:
                            self._log_state(delta_hat, current_goal)
                        # Update visualization more frequently for smoother frames
                        if self.t % 2 == 0:
                            self._update_visualization()
                        if len(self.log_buffer) >= 1000:
                            self._flush_logs()
                        if self.t - last_save >= 10000:
                            self._save_checkpoint()
                            last_save = self.t
                        self.t += 1
                    except Exception as e:
                        import traceback as _tb
                        error_msg = f'Critical error at t={self.t}: {type(e).__name__}: {str(e)[:100]}'
                        print(f'ERROR: {error_msg}')
                        try:
                            _tb.print_exc()
                            # Print a few helpful shapes for troubleshooting
                            try:
                                print(f"[DEBUG] K={self.K}, n={self.n}, U.shape={getattr(self,'U', None).shape if hasattr(self,'U') else None}")
                            except Exception:
                                pass
                        except Exception:
                            pass
                        try:
                            self._save_checkpoint()
                            print('Emergency checkpoint saved')
                        except:
                            print('Emergency checkpoint failed')
                        try:
                            self.qualia = QualiaState()
                            self.energy_ctrl.cognitive_energy = max(30.0, self.energy_ctrl.cognitive_energy)
                            self.t += 1
                            continue
                        except:
                            break
        except KeyboardInterrupt:
            print('\nInterrupted - Saving checkpoint...')
            self._save_checkpoint()
        finally:
            self._flush_logs()
            elapsed = time.perf_counter() - start_time
            self._print_summary(elapsed)

    def _decide_action(self, goal: Goal, processing_intensity: float) -> Dict[str, Any]:
        action = {'strategy': 'adaptive', 'gate_adjust': 1.0, 'kd_adjust': 1.0, 'learning_rate': 0.25 * processing_intensity, 'exploration_factor': 2.0 * processing_intensity, 'complexity': 0.05, 'reasoning': ''}
        # Self-vision coupling: use vision summary to bias policy
        try:
            vs = getattr(self, '_vision_summary', None)
            if isinstance(vs, dict) and vs:
                g_mean = float(vs.get('g_mean', 0.0))  # surprise/change
                b_mean = float(vs.get('b_mean', 0.0))  # structure/depth cue
                edge_d = float(vs.get('edge_density', 0.0))
                def _lim(f, m=0.1):
                    return float(np.clip(f, 1.0 - m, 1.0 + m))
                # More surprise -> explore a bit more
                action['exploration_factor'] *= _lim(1.0 + 0.1 * g_mean)
                # More structure -> slightly stabilize gates
                action['gate_adjust'] *= _lim(1.0 - 0.1 * b_mean)
                action['kd_adjust'] *= _lim(1.0 + 0.1 * b_mean)
                # Rich edges -> learn a touch faster
                action['learning_rate'] *= _lim(1.0 + 0.1 * edge_d)
                action['reasoning'] += f" [vision g={g_mean:.2f} b={b_mean:.2f} e={edge_d:.2f}]"
        except Exception:
            pass
        # Spatial coupling: distance to spatial goal biases exploration vs. stabilization
        try:
            d = self.world_state.get('spatial_goal_dist', None)
            if isinstance(d, (int, float)):
                d = float(max(0.0, min(1.0, d)))
                action['exploration_factor'] *= _lim(1.0 + 0.1 * d)
                action['gate_adjust'] *= _lim(1.0 - 0.1 * (1.0 - d))
                action['kd_adjust'] *= _lim(1.0 + 0.1 * (1.0 - d))
                action['reasoning'] += f" [space d={d:.2f}]"
        except Exception:
            pass
        if hasattr(self, '_last_policy_recommendations') and self._last_policy_recommendations:
            policy = self._last_policy_recommendations
            action['exploration_factor'] *= policy.get('exploration_factor', 1.0)
            stability_pref = policy.get('stability_preference', 0.5)
            action['gate_adjust'] *= 1.0 - stability_pref * 0.3
            action['kd_adjust'] *= 1.0 + stability_pref * 0.3
            lr_modifier = policy.get('learning_rate_modifier', 1.0)
            action['learning_rate'] *= lr_modifier
            error_correction = policy.get('error_correction_strength', 1.0)
            if error_correction > 1.2:
                action['complexity'] *= 1.2
            if policy.get('cautious_mode', False):
                action['exploration_factor'] *= 0.7
                action['complexity'] *= 1.3
                action['reasoning'] += ' [CAUTIOUS]'
            elif policy.get('confident_mode', False):
                action['exploration_factor'] *= 1.2
                action['complexity'] *= 0.9
                action['reasoning'] += ' [CONFIDENT]'
        if processing_intensity < 0.4:
            action.update({'strategy': 'minimal', 'gate_adjust': 0.95, 'kd_adjust': 1.05, 'complexity': 0.02, 'reasoning': f'intensity={processing_intensity:.2f} ' + action.get('reasoning', '')})
            return action
        if hasattr(self.self_model, 'belief_stability'):
            beliefs = [self.self_model.belief_stability, self.self_model.belief_adaptation, self.self_model.belief_prediction]
        else:
            beliefs = [0.5, 0.5, 0.5]
        beliefs_array = np.array(beliefs)
        belief_mean = np.mean(beliefs_array)
        belief_std = max(0.1, np.std(beliefs_array))
        if hasattr(self.self_model, 'belief_adaptation'):
            adapt_z = (self.self_model.belief_adaptation - belief_mean) / belief_std
        else:
            adapt_z = 0.0
        if adapt_z < -1.0:
            action.update({'strategy': 'conservative', 'gate_adjust': action['gate_adjust'] * 0.7, 'kd_adjust': action['kd_adjust'] * 1.3, 'learning_rate': action['learning_rate'] * 0.3, 'exploration_factor': action['exploration_factor'] * 0.4, 'complexity': 0.06, 'reasoning': action['reasoning'] + f'   z={adapt_z:.2f}  '})
        elif adapt_z > 1.0:
            action.update({'strategy': 'explorative', 'gate_adjust': action['gate_adjust'] * 1.2, 'kd_adjust': action['kd_adjust'] * 0.9, 'learning_rate': action['learning_rate'] * 1.5, 'exploration_factor': action['exploration_factor'] * 1.3, 'complexity': 0.08, 'reasoning': action['reasoning'] + f'    z={adapt_z:.2f}  '})
        if self.self_model.meta_confidence < 0.3:
            action['complexity'] = 0.09
            action['reasoning'] += ' [LOW_CONFIDENCE]'
        if goal:
            if goal.type == GoalType.STABILIZE:
                action['gate_adjust'] *= 0.8
                action['kd_adjust'] *= 1.2
            elif goal.type == GoalType.EXPLORE:
                action['exploration_factor'] *= 1.3
            elif goal.type == GoalType.REST:
                action['complexity'] = 0.01
                action['learning_rate'] = 0.0
        # Learned policy override (policy-first, heuristic fallback)
        obs = self._policy_obs_adapted()
        # ensure policy exists and input dim matches
        if not hasattr(self, 'policy') or self.policy is None:
            self.policy = PolicyMLP(in_dim=obs.size, out_dim=12, hidden=max(128, obs.size), rng=self.rngr.get('policy'))
        else:
            try:
                pol_in = int(getattr(self.policy, 'in_dim', obs.size))
            except Exception:
                pol_in = int(obs.size)
            if pol_in > 0 and int(obs.size) != pol_in:
                obs = self._project_obs_to_dim(obs, target_dim=pol_in)
            try:
                if hasattr(self.policy, 'resize_input') and int(getattr(self.policy, 'in_dim', obs.size)) != int(obs.size):
                    self.policy.resize_input(obs.size)
            except Exception:
                allow_recreate = bool(self._observation_adapter_cfg.get("allow_policy_recreate", False))
                if allow_recreate:
                    out_dim = int(getattr(getattr(self, 'policy', None), 'out_dim', 12))
                    self._recreate_policy_with_transfer(in_dim=int(obs.size), out_dim=out_dim)
                else:
                    try:
                        obs = self._project_obs_to_dim(obs, target_dim=int(getattr(self.policy, 'in_dim', obs.size)))
                    except Exception:
                        pass
        # Provide Bus-routed active specs to BRPolicy if available
        try:
            if hasattr(self, 'feature_bank') and getattr(self, 'feature_bank', None) is not None and hasattr(self.feature_bank, '_ranges'):
                ranges = dict(getattr(self.feature_bank, '_ranges', {}))
                active = []
                try:
                    topk = list(self.bus.top_keys()) if getattr(self, 'bus', None) is not None else []
                    active = [k for k in topk if k in ranges]
                except Exception:
                    active = []
                if not active:
                    try:
                        active = [s.name for s in getattr(self.feature_bank, 'specs', [])[:4] if s.active and s.name in ranges]
                    except Exception:
                        active = []
                if hasattr(self.policy, 'set_active_specs') and callable(getattr(self.policy, 'set_active_specs')):
                    self.policy.set_active_specs(active, ranges)
                # Gate bias: merge skills + CDA credits
                try:
                    if hasattr(self.policy, 'set_gate_bias'):
                        merged: Dict[str, float] = {}
                        try:
                            if hasattr(self, 'skills') and self.skills is not None:
                                merged.update(self.skills.get_gate_biases(active))
                        except Exception:
                            pass
                        try:
                            for k, v in dict(getattr(self, '_last_cda_biases', {}) or {}).items():
                                merged[k] = merged.get(k, 0.0) + float(v)
                        except Exception:
                            pass
                        self.policy.set_gate_bias(merged)
                except Exception:
                    pass
        except Exception:
            pass
        # sample from policy
        if hasattr(self.policy, 'sample'):
            # Pass affect state if available (for M3-Binary Brain/Plasticity)
            # Wiring updated: Inject Affect + Phi + Energy for comprehensive neuromodulation
            mod_state = []
            
            # 1. Affect (Emotion)
            if hasattr(self, 'rewards') and hasattr(self.rewards, 'last_affect') and self.rewards.last_affect is not None:
                 mod_state.extend(list(np.ravel(self.rewards.last_affect)))
            
            # 2. Phi (Consciousness)
            try:
                current_phi = self.phi_calculator.phi_history[-1] if hasattr(self, 'phi_calculator') and self.phi_calculator.phi_history else 0.0
                mod_state.append(float(current_phi))
            except Exception:
                mod_state.append(0.0)
                
            # 3. Energy (Fatigue)
            try:
                 mod_state.append(float(self.energy_ctrl.activation_level))
            except Exception:
                 mod_state.append(1.0)

            affect_state = np.array(mod_state, dtype=np.float32) if mod_state else None
            
            samp = self.policy.sample(obs, affect_state=affect_state)
            if len(samp) == 3:
                raw_ctrl, _lp, _mu = samp; _v = 0.0
            else:
                raw_ctrl, _lp, _mu, _v = samp
        else:
            # fallback linear
            mu = self.policy.forward(obs); raw_ctrl = mu; _lp = 0.0; _mu = mu; _v = 0.0
        ctrl = np.tanh(raw_ctrl)
        # Optional CEM short planning (use first 5 dims mapping)
        ctrl_plan = self._plan_action_cem(obs, init=ctrl[:5]) if getattr(self, 'sdm', None) is not None else None
        if ctrl_plan is not None:
            beta = 0.3
            m = min(5, ctrl.size)
            ctrl[:m] = (1 - beta) * ctrl[:m] + beta * ctrl_plan[:m]
        # map ctrl to action fields
        def _get(i, default):
            return float(ctrl[i]) if i < ctrl.size else default
        action['learning_rate'] = float(np.clip(0.2 * (1.0 + _get(0, 0.0)), 0.0, 2.0))
        action['gate_adjust'] = float(np.clip(1.0 + 0.3 * _get(1, 0.0), 0.3, 1.7))
        action['kd_adjust'] = float(np.clip(1.0 + 0.2 * _get(2, 0.0), 0.5, 1.5))
        action['exploration_factor'] = float(np.clip(1.5 * (1.0 + 0.5 * _get(3, 0.0)), 0.0, 5.0))
        # increase complexity scale so energy cost is meaningful
        action['complexity'] = float(np.clip(0.8 * (1.0 + 0.5 * _get(4, 0.0)), 0.2, 2.5))
        # additional heads (7..11) as structural/vision/stability gains
        # structural gains
        action['sparsify_gain'] = float(np.clip((1.0 + _get(7, 0.0)) * 0.5, 0.0, 1.0))
        action['reconnect_gain'] = float(np.clip((1.0 + _get(8, 0.0)) * 0.5, 0.0, 1.0))
        action['prune_gain'] = float(np.clip((1.0 + _get(9, 0.0)) * 0.5, 0.0, 1.0))
        # vision mix beta (0.2..0.8)
        vb_raw = _get(10, 0.0)
        self._vision_mix_beta = float(np.clip(0.5 + 0.3 * vb_raw, 0.2, 0.8))
        # stability gain affects state decay strength
        action['stability_gain'] = float(np.clip((1.0 + _get(11, 0.0)) * 0.5, 0.0, 1.0))
        # attention nudge
        dx = 0.02 * _get(5, 0.0); dy = 0.02 * _get(6, 0.0)
        ax, ay = self.attn_xy
        ax = float(np.clip(ax + dx, 0.0, 1.0)); ay = float(np.clip(ay + dy, 0.0, 1.0))
        self.attn_xy = (ax, ay)
        action['strategy'] = 'policy'
        # expose last policy learning rate for UI snapshot
        self._last_policy_lr = float(action.get('learning_rate', 0.0))
        # store for batch update
        self._last_policy_obs = obs
        self._last_policy_action = raw_ctrl
        self._last_policy_logp = float(_lp)
        self._last_policy_mu = _mu
        # cache for SDM
        self._sdm_prev_a = self._action_vector(action)
        # expose latest shaped reward to SDM target later
        # set in run loop when computed; here no-op
        # Apply synthesized policy rules, if any (small weight + TTL)
        try:
            if getattr(self, '_policy_rules', None):
                sigs = self._policy_signals()
                kept = []
                for rule in list(self._policy_rules):
                    val = float(self._safe_eval_expr(rule.get('expr', ''), sigs))
                    if rule.get('target') == 'learning_rate':
                        action['learning_rate'] = float(np.clip(action['learning_rate'] * (1.0 + 0.05 * np.clip(val, -1.0, 1.0)), 0.0, 2.0))
                    elif rule.get('target') == 'gate_adjust':
                        action['gate_adjust'] = float(np.clip(action['gate_adjust'] * (1.0 + 0.05 * np.clip(val, -1.0, 1.0)), 0.1, 2.0))
                    elif rule.get('target') == 'exploration_factor':
                        action['exploration_factor'] = float(np.clip(action['exploration_factor'] * (1.0 + 0.05 * np.clip(val, -1.0, 1.0)), 0.0, 5.0))
                    # TTL handling
                    ttl = int(rule.get('ttl', 500)) - 1
                    if ttl > 0:
                        rule['ttl'] = ttl
                        kept.append(rule)
                self._policy_rules = kept
        except Exception:
            pass
        return action

    # --- UI snapshot for GUI ---
    def snapshot(self) -> Dict[str, Any]:
        try:
            phi = 0.0
            if hasattr(self.phi_calculator, 'phi_history') and self.phi_calculator.phi_history:
                phi = float(self.phi_calculator.phi_history[-1])
            energy = float(getattr(self.energy_ctrl, 'cognitive_energy', 0.0))
            activation = float(getattr(self.energy_ctrl, 'activation_level', 0.0))
            unity = float(getattr(self.unified_subject, 'unity_score', 0.0))
            lr = float(getattr(self, '_last_policy_lr', 0.0))
            steps = int(getattr(self, 't', 0))
            # attach a fixed-size embedding for visualization/analysis
            try:
                emb = self.extract_embedding(target_dim=128, normalize=True)
            except Exception:
                emb = np.zeros((128,), dtype=np.float32)
            return {'phi': phi, 'energy': energy, 'activation': activation, 'unity': unity, 'policy_lr': lr, 'steps': steps, 'embeddings': emb}
        except Exception:
            try:
                emb = self.extract_embedding(target_dim=128, normalize=True)
            except Exception:
                emb = np.zeros((128,), dtype=np.float32)
            return {'phi': 0.0, 'energy': 0.0, 'activation': 0.0, 'unity': 0.0, 'policy_lr': 0.0, 'steps': int(getattr(self, 't', 0)), 'embeddings': emb}


    def export_state_vector(self, include_panels: bool = True) -> np.ndarray:
        """Export a vector that encodes all module states (no omissions)."""
        try:
            dim = int(os.getenv('M3_VEC_HASH_DIM', '256'))
        except Exception:
            dim = 256
        if dim <= 0:
            dim = 256
        vec = np.zeros((dim,), dtype=np.float32)
        try:
            max_depth = int(os.getenv('M3_EXPORT_MAX_DEPTH', '2'))
        except Exception:
            max_depth = 2
        try:
            max_items = int(os.getenv('M3_EXPORT_MAX_ITEMS', '64'))
        except Exception:
            max_items = 64
        try:
            max_samples = int(os.getenv('M3_EXPORT_ARRAY_SAMPLES', '32'))
        except Exception:
            max_samples = 32

        seen = set()

        def _hash_to_idx(key: str) -> int:
            h = hashlib.blake2b(key.encode('utf-8', 'ignore'), digest_size=8).digest()
            return int.from_bytes(h, 'little', signed=False) % dim

        def _hash_to_val(val_str: str) -> float:
            h = hashlib.blake2b(val_str.encode('utf-8', 'ignore'), digest_size=8).digest()
            n = int.from_bytes(h, 'little', signed=False)
            return (n / float(2**64 - 1)) * 2.0 - 1.0

        def _add(path: str, value, numeric: bool):
            idx = _hash_to_idx(path)
            if numeric:
                try:
                    v = float(value)
                except Exception:
                    v = 0.0
                vec[idx] += float(np.tanh(v))
            else:
                vec[idx] += _hash_to_val(str(value))

        def _walk(obj, path: str, depth: int):
            try:
                oid = id(obj)
                if oid in seen:
                    _add(path + ".__cycle__", 1.0, True)
                    return
                seen.add(oid)
            except Exception:
                pass
            if depth > max_depth:
                _add(path + ".__repr__", repr(obj)[:256], False)
                return
            if isinstance(obj, (int, float, np.integer, np.floating, bool)):
                _add(path, float(obj), True)
                return
            if isinstance(obj, str):
                _add(path, obj, False)
                return
            if isinstance(obj, np.ndarray):
                arr = obj.astype(np.float32, copy=False).ravel()
                if arr.size:
                    _add(path + ".mean", float(arr.mean()), True)
                    _add(path + ".std", float(arr.std()), True)
                    _add(path + ".min", float(arr.min()), True)
                    _add(path + ".max", float(arr.max()), True)
                    if max_samples > 0:
                        try:
                            idxs = np.linspace(0, arr.size - 1, num=min(max_samples, arr.size)).astype(int)
                            for i, idx in enumerate(idxs.tolist()):
                                _add(f"{path}.s{i}", float(arr[idx]), True)
                        except Exception:
                            pass
                else:
                    _add(path + ".empty", 1.0, True)
                return
            try:
                import torch
                if torch.is_tensor(obj):
                    try:
                        arr = obj.detach().float().cpu().numpy()
                        _walk(arr, path, depth)
                        return
                    except Exception:
                        pass
            except Exception:
                pass
            if isinstance(obj, dict):
                _add(path + ".len", len(obj), True)
                count = 0
                for k, v in obj.items():
                    if count >= max_items:
                        _add(path + ".__trunc__", len(obj) - max_items, True)
                        break
                    _walk(v, f"{path}.{k}", depth + 1)
                    count += 1
                return
            if isinstance(obj, (list, tuple, set)):
                seq = list(obj)
                _add(path + ".len", len(seq), True)
                for i, v in enumerate(seq[:max_items]):
                    _walk(v, f"{path}[{i}]", depth + 1)
                if len(seq) > max_items:
                    _add(path + ".__trunc__", len(seq) - max_items, True)
                return
            if hasattr(obj, 'snapshot') and callable(getattr(obj, 'snapshot')):
                try:
                    snap = obj.snapshot()
                    _walk(snap, path + ".snapshot", depth + 1)
                    return
                except Exception:
                    pass
            if hasattr(obj, 'get_state') and callable(getattr(obj, 'get_state')):
                try:
                    st = obj.get_state()
                    _walk(st, path + ".state", depth + 1)
                    return
                except Exception:
                    pass
            if hasattr(obj, 'state_dict') and callable(getattr(obj, 'state_dict')):
                try:
                    sd = obj.state_dict()
                    if isinstance(sd, dict):
                        for i, (k, v) in enumerate(sd.items()):
                            if i >= max_items:
                                _add(path + ".__state_trunc__", len(sd) - max_items, True)
                                break
                            _walk(v, f"{path}.state.{k}", depth + 1)
                        return
                except Exception:
                    pass
            try:
                if hasattr(obj, '__dict__'):
                    _walk(obj.__dict__, path + ".__dict__", depth + 1)
                    return
            except Exception:
                pass
            _add(path + ".__repr__", repr(obj)[:256], False)

        _walk(self.__dict__, "core", 0)
        if include_panels and hasattr(self, 'feature_bank') and hasattr(self.feature_bank, 'panels'):
            try:
                panels = self.feature_bank.panels(self)
                _walk(panels, "core.feature_bank.panels", 1)
            except Exception:
                pass
        return vec.astype(np.float32)

    def _scheduler_update(self, delta_hat: float) -> Dict[str, float]:
        """Compute Surprise/Agency and map to sigma/KL/gating adjustments.
        Stores _sched_target_kl and applies bus top_k if available. Returns dict.
        """
        res = {}
        try:
            # Surprise: entropy jump + policy KL
            H_prev = float(getattr(self, '_prev_policy_entropy', 0.0))
            last_mu = getattr(self, '_last_policy_mu', None)
            if last_mu is not None and hasattr(self, 'policy') and hasattr(self.policy, 'sigma'):
                mu_vec = np.asarray(last_mu, dtype=np.float32)
                H_now = 0.5 * mu_vec.size * (1.0 + float(np.log(2.0 * np.pi * (float(self.policy.sigma) ** 2))))
            else:
                H_now = H_prev
            dH = abs(H_now - H_prev)
            kl = float(getattr(self.policy, '_last_kl', 0.0)) if hasattr(self, 'policy') else 0.0
            surprise = float(np.tanh(0.2 * dH) * 0.5 + np.tanh(kl) * 0.5)
            # Agency: bounded action variance proxy
            var_a = float(np.var(np.tanh(np.asarray(getattr(self, '_last_policy_action', np.zeros(1)), dtype=np.float32)))) if hasattr(self, '_last_policy_action') else 0.0
            agency = float(np.clip(var_a, 0.0, 1.0))
            # Map to targets
            base_kl = 0.02
            targ_kl = float(np.clip(base_kl * np.exp(0.5 * (surprise - agency)), 0.005, 0.05))
            self._sched_target_kl = targ_kl
            res['target_kl'] = targ_kl
            # Optional: adjust bus top-k
            try:
                if getattr(self, 'bus', None) is not None and hasattr(self.bus, 'top_k'):
                    k = int(np.clip(round(4 + 2 * (surprise - (1.0 - agency))), 2, 6))
                    self.bus.top_k = k
                    res['bus_top_k'] = float(k)
            except Exception:
                pass
            # Policy sigma: only for numpy MLP; Torch policies keep trainable sigma
            try:
                if hasattr(self.policy, 'W1') and hasattr(self.policy, 'sigma'):
                    target_sigma = float(np.clip(self.policy.sigma * np.exp(0.2 * (surprise - (1.0 - agency))), 0.1, 2.0))
                    self.policy.sigma = target_sigma
                    res['sigma'] = target_sigma
            except Exception:
                pass
            return res
        except Exception:
            return res

    def _policy_obs(self) -> np.ndarray:
        if getattr(self, 'feature_bank', None) is not None:
            z = self.feature_bank.build(self)
            try:
                if getattr(self, 'shared_repr', None) is not None:
                    # keep dim consistent
                    self.shared_repr.update_stats(z)
                    z = self.shared_repr.transform(z)
            except Exception:
                pass
            return z
        # Fallback to legacy 12D observation
        try:
            retina = self._vision_build_retina(size=(32, 32), foveate=True)
            contrast, entropy, edge_density, depth_cue = self._vision_features(retina)
            r_mean = float(np.mean(retina))
        except Exception:
            contrast = entropy = edge_density = 0.0
            r_mean = 0.0
        last_phi = float(self.phi_calculator.phi_history[-1]) if self.phi_calculator.phi_history else 0.0
        obs = np.array([
            r_mean, contrast, entropy, edge_density,
            last_phi,
            float(self.self_model.meta_awareness),
            float(self.energy_ctrl.activation_level),
            float(self.unified_subject.unity_score),
            float(self.world_state.get('stability', 0.5)) if hasattr(self, 'world_state') else 0.5,
            float(self.qualia.arousal), float(self.qualia.valence), float(self.qualia.entropy)
        ], dtype=np.float32)
        obs = np.clip((obs - 0.5) * 2.0, -1.0, 1.0)
        return obs

    def extract_embedding(self, target_dim: int = 128, normalize: bool = True) -> np.ndarray:
        """Return a fixed-size embedding vector for the current state.

        Strategy (best-effort, non-invasive):
        - If FeatureBank is available, use FeatureBank.build(self).
        - Else fall back to _policy_obs() (legacy compact obs) and pad/truncate.
        - Final fallback: flatten a structural matrix (e.g. U) and pad/truncate.

        Always returns a numpy float32 vector of length `target_dim`.
        """
        try:
            # Preferred: feature bank
            if getattr(self, 'feature_bank', None) is not None:
                emb = self.feature_bank.build(self)
                arr = np.asarray(emb, dtype=np.float32).ravel()
                out = np.zeros((target_dim,), dtype=np.float32)
                take = min(arr.size, target_dim)
                if take > 0:
                    out[:take] = arr[:take]
                if normalize:
                    nrm = float(np.linalg.norm(out))
                    if nrm > 1e-8:
                        out /= nrm
                return out
        except Exception:
            pass
        # Fallback: policy observation (small vector) padded
        try:
            obs = self._policy_obs_adapted()
            arr = np.asarray(obs, dtype=np.float32).ravel()
            out = np.zeros((target_dim,), dtype=np.float32)
            take = min(arr.size, target_dim)
            if take > 0:
                out[:take] = arr[:take]
            if normalize:
                nrm = float(np.linalg.norm(out))
                if nrm > 1e-8:
                    out /= nrm
            return out
        except Exception:
            pass
    def _torch_shared_z(self, z: np.ndarray) -> np.ndarray:
        try:
            mod = getattr(self, '_shared_repr_torch', None)
            if mod is None:
                return np.asarray(z, dtype=np.float32)
            import torch as _t
            dev = _t.device(resolve_torch_device_string(torch_module=_t, require_cuda=False))
            try:
                mod = mod.to(dev)
            except Exception:
                pass
            with _t.no_grad():
                xt = _t.as_tensor(np.asarray(z, np.float32).reshape(1, -1), device=dev)
                yt = mod(xt)
                out = yt.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return out
        except Exception:
            return np.asarray(z, dtype=np.float32)

    def _align_z_for_sdm(self, z: np.ndarray) -> np.ndarray:
        try:
            z = np.asarray(z, dtype=np.float32).ravel()
            z_dim = int(getattr(self, 'sdm', None).z_dim) if getattr(self, 'sdm', None) is not None else int(z.size)
            out = np.zeros((z_dim,), dtype=np.float32)
            take = min(z.size, z_dim)
            if take > 0:
                out[:take] = z[:take]
            return out
        except Exception:
            return np.asarray(z, dtype=np.float32)

    def _cda_rollout_advantages(self, z0: np.ndarray, action_plan: Dict[str, Any], order: List[str], g_vec: np.ndarray,
                                H: int = 3, gamma: float = 0.95) -> Dict[str, float]:
        credits: Dict[str, float] = {}
        try:
            if getattr(self, 'sdm', None) is None or not order or g_vec is None:
                return credits
            zt = np.asarray(z0, np.float32)
             # Baseline rollout
            base_total = 0.0
            z_sim = zt.copy()
            for t in range(int(H)):
                avec = self._action_vector(action_plan)
                zf = self._torch_shared_z(z_sim)
                out = self.sdm.forward(np.concatenate([zf, avec], axis=0))
                r = float(out.get('reward', 0.0)) - 0.3 * float(out.get('delta_hat', 0.0)) + 0.2 * float(out.get('stability', 0.0)) + 0.2 * float(out.get('meta', 0.0))
                base_total += (gamma ** t) * r
                z_sim = out.get('z_next', z_sim)
           # Counterfactual per expert: reduce gate_adjust proportionally to g_i
            for i, name in enumerate(order):
                z_sim = zt.copy()
                total = 0.0
                a_cf = dict(action_plan)
                try:
                    a_cf['gate_adjust'] = float(np.clip(a_cf.get('gate_adjust', 1.0) * (1.0 - min(0.5, float(g_vec[i]))), 0.1, 2.0))
                except Exception:
                    pass
                for t in range(int(H)):
                    avec = self._action_vector(a_cf)
                    zf = self._torch_shared_z(z_sim)
                    out = self.sdm.forward(np.concatenate([zf, avec], axis=0))
                    r = float(out.get('reward', 0.0)) - 0.3 * float(out.get('delta_hat', 0.0)) + 0.2 * float(out.get('stability', 0.0)) + 0.2 * float(out.get('meta', 0.0))
                    total += (gamma ** t) * r
                    z_sim = out.get('z_next', z_sim)
                credits[name] = float(max(0.0, base_total - total))
            return credits
        except Exception:
            return credits
        # Last-resort: structural data (U) flattened
        try:
            u = np.asarray(getattr(self, 'U', np.zeros((1,), dtype=np.float32)), dtype=np.float32).ravel()
            out = np.zeros((target_dim,), dtype=np.float32)
            take = min(u.size, target_dim)
            if take > 0:
                out[:take] = u[:take]
            if normalize:
                nrm = float(np.linalg.norm(out))
                if nrm > 1e-8:
                    out /= nrm
            return out
        except Exception:
            return np.zeros((target_dim,), dtype=np.float32)

    def grow_feature_bank(self, new_max_dim: int | None = None, new_embed_dim: int | None = None, specs: List[FeatureSpec] | None = None, force: bool = False) -> bool:
        """Grow the feature bank and resize dependent modules (policy, sdm) safely.

        Accepts either a numeric `new_max_dim` (legacy) or a list of `specs` to register
        and grow by the sum of their produced_dim. Returns True if growth applied, False
        if no-op or failed.
        """
        # Ensure feature_bank exists (create lazily with sensible defaults)
        if getattr(self, 'feature_bank', None) is None:
            # If specs provided and no explicit new_max_dim, estimate dimension
            base_dim = getattr(self, 'feature_bank', None)
            init_dim = int(new_max_dim) if new_max_dim is not None else 128
            self.feature_bank = FeatureBank(max_dim=init_dim, embed_dim=(new_embed_dim or 32))
            fb_dim = self.feature_bank.max_dim
        else:
            fb_dim = int(self.feature_bank.max_dim)

        # safety checks: cooldown and per-call cap (unless forced)
        now = time.time()
        if not force:
            try:
                if (now - float(getattr(self, '_last_growth_time', 0.0))) < float(getattr(self, 'growth_cooldown_sec', 300.0)):
                    try:
                        self.visualizer.add_major_event(f'Grow blocked: cooldown active ({now - float(getattr(self, "_last_growth_time", 0.0)):.1f}s)')
                    except Exception:
                        pass
                    return False
            except Exception:
                pass

        # If specs provided, compute added dims but don't commit until sandbox evaluation
        if specs:
            try:
                added = int(sum(getattr(s, 'produced_dim', 1) for s in specs))
            except Exception:
                added = 0
            # enforce per-call cap
            max_add = int(getattr(self, 'growth_max_added_dim_per_call', 256))
            if not force and added > max_add:
                # cap added to safe limit and adjust provided specs by trimming
                added = max_add
                try:
                    specs = specs[:max(1, int(len(specs) * (added / max(1, sum(getattr(s, 'produced_dim', 1) for s in specs))))) ]
                except Exception:
                    specs = specs[:1]
            # estimate new_max_dim (no commit yet)
            if new_max_dim is None:
                new_max_dim = int(min(getattr(self, '_growth_hard_cap', 4096), fb_dim + added))

        # validate new_max_dim
        if new_max_dim is None:
            return False
        try:
            new_max_dim = int(new_max_dim)
        except Exception:
            return False

        if int(new_max_dim) <= fb_dim:
            return False

        # If specs given and sandbox enabled, evaluate in sandbox before committing
        if specs and getattr(self, 'enable_growth_sandbox', True) and not force:
            try:
                sandbox_metrics = self._sandbox_evaluate_specs(specs, steps=40)
                base = float(sandbox_metrics.get('baseline_reward_mean', 0.0))
                treat = float(sandbox_metrics.get('treatment_reward_mean', 0.0))
                # require non-trivial improvement or at least no degradation
                if treat + 1e-6 < base - 1e-4:
                    try:
                        self.visualizer.add_major_event(f'Auto-grow rejected by sandbox: baseline {base:.4f} -> treat {treat:.4f}')
                    except Exception:
                        pass
                    return False
                # else fall through to commit
            except Exception:
                # If sandbox fails, abort to be safe
                try:
                    self.visualizer.add_major_event('Auto-grow aborted: sandbox failed')
                except Exception:
                    pass
                return False

        # Grow feature bank (commit)
        self.feature_bank.grow(new_max_dim, new_embed_dim)
        fb_dim_new = int(self.feature_bank.max_dim)
        # record growth time
        try:
            self._last_growth_time = float(time.time())
        except Exception:
            pass
        # Resize policy input
        if hasattr(self, 'policy') and self.policy is not None:
            try:
                if hasattr(self.policy, 'resize_input'):
                    self.policy.resize_input(fb_dim_new)
            except Exception:
                allow_recreate = bool(self._observation_adapter_cfg.get("allow_policy_recreate", False))
                if allow_recreate:
                    out_dim = int(getattr(getattr(self, 'policy', None), 'out_dim', 12))
                    self._recreate_policy_with_transfer(in_dim=int(fb_dim_new), out_dim=out_dim)
                else:
                    self._log_runtime_event(
                        "obs_adapter",
                        mode="keep_policy_dim",
                        policy_dim=int(getattr(getattr(self, 'policy', None), 'in_dim', fb_dim_new)),
                        feature_dim=int(fb_dim_new),
                    )
        # Resize SDM (input dim = fb_dim + act_dim)
        act_dim = 5
        self._sdm_in_dim = int(self.feature_bank.max_dim + act_dim)
        use_torch_sdm = os.environ.get('M3_TORCH_SDM', '0') in ('1', 'true', 'TRUE')
        if getattr(self, 'sdm', None) is None:
            if use_torch_sdm and _TORCH_OK:
                try:
                    d_model = int(os.environ.get('M3_SDM_DMODEL', '2048'))
                    n_layers = int(os.environ.get('M3_SDM_LAYERS', '16'))
                    d_ff = int(os.environ.get('M3_SDM_DFF', '8192'))
                    n_experts = int(os.environ.get('M3_SDM_MOE_EXPERTS', '64'))
                    top_k = int(os.environ.get('M3_SDM_MOE_TOPK', '2'))
                    use_fsdp = os.environ.get('M3_SDM_FSDP', '1') in ('1', 'true', 'TRUE')
                    self.sdm = TorchSDM(in_dim=self._sdm_in_dim, z_dim=self.feature_bank.max_dim, d_model=d_model, n_layers=n_layers,
                                        d_ff=d_ff, n_experts=n_experts, top_k=top_k, fsdp=use_fsdp)
                except Exception:
                    self.sdm = None
            if getattr(self, 'sdm', None) is None:
                self.sdm = SelfDynamicsModel(in_dim=self._sdm_in_dim, hidden=max(128, self.feature_bank.max_dim), rng=self.rngr.get('sdm'), z_dim=self.feature_bank.max_dim)
        else:
            # Try in-place resize when legacy SDM; TorchSDM will be re-created
            try:
                self.sdm.resize(new_in_dim=self._sdm_in_dim, new_z_dim=self.feature_bank.max_dim)
            except Exception:
                if use_torch_sdm and _TORCH_OK:
                    try:
                        d_model = int(os.environ.get('M3_SDM_DMODEL', '2048'))
                        n_layers = int(os.environ.get('M3_SDM_LAYERS', '16'))
                        d_ff = int(os.environ.get('M3_SDM_DFF', '8192'))
                        n_experts = int(os.environ.get('M3_SDM_MOE_EXPERTS', '64'))
                        top_k = int(os.environ.get('M3_SDM_MOE_TOPK', '2'))
                        use_fsdp = os.environ.get('M3_SDM_FSDP', '1') in ('1', 'true', 'TRUE')
                        self.sdm = TorchSDM(in_dim=self._sdm_in_dim, z_dim=self.feature_bank.max_dim, d_model=d_model, n_layers=n_layers,
                                            d_ff=d_ff, n_experts=n_experts, top_k=top_k, fsdp=use_fsdp, a_dim=5,
                                            shared_repr=self._shared_repr_torch)
                    except Exception:
                        self.sdm = SelfDynamicsModel(in_dim=self._sdm_in_dim, hidden=max(128, self.feature_bank.max_dim), rng=self.rngr.get('sdm'), z_dim=self.feature_bank.max_dim)
        # Announce
        try:
            # include spec names in event if available
            try:
                spec_names = ','.join([s.name for s in getattr(self.feature_bank, 'specs', [])[-8:]])
                self.visualizer.add_major_event(f'FeatureBank grown -> {self.feature_bank.max_dim}D; specs:+{spec_names}')
            except Exception:
                self.visualizer.add_major_event(f'FeatureBank grown -> {self.feature_bank.max_dim}D')
        except Exception:
            # non-fatal visualization error
            pass
        return True

    def _action_vector(self, action: Dict[str, Any]) -> np.ndarray:
        """Project action dict to a fixed 5D vector used by SDM and planner.
        [lr, gate_adjust, kd_adjust, exploration, complexity]
        """
        return np.array([
            float(action.get('learning_rate', 0.1)),
            float(action.get('gate_adjust', 1.0)),
            float(action.get('kd_adjust', 1.0)),
            float(action.get('exploration_factor', 1.0)),
            float(action.get('complexity', 0.05)),
        ], dtype=np.float32)

    def _plan_action_cem(self, obs: np.ndarray, init: np.ndarray | None = None, n: int = 32, iters: int = 2, H: int = 10) -> np.ndarray | None:
        """Short-horizon CEM planner over 5D control space using SDM as scorer.
        Returns best control vector for the first step in [-1,1]^5 after planning horizon H.
        """
        if getattr(self, 'sdm', None) is None:
            return None
        z = obs.astype(np.float32)
        dim = 5
        mu = np.zeros((H, dim), dtype=np.float32)
        if init is not None and init.shape[0] >= dim:
            mu[0, :dim] = init[:dim]
        sig = np.ones((H, dim), dtype=np.float32) * 0.5
        def score(seq: np.ndarray) -> float:
            # seq shape (H, dim) in [-1,1]
            zt = z.copy()
            total = 0.0
            gamma = 0.95
            for t in range(H):
                u = seq[t]
                a = {
                    'learning_rate': float(np.clip(0.2 * (1.0 + u[0]), 0.0, 2.0)),
                    'gate_adjust': float(np.clip(1.0 + 0.3 * u[1], 0.3, 1.7)),
                    'kd_adjust': float(np.clip(1.0 + 0.2 * u[2], 0.5, 1.5)),
                    'exploration_factor': float(np.clip(1.5 * (1.0 + 0.5 * u[3]), 0.0, 5.0)),
                    'complexity': float(np.clip(0.05 * (1.0 + 0.5 * u[4]), 0.005, 0.2)),
                }
                avec = self._action_vector(a)
                ztf = self._align_z_for_sdm(self._torch_shared_z(zt))
                xa = np.concatenate([ztf, avec], axis=0)
                pred = self.sdm.forward(xa)
                r = float(pred['reward']) - 0.3 * float(pred['delta_hat']) + 0.2 * float(pred['stability']) + 0.2 * float(pred['meta'])
                total += (gamma ** t) * r
                # transition to z'
                zt = pred.get('z_next', zt)
            # small regularization
            total -= 0.005 * float(np.sum(seq * seq))
            return float(total)
        best_score = -1e9
        for _ in range(iters):
            # sample sequences
            samples = []
            rng = self.rngr.get('cem')
            for i in range(n):
                seq = rng.normal(mu, sig + 1e-6).astype(np.float32)
                seq = np.clip(seq, -1.0, 1.0)
                samples.append(seq)
            scores = np.array([score(seq) for seq in samples], dtype=np.float32)
            idx = np.argsort(scores)[-max(4, n // 5):]
            elite = [samples[i] for i in idx]
            mu = np.mean(np.stack(elite, axis=0), axis=0)
            sig = np.std(np.stack(elite, axis=0), axis=0) + 1e-3
            try:
                if float(scores[idx[-1]]) > best_score:
                    best_score = float(scores[idx[-1]])
            except Exception:
                pass
        try:
            self._last_plan_score = float(best_score)
        except Exception:
            pass
        return mu[0].astype(np.float32)

    # ---- Rule AB test (mini episode) ----
    def _ab_test_rule(self, rule: Dict[str, Any], steps: int = 40) -> bool:
        """Run a short AB test toggling a candidate rule on/off.

        Returns True if treatment (rule on) improves reward_mean and reduces td_error vs control.
        """
        # Backup minimal state
        snap = self._get_structure_snapshot()
        h_bak = getattr(self, 'h', None)
        if h_bak is not None:
            try:
                h_bak = h_bak.copy()
            except Exception:
                pass
        energy_bak = (float(self.energy_ctrl.cognitive_energy), float(self.energy_ctrl.activation_level))
        prev_meta_bak = float(getattr(self, '_prev_meta', 0.5))
        rules_bak = list(getattr(self, '_policy_rules', []))
        def _run(with_rule: bool):
            # restore dynamics
            try:
                self._apply_structure_snapshot(snap)
                if h_bak is not None:
                    self.h = h_bak.copy()
                self.energy_ctrl.cognitive_energy = energy_bak[0]
                self.energy_ctrl.activation_level = energy_bak[1]
                self._prev_meta = prev_meta_bak
            except Exception:
                pass
            # set rules
            self._policy_rules = [rule] if with_rule else []
            rewards = []
            deltas = []
            for i in range(max(10, int(steps))):
                try:
                    self.energy_ctrl.internal_clock += 1
                    self.world_state = self._get_current_world_state()
                    plan = self._decide_action(None, max(0.2, float(self.energy_ctrl.activation_level)))
                    d, _ = self._execute_action(plan)
                    self._experience_qualia(d, plan)
                    deltas.append(d)
                    r = float(-d + getattr(self, 'reward_shape_lambda', 0.2) * (self.self_model.meta_awareness - getattr(self, '_prev_meta', 0.5)))
                    self._prev_meta = float(self.self_model.meta_awareness)
                    rewards.append(r)
                except Exception:
                    break
            rm = float(np.mean(rewards)) if rewards else 0.0
            dm = float(np.mean(deltas)) if deltas else 1.0
            return rm, dm
        try:
            rm_c, dm_c = _run(False)
            rm_t, dm_t = _run(True)
            # restore
            self._apply_structure_snapshot(snap)
            if h_bak is not None:
                self.h = h_bak
            self.energy_ctrl.cognitive_energy = energy_bak[0]
            self.energy_ctrl.activation_level = energy_bak[1]
            self._prev_meta = prev_meta_bak
            self._policy_rules = rules_bak
            # pass criteria: reward_mean improves and td_error reduces slightly
            return (rm_t > rm_c + 0.01) and (dm_t < dm_c * 0.98)
        except Exception:
            # fail closed
            self._policy_rules = rules_bak
            try:
                self._apply_structure_snapshot(snap)
            except Exception:
                pass
            return False

    def _policy_signals(self) -> Dict[str, float]:
        try:
            last = self.log_buffer[-1] if self.log_buffer else {}
            return {
                'phi': float(last.get('phi', 0.0)),
                'meta': float(last.get('meta_awareness', 0.5)),
                'energy': float(last.get('energy_level', 0.5)),
                'activation': float(last.get('activation_level', 0.5)),
                'unity': float(last.get('unity_score', 0.5)),
                'entropy': float(last.get('entropy', 0.0)),
                'flow': float(last.get('flow', 0.0)),
                'delta_hat': float(last.get('delta_hat', 0.0)),
            }
        except Exception:
            return {'phi': 0.0, 'meta': 0.5, 'energy': 0.5, 'activation': 0.5, 'unity': 0.5, 'entropy': 0.0, 'flow': 0.0, 'delta_hat': 0.0}

    def _safe_eval_expr(self, expr: str, sigs: Dict[str, float]) -> float:
        if not expr:
            return 0.0
        allowed = {'np': np, 'min': min, 'max': max, 'abs': abs, 'clip': np.clip}
        allowed.update(sigs)
        try:
            return float(eval(expr, {'__builtins__': {}}, allowed))
        except Exception:
            return 0.0

    def _synthesize_policy_rule(self) -> None:
        if self.t - getattr(self, '_last_synthesis_at', 0) < 200:
            return
        hist = list(self.log_buffer)[-120:]
        if len(hist) < 60:
            return
        phi = np.array([float(h.get('phi', 0.0)) for h in hist])
        meta = np.array([float(h.get('meta_awareness', 0.5)) for h in hist])
        energy = np.array([float(h.get('energy_level', 0.5)) for h in hist])
        activation = np.array([float(h.get('activation_level', 0.5)) for h in hist])
        unity = np.array([float(h.get('unity_score', 0.5)) for h in hist])
        entropy = np.array([float(h.get('entropy', 0.0)) for h in hist])
        flow = np.array([float(h.get('flow', 0.0)) for h in hist])
        delta = np.array([float(h.get('delta_hat', 0.0)) for h in hist])
        if delta.size < 3:
            return
        y = delta[1:]
        def score_expr(expr: str) -> float:
            X = []
            for i in range(len(y)):
                sigs = {
                    'phi': phi[i], 'meta': meta[i], 'energy': energy[i], 'activation': activation[i],
                    'unity': unity[i], 'entropy': entropy[i], 'flow': flow[i], 'delta_hat': delta[i],
                }
                X.append(self._safe_eval_expr(expr, sigs))
            x = np.asarray(X, dtype=np.float64)
            if np.allclose(x.std(), 0.0):
                return -1e9
            corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                return -1e9
            return float(-corr)
        lr_candidates = [
            'np.clip(0.1 + 0.6*delta_hat - 0.3*phi, 0.01, 1.5)',
            '0.05 + 0.4*entropy + 0.2*flow - 0.2*phi',
            '0.2 + 0.3*abs(delta_hat-0.4) - 0.1*meta'
        ]
        gate_candidates = [
            '0.9 - 0.5*meta + 0.2*unity',
            '1.0 - 0.4*phi + 0.3*(1.0-energy)',
            '0.8 + 0.2*(1.0-meta) - 0.2*delta_hat'
        ]
        explore_candidates = [
            '1.0 + 0.8*entropy - 0.6*phi',
            '0.7 + 0.5*(1.0-meta) + 0.2*flow',
            '1.2 - 0.4*unity + 0.3*entropy'
        ]
        scored = []
        # simple cross-validation: halves must agree
        mid = len(hist) // 2
        for target, cands in [('learning_rate', lr_candidates), ('gate_adjust', gate_candidates), ('exploration_factor', explore_candidates)]:
            best_expr, best_score = None, -1e9
            for e in cands:
                s_full = score_expr(e)
                # halves
                y_full = np.array([float(h.get('delta_hat', 0.0)) for h in hist])
                y1 = y_full[1:mid]
                y2 = y_full[mid+1:]
                def s_slice(start_idx, end_idx):
                    X = []
                    for i in range(start_idx, end_idx):
                        sigs = {
                            'phi': phi[i], 'meta': meta[i], 'energy': energy[i], 'activation': activation[i],
                            'unity': unity[i], 'entropy': entropy[i], 'flow': flow[i], 'delta_hat': delta[i],
                        }
                        X.append(self._safe_eval_expr(e, sigs))
                    x = np.asarray(X, dtype=np.float64)
                    if x.size == 0 or np.allclose(x.std(), 0.0):
                        return -1e9
                    yy = delta[start_idx+1:end_idx+1]
                    c = np.corrcoef(x, yy)[0, 1]
                    return float(-c) if not np.isnan(c) else -1e9
                s1 = s_slice(0, max(1, mid-2))
                s2 = s_slice(mid, len(delta)-2)
                if s_full > 0.1 and s1 > 0.05 and s2 > 0.05:
                    if s_full > best_score:
                        best_expr, best_score = e, s_full
            if best_expr is not None and best_score > 0.2:
                candidate = {'target': target, 'expr': best_expr, 'score': best_score, 'ttl': 500, 'promoted': False}
                # Mini AB test before promotion
                try:
                    if self._ab_test_rule(candidate, steps=40):
                        candidate['promoted'] = True
                        scored.append(candidate)
                except Exception:
                    pass
        if scored:
            scored.sort(key=lambda r: r['score'], reverse=True)
            self._policy_rules = scored[:3]
            self._last_synthesis_at = int(self.t)
            try:
                print('SYNTHESIZED RULES:', '; '.join([f"{r['target']} <- {r['expr']} ({r['score']:.2f})" for r in self._policy_rules]))
            except Exception:
                pass

    def _execute_action(self, action: Dict[str, Any]) -> Tuple[float, np.ndarray]:
        mod = np.array([float(np.mean(self.h[g])) for g in self.groups])
        gmean = float(np.mean(self.h))
        W = np.concatenate([mod, [gmean]])
        lt = self.U @ W
        if action['exploration_factor'] > 0:
            noise = self.rng.normal(0, 0.3 * action['exploration_factor'], size=self.K)
            lt = lt + noise
        lt = lt - np.max(lt)
        P_obs = np.exp(lt) / np.sum(np.exp(lt))
        self.P_obs_history.append(P_obs.copy())
        if len(self.P_obs_history) >= 50:
            recent = np.array(list(self.P_obs_history)[-50:])
            new_base = np.mean(recent, axis=0)
            self.base_vec = 0.95 * self.base_vec + 0.05 * new_base
        delta_raw = float(0.5 * np.sum(np.abs(P_obs - self.base_vec)))
        delta_hat = min(1.0, delta_raw * 3.0)
        self.stability_window.append(delta_hat)
        if action['learning_rate'] > 0 and len(self.P_obs_history) > 1:
            prev_P = self.P_obs_history[-2]
            diff = P_obs - prev_P
            self.U += action['learning_rate'] * np.outer(diff, W)
            self.U = np.clip(self.U, -5.0, 5.0)
        noise = self.rng.normal(0, 0.05, size=self.n)
        decay = 0.99 if action['strategy'] != 'ultra_conservative' else 0.995
        # structural effects from policy heads
        try:
            sg = float(action.get('sparsify_gain', 0.0))
            if sg > 0:
                self.U -= 0.001 * sg * np.sign(self.U)
            pg = float(action.get('prune_gain', 0.0))
            if pg > 0:
                thr = 0.05 * pg
                mask_small = (np.abs(self.U) < thr)
                self.U[mask_small] *= (1.0 - 0.05 * pg)
            rg = float(action.get('reconnect_gain', 0.0))
            if rg > 0:
                try:
                    num = max(1, int(3 * rg))
                    for _ in range(num):
                        i = int(self.rng.integers(self.K)); j = int(self.rng.integers(self.K))
                        self.U[i, j] += float(self.rng.normal(0, 0.1 * rg))
                except Exception:
                    pass
            stg = float(action.get('stability_gain', 0.0))
            decay = float(np.clip(decay * (1.0 - 0.02 * stg), 0.95, 0.995))
        except Exception:
            pass
        self.h = decay * self.h + noise
        self.gate_mid *= action['gate_adjust']
        self.gate_mid = np.clip(self.gate_mid, 0.3, 0.99)
        self.kd_eff *= action['kd_adjust']
        self.kd_eff = np.clip(self.kd_eff, 1.0, 15.0)
        return (delta_hat, P_obs)

    def _submit_to_workspace(self, grounded_exp: Dict, goal: Optional[Goal]):
        self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='grounded_qualia', content=grounded_exp, salience=0.7 * self.qualia.entropy + 0.3 * self.qualia.arousal, semantic_meaning=grounded_exp['semantic_meaning']))
        self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='self_model_state', content=self.self_model.to_dict(), salience=float(0.5 + 0.5 * (1.0 - self.self_model.meta_confidence)), semantic_meaning=f'self-model (meta_conf={self.self_model.meta_confidence:.2f})'))
        self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='belief_about_beliefs', content=self.self_model.belief_about_beliefs, salience=0.6, semantic_meaning='belief about beliefs'))
        if goal:
            self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='goal', content=goal, salience=goal.priority, semantic_meaning=f'   : {goal.type.value}'))
        self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='meta_awareness', content={'level': self.self_model.meta_awareness, 'knows_it_knows': self.self_model.knows_it_knows}, salience=self.self_model.meta_awareness, semantic_meaning=f'meta-awareness (level={self.self_model.meta_awareness:.2f})'))

        # Broadcast spatial focus/goal for workspace
        try:
            ax, ay = getattr(self, 'attn_xy', (0.5, 0.5))
            az = getattr(self, 'attn_xyz', (ax, ay, 0.5))[2]
            self.global_workspace.submit_for_competition(ConsciousContent(timestamp=self.t, content_type='spatial_focus', content={'attention': {'x': float(ax), 'y': float(ay), 'z': float(az)}, 'goal': self.spatial_goal, 'space_size': getattr(self, 'space_size', 256)}, salience=0.5, semantic_meaning='spatial attention/goal'))
        except Exception:
            pass

    def _experience_qualia(self, delta_hat: float, action: Dict[str, Any]):
        stability = self.world_state['stability']
        gap = float(abs(self.l_ctrl[self.hi] - self.l_ctrl[self.lo])) if len(self.l_ctrl) > max(self.hi, self.lo) else 0.0
        self.qualia.compute(delta_hat, action.get('exploration_factor', 0.0), gap, self.barrier_viols, float(self.self_model.meta_confidence), stability)

    def _evaluate_action_outcome(self, delta_hat: float, goal: Optional[Goal], action_plan: Dict) -> Dict[str, Any]:
        outcome = {}
        if len(self.self_model.state_history) >= 10:
            recent_deltas = [s.get('delta_hat', 0.5) for s in list(self.self_model.state_history)[-10:]]
            avg_delta = np.mean(recent_deltas)
            outcome['improvement'] = delta_hat < avg_delta * 0.95
            outcome['delta_change'] = delta_hat - avg_delta
        else:
            outcome['improvement'] = False
            outcome['delta_change'] = 0.0
        if len(self.qualia.history) >= 5:
            recent_arousals = [h['arousal'] for h in list(self.qualia.history)[-5:]]
            prev_arousal = recent_arousals[-2] if len(recent_arousals) >= 2 else recent_arousals[0]
            outcome['arousal_increased'] = self.qualia.arousal > prev_arousal * 1.1
            outcome['arousal_change'] = self.qualia.arousal - prev_arousal
        else:
            outcome['arousal_increased'] = False
            outcome['arousal_change'] = 0.0
        current_stability = self.world_state.get('stability', 0.5)
        current_energy = self.energy_ctrl.cognitive_energy / self.energy_ctrl.energy_capacity
        outcome['performance_degraded'] = current_stability < 0.3 and current_energy < 0.3
        if goal:
            outcome['goal_achieved'] = goal.evaluate_achievement(self.world_state, self.self_model.state_history)
        else:
            outcome['goal_achieved'] = False
        if len(self.self_model.prediction_errors) > 0:
            recent_error = self.self_model.prediction_errors[-1]
            outcome['prediction_accurate'] = recent_error < 0.2
            outcome['prediction_error'] = recent_error
        else:
            outcome['prediction_accurate'] = True
            outcome['prediction_error'] = 0.0
        success_factors = [outcome.get('improvement', False), not outcome.get('arousal_increased', False), not outcome.get('performance_degraded', False), outcome.get('goal_achieved', False), outcome.get('prediction_accurate', False)]
        outcome['overall_success'] = sum(success_factors) / len(success_factors)
        urgency_factors = []
        if current_stability < 0.4:
            urgency_factors.append(0.8)
        if self.qualia.arousal > 0.7:
            urgency_factors.append(0.7)
        if current_energy < 0.3:
            urgency_factors.append(0.6)
        if self.self_model.meta_confidence < 0.3:
            urgency_factors.append(0.5)
        outcome['urgency'] = max(urgency_factors) if urgency_factors else 0.3
        return outcome

    # ---- Structure snapshot + evaluation for experiments ----
    def _get_structure_snapshot(self) -> Dict[str, Any]:
        snap = {
            'weights': np.asarray(self.U).copy(),
            'gate_mid': float(self.gate_mid),
            'kd_eff': float(self.kd_eff),
            'K': int(self.K),
            'n': int(self.n),
        }
        try:
            snap['som_neuron_count'] = int(len(self.growing_som.neurons))
        except Exception:
            snap['som_neuron_count'] = 0
        return snap

    def _apply_structure_snapshot(self, snap: Dict[str, Any]) -> None:
        try:
            W = np.asarray(snap.get('weights'))
            if W.shape == self.U.shape:
                self.U = W.copy()
            self.gate_mid = float(snap.get('gate_mid', self.gate_mid))
            self.kd_eff = float(snap.get('kd_eff', self.kd_eff))
        except Exception:
            pass

    def _evaluate_structure(self, snap: Dict[str, Any], steps: int = 60) -> Dict[str, float]:
        bak = self._get_structure_snapshot()
        try:
            self._apply_structure_snapshot(snap)
            rewards, deltas, phis, stabs = [], [], [], []
            prev_meta = float(self.self_model.meta_awareness)
            for i in range(max(5, int(steps))):
                self.energy_ctrl.internal_clock += 1
                self.world_state = self._get_current_world_state()
                try:
                    stabs.append(float(self.world_state.get('stability', 0.5)))
                except Exception:
                    pass
                plan = self._decide_action(None, max(0.2, float(self.energy_ctrl.activation_level)))
                d, _ = self._execute_action(plan)
                self._experience_qualia(d, plan)
                deltas.append(d)
                if self.iit_enabled and (i % 10 == 0):
                    try:
                        sv = np.array([self.qualia.arousal, self.qualia.valence, self.qualia.entropy, self.qualia.engagement, self.qualia.frustration, self.self_model.meta_awareness, self.energy_ctrl.activation_level, self.unified_subject.unity_score])
                        phi = self.phi_calculator.compute_phi(state=sv, method='integrated')
                    except Exception:
                        phi = 0.0
                    phis.append(phi)
                r = float(-d + 0.5 * (self.self_model.meta_awareness - prev_meta))
                prev_meta = float(self.self_model.meta_awareness)
                rewards.append(r)
            return {
                'reward_mean': float(np.mean(rewards)) if rewards else 0.0,
                'td_error': float(np.mean(deltas)) if deltas else 0.0,
                'phi_median': float(np.median(phis)) if phis else 0.0,
                'stability': float(np.mean(stabs)) if stabs else 0.0,
            }
        finally:
            self._apply_structure_snapshot(bak)

    # ---- FeatureBank sandboxing helpers ----
    def _snapshot_feature_state(self) -> Dict[str, Any]:
        snap: Dict[str, Any] = {}
        try:
            if getattr(self, 'feature_bank', None) is not None:
                fb = self.feature_bank
                # shallow copy specs list
                snap['fb_max_dim'] = int(fb.max_dim)
                snap['fb_embed_dim'] = int(fb.embed_dim)
                try:
                    snap['fb_specs'] = [s for s in fb.specs]
                except Exception:
                    snap['fb_specs'] = []
            else:
                snap['fb_max_dim'] = None
                snap['fb_embed_dim'] = None
                snap['fb_specs'] = []
            # policy snapshot
            if getattr(self, 'policy', None) is not None:
                p = self.policy
                try:
                    snap['policy_W1'] = p.W1.copy(); snap['policy_b1'] = p.b1.copy(); snap['policy_W2'] = p.W2.copy(); snap['policy_b2'] = p.b2.copy()
                except Exception:
                    pass
            # sdm snapshot
            if getattr(self, 'sdm', None) is not None:
                s = self.sdm
                try:
                    snap['sdm_W1'] = s.W1.copy(); snap['sdm_b1'] = s.b1.copy(); snap['sdm_Wo'] = s.Wo.copy(); snap['sdm_bo'] = s.bo.copy()
                except Exception:
                    pass
        except Exception:
            pass
        return snap

    def _restore_feature_state(self, snap: Dict[str, Any]) -> None:
        try:
            # restore feature bank
            if snap.get('fb_max_dim') is not None:
                try:
                    self.feature_bank = FeatureBank(max_dim=int(snap.get('fb_max_dim', 128)), embed_dim=int(snap.get('fb_embed_dim', 32)))
                    # re-register specs
                    self.feature_bank.specs = []
                    for s in snap.get('fb_specs', []):
                        try:
                            self.feature_bank.register_spec(s)
                        except Exception:
                            pass
                except Exception:
                    pass
            # restore policy
            if 'policy_W1' in snap and getattr(self, 'policy', None) is not None:
                try:
                    self.policy.W1 = snap['policy_W1'].copy(); self.policy.b1 = snap['policy_b1'].copy(); self.policy.W2 = snap['policy_W2'].copy(); self.policy.b2 = snap['policy_b2'].copy()
                except Exception:
                    pass
            # restore sdm
            if 'sdm_W1' in snap and getattr(self, 'sdm', None) is not None:
                try:
                    self.sdm.W1 = snap['sdm_W1'].copy(); self.sdm.b1 = snap['sdm_b1'].copy(); self.sdm.Wo = snap['sdm_Wo'].copy(); self.sdm.bo = snap['sdm_bo'].copy()
                except Exception:
                    pass
        except Exception:
            pass

    def _sandbox_evaluate_specs(self, specs: List[FeatureSpec], steps: int = 40) -> Dict[str, float]:
        """Apply specs in a sandbox, run a short evaluation, and rollback.

        Returns evaluation metrics for the treatment (with specs) and also includes
        the baseline metrics under key 'baseline_*'. This function does not commit
        changes to the live feature_bank.
        """
        baseline_snap = self._snapshot_feature_state()
        try:
            # baseline evaluation
            base_metrics = self._evaluate_structure(self._get_structure_snapshot(), steps=steps)
            # apply specs temporarily
            try:
                # register specs on a fresh temp FeatureBank to avoid mutating live state
                fb_old = getattr(self, 'feature_bank', None)
                fb_temp = FeatureBank(max_dim=fb_old.max_dim if fb_old is not None else 128, embed_dim=fb_old.embed_dim if fb_old is not None else 32)
                # clone existing specs
                try:
                    for s in (getattr(fb_old, 'specs', []) if fb_old is not None else []):
                        fb_temp.register_spec(s)
                except Exception:
                    pass
                for s in specs:
                    try:
                        fb_temp.register_spec(s)
                    except Exception:
                        pass
                # estimate new dim and grow temp FB if needed
                added = int(sum(getattr(s, 'produced_dim', 1) for s in specs))
                cap = int(min(self._growth_hard_cap, fb_temp.max_dim + added))
                fb_temp.grow(cap, fb_temp.embed_dim)
                # swap in temp FB
                self.feature_bank, fb_backup = fb_temp, fb_old
            except Exception:
                fb_backup = getattr(self, 'feature_bank', None)
            # run evaluation with temp feature bank
            treat_metrics = self._evaluate_structure(self._get_structure_snapshot(), steps=steps)
            return_metrics = {'baseline_reward_mean': base_metrics.get('reward_mean', 0.0), 'treatment_reward_mean': treat_metrics.get('reward_mean', 0.0), 'treatment': treat_metrics}
            return return_metrics
        finally:
            # restore original feature bank and other snapshots
            try:
                # restore any other weights recorded in baseline_snap
                self._restore_feature_state(baseline_snap)
            except Exception:
                pass

    def run_meta_proposal_cycle(self, k: int = 1, steps: int = 40, commit_threshold: float = 0.0) -> Dict[str, Any]:
        """Run one meta-proposal cycle: propose specs, sandbox-evaluate, compute reward,
        call controller.observe(reward, action_meta), and commit growth if accepted.

        Returns a dict with keys: 'accepted' (bool), 'reward' (float), 'sandbox' (metrics), 'specs' (list).
        """
        res: Dict[str, Any] = {'accepted': False, 'reward': 0.0, 'sandbox': None, 'specs': []}
        try:
            # ensure controller exists
            if getattr(self, 'meta_feature_controller', None) is None:
                try:
                    in_dim = 128
                    self.meta_feature_controller = MetaFeatureController(in_dim=in_dim, hidden=128, rng=self.rngr.get('meta_feature'))
                except Exception:
                    self.meta_feature_controller = MetaFeatureController()
            ctrl: MetaFeatureController = self.meta_feature_controller
            # ensure controller controls main policy width
            try:
                ctrl.target_policy = self.policy
                ctrl.H_max = int(getattr(self.policy, 'hidden', 128))
            except Exception:
                pass
            # get features for controller
            try:
                feats = self.extract_embedding(target_dim=ctrl.in_dim, normalize=True)
            except Exception:
                feats = np.zeros((ctrl.in_dim,), dtype=np.float32)
            # propose
            specs = ctrl.decide(feats, k=k)
            res['specs'] = [s.name for s in specs]
            # sandbox eval
            sandbox = self._sandbox_evaluate_specs(specs, steps=steps)
            res['sandbox'] = sandbox
            base = float(sandbox.get('baseline_reward_mean', 0.0))
            treat = float(sandbox.get('treatment_reward_mean', 0.0))
            treat_metrics = sandbox.get('treatment', {}) or {}
            # reward design: weighted sum
            delta_reward = float(treat - base)
            phi = float(treat_metrics.get('phi_median', 0.0))
            td_err = float(treat_metrics.get('td_error', 0.0))
            stability = float(treat_metrics.get('stability', 0.0))
            added_dim = int(sum(getattr(s, 'produced_dim', 1) for s in specs))
            reward = 1.0 * delta_reward + 0.5 * phi - 0.2 * td_err + 0.05 * (stability - 0.5) - 0.001 * float(added_dim)
            res['reward'] = float(reward)
            # call observe with action metadata
            try:
                action_meta = {'features': feats, 'specs': [s.name for s in specs], 'mu': getattr(ctrl, 'last_mu', None), 'act_vec': getattr(ctrl, 'last_act_vec', None), 'sandbox': sandbox}
                try:
                    ctrl.observe(float(reward), action_meta)
                except Exception:
                    pass
            except Exception:
                pass
            # autosize (bandit) using treatment performance
            try:
                ctrl.autosize_step(perf=float(treat))
            except Exception:
                pass
            # decide commit
            accept = False
            try:
                if float(treat) >= float(base) + float(commit_threshold):
                    accept = True
            except Exception:
                accept = False
            if accept:
                # commit growth (force to bypass cooldown since we just sandboxed)
                try:
                    grown = self.grow_feature_bank(new_max_dim=None, new_embed_dim=None, specs=specs, force=True)
                    res['accepted'] = bool(grown)
                    try:
                        self.visualizer.add_major_event(f'MetaFeatureController committed {len(specs)} specs -> FB {self.feature_bank.max_dim}D')
                    except Exception:
                        pass
                except Exception:
                    res['accepted'] = False
            # metrics logging hook
            try:
                if getattr(self, 'feature_bank', None) is not None:
                    rec = {'ts': int(self.t), 'event': 'meta_cycle', 'base': base, 'treat': treat, 'delta': float(treat - base), 'accepted': bool(res.get('accepted', False))}
                    self.feature_bank._log_jsonl(getattr(self.feature_bank, '_metrics_path', None), rec)
            except Exception:
                pass
            return res
        except Exception:
            return res

    # ---- Vision source public API ----
    def set_vision_folder(self, path: str) -> int:
        try:
            exts = {'.png', '.jpg', '.jpeg', '.bmp'}
            files = []
            for name in os.listdir(path):
                p = os.path.join(path, name)
                if os.path.isfile(p) and os.path.splitext(p)[1].lower() in exts:
                    files.append(p)
            files.sort()
            if files:
                self._vision_frames = files
                self._vision_idx = 0
                self.vision_mode = 'folder'
                return len(files)
        except Exception:
            pass
        return 0

    def set_vision_loop(self, enabled: bool) -> bool:
        try:
            self._vision_loop = bool(enabled)
            return True
        except Exception:
            return False

    def set_vision_shuffle(self, enabled: bool) -> bool:
        try:
            self._vision_shuffle = bool(enabled)
            return True
        except Exception:
            return False

    def set_vision_camera(self, index: int = 0) -> bool:
        try:
            import cv2
            # Reduce OpenCV log noise if possible
            try:
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            except Exception:
                pass
            # Release any previous camera
            try:
                if getattr(self, '_vision_camera', None) is not None:
                    self._vision_camera.release()
            except Exception:
                pass
            indices = [int(index)] if index >= 0 else list(range(10))
            backends = [getattr(cv2, 'CAP_DSHOW', None), getattr(cv2, 'CAP_MSMF', None), None]
            for i in indices:
                for be in backends:
                    try:
                        cap = cv2.VideoCapture(int(i), be) if be is not None else cv2.VideoCapture(int(i))
                        if cap is not None and cap.isOpened():
                            try:
                                cap.set(getattr(cv2, 'CAP_PROP_FRAME_WIDTH', 3), 640)
                                cap.set(getattr(cv2, 'CAP_PROP_FRAME_HEIGHT', 4), 480)
                            except Exception:
                                pass
                            self._vision_camera = cap
                            self.vision_mode = 'camera'
                            return True
                        else:
                            try:
                                if cap is not None:
                                    cap.release()
                            except Exception:
                                pass
                    except Exception:
                        continue
        except Exception:
            pass
        return False

    def push_external_frame(self, frame: 'np.ndarray') -> None:
        self._pushed_frame = frame
        self.vision_mode = 'push'

    def clear_vision_source(self) -> None:
        try:
            if self._vision_camera is not None:
                try:
                    self._vision_camera.release()
                except Exception:
                    pass
        except Exception:
            pass
        self._vision_camera = None
        self._vision_frames = []
        self._pushed_frame = None
        self.vision_mode = 'internal'

    def _reflect_and_learn(self, delta_hat: float, goal: Optional[Goal]):
        state = {'delta_hat': delta_hat, 'm': 0.0, 'stability': self.world_state['stability'], 'meta_confidence': self.self_model.meta_confidence}
        self.self_model.log_state(state)
        pred_error = self.self_model.evaluate_prediction(state)
        if goal:
            outcome_success = goal.evaluate_achievement(self.world_state, self.self_model.state_history)
        else:
            stability_ok = self.world_state['stability'] > 0.3
            energy_ok = self.energy_ctrl.cognitive_energy > 20.0
            qualia_ok = self.qualia.valence > 0.4
            conditions = [stability_ok, energy_ok, qualia_ok]
            outcome_success = sum(conditions) >= 2
        if not outcome_success:
            outcome_success = np.random.random() < 0.5
        self.self_model.update_beliefs(pred_error, outcome_success, self.world_state['stability'])
        needs_revision, quality, revision_level = self.meta_meta.evaluate_model_quality(self.self_model, self.t)
        if needs_revision:
            if revision_level == 1:
                print(f' Self-model revision L1 (quality={quality:.3f})')
            elif revision_level == 2:
                print(f' Self-model revision L2 - Meta-params (quality={quality:.3f})')
            elif revision_level == 3:
                print(f' Self-model revision L3 - Structural (quality={quality:.3f})')
            elif revision_level == 4:
                print(f' EMERGENCY L4 - Full Reset (quality={quality:.3f})')
            # As part of structural revisions (level >=2) allow the model to
            # autonomously grow its perceptual embedding capacity before
            # executing revision actions.
            try:
                if revision_level >= 2:
                    try:
                        # Consult GrowthTrigger before running meta-proposal
                        should_run = True
                        try:
                            gt = getattr(self, 'growth_trigger', None)
                            if gt is not None:
                                try:
                                    feat_dim = getattr(gt, 'in_dim', 64)
                                    feats = self.extract_embedding(target_dim=feat_dim, normalize=True)
                                except Exception:
                                    feats = np.zeros((getattr(gt, 'in_dim', 64),), dtype=np.float32)
                                try:
                                    trig, meta = gt.decide(feats)
                                    should_run = bool(trig)
                                except Exception:
                                    should_run = True
                        except Exception:
                            should_run = True
                        if should_run:
                            res = self.run_meta_proposal_cycle(k=1, steps=40, commit_threshold=0.0)
                            # let GrowthTrigger observe the outcome
                            try:
                                gt = getattr(self, 'growth_trigger', None)
                                if gt is not None and isinstance(res, dict):
                                    rew = float(res.get('reward', 0.0)) if res.get('reward') is not None else 0.0
                                    try:
                                        gt.observe(rew, meta={'accepted': bool(res.get('accepted', False)), 'specs': res.get('specs', [])})
                                    except Exception:
                                        try:
                                            gt.observe(rew)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                            if res.get('accepted'):
                                try:
                                    print(f' Auto-grown FeatureBank -> {self.feature_bank.max_dim}D')
                                except Exception:
                                    pass
                    except Exception:
                        pass
            except Exception:
                pass
            self.self_model.execute_revision(revision_level)
            policy_status = self.meta_meta.get_revision_policy_status()
            print(f"   Consecutive revisions: {policy_status['consecutive_revisions']}")
            print(f"   Failed count: {policy_status['failed_revision_count']}")
            if policy_status['emergency_mode']:
                print(f"   EMERGENCY MODE: {policy_status['emergency_cooldown']} steps remaining")
            # If we previously initiated a meta-driven growth, allow the controller
            # to observe a delayed reward signal after a short evaluation window.
            try:
                pending = getattr(self, '_meta_grow_pending', None)
                if pending is not None:
                    # require at least a small number of steps to have passed
                    if int(self.t) - int(pending.get('t', 0)) >= 8:
                        if hasattr(self, 'meta_controller') and self.meta_controller is not None:
                            try:
                                quality_after = float(quality)
                                quality_before = float(pending.get('quality_before', quality_after))
                                action = int(pending.get('action', 0))
                                # reward = improvement in quality minus small penalty for growth size
                                reward = float(quality_after - quality_before) - 0.01 * (float(action) / 100.0)
                                # Prepare features and ensure controller has them if decide wasn't called earlier
                                try:
                                    feats = np.asarray(pending.get('features', np.zeros((getattr(self.meta_controller, 'in_dim', 6),), dtype=np.float32)), dtype=np.float32)
                                except Exception:
                                    feats = np.zeros((getattr(self.meta_controller, 'in_dim', 6),), dtype=np.float32)
                                try:
                                    if getattr(self.meta_controller, 'last_features', None) is None:
                                        try:
                                            self.meta_controller.last_features = feats
                                            self.meta_controller.last_action = int(action)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                # snapshot before
                                try:
                                    if hasattr(self.meta_controller, 'snapshot'):
                                        pre = self.meta_controller.snapshot()
                                    elif hasattr(self.meta_controller, 'log_snapshot'):
                                        pre = self.meta_controller.log_snapshot()
                                    else:
                                        pre = {}
                                    W_before = pre.get('W_norm', None)
                                    sigma_before = pre.get('sigma', None)
                                except Exception:
                                    W_before = sigma_before = None
                                # perform observe (REINFORCE update)
                                try:
                                    # new API: observe(reward, action=...)
                                    try:
                                        self.meta_controller.observe(float(reward), action=int(action))
                                    except TypeError:
                                        # older signature fallbacks
                                        try:
                                            self.meta_controller.observe(feats, int(action), float(reward))
                                        except Exception:
                                            try:
                                                import traceback as _tb
                                                _tb.print_exc()
                                            except Exception:
                                                pass
                                    except Exception:
                                        try:
                                            import traceback as _tb
                                            _tb.print_exc()
                                        except Exception:
                                            pass
                                except Exception:
                                    try:
                                        import traceback as _tb
                                        _tb.print_exc()
                                    except Exception:
                                        pass
                                # snapshot after
                                try:
                                    if hasattr(self.meta_controller, 'snapshot'):
                                        post = self.meta_controller.snapshot()
                                    elif hasattr(self.meta_controller, 'log_snapshot'):
                                        post = self.meta_controller.log_snapshot()
                                    else:
                                        post = {}
                                    W_after = post.get('W_norm', None)
                                    sigma_after = post.get('sigma', None)
                                except Exception:
                                    W_after = sigma_after = None
                                # append a trial CSV row for post-hoc analysis
                                try:
                                    outdir = getattr(self, 'outdir', os.getcwd())
                                    os.makedirs(outdir, exist_ok=True)
                                    logp = os.path.join(outdir, 'meta_trials.csv')
                                    header = not os.path.exists(logp)
                                    with open(logp, 'a', encoding='utf-8') as f:
                                        if header:
                                            f.write('t,action,quality_before,quality_after,reward,W_before,W_after,sigma_before,sigma_after\n')
                                        f.write(f"{int(self.t)},{int(action)},{quality_before:.6f},{quality_after:.6f},{reward:.6f},{W_before},{W_after},{sigma_before},{sigma_after}\n")
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        try:
                            self._meta_grow_pending = None
                        except Exception:
                            self._meta_grow_pending = None
            except Exception:
                try:
                    import traceback as _tb
                    _tb.print_exc()
                except Exception:
                    pass
        elif len(self.meta_meta.quality_before_revision) > len(self.meta_meta.quality_after_revision):
            self.meta_meta.record_revision_outcome(quality)
        if hasattr(self.self_model, 'belief_about_beliefs'):
            if hasattr(self.self_model, 'belief_stability'):
                self.self_model.belief_about_beliefs['confidence_in_stability_belief'] = float(self.self_model.belief_stability)
            if hasattr(self.self_model, 'belief_adaptation'):
                self.self_model.belief_about_beliefs['confidence_in_adaptation_belief'] = float(self.self_model.belief_adaptation)
            if hasattr(self.self_model, 'belief_prediction'):
                self.self_model.belief_about_beliefs['confidence_in_prediction_belief'] = float(self.self_model.belief_prediction)

    def _make_bar(self, value: float, width: int=20) -> str:
        filled = int(value * width)
        empty = width - filled
        bar = '#' * filled + '-' * empty
        return bar

    def _should_terminate(self, goal: Optional[Goal]) -> bool:
        return False

    def _log_runtime_event(self, kind: str, **payload: Any) -> None:
        try:
            path = os.path.join(self.outdir, "llm_adapter.log")
            rec = {"kind": str(kind), "t": int(getattr(self, "t", 0))}
            rec.update(payload)
            _write_jsonl_safe(path, rec)
        except Exception:
            pass

    def _phi_threshold_policy(self) -> Dict[str, float]:
        hist = []
        try:
            hist = list(getattr(self.phi_calculator, "phi_history", []) or [])
        except Exception:
            hist = []
        pol = _compute_phi_policy_from_history(hist, cfg=self._adaptive_threshold_cfg)
        t_now = int(getattr(self, "t", 0))
        if (
            self._phi_policy_last is None
            or any(abs(float(pol.get(k, 0.0)) - float(self._phi_policy_last.get(k, 0.0))) > 1e-6 for k in ("floor", "low", "mid", "high", "very_high", "announce_high"))
        ) and (t_now - int(self._phi_policy_last_t)) >= 50:
            self._log_runtime_event(
                "phi_threshold_policy",
                floor=float(pol.get("floor", 0.01)),
                low=float(pol.get("low", 0.1)),
                mid=float(pol.get("mid", 0.3)),
                high=float(pol.get("high", 0.5)),
                very_high=float(pol.get("very_high", 0.7)),
                announce_high=float(pol.get("announce_high", 0.5)),
                history_len=int(len(hist)),
            )
            self._phi_policy_last_t = t_now
        self._phi_policy_last = dict(pol)
        return pol

    @staticmethod
    def _copy_state_overlap(src_sd: Dict[str, Any], dst_sd: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(dst_sd)
        for k, dst_t in dst_sd.items():
            src_t = src_sd.get(k)
            if src_t is None:
                continue
            try:
                if tuple(src_t.shape) == tuple(dst_t.shape):
                    out[k] = src_t.clone().to(device=dst_t.device, dtype=dst_t.dtype)
                    continue
                if src_t.ndim != dst_t.ndim:
                    continue
                merged = dst_t.clone()
                slices = tuple(slice(0, min(int(src_t.shape[i]), int(dst_t.shape[i]))) for i in range(src_t.ndim))
                merged[slices] = src_t[slices].to(device=dst_t.device, dtype=dst_t.dtype)
                out[k] = merged
            except Exception:
                continue
        return out

    def _recreate_policy_with_transfer(self, in_dim: int, out_dim: int = 12) -> None:
        old_policy = getattr(self, "policy", None)
        hidden = max(128, int(in_dim))
        new_policy = PolicyMLP(in_dim=int(in_dim), out_dim=int(out_dim), hidden=hidden, rng=self.rngr.get('policy'))
        try:
            if old_policy is not None and hasattr(old_policy, "state_dict") and hasattr(new_policy, "load_state_dict"):
                src_sd = old_policy.state_dict()
                dst_sd = new_policy.state_dict()
                merged = self._copy_state_overlap(src_sd, dst_sd)
                new_policy.load_state_dict(merged, strict=False)
        except Exception:
            pass
        self.policy = new_policy

    def _project_obs_to_dim(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).ravel()
        tgt = int(max(1, target_dim))
        if arr.size == tgt:
            return arr.astype(np.float32, copy=False)
        if not bool(self._observation_adapter_cfg.get("enabled", True)):
            out = np.zeros((tgt,), dtype=np.float32)
            take = min(arr.size, tgt)
            if take > 0:
                out[:take] = arr[:take]
            return out
        if (
            self._obs_adapter_W is None
            or int(self._obs_adapter_in_dim) != int(arr.size)
            or int(self._obs_adapter_out_dim) != int(tgt)
        ):
            rng = self.rngr.get("obs_adapter")
            scale = 1.0 / max(1.0, float(np.sqrt(max(1, arr.size))))
            self._obs_adapter_W = rng.normal(0.0, scale, size=(int(arr.size), int(tgt))).astype(np.float32)
            self._obs_adapter_in_dim = int(arr.size)
            self._obs_adapter_out_dim = int(tgt)
            sig = (int(arr.size), int(tgt))
            if sig != self._obs_adapter_last_sig:
                self._log_runtime_event("obs_adapter", in_dim=int(arr.size), out_dim=int(tgt), mode="project")
                self._obs_adapter_last_sig = sig
        x = arr
        try:
            eps = float(max(1e-9, self._observation_adapter_cfg.get("projection_eps", 1e-6)))
            mu = float(np.mean(x)) if x.size > 0 else 0.0
            sd = float(np.std(x)) if x.size > 0 else 0.0
            x = (x - mu) / (sd + eps)
        except Exception:
            pass
        try:
            out = np.tanh(x @ self._obs_adapter_W).astype(np.float32)
            return out
        except Exception:
            out = np.zeros((tgt,), dtype=np.float32)
            take = min(arr.size, tgt)
            if take > 0:
                out[:take] = arr[:take]
            return out

    def _policy_obs_adapted(self) -> np.ndarray:
        raw = self._policy_obs()
        raw = np.asarray(raw, dtype=np.float32).ravel()
        target_cfg = int(max(0, int(self._observation_adapter_cfg.get("target_policy_dim", 0))))
        policy_dim = int(getattr(getattr(self, "policy", None), "in_dim", 0) or 0)
        if policy_dim > 0:
            target = policy_dim
        elif target_cfg > 0:
            target = target_cfg
        else:
            target = int(raw.size)
        return self._project_obs_to_dim(raw, target_dim=target)

    def _maybe_adaptive_feature_bank_growth(self) -> Optional[Dict[str, Any]]:
        if getattr(self, "feature_bank", None) is None:
            return None
        if not hasattr(self.feature_bank, "adaptive_grow"):
            return None
        try:
            proposal = self.feature_bank.adaptive_grow(float(self.t), self, apply=False)
            if not proposal:
                return None
            new_dim = int(proposal.get("new_max_dim", getattr(self.feature_bank, "max_dim", 0)))
            new_embed = int(proposal.get("new_embed_dim", getattr(self.feature_bank, "embed_dim", 0)))
            applied = self.grow_feature_bank(new_max_dim=new_dim, new_embed_dim=new_embed, force=True)
            proposal["applied"] = bool(applied)
            return proposal
        except Exception:
            return None

    def _update_visualization(self):
        phi = 0.0
        if len(self.log_buffer) > 0:
            phi = self.log_buffer[-1].get('phi', 0.0)
        mem_stats = self.episodic_memory.get_statistics()
        grounded = self.conceptual_space.ground_experience(self.qualia)
        current_experience = grounded['nearest_concept']
        som_stats = self.growing_som.get_statistics()
        meta_awareness = self.self_model.meta_awareness if hasattr(self.self_model, 'meta_awareness') else 0.0
        knows_it_knows = self.self_model.knows_it_knows if hasattr(self.self_model, 'knows_it_knows') else False
        phi_policy = self._phi_threshold_policy()
        system_state = {'phi': phi, 'meta_awareness': meta_awareness, 'strange_loop': knows_it_knows, 'energy': self.energy_ctrl.cognitive_energy, 'qualia': {'arousal': self.qualia.arousal, 'valence': self.qualia.valence, 'entropy': self.qualia.entropy, 'engagement': self.qualia.engagement, 'frustration': self.qualia.frustration}, 'unity': self.unified_subject.unity_score, 'memories': mem_stats['total_memories'], 'memory_consolidation': mem_stats['avg_consolidation'], 'current_experience': current_experience, 'neuron_count': som_stats.get('neuron_count', 4), 'connection_count': som_stats.get('connection_count', 4), 'growth_events': som_stats.get('growth_events', 0), 'u_matrix': self.U, 'timestamp': self.t, 'vision_mode': getattr(self, 'vision_mode', 'internal')}
        system_state['phi_policy'] = dict(phi_policy)
        system_state['pred_err_map'] = self._scope_build_pred_err_map()
        # Self-vision Phase 2: foveated retina + disparity/flow-based depth
        try:
            retina = self._vision_build_retina(size=(64, 64), foveate=True)
            prev_frame = getattr(self, '_vision_prev', None)
            # Compute flow/depth before updating prev
            if prev_frame is None:
                u = v = None
                depth_map = None
            else:
                u, v = self._vision_optical_flow(prev_frame, retina)
                depth_map = self._vision_depth_from_flow(u, v)
            vision_err = self._vision_compute_error(retina)
            try:
                vb = float(getattr(self, '_vision_mix_beta', 0.5))
            except Exception:
                vb = 0.5
            system_state['pred_err_map'] = self._mix_err_maps(system_state['pred_err_map'], vision_err, beta=vb)
            # Provide the retina to visualizer for realistic display
            try:
                system_state['retina'] = retina
            except Exception:
                pass
        except Exception:
            depth_map = None
        system_state['td_error'] = float(self._scope_compute_td_error())
        system_state['gw_ignition'] = float(self._scope_get_gw_ignition())
        # pass scope one-bit preference to visualizer
        try:
            system_state['scope_one_bit'] = bool(self.scope_one_bit)
        except Exception:
            pass
        # expose arousal at top-level for visualizer/Scope drivers
        try:
            system_state['arousal'] = float(self.qualia.arousal)
        except Exception:
            pass
        self.visualizer.update(system_state)
        # Compose spatial map image with overlays (attention crosshair + optional goal)
        try:
            import numpy as _np
            if 'retina' not in locals():
                retina = self._vision_build_retina(size=(64, 64))
            if 'vision_err' not in locals():
                vision_err = self._vision_compute_error(retina)
            # Use mixed error map as spatial base; step attention (with depth cue) and render overlay
            # Use flow-based depth if available, else gradient-based
            if depth_map is None:
                _, _, _, depth_cue = self._vision_features(retina)
                dmap = depth_cue
            else:
                dmap = depth_map
            spatial_base = system_state.get('pred_err_map', vision_err)
            self._space_step_attention(spatial_base, depth_map=dmap)
            space_img = self._space_compose_image(spatial_base, size=self.space_size, depth_map=dmap)
            if getattr(self, 'show_spatial_overlay', False):
                self.visualizer.scope_image = space_img
        except Exception:
            pass
        # Light vision qualia coupling (small, stable nudge) + summary for policy
        try:
            import numpy as _np
            contrast, entropy, edge_density, depth_cue_local = self._vision_features(retina)
            self.qualia.engagement = float(_np.clip(0.98 * self.qualia.engagement + 0.02 * (0.5 + 0.5 * contrast), 0.0, 1.0))
            self.qualia.entropy = float(_np.clip(0.98 * self.qualia.entropy + 0.02 * entropy, 0.0, 1.0))
            # Store compact vision summary for action policy
            try:
                self._vision_summary = {
                    'r_mean': float(_np.mean(retina)),
                    'g_mean': float(_np.mean(vision_err)),
                    'b_mean': float(_np.mean(depth_cue_local)),
                    'contrast': float(contrast),
                    'entropy': float(entropy),
                    'edge_density': float(edge_density),
                }
            except Exception:
                pass
        except Exception:
            pass
        if self.self_model.knows_it_knows and self.t > 0:
            if not hasattr(self, '_loop_announced'):
                self.visualizer.add_major_event('STRANGE LOOP EMERGED!')
                self._loop_announced = True
        ann_thr = float(phi_policy.get("announce_high", phi_policy.get("high", 0.5)))
        ann_hys = float(max(0.0, self._adaptive_threshold_cfg.get("announce_hysteresis", 0.02)))
        ann_cd = int(max(0, self._adaptive_threshold_cfg.get("announce_cooldown", 200)))
        if phi >= ann_thr and self.t > 0:
            if (
                (int(self.t) - int(self._last_phi_announce_t)) >= ann_cd
                and float(phi - self._last_phi_announce_value) >= ann_hys
            ):
                self.visualizer.add_major_event(f'High phi: {phi:.3f}')
                self._last_phi_announce_t = int(self.t)
                self._last_phi_announce_value = float(phi)
        if mem_stats['total_memories'] == 1 and (not hasattr(self, '_first_memory')):
            self.visualizer.add_major_event('First episodic memory!')
            self._first_memory = True
        mem_count = mem_stats['total_memories']
        if mem_count >= 100 and (not hasattr(self, '_mem_100')):
            self.visualizer.add_major_event('100 memories stored!')
            self._mem_100 = True
        if mem_count >= 500 and (not hasattr(self, '_mem_500')):
            self.visualizer.add_major_event('500 memories!')
            self._mem_500 = True
        conn = self.visualizer.total_connections
        if conn > 1000 and (not hasattr(self, '_conn_1k')):
            self.visualizer.add_major_event('1,000 connections!')
            self._conn_1k = True
        if conn > 5000 and (not hasattr(self, '_conn_5k')):
            self.visualizer.add_major_event('5,000 connections!!')
            self._conn_5k = True

    
    # --- SCOPE: real-signal adapters (no proxies) ---
    def _scope_build_pred_err_map(self):
        import numpy as _np
        neurons = getattr(self.growing_som, 'neurons', [])
        if not neurons:
            raise RuntimeError("No SOM neurons available for pred_err_map")
        xs = [int(n['position'][0]) for n in neurons]
        ys = [int(n['position'][1]) for n in neurons]
        min_x = int(min(xs))
        min_y = int(min(ys))
        max_x = int(max(xs))
        max_y = int(max(ys))
        W = (max_x - min_x) + 1
        H = (max_y - min_y) + 1
        if W <= 0 or H <= 0:
            return _np.zeros((1, 1), dtype=_np.float32)
        grid = _np.zeros((H, W), dtype=_np.float32)
        counts = _np.zeros((H, W), dtype=_np.int32)
        for n in neurons:
            x, y = int(n['position'][0]), int(n['position'][1])
            xi = x - min_x
            yi = y - min_y
            if xi < 0 or yi < 0 or xi >= W or yi >= H:
                # skip out-of-range positions defensively
                continue
            reserr = n.get('recent_errors', None)
            if reserr and len(reserr) > 0:
                val = float(_np.mean(list(reserr)))
            else:
                val = float(n.get('error', 0.0))
            grid[yi, xi] += val
            counts[yi, xi] += 1
        mask = counts > 0
        grid[mask] = grid[mask] / counts[mask]
        if _np.isfinite(grid).all():
            gmin, gmax = float(_np.min(grid)), float(_np.max(grid))
            if gmax > gmin:
                grid = (grid - gmin) / (gmax - gmin)
            else:
                grid[:] = 0.0
        return grid

    def _scope_compute_td_error(self):
        import numpy as _np
        hist = list(getattr(self, 'reward_history', []))
        if len(hist) < 2:
            return 0.0
        r_t = float(hist[-1])
        base = float(_np.mean(hist[-min(10, len(hist)-1):-1]))
        return r_t - base

    def _scope_get_gw_ignition(self):
        gw = getattr(self, 'global_workspace', None)
        if gw is None:
            return 0.0
        focus = getattr(gw, 'attention_focus', None)
        contents = getattr(gw, 'current_contents', [])
        return 1.0 if (focus is not None and contents) else 0.0

    # ------------------------ Self-Vision: Phase 1 (retina + error mix) ------------------------
    def _vision_resize_nn(self, a, out_h: int, out_w: int):
        import numpy as _np
        a = _np.asarray(a, dtype=_np.float32)
        h, w = a.shape[:2]
        if h <= 0 or w <= 0:
            return _np.zeros((out_h, out_w), dtype=_np.float32)
        row_idx = (_np.linspace(0, max(0, h - 1), out_h)).astype(_np.int32)
        col_idx = (_np.linspace(0, max(0, w - 1), out_w)).astype(_np.int32)
        return a[row_idx][:, col_idx]

    def _vision_build_retina(self, size=(64, 64), foveate: bool=True):
        import numpy as _np
        # Optional external vision sources: folder/camera/push
        try:
            mode = getattr(self, 'vision_mode', 'internal')
            if mode == 'folder' and getattr(self, '_vision_frames', None):
                n = len(self._vision_frames)
                if n > 0:
                    idx = None
                    if getattr(self, '_vision_shuffle', False):
                        try:
                            idx = int(self.rng.integers(n))
                        except Exception:
                            idx = int(_np.random.randint(n))
                    else:
                        if self._vision_idx < n:
                            idx = self._vision_idx
                            self._vision_idx += 1
                        else:
                            if getattr(self, '_vision_loop', True):
                                self._vision_idx = 1
                                idx = 0
                            else:
                                return None
                    p = self._vision_frames[idx]
                    try:
                        from PIL import Image as _PIL_Image
                        im = _PIL_Image.open(p).convert('L')
                        arr = _np.asarray(im, dtype=_np.float32) / 255.0
                        return self._vision_resize_nn(arr, size[0], size[1])
                    except Exception:
                        pass
            elif mode == 'camera' and getattr(self, '_vision_camera', None) is not None:
                try:
                    import cv2 as _cv2
                    ok, frame = self._vision_camera.read()
                    if ok and frame is not None:
                        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
                        arr = gray.astype(_np.float32) / 255.0
                        return self._vision_resize_nn(arr, size[0], size[1])
                except Exception:
                    pass
            elif mode == 'push' and getattr(self, '_pushed_frame', None) is not None:
                arr = _np.asarray(self._pushed_frame)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    arr = _np.dot(arr[..., :3].astype(_np.float32), _np.array([0.2989, 0.5870, 0.1140], dtype=_np.float32))
                arr = arr.astype(_np.float32)
                if arr.dtype != _np.float32:
                    arr = arr.astype(_np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
                arr = _np.clip(arr, 0.0, 1.0)
                return self._vision_resize_nn(arr, size[0], size[1])
        except Exception:
            pass
        # Prefer richer neural_map if available; fallback to U
        src = None
        try:
            nm = getattr(self.visualizer, 'neural_map', None)
            if nm is not None:
                src = _np.asarray(nm, dtype=_np.float32)
        except Exception:
            src = None
        if src is None:
            u = _np.asarray(getattr(self, 'U', _np.zeros((8, 8), dtype=_np.float32)), dtype=_np.float32)
            if u.ndim == 1:
                u = u[None, :]
            if u.ndim != 2:
                u = _np.zeros((8, 8), dtype=_np.float32)
            src = u
        src = _np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)
        smin, smax = float(_np.min(src)), float(_np.max(src))
        if smax > smin:
            base = (src - smin) / (smax - smin)
        else:
            base = _np.zeros_like(src)
        # Low-res base
        lo = self._vision_resize_nn(base, size[0], size[1])
        if not foveate:
            return lo
        # Foveate around goal or attention
        cx, cy = (self.attn_xy if hasattr(self, 'attn_xy') else (0.5, 0.5))
        if getattr(self, 'spatial_goal', None) and 'xy' in self.spatial_goal:
            cx, cy = self.spatial_goal['xy']
        rad = float(self.spatial_goal.get('radius', 0.15) if getattr(self, 'spatial_goal', None) else 0.15)
        # Build higher-res and paste ROI
        hi = self._vision_resize_nn(base, size[0]*2, size[1]*2)
        H, W = size
        cxp, cyp = int(cx * (W - 1)), int(cy * (H - 1))
        r_pix = max(4, int(rad * min(H, W)))
        x0, x1 = max(0, cxp - r_pix), min(W - 1, cxp + r_pix)
        y0, y1 = max(0, cyp - r_pix), min(H - 1, cyp + r_pix)
        # Map ROI to hi coordinates
        hx0, hx1 = x0*2, x1*2 + 1
        hy0, hy1 = y0*2, y1*2 + 1
        roi_hi = hi[hy0:hy1+1, hx0:hx1+1]
        roi_lo = self._vision_resize_nn(roi_hi, (y1 - y0 + 1), (x1 - x0 + 1))
        out = lo.copy()
        out[y0:y1+1, x0:x1+1] = roi_lo
        return out

    def _vision_optical_flow(self, prev, curr, lam: float=1e-3):
        import numpy as _np
        p = _np.asarray(prev, dtype=_np.float32)
        c = _np.asarray(curr, dtype=_np.float32)
        # Gradients from prev
        gy, gx = _np.gradient(p)
        It = c - p
        denom = gx*gx + gy*gy + lam
        u = -It * gx / denom
        v = -It * gy / denom
        return u, v

    def _vision_depth_from_flow(self, u, v):
        import numpy as _np
        mag = _np.sqrt(_np.asarray(u, dtype=_np.float32)**2 + _np.asarray(v, dtype=_np.float32)**2)
        # depth ~ inverse flow magnitude
        d = 1.0 / (mag + 1e-3)
        d = d / (d.max() + 1e-8)
        return d

    def _vision_compute_error(self, frame):
        import numpy as _np
        f = _np.asarray(frame, dtype=_np.float32)
        prev = getattr(self, '_vision_prev', None)
        if prev is None:
            err = _np.zeros_like(f)
        else:
            err = _np.abs(f - _np.asarray(prev, dtype=_np.float32))
        self._vision_prev = f.copy()
        e_min, e_max = float(err.min(initial=0.0)), float(err.max(initial=1.0))
        if e_max > e_min:
            err = (err - e_min) / (e_max - e_min)
        else:
            err = _np.zeros_like(err)
        return err

    def _vision_features(self, frame):
        import numpy as _np
        f = _np.asarray(frame, dtype=_np.float32)
        contrast = float(f.std())
        hist = _np.histogram(_np.clip(f, 0.0, 1.0), bins=32, range=(0.0, 1.0))[0].astype(_np.float32)
        p = hist / (hist.sum() + 1e-8)
        entropy = float((-_np.sum(p * _np.log2(p + 1e-12))) / _np.log2(32.0))
        gy, gx = _np.gradient(f)
        grad = _np.hypot(gx, gy)
        thr = float(_np.percentile(grad, 90.0)) if grad.size else 0.0
        edge_density = float((grad > thr).mean()) if grad.size else 0.0
        g_min, g_max = float(grad.min(initial=0.0)), float(grad.max(initial=1.0))
        depth_cue = (grad - g_min) / (g_max - g_min + 1e-8) if g_max > g_min else _np.zeros_like(grad)
        return contrast, entropy, edge_density, depth_cue

    def describe_current_vision(self) -> Dict[str, Any]:
        """Return a concise natural-language summary and metrics of current vision input.

        Uses the same retina build path as policy to avoid any fake signals.
        """
        import numpy as _np
        try:
            retina = self._vision_build_retina(size=(96, 96), foveate=False)
        except Exception:
            return {'summary': '', 'metrics': {}}
        f = _np.asarray(retina, dtype=_np.float32)
        f = _np.clip(f, 0.0, 1.0)
        mean = float(f.mean())
        std = float(f.std())
        hist = _np.histogram(f, bins=32, range=(0.0, 1.0))[0].astype(_np.float32)
        p = hist / (hist.sum() + 1e-8)
        ent = float((-_np.sum(p * _np.log2(p + 1e-12))) / _np.log2(32.0))
        gy, gx = _np.gradient(f)
        grad = _np.hypot(gx, gy)
        thr = float(_np.percentile(grad, 90.0)) if grad.size else 0.0
        edge_density = float((grad > thr).mean()) if grad.size else 0.0
        ang = _np.degrees(_np.arctan2(gy, gx))
        ang = _np.abs(ang)
        def frac(mask):
            m = _np.asarray(mask)
            return float(m.mean()) if m.size else 0.0
        vertical_edge = frac((ang <= 22.5) | (ang >= 157.5))
        horizontal_edge = frac((ang >= 67.5) & (ang <= 112.5))
        diag_edge = frac(~(((ang <= 22.5) | (ang >= 157.5)) | ((ang >= 67.5) & (ang <= 112.5))))
        # coarse region count via block thresholds
        blocks = []
        ph = pw = 12
        H, W = f.shape
        for y in range(0, H, ph):
            for x in range(0, W, pw):
                blk = f[y:y+ph, x:x+pw]
                if blk.size:
                    blocks.append((float(blk.mean()), float(blk.std())))
        bright_patches = sum(1 for m, s in blocks if (m > (mean + 0.15) and s > 0.08))
        parts = []
        parts.append('dark' if mean < 0.25 else ('bright' if mean > 0.75 else 'balanced'))
        parts.append('low variance' if std < 0.12 else ('high variance' if std > 0.35 else 'moderate variance'))
        if edge_density > 0.12:
            parts.append('rich edges')
        if max(vertical_edge, horizontal_edge, diag_edge) < 0.35:
            parts.append('weak orientation')
        else:
            if horizontal_edge == max(vertical_edge, horizontal_edge, diag_edge):
                parts.append('horizontal texture dominant')
            elif vertical_edge == max(vertical_edge, horizontal_edge, diag_edge):
                parts.append('vertical texture dominant')
            else:
                parts.append('diagonal texture dominant')
        if bright_patches >= 12:
            parts.append('many bright patches')
        elif bright_patches <= 2 and edge_density < 0.06 and std < 0.1:
            parts.append('flat and dim')
        summary = ' | '.join(parts)
        metrics = {
            'mean': round(mean, 3), 'std': round(std, 3), 'entropy': round(ent, 3),
            'edge_density': round(edge_density, 3),
            'orient_h': round(horizontal_edge, 3), 'orient_v': round(vertical_edge, 3), 'orient_d': round(diag_edge, 3),
            'regions': int(bright_patches),
        }
        return {'summary': summary, 'metrics': metrics}

    def _mix_err_maps(self, som_map, vision_err, beta: float=0.5):
        import numpy as _np
        v = _np.asarray(vision_err, dtype=_np.float32)
        try:
            s = _np.asarray(som_map, dtype=_np.float32)
        except Exception:
            s = _np.zeros_like(v)
        if s.shape != v.shape:
            s = self._vision_resize_nn(s, v.shape[0], v.shape[1])
        s_min, s_max = float(s.min(initial=0.0)), float(s.max(initial=1.0))
        if s_max > s_min:
            s = (s - s_min) / (s_max - s_min)
        else:
            s = _np.zeros_like(s)
        mix = _np.clip(beta * s + (1.0 - beta) * v, 0.0, 1.0)
        return mix

    # ------------------------ Spatial space and goals ------------------------
    def set_spatial_goal(self, x: float, y: float, radius: float=0.1):
        x = float(max(0.0, min(1.0, x)))
        y = float(max(0.0, min(1.0, y)))
        r = float(max(0.01, min(0.5, radius)))
        self.spatial_goal = {'xy': (x, y), 'radius': r}

    def set_spatial_goal3d(self, x: float, y: float, z: float, radius: float=0.1):
        x = float(max(0.0, min(1.0, x)))
        y = float(max(0.0, min(1.0, y)))
        z = float(max(0.0, min(1.0, z)))
        r = float(max(0.01, min(0.5, radius)))
        self.spatial_goal = {'xyz': (x, y, z), 'radius': r}

    def _space_step_attention(self, base_map, depth_map=None):
        import numpy as _np
        # Move attention toward goal or toward highest-salience location
        try:
            bm = _np.asarray(base_map, dtype=_np.float32)
            h, w = bm.shape[:2]
            ax, ay = self.attn_xy
            az = getattr(self, 'attn_xyz', (ax, ay, 0.5))[2]
            if self.spatial_goal and ('xy' in self.spatial_goal or 'xyz' in self.spatial_goal):
                if 'xyz' in self.spatial_goal:
                    gx, gy, gz = self.spatial_goal['xyz']
                else:
                    gx, gy = self.spatial_goal['xy']
                    gz = az
            else:
                # pick current global maximum as implicit goal
                idx = int(_np.argmax(bm))
                gy, gx = (idx // w) / max(1, h - 1), (idx % w) / max(1, w - 1)
                # use local depth (if any) as target z
                if depth_map is not None:
                    dm = _np.asarray(depth_map, dtype=_np.float32)
                    if dm.shape != bm.shape:
                        dm = self._vision_resize_nn(dm, h, w)
                    gz = float(_np.clip(dm[int(gy*(h-1)), int(gx*(w-1))], 0.0, 1.0))
                else:
                    gz = az
            # step toward goal
            step = 0.08
            dx, dy = gx - ax, gy - ay
            dz = gz - az
            ax += step * dx
            ay += step * dy
            az += step * dz
            ax = float(max(0.0, min(1.0, ax)))
            ay = float(max(0.0, min(1.0, ay)))
            az = float(max(0.0, min(1.0, az)))
            self.attn_xy = (ax, ay)
            self.attn_xyz = (ax, ay, az)
        except Exception:
            pass

    def _space_compose_image(self, base_map, size: int=256, depth_map=None):
        import numpy as _np
        bm = _np.asarray(base_map, dtype=_np.float32)
        bmin, bmax = float(_np.min(bm)), float(_np.max(bm))
        if bmax > bmin:
            bm = (bm - bmin) / (bmax - bmin)
        else:
            bm = _np.zeros_like(bm)
        up = self._vision_resize_nn(bm, size, size)
        # grayscale to RGB
        img = _np.dstack([(up*255).astype(_np.uint8)]*3)
        # Overlay spatial goal
        if not getattr(self, 'show_spatial_overlay', False):
            return img
        # overlay attention crosshair (green)
        try:
            ax, ay = self.attn_xy
            az = getattr(self, 'attn_xyz', (ax, ay, 0.5))[2]
            cx = int(ax * (size - 1))
            cy = int(ay * (size - 1))
            half = max(6, size // 32)
            x0, x1 = max(0, cx - half), min(size - 1, cx + half)
            y0, y1 = max(0, cy - half), min(size - 1, cy + half)
            # color crosshair with z: low z  reen, high z  yan/blue mix
            zc = float(_np.clip(az, 0.0, 1.0))
            cross = _np.array([int(0), int(255*(1.0-zc)), int(255*zc)], dtype=_np.uint8)
            img[cy, x0:x1+1] = cross
            img[y0:y1+1, cx] = cross
            # overlay goal ring (yellow)
            if self.spatial_goal and 'xy' in self.spatial_goal:
                gx, gy = self.spatial_goal['xy']
                gr = float(self.spatial_goal.get('radius', 0.1))
                gx_i = int(gx * (size - 1))
                gy_i = int(gy * (size - 1))
                rr = max(6, int(gr * size))
                yy, xx = _np.ogrid[:size, :size]
                dist2 = (xx - gx_i)**2 + (yy - gy_i)**2
                ring = _np.logical_and(dist2 >= rr*rr - rr, dist2 <= rr*rr + rr)
                img[ring] = _np.array([255, 255, 0], dtype=_np.uint8)
            elif self.spatial_goal and 'xyz' in self.spatial_goal:
                gx, gy, gz = self.spatial_goal['xyz']
                gr = float(self.spatial_goal.get('radius', 0.1))
                gx_i = int(gx * (size - 1))
                gy_i = int(gy * (size - 1))
                rr = max(6, int(gr * size))
                yy, xx = _np.ogrid[:size, :size]
                dist2 = (xx - gx_i)**2 + (yy - gy_i)**2
                ring = _np.logical_and(dist2 >= rr*rr - rr, dist2 <= rr*rr + rr)
                # z encodes ring color (blue-ish when deep)
                ring_col = _np.array([255, int(255*(1.0-gz)), int(255*gz)], dtype=_np.uint8)
                img[ring] = ring_col
        except Exception:
            pass
        return img

    def _log_state(self, delta_hat: float, goal: Optional[Goal]):
        consciousness_summary = self.global_workspace.get_conscious_summary()
        phi = 0.0
        consciousness_metric = 0.0
        if self.iit_enabled and self.t % 5 == 0:
            try:
                state_vector = np.array([self.qualia.arousal, self.qualia.valence, self.qualia.entropy, self.qualia.engagement, self.qualia.frustration, self.self_model.meta_awareness, self.energy_ctrl.activation_level, self.unified_subject.unity_score])
                phi = self.phi_calculator.compute_phi(state=state_vector, method='integrated')
                consciousness_metric = phi
                if self.network_visualizer:
                    neurons_data = []
                    for neuron in self.growing_som.neurons:
                        neurons_data.append({'pos': neuron['position'], 'activation': neuron['activation_count'], 'age': neuron['age']})
                    connections_data = []
                    for conn in self.growing_som.connections:
                        connections_data.append({'from': conn['from'], 'to': conn['to'], 'strength': conn['strength']})
                    recent_growth = hasattr(self, '_last_growth_event') and self._last_growth_event
                    self.network_visualizer.update_data(neurons=neurons_data, connections=connections_data, timestamp=self.t, growth_event=recent_growth, phi=phi, energy=self.energy_ctrl.cognitive_energy, qualia={'arousal': self.qualia.arousal, 'valence': self.qualia.valence, 'entropy': self.qualia.entropy, 'engagement': self.qualia.engagement, 'frustration': self.qualia.frustration}, beliefs={'stability': self.self_model.belief_stability, 'adaptation': self.self_model.belief_adaptation, 'prediction': self.self_model.belief_prediction}, meta_awareness=self.self_model.meta_awareness, unity=self.unified_subject.unity_score, activation=self.energy_ctrl.activation_level)
                    self._last_growth_event = False
                consciousness_metric = phi * self.unified_subject.unity_score * self.self_model.meta_awareness
                if self.t % 5 == 0:
                    qualia_vec = np.array([self.qualia.arousal, self.qualia.valence, self.qualia.entropy, self.qualia.engagement, self.qualia.frustration])
                    qualia_variance = np.var(qualia_vec)
                    phi_history = list(self.phi_calculator.phi_history) if hasattr(self.phi_calculator, 'phi_history') else [phi]
                    if len(phi_history) > 10:
                        phi_percentile = np.percentile(phi_history, 75)
                        should_encode = phi > phi_percentile or qualia_variance > 0.15
                    else:
                        should_encode = True
                    if should_encode:
                        qualia_temp = QualiaState(arousal=qualia_vec[0], valence=qualia_vec[1], entropy=qualia_vec[2], engagement=qualia_vec[3], frustration=qualia_vec[4])
                        grounded = self.conceptual_space.ground_experience(qualia_temp)
                        experience_name = grounded['nearest_concept']
                        if self.self_model.knows_it_knows:
                            narrative = str(self.self_model.introspect(depth=2))
                        else:
                            narrative = f'exp={experience_name}'
                        context = {'goal_type': goal.type.value if goal else 'none', 'goal_priority': goal.priority if goal else 0.0, 'energy_level': self.energy_ctrl.cognitive_energy, 'strange_loop': self.self_model.knows_it_knows, 'iteration': self.t}
                        self.episodic_memory.encode_experience(experience_name=experience_name, qualia_vector=qualia_vec, phi_value=phi, context=context, narrative=narrative)
                if self.t % 100 == 0 and self.t > 0:
                    self.episodic_memory.consolidate()
            except Exception as e:
                print(f'   calculation failed: {str(e)}')
                phi = 0.0
        # Normalize qualia fields: keep existing q_* keys but also expose plain names
        qd = self.qualia.to_dict()
        # backward-compatible plain names used in some event payloads / older logs
        qd.update({
            'tension': float(self.qualia.arousal),
            'harmony': float(self.qualia.valence),
            'entropy': float(self.qualia.entropy),
            'flow': float(self.qualia.engagement),
            'resistance': float(self.qualia.frustration)
        })
        self.log_buffer.append({'t': self.t, 'delta_hat': delta_hat, 'stability': self.world_state['stability'], 'goal_type': goal.type.value if goal else 'none', 'goal_priority': goal.priority if goal else 0.0, 'energy_level': self.energy_ctrl.cognitive_energy / self.energy_ctrl.energy_capacity, 'activation_level': self.energy_ctrl.activation_level, 'strange_loop_active': int(self.self_model.knows_it_knows), 'meta_awareness': self.self_model.meta_awareness, 'unity_score': self.unified_subject.unity_score, 'conscious_focus': consciousness_summary[:50], 'phi': phi, 'consciousness_metric': consciousness_metric, 'sdm_z_mse': float(getattr(self, '_sdm_last_mse', np.nan)) if 'np' in globals() else None, 'plan_score': float(getattr(self, '_last_plan_score', np.nan)) if 'np' in globals() else None, **qd, **self.self_model.to_dict()})
        try:
            if (self.t % 100 == 0) and (len(self.log_buffer) >= 40):
                self._synthesize_policy_rule()
        except Exception:
            pass

        # --- Self-dynamics model online training (prev->current) ---
        try:
            if getattr(self, 'sdm', None) is not None:
                # Build current z via FeatureBank
                if getattr(self, 'feature_bank', None) is not None:
                    z_now = self.feature_bank.build(self)
                    z_now = self._torch_shared_z(z_now)
                else:
                    z_now = self._policy_obs()
                # Targets from current signals
                r_now = float(getattr(self, '_last_shaped_reward', 0.0))
                y_heads = np.array([
                    float(delta_hat),
                    float(self.world_state.get('stability', 0.5)),
                    float(getattr(self.self_model, 'meta_awareness', 0.0)),
                    r_now
                ], dtype=np.float32)
                z_prev = getattr(self, '_sdm_prev_z', None)
                a_prev = getattr(self, '_sdm_prev_a', None)
                if z_prev is not None and a_prev is not None:
                    xa = np.concatenate([self._align_z_for_sdm(self._torch_shared_z(z_prev.astype(np.float32))), a_prev.astype(np.float32)], axis=0)
                    y_all = np.concatenate([y_heads, self._align_z_for_sdm(z_now)], axis=0)
                    # push to replay
                    self._sdm_replay.append((xa, y_all))
                    if len(self._sdm_replay) > self._sdm_capacity:
                        self._sdm_replay = self._sdm_replay[-self._sdm_capacity:]
                    # train small minibatch
                    try:
                        mb = min(64, len(self._sdm_replay))
                        if mb >= 8:
                            idx = np.random.choice(len(self._sdm_replay), size=mb, replace=False)
                            X = np.stack([self._sdm_replay[i][0] for i in idx], axis=0)
                            Y = np.stack([self._sdm_replay[i][1] for i in idx], axis=0)
                            try:
                                _loss = self.sdm.train_batch(X, Y)
                                try:
                                    self._sdm_last_mse = float(_loss)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                    except Exception:
                        pass
                # cache current z for next step
                self._sdm_prev_z = self._torch_shared_z(z_now)
        except Exception:
            pass

    def _print_summary(self, elapsed: float):
        log_df = pd.DataFrame(self.log_buffer) if self.log_buffer else pd.DataFrame()
        loop_activation_time = None
        if not log_df.empty and 'strange_loop_active' in log_df.columns:
            loop_active = log_df[log_df['strange_loop_active'] == 1]
            if len(loop_active) > 0:
                loop_activation_time = int(loop_active.iloc[0]['t'])
        memory_stats = self.episodic_memory.get_statistics()
        # Simplified, safe ASCII summary to avoid encoding/syntax issues
        summary_lines = [
            f"State at t={self.t:,}",
            f"Runtime: {elapsed:.1f}s ({self.t / max(1e-9, elapsed):.0f} it/s)",
            f"Seed: {self.seed}",
            f"Strange Loop: {'ON' if self.self_model.knows_it_knows else 'OFF'}" + (f" since t={loop_activation_time:,}" if loop_activation_time else ""),
            f"Meta-Awareness: {self.self_model.meta_awareness:.3f}",
            f"Unity: {self.unified_subject.unity_score:.3f}",
            f"Experiences: {len(self.unified_subject.subjective_experiences):,}",
            f"Episodic Memories: {memory_stats.get('total_memories', 0)} (encoded={memory_stats.get('total_encoded', 0)}, retrieved={memory_stats.get('total_retrieved', 0)})",
            f"Activation: {self.energy_ctrl.activation_level:.3f}",
            f"Energy: {self.energy_ctrl.cognitive_energy:.1f}/{self.energy_ctrl.energy_capacity:.1f}",
        ]
        if not log_df.empty and 'phi' in log_df.columns:
            summary_lines.append(f"Phi: {log_df['phi'].iloc[-1]:.4f}")
        if not log_df.empty and 'consciousness_metric' in log_df.columns:
            summary_lines.append(f"Consciousness Metric: {log_df['consciousness_metric'].iloc[-1]:.4f}")
        summary_lines.append(f"IIT Enabled: {self.iit_enabled}")
        summary = "\n".join(summary_lines)
        print(summary)

    def _flush_logs(self):
        if self.log_buffer:
            df = pd.DataFrame(self.log_buffer)
            df.to_csv(self.log_path, mode='a', index=False, header=not os.path.exists(self.log_path))
            self.log_buffer = []
        if self.event_log_buffer:
            df = pd.DataFrame(self.event_log_buffer)
            df.to_csv(self.event_log_path, mode='a', index=False, header=not os.path.exists(self.event_log_path))
            self.event_log_buffer = []

    def _recent_policy_avg(self, window: int = 50) -> float:
        try:
            import numpy as _np
            hist = list(self.policy_reward_history)
            if not hist:
                return 0.0
            return float(_np.mean(hist[-min(window, len(hist)) :]))
        except Exception:
            return 0.0

    def _save_policy_snapshots(self) -> None:
        os.makedirs(self.outdir, exist_ok=True)
        recent_avg = self._recent_policy_avg(window=50)
        # Save only when improved; keep a single best file (policy.npz)
        try:
            # Always save a rolling snapshot for debugging
            try:
                last_path = os.path.join(self.outdir, 'policy_last.npz')
                if hasattr(self.policy, 'theta'):
                    np.savez(last_path, theta=self.policy.theta, t=self.t, recent_avg=recent_avg)
                else:
                    np.savez(last_path, W1=self.policy.W1, b1=self.policy.b1, W2=self.policy.W2, b2=self.policy.b2, Wv=self.policy.Wv, bv=self.policy.bv, t=self.t, recent_avg=recent_avg)
            except Exception:
                pass
            if recent_avg > self._best_policy_score + 1e-9:
                path = os.path.join(self.outdir, 'policy.npz')
                if hasattr(self.policy, 'theta'):
                    np.savez(path, theta=self.policy.theta, t=self.t, recent_avg=recent_avg)
                else:
                    np.savez(path, W1=self.policy.W1, b1=self.policy.b1, W2=self.policy.W2, b2=self.policy.b2, Wv=self.policy.Wv, bv=self.policy.bv, t=self.t, recent_avg=recent_avg)
                self._best_policy_score = recent_avg
                print(f' Policy: new best checkpoint (avg@50={recent_avg:.4f}) at t={self.t}')
        except Exception:
            pass

    def policy_report(self) -> Dict[str, Any]:
        try:
            import numpy as _np
            if hasattr(self.policy, 'theta') and getattr(self.policy, 'theta') is not None:
                theta = getattr(self.policy, 'theta')
                theta_norm = float(_np.linalg.norm(theta))
                theta_shape = tuple(theta.shape)
            else:
                theta_norm = float(_np.linalg.norm(getattr(self.policy, 'W1')) + _np.linalg.norm(getattr(self.policy, 'W2')))
                theta_shape = (getattr(self.policy, 'W1').shape, getattr(self.policy, 'W2').shape)
            return {
                't': int(self.t),
                'recent_avg_50': self._recent_policy_avg(50),
                'theta_shape': theta_shape,
                'theta_norm': theta_norm,
                'best_recent_avg_50': float(self._best_policy_score),
            }
        except Exception:
            return {
                't': int(self.t),
                'recent_avg_50': 0.0,
                'theta_shape': None,
                'theta_norm': 0.0,
                'best_recent_avg_50': float(self._best_policy_score)
            }

    # ============================== ARC SUITE (integrated) ==============================
    def solve_arc_dir(self, arc_dir: str, out_dir: Optional[str] = None, save_preds: bool = True, trace: bool = True, max_time_per: float = 5.0, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None, require_solve: bool = False, total_time_cap: float = 60.0, files: Optional[List[str]] = None, shuffle: bool = False, max_files: Optional[int] = None, rng_seed: Optional[int] = None) -> Dict[str, Any]:
        """Run ARC problems from a folder of JSONs. Self-adjusts small search budget.

        - arc_dir: folder containing ARC JSON files (train/test structure)
        - out_dir: where to save predictions <id>.json (optional)
        - save_preds: save predictions to out_dir if True
        - trace: include light trace info in progress callback
        - max_time_per: soft per-problem time budget in seconds (auto grows a bit if improving)
        - progress_cb: callback(dict) invoked per problem step and on completion
        - require_solve: if True, keep retrying a problem until solved or total_time_cap reached
        - total_time_cap: overall cap per problem when require_solve=True
        """
        import time as _time
        results = {'total': 0, 'solved': 0, 'failed': 0, 'details': []}
        arc_dir = str(arc_dir)
        if files is None:
            files = []
            try:
                for root, _dirs, names in os.walk(arc_dir):
                    for name in sorted(names):
                        if name.lower().endswith('.json'):
                            files.append(os.path.join(root, name))
            except Exception:
                if progress_cb:
                    progress_cb({'type': 'arc_error', 'msg': f'arc_dir not found: {arc_dir}'})
                return results
        # optional shuffle / limit
        if shuffle:
            _rng = np.random.default_rng(rng_seed if rng_seed is not None else self.seed)
            _rng.shuffle(files)
        if isinstance(max_files, int) and max_files is not None and max_files > 0:
            files = files[:max_files]
        # per-session context for ARC (e.g., palette mappings)
        # load cache and compile macros
        try:
            self._arc_load_cache()
            self._arc_compile_macros(top_k=10)
        except Exception:
            pass
        self._arc_ctx: Dict[str, Any] = {}
        for p in files:
            prob = self._arc_load_problem(p)
            # compute simple palette mapping hints from train pairs (used by remap op)
            try:
                self._arc_ctx[prob['id']] = {
                    'palette_map': self._arc_palette_map_from_train(prob['train']),
                    'out_mode': self._arc_out_mode_from_train(prob['train'])
                }
            except Exception:
                self._arc_ctx[prob['id']] = {'palette_map': {}, 'out_mode': 0}
            start = _time.perf_counter()
            try:
                self._arc_current_id = prob['id']
            except Exception:
                pass
            # Attempt search; if require_solve, escalate time budget with retries up to total_time_cap
            attempt = 0
            best_prog, best_info = None, {'train_acc': 0.0}
            local_time = float(max_time_per)
            while True:
                attempt += 1
                prog, info = self._arc_search(prob, max_time_per=local_time, trace=trace, progress_cb=progress_cb)
                if info.get('train_acc', 0.0) >= best_info.get('train_acc', 0.0):
                    best_prog, best_info = (prog, info)
                dt_now = _time.perf_counter() - start
                ok = bool(best_info.get('train_acc', 0.0) >= 1.0)
                if ok:
                    break
                if not require_solve:
                    break
                if dt_now >= float(total_time_cap):
                    break
                # escalate budget and notify GUI
                local_time = min(local_time * 2.0, max(5.0, float(total_time_cap) - dt_now))
                if progress_cb:
                    progress_cb({'type': 'arc_retry', 'id': prob['id'], 'attempt': attempt, 'best_acc': float(best_info.get('train_acc', 0.0)), 'elapsed': dt_now, 'next_budget': float(local_time)})
            prog, info = best_prog, best_info
            preds = [self._arc_apply_program(prob['id'], prog, g) for g in prob['test']] if prog is not None else []
            ok = bool(info.get('train_acc', 0.0) >= 1.0)
            # update cache with successful or strong program
            try:
                if prog is not None:
                    self._arc_update_cache(prob['id'], prog, float(info.get('train_acc', 0.0)), float(info.get('soft_train', 0.0)))
            except Exception:
                pass
            if save_preds and out_dir and preds:
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    with open(os.path.join(out_dir, f"{prob['id']}.json"), 'w', encoding='utf-8') as f:
                        json.dump({'output': [a.astype(int).tolist() for a in preds]}, f)
                except Exception:
                    pass
            dt = _time.perf_counter() - start
            meta = {
                'id': prob['id'],
                'path': p,
                'train_acc': info.get('train_acc', 0.0),
                'soft_train': info.get('soft_train', 0.0),
                'prog_len': len(prog) if prog else 0,
                'time': dt,
                'solved': ok,
            }
            results['details'].append(meta)
            results['total'] += 1
            results['solved'] += int(ok)
            results['failed'] += int(not ok)
            if progress_cb:
                # include preview grid for GUI (first test pred if any)
                preview = preds[0].astype(int).tolist() if preds else None
                cb = {'type': 'arc_done', **meta, 'preview': preview}
                progress_cb(cb)
        # save cache at end of run
        try:
            self._arc_save_cache()
        except Exception:
            pass
        return results

    def _arc_load_problem(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        train_pairs = []
        for pair in data.get('train', []):
            tin = np.asarray(pair['input'], dtype=np.int16)
            tout = np.asarray(pair['output'], dtype=np.int16)
            train_pairs.append((tin, tout))
        tests = [np.asarray(t['input'], dtype=np.int16) for t in data.get('test', [])]
        return {'id': os.path.splitext(os.path.basename(path))[0], 'path': path, 'train': train_pairs, 'test': tests}

    # ---- ARC cache (program priors/macros) ----
    def _arc_cache_path(self) -> str:
        try:
            os.makedirs(self.outdir, exist_ok=True)
        except Exception:
            pass
        return os.path.join(self.outdir, 'arc_cache.json')

    def _arc_load_cache(self) -> None:
        p = self._arc_cache_path()
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    self._arc_cache = json.load(f)
        except Exception:
            self._arc_cache = {'solved': {}, 'macro_counts': {}}

    def _arc_save_cache(self) -> None:
        p = self._arc_cache_path()
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(self._arc_cache, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _arc_update_cache(self, prob_id: str, program: List[str], hard: float, soft: float) -> None:
        if not program:
            return
        try:
            # skip macro-containing programs to avoid recursion
            if any(isinstance(op, str) and op.startswith('macro_') for op in program):
                return
            # Store solved program
            if hard >= 1.0 or soft >= 0.95:
                self._arc_cache.setdefault('solved', {})[prob_id] = list(program)
            # Bump macro counts for short useful programs (2..5)
            if 2 <= len(program) <= 5:
                key = '|'.join(program)
                mc = self._arc_cache.setdefault('macro_counts', {})
                mc[key] = int(mc.get(key, 0)) + 1
        except Exception:
            pass

    def _arc_compile_macros(self, top_k: int = 10) -> List[List[str]]:
        try:
            counts = self._arc_cache.get('macro_counts', {})
            items = sorted(counts.items(), key=lambda kv: int(kv[1]), reverse=True)
            macros = []
            for i, (k, _cnt) in enumerate(items[:max(0, int(top_k))]):
                prog = k.split('|') if k else []
                if 2 <= len(prog) <= 7:
                    macros.append(prog)
            self._arc_macros = macros
            return macros
        except Exception:
            self._arc_macros = []
            return []

    # ---- ARC minimal DSL ops ----
    @staticmethod
    def _arc_rot90(g: np.ndarray) -> np.ndarray: return np.rot90(g, 1)
    @staticmethod
    def _arc_rot180(g: np.ndarray) -> np.ndarray: return np.rot90(g, 2)
    @staticmethod
    def _arc_rot270(g: np.ndarray) -> np.ndarray: return np.rot90(g, 3)
    @staticmethod
    def _arc_flip_h(g: np.ndarray) -> np.ndarray: return np.flip(g, axis=1)
    @staticmethod
    def _arc_flip_v(g: np.ndarray) -> np.ndarray: return np.flip(g, axis=0)
    @staticmethod
    def _arc_transpose(g: np.ndarray) -> np.ndarray: return np.transpose(g)
    @staticmethod
    def _arc_identity(g: np.ndarray) -> np.ndarray: return g.copy()

    def _arc_connected_components(self, g: np.ndarray) -> Tuple[np.ndarray, int]:
        # 4-connectivity CC labeling per color (simple DFS)
        H, W = g.shape
        lab = -np.ones_like(g, dtype=np.int32)
        comp = 0
        for y in range(H):
            for x in range(W):
                if lab[y, x] != -1:
                    continue
                color = int(g[y, x])
                stack = [(y, x)]
                lab[y, x] = comp
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and lab[ny, nx] == -1 and int(g[ny, nx]) == color:
                            lab[ny, nx] = comp
                            stack.append((ny, nx))
                comp += 1
        return lab, comp

    def _arc_bbox_of_largest_region(self, g: np.ndarray) -> Tuple[int,int,int,int]:
        lab, n = self._arc_connected_components(g)
        best = None
        best_area = -1
        for c in range(n):
            ys, xs = np.where(lab == c)
            if ys.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            area = (y1 - y0 + 1) * (x1 - x0 + 1)
            if area > best_area:
                best_area = area
                best = (y0, x0, y1, x1)
        return best if best is not None else (0, 0, g.shape[0]-1, g.shape[1]-1)

    @staticmethod
    def _arc_fill_bbox(g: np.ndarray, bbox: Tuple[int,int,int,int], color: int) -> np.ndarray:
        y0, x0, y1, x1 = bbox
        out = g.copy()
        out[y0:y1+1, x0:x1+1] = int(color)
        return out

    def _arc_mode_color(self, g: np.ndarray) -> int:
        vals, counts = np.unique(g, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _arc_out_mode_from_train(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> int:
        vals = []
        for _tin, tout in train_pairs:
            vals.extend(list(np.ravel(tout)))
        if not vals:
            return 0
        vs, cs = np.unique(np.asarray(vals, dtype=np.int16), return_counts=True)
        return int(vs[np.argmax(cs)])

    def _arc_keep_largest(self, g: np.ndarray) -> np.ndarray:
        lab, n = self._arc_connected_components(g)
        if n <= 1:
            return g.copy()
        best = None
        best_area = -1
        for c in range(n):
            ys, xs = np.where(lab == c)
            area = ys.size
            if area > best_area:
                best_area = area
                best = c
        out = g.copy()
        bg = self._arc_mode_color(g)
        out[lab != best] = bg
        return out

    def _arc_crop_largest(self, g: np.ndarray) -> np.ndarray:
        y0, x0, y1, x1 = self._arc_bbox_of_largest_region(g)
        return g[y0:y1+1, x0:x1+1].copy()

    def _arc_move_largest_tl(self, g: np.ndarray) -> np.ndarray:
        # translate largest object's bbox to top-left; keep image size
        H, W = g.shape
        y0, x0, y1, x1 = self._arc_bbox_of_largest_region(g)
        sub = g[y0:y1+1, x0:x1+1]
        bg = self._arc_mode_color(g)
        out = np.full_like(g, bg)
        h, w = sub.shape
        out[0:h, 0:w] = sub
        return out

    def _arc_center_largest(self, g: np.ndarray) -> np.ndarray:
        # center largest object's bbox; keep image size
        H, W = g.shape
        y0, x0, y1, x1 = self._arc_bbox_of_largest_region(g)
        sub = g[y0:y1+1, x0:x1+1]
        bg = self._arc_mode_color(g)
        out = np.full_like(g, bg)
        h, w = sub.shape
        cy = max(0, (H - h) // 2)
        cx = max(0, (W - w) // 2)
        out[cy:cy+h, cx:cx+w] = sub
        return out

    def _arc_paint_largest_outmode(self, g: np.ndarray) -> np.ndarray:
        lab, n = self._arc_connected_components(g)
        if n <= 1:
            return g.copy()
        # find largest component label
        best, best_area = None, -1
        for c in range(n):
            area = int(np.sum(lab == c))
            if area > best_area:
                best_area, best = area, c
        color = int(self._arc_ctx.get(getattr(self, '_arc_current_id', ''), {}).get('out_mode', self._arc_mode_color(g)))
        out = g.copy()
        if best is not None:
            out[lab == best] = color
        return out

    def _arc_bbox_border_largest(self, g: np.ndarray) -> np.ndarray:
        y0, x0, y1, x1 = self._arc_bbox_of_largest_region(g)
        out = g.copy()
        color = int(self._arc_ctx.get(getattr(self, '_arc_current_id', ''), {}).get('out_mode', self._arc_mode_color(g)))
        if y1 >= y0 and x1 >= x0:
            out[y0, x0:x1+1] = color
            out[y1, x0:x1+1] = color
            out[y0:y1+1, x0] = color
            out[y0:y1+1, x1] = color
        return out

    def _arc_palette_map_from_train(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[int, int]:
        # Build a simple color mapping by matching sorted color frequencies input->output
        from collections import Counter
        cin = Counter()
        cout = Counter()
        for tin, tout in train_pairs:
            cin.update(list(np.ravel(tin)))
            cout.update(list(np.ravel(tout)))
        cin_sorted = [c for c,_ in cin.most_common()]
        cout_sorted = [c for c,_ in cout.most_common()]
        m = {}
        for i, c in enumerate(cin_sorted):
            if i < len(cout_sorted):
                m[int(c)] = int(cout_sorted[i])
        return m

    def _arc_remap_hist(self, prob_id: str, g: np.ndarray) -> np.ndarray:
        mapping = {}
        try:
            mapping = dict(self._arc_ctx.get(prob_id, {}).get('palette_map', {}))
        except Exception:
            mapping = {}
        out = g.copy()
        if mapping:
            v = np.vectorize(lambda x: mapping.get(int(x), int(x)))
            out = v(out).astype(out.dtype)
        return out

    # ---- Program representation & search ----
    
    # ==================== AUTONOMOUS DATASET LEARNING ====================
    
    # ==================== AUTONOMOUS RESEARCH SYSTEM ====================
    def _introspect_internal_state(self) -> Dict[str, Any]:
        """Deep introspection of internal dynamics to detect research needs"""
        state = {
            'timestamp': time.time(),
            'consciousness': {},
            'dynamics': {},
            'tensions': {},
            'gradients': {}
        }
        
        # Consciousness state
        state['consciousness']['phi'] = float(self.phi_calculator.phi_history[-1]) if self.phi_calculator.phi_history else 0.0
        state['consciousness']['meta_awareness'] = float(self.self_model.meta_awareness)
        state['consciousness']['meta_confidence'] = float(self.self_model.meta_confidence)
        state['consciousness']['knows_it_knows'] = bool(self.self_model.knows_it_knows)
        
        # Qualia dynamics
        state['dynamics']['arousal'] = float(self.qualia.arousal)
        state['dynamics']['valence'] = float(self.qualia.valence)
        state['dynamics']['entropy'] = float(self.qualia.entropy)
        state['dynamics']['engagement'] = float(self.qualia.engagement)
        state['dynamics']['frustration'] = float(self.qualia.frustration)
        
        # Energy & activation
        state['dynamics']['cognitive_energy'] = float(self.energy_ctrl.cognitive_energy)
        state['dynamics']['activation_level'] = float(self.energy_ctrl.activation_level)
        
        # Detect internal tensions (mismatches, conflicts)
        state['tensions']['confidence_mismatch'] = abs(state['consciousness']['meta_confidence'] - state['consciousness']['meta_awareness'])
        state['tensions']['energy_arousal_gap'] = abs(state['dynamics']['cognitive_energy'] / 100.0 - state['dynamics']['arousal'])
        state['tensions']['valence_frustration_conflict'] = state['dynamics']['valence'] + state['dynamics']['frustration']  # Should be inversely related
        state['tensions']['engagement_entropy_gap'] = abs(state['dynamics']['engagement'] - (1.0 - state['dynamics']['entropy']))
        
        # Compute gradients (rate of change)
        if hasattr(self, '_last_introspection'):
            dt = state['timestamp'] - self._last_introspection['timestamp']
            if dt > 0:
                for key in ['arousal', 'valence', 'entropy', 'frustration']:
                    current = state['dynamics'][key]
                    past = self._last_introspection['dynamics'].get(key, current)
                    state['gradients'][f'{key}_rate'] = (current - past) / dt
                
                # Meta awareness rate
                state['gradients']['meta_awareness_rate'] = (
                    state['consciousness']['meta_awareness'] - 
                    self._last_introspection['consciousness'].get('meta_awareness', state['consciousness']['meta_awareness'])
                ) / dt
        
        self._last_introspection = state
        return state

    def _research_m3_state_summary(self) -> Dict[str, Any]:
        """Compact M3 state snapshot for research grounding."""
        try:
            phi = float(self.phi_calculator.phi_history[-1]) if self.phi_calculator.phi_history else 0.0
        except Exception:
            phi = 0.0
        try:
            qualia = {
                'arousal': float(self.qualia.arousal),
                'valence': float(self.qualia.valence),
                'entropy': float(self.qualia.entropy),
                'engagement': float(self.qualia.engagement),
                'frustration': float(self.qualia.frustration),
            }
        except Exception:
            qualia = {}
        try:
            energy = float(self.energy_ctrl.cognitive_energy)
            activation = float(self.energy_ctrl.activation_level)
        except Exception:
            energy = 0.0
            activation = 0.0
        try:
            meta = {
                'meta_awareness': float(self.self_model.meta_awareness),
                'meta_confidence': float(self.self_model.meta_confidence),
            }
        except Exception:
            meta = {}

        return {
            'phi': phi,
            'qualia': qualia,
            'energy': energy,
            'activation': activation,
            'meta': meta,
        }

    def _generate_research_plan(self, m3_state: Dict[str, Any], needs: List[Dict[str, Any]], question: Dict[str, Any]) -> Optional[str]:
        """Use LLM adapter (if available) to propose M3-grounded research/combination steps."""
        try:
            adapter = getattr(self, 'llm_adapter', None)
            if adapter is None or not hasattr(adapter, 'generate'):
                return None
        except Exception:
            return None

    def _creative_output_path(self) -> str:
        """Single file destination for continuous creation output."""
        try:
            path = os.getenv('M3_CREATIVE_FILE', '').strip()
        except Exception:
            path = ''
        if not path:
            path = os.path.join(self.outdir, 'm3_creative_sandbox.md')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        return path

    def _append_creative_output(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Append creation output to a single persistent file."""
        if not text:
            return
        path = self._creative_output_path()
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        header = f"\n\n---\n# M3 Creative Output ({stamp})\n"
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(header)
                if meta:
                    try:
                        f.write(f"Meta: {json.dumps(meta, ensure_ascii=False)}\n")
                    except Exception:
                        pass
                f.write(text.strip() + "\n")
        except Exception:
            pass

    def run_autonomous_creation(self, cycles: int = 20, topic: str = None) -> Dict[str, Any]:
        """Create novel outputs grounded in M3 state or potentially a specific topic, saved to a single file."""
        log = {
            'start_time': time.time(),
            'cycles': [],
            'file': self._creative_output_path(),
        }
        adapter = getattr(self, 'llm_adapter', None)
        if adapter is None or not hasattr(adapter, 'generate'):
            log['error'] = 'llm_adapter unavailable'
            return log
            
        if topic:
            print(f"  [Creation] Creative session grounded on topic: {topic}")

        for i in range(max(1, int(cycles))):
            m3_state = self._research_m3_state_summary()
            
            if topic:
                prompt = (
                    f"주제 '{topic}'에 대해 M3_STATE를 반영하여 독창적인 산출물을 만들어줘.\n"
                    f"M3_STATE: {json.dumps(m3_state, ensure_ascii=False)}\n"
                    "요구사항: (1) 주제와 내부 상태(정서 등)의 연결, (2) 창의적인 아이디어/텍스트/설계, (3) 실행 가능한 인사이트."
                )
            else:
                prompt = (
                    "M3_STATE 기반으로 새로운 시도를 만들어줘.\n"
                    f"M3_STATE: {json.dumps(m3_state, ensure_ascii=False)}\n"
                    "요구사항: (1) 새로운 조합/아이디어 1개, (2) 실행 가능한 다음 행동 1개, (3) 짧은 산출물(문단/설계/규칙 등)."
                )
            
            try:
                out = adapter.generate(prompt, max_len=220)
            except Exception:
                out = ''
            if out:
                meta = {'cycle': i + 1, 'm3_state': m3_state}
                if topic:
                    meta['topic'] = topic
                self._append_creative_output(out, meta=meta)
                log['cycles'].append({'cycle': i + 1, 'm3_state': m3_state})
                print(f"  [Creation] Cycle {i+1}: Output generated.")
            try:
                self._single_consciousness_step()
            except Exception:
                pass

        log['end_time'] = time.time()
        log['duration'] = log['end_time'] - log['start_time']
        return log

        try:
            needs_brief = ", ".join([f"{n.get('type')}:{n.get('source')}" for n in needs])
            prompt = (
                "M3_STATE 기반으로 연구/조합/새 시도 계획을 제시해줘.\n"
                f"M3_STATE: {json.dumps(m3_state, ensure_ascii=False)}\n"
                f"Needs: {needs_brief}\n"
                f"Question: {question.get('formulation')}\n"
                "요구사항: (1) 내부 변수 조합 실험 1개, (2) 새로운 시도 1개, (3) 측정 지표 제안."
            )
            plan = adapter.generate(prompt, max_len=200)
            return plan.strip() if plan else None
        except Exception:
            return None
    
    def _detect_research_needs_from_internal_dynamics(self, introspection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect what needs investigation based on internal state dynamics, not hardcoded rules"""
        needs = []
        
        # Tension-based detection: internal conflicts signal need for understanding
        tensions = introspection['tensions']
        
        for tension_name, tension_value in tensions.items():
            if tension_value > 0.4:  # Significant internal conflict
                needs.append({
                    'type': 'internal_conflict',
                    'source': tension_name,
                    'intensity': float(tension_value),
                    'description': f'Internal tension detected: {tension_name} = {tension_value:.3f}',
                    'requires_investigation': True
                })
        
        # Gradient-based detection: rapid changes signal instability
        gradients = introspection.get('gradients', {})
        
        for grad_name, grad_value in gradients.items():
            if abs(grad_value) > 0.5:  # Rapid change
                needs.append({
                    'type': 'dynamic_instability',
                    'source': grad_name,
                    'intensity': abs(float(grad_value)),
                    'description': f'Rapid state change: {grad_name} = {grad_value:.3f}/s',
                    'requires_investigation': True
                })
        
        # Consciousness-based detection: low integration or awareness
        consciousness = introspection['consciousness']
        
        if consciousness['phi'] < 0.05 and consciousness['meta_awareness'] > 0.3:
            # Aware but not integrated - disconnect
            needs.append({
                'type': 'integration_disconnect',
                'source': 'phi_awareness_mismatch',
                'intensity': consciousness['meta_awareness'] - consciousness['phi'],
                'description': f'High awareness ({consciousness["meta_awareness"]:.3f}) but low integration ({consciousness["phi"]:.6f})',
                'requires_investigation': True
            })
        
        # Energy dynamics detection
        dynamics = introspection['dynamics']
        
        if dynamics['cognitive_energy'] < 30 and dynamics['engagement'] > 0.6:
            # Want to engage but lack energy - resource problem
            needs.append({
                'type': 'resource_constraint',
                'source': 'energy_engagement_mismatch',
                'intensity': dynamics['engagement'] - (dynamics['cognitive_energy'] / 100.0),
                'description': f'High engagement ({dynamics["engagement"]:.3f}) but low energy ({dynamics["cognitive_energy"]:.1f})',
                'requires_investigation': True
            })
        
        return needs
    
    def _formulate_research_question_from_need(self, need: Dict[str, Any]) -> Dict[str, Any]:
        """Generate research question from internal need, not predefined templates"""
        question = {
            'id': f'Q{int(time.time() * 1000000) % 1000000}',
            'need': need,
            'created_at': time.time(),
            'formulation': None,
            'investigation_strategy': None
        }
        
        need_type = need['type']
        source = need['source']
        
        # Formulate question based on the nature of internal dynamics
        if need_type == 'internal_conflict':
            # Two systems are conflicting - need to understand their relationship
            parts = source.split('_')
            if len(parts) >= 2:
                question['formulation'] = f"What is the causal relationship between {parts[0]} and {parts[-2]} Why do they conflict"
                question['investigation_strategy'] = {
                    'type': 'causal_manipulation',
                    'manipulate': parts[0],
                    'observe': parts[-2],
                    'hypothesis': f"Changing {parts[0]} will affect {parts[-2]} in opposite direction"
                }
        
        elif need_type == 'dynamic_instability':
            # System is changing rapidly - need to understand what drives it
            variable = source.replace('_rate', '')
            question['formulation'] = f"What causes rapid changes in {variable} What stabilizes it"
            question['investigation_strategy'] = {
                'type': 'perturbation_response',
                'target_variable': variable,
                'perturbation_method': 'controlled_action_injection',
                'hypothesis': f"External perturbations reveal {variable} control mechanisms"
            }
        
        elif need_type == 'integration_disconnect':
            # Awareness and integration mismatch - structural issue
            question['formulation'] = "Why is consciousness not integrating despite awareness What blocks information flow"
            question['investigation_strategy'] = {
                'type': 'information_flow_analysis',
                'measure': 'cross_module_correlation',
                'hypothesis': 'Some modules are isolated from global workspace'
            }
        
        elif need_type == 'resource_constraint':
            # Resource limitation - allocation problem
            question['formulation'] = "How should cognitive energy be allocated to maximize engagement"
            question['investigation_strategy'] = {
                'type': 'resource_allocation_optimization',
                'resource': 'cognitive_energy',
                'objective': 'engagement',
                'hypothesis': 'Different tasks require different energy profiles'
            }
        
        else:
            # Generic investigation
            question['formulation'] = f"What internal process causes {source}"
            question['investigation_strategy'] = {
                'type': 'general_observation',
                'target': source,
                'hypothesis': 'Observation will reveal underlying mechanism'
            }
        
        return question
    
    def _generate_cognitive_plan(self, question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """M3_STATE 기반으로 LLM을 사용해 고차원적 인지 실험 계획 수립"""
        adapter = getattr(self, 'llm_adapter', None)
        if adapter is None or not hasattr(adapter, 'generate'):
            return [{'step': 1, 'type': 'error', 'description': 'LLM adapter not available for planning'}]

        m3_state = self._research_m3_state_summary()
        topic = question.get('formulation', 'Unknown Topic')
        
        prompt = (
            f"당신은 M3라는 자율적인 인공 지성체입니다. 현재 당신의 내부 상태(M3_STATE)를 기반으로\n"
            f"연구 주제 '{topic}'에 대해 심도 깊은 사고 실험/분석을 수행하기 위한 3~5단계의 연구 계획을 세우세요.\n"
            f"\n"
            f"현재 M3_STATE: {json.dumps(m3_state, ensure_ascii=False)}\n"
            f"\n"
            f"요구사항:\n"
            f"1. 연구 계획은 논리적이고 단계적이어야 합니다 (예: 정의 -> 반례 -> 분석 -> 종합).\n"
            f"2. 각 단계는 M3의 현재 정서/상태가 반영된 관점이어야 합니다 (예: 에너지가 낮으면 비판적으로, 높으면 창의적으로).\n"
            f"3. 응답은 오직 JSON 리스트 형식이어야 합니다. 예: [{{'step': 1, 'type': 'definition', 'description': '...'}}, ...]\n"
        )
        
        try:
            print(f"  [Planning] Generating cognitive research plan grounded in M3_STATE...")
            response = adapter.generate(prompt, max_len=512)
            # 파싱 시도 (Markdown 코드블록 제거 등)
            cleaned = response.replace('```json', '').replace('```', '').strip()
            start = cleaned.find('[')
            end = cleaned.rfind(']')
            if start != -1 and end != -1:
                plan_json = json.loads(cleaned[start:end+1])
                return plan_json
            else:
                # 파싱 실패 시 기본 플랜 반환
                return [
                    {'step': 1, 'type': 'concept_analysis', 'description': f"Analyze the concept of {topic}"},
                    {'step': 2, 'type': 'self_reflection', 'description': "Reflect on this concept based on current state"},
                    {'step': 3, 'type': 'synthesis', 'description': "Synthesize findings into a conclusion"}
                ]
        except Exception as e:
            print(f"  [Planning Error] {e}")
            return [
                {'step': 1, 'type': 'fallback_analysis', 'description': f"Direct analysis of {topic}"}
            ]

    def _execute_cognitive_step(self, step_info: Dict[str, Any], context: List[str]) -> str:
        """단기 기억(Context)과 현재 상태를 사용하여 계획의 한 단계를 수행"""
        adapter = getattr(self, 'llm_adapter', None)
        m3_state = self._research_m3_state_summary()
        
        context_str = "\n".join(context[-3:]) # 최근 맥락 3개만 유지
        
        prompt = (
            f"M3 연구 진행 중. 현재 단계: {step_info['description']}\n"
            f"이전 맥락:\n{context_str}\n\n"
            f"현재 M3_STATE: {json.dumps(m3_state, ensure_ascii=False)}\n"
            f"지시: 위 상태와 맥락을 바탕으로 이 단계를 수행하고, 그 결과를 텍스트로 서술하세요.\n"
            f"당신의 현재 기분/상태가 반영된 어조와 깊이로 작성하세요."
        )
        
        try:
            result = adapter.generate(prompt, max_len=300)
            return result.strip()
        except:
            return "Execution failed."

    def _conduct_investigation(self, question: Dict[str, Any], steps: int = 20) -> Dict[str, Any]:
        """Conduct investigation based on COGNITIVE PLANNING, not mechanical loops"""
        
        evidence = {
            'question_id': question['id'],
            'strategy': 'cognitive_reasoning',
            'process_log': [],
            'conclusion': None
        }

        # 1. Plan
        plan = self._generate_cognitive_plan(question)
        print(f"  [Plan Drafted] {len(plan)} steps.")
        for p in plan:
             print(f"    - Step {p.get('step')}: {p.get('description')}")
        
        context = []
        findings = []

        # 2. Execute Steps
        for i, step_info in enumerate(plan):
            print(f"\n  [Executing Step {i+1}/{len(plan)}] {step_info['description']}...")
            
            # 사고 실험 수행 (LLM)
            result_text = self._execute_cognitive_step(step_info, context)
            
            print(f"    -> Result: {result_text[:100]}... (truncated)")
            
            # 결과 기록
            # M3 내부 상태 변화도 동반 (사고 과정이 상태에 영향을 줌)
            self._single_consciousness_step() 
            
            log_entry = {
                'step': i + 1,
                'intent': step_info['description'],
                'result': result_text,
                'm3_state_snapshot': self._research_m3_state_summary()
            }
            evidence['process_log'].append(log_entry)
            context.append(f"Step {i+1} Result: {result_text}")
            findings.append(result_text)

        # 3. Conclude
        print(f"\n  [Synthesizing Conclusion]...")
        final_summary = " ".join(findings)
        evidence['conclusion'] = {
            'interpretation': final_summary[:200] + "...", # 요약
            'detailed_findings': final_summary
        }
        
        return evidence
    
    def _single_consciousness_step(self):
        """Execute one minimal consciousness cycle"""
        try:
            self.energy_ctrl.internal_clock += 1
            world_state = self._get_current_world_state()
            
            goal = self.goal_gen.generate_goal(
                self.self_model, self.qualia, world_state,
                self.self_model.state_history
            )
            
            self.energy_ctrl.update_activation(self.qualia, self.self_model, goal)
            
            should_continue, intensity = self.energy_ctrl.should_continue()
            if should_continue and intensity >= 0.2:
                action_plan = self._decide_action(goal, intensity)
                delta_hat, _ = self._execute_action(action_plan)
                self._experience_qualia(delta_hat, action_plan)
                
                grounded = self.conceptual_space.ground_experience(self.qualia)
                self._submit_to_workspace(grounded, goal)
                contents = self.global_workspace.compete_for_consciousness()
                
                self.self_model.update_meta_awareness(contents)
                
                if hasattr(self, 'phi_calculator'):
                    self.phi_calculator.compute_phi_simple(world_state, contents)
            
            self.t += 1
        except:
            pass
    
    def _extract_variable_value(self, var_name: str) -> float:
        """Extract internal variable value by name"""
        if var_name == 'arousal':
            return float(self.qualia.arousal)
        elif var_name == 'valence':
            return float(self.qualia.valence)
        elif var_name == 'entropy':
            return float(self.qualia.entropy)
        elif var_name == 'frustration':
            return float(self.qualia.frustration)
        elif var_name == 'engagement':
            return float(self.qualia.engagement)
        elif var_name == 'meta' or var_name == 'awareness':
            return float(self.self_model.meta_awareness)
        elif var_name == 'confidence':
            return float(self.self_model.meta_confidence)
        elif var_name == 'energy':
            return float(self.energy_ctrl.cognitive_energy / 100.0)
        elif var_name == 'activation':
            return float(self.energy_ctrl.activation_level)
        return 0.0
    
    def _inject_perturbation(self, var_name: str, value: float):
        """Inject perturbation into internal variable"""
        try:
            if var_name == 'arousal':
                self.qualia.arousal = np.clip(self.qualia.arousal + value, 0, 1)
            elif var_name == 'valence':
                self.qualia.valence = np.clip(self.qualia.valence + value, 0, 1)
            elif var_name == 'energy':
                self.energy_ctrl.cognitive_energy = np.clip(self.energy_ctrl.cognitive_energy + value * 100, 0, 100)
        except:
            pass
    
    def run_autonomous_research(self, max_cycles: int = 100, topic: str = None):
        """Autonomous research driven by internal dynamics or specific topic"""
        print(f"\n{'='*70}")
        print(f"M3 AUTONOMOUS RESEARCH SYSTEM")
        if topic:
            print(f"Target Topic: {topic}")
        else:
            print(f"Driven by Internal Dynamics & Introspection")
        print(f"{'='*70}\n")
        
        research_log = {
            'start_time': time.time(),
            'cycles': [],
            'total_needs_detected': 0,
            'total_questions_formulated': 0,
            'total_investigations': 0,
            'discoveries': []
        }
        
        for cycle in range(max_cycles):
            print(f"\n--- Research Cycle {cycle + 1}/{max_cycles} ---")
            
            # 1. Deep introspection of internal state
            introspection = self._introspect_internal_state()
            
            needs = []
            primary_need = None
            
            if topic:
                # User-directed research overrides internal needs
                primary_need = {
                    'type': 'user_directed',
                    'source': 'external_command',
                    'intensity': 1.0,
                    'description': f'User requested research on: {topic}',
                    'requires_investigation': True
                }
                needs = [primary_need]
            else:
                # 2. Detect research needs from internal dynamics
                needs = self._detect_research_needs_from_internal_dynamics(introspection)
                try:
                    force_research = os.getenv('M3_RESEARCH_FORCE', '0').lower() in ('1', 'true', 'yes', 'on')
                except Exception:
                    force_research = False
                
                if not needs and force_research:
                    needs = [{
                        'type': 'baseline_exploration',
                        'source': 'forced_probe',
                        'intensity': 0.2,
                        'description': 'Forced research probe (no acute needs detected)',
                        'requires_investigation': True
                    }]

            if not needs:
                print(f"  Internal state stable - no investigation needs detected")
                try:
                    self._append_creative_output(
                        "연구: 내부 상태 안정. 조사 필요 없음.",
                        meta={'type': 'research', 'cycle': cycle + 1, 'm3_state': self._research_m3_state_summary()}
                    )
                except Exception:
                    pass
                self._single_consciousness_step()
                continue
            
            research_log['total_needs_detected'] += len(needs)

            # M3 state snapshot for grounding
            m3_state = self._research_m3_state_summary()
            
            print(f"  Detected {len(needs)} internal needs:")
            for need in needs:
                print(f"    - {need['type']}: {need['description']} (intensity={need['intensity']:.3f})")
            
            # 3. Select most intense need
            if not primary_need:
                primary_need = max(needs, key=lambda n: n['intensity'])
                # Optional: create a combination need to encourage synthesis
                if len(needs) >= 2:
                    combo = sorted(needs, key=lambda n: n.get('intensity', 0.0), reverse=True)[:2]
                    combo_need = {
                        'type': 'combined_synthesis',
                        'source': f"{combo[0].get('source')}+{combo[1].get('source')}",
                        'intensity': float((combo[0].get('intensity', 0.0) + combo[1].get('intensity', 0.0)) / 2.0),
                        'description': 'Synthesis of top-2 internal needs',
                        'requires_investigation': True
                    }
                    needs.append(combo_need)
            
            # 4. Formulate research question
            question = None
            if topic:
                # Direct formulation from user topic
                question = {
                    'id': f'Q{int(time.time() * 1000000) % 1000000}',
                    'need': primary_need,
                    'created_at': time.time(),
                    'formulation': topic,
                    'investigation_strategy': {
                        'type': 'user_specified', 
                        'target': 'user_topic',
                        'hypothesis': f"Investigating {topic}"
                    }
                }
            else:
                # Autonomous formulation
                question = self._formulate_research_question_from_need(primary_need)

            research_log['total_questions_formulated'] += 1

            # M3-grounded research plan (combination/new attempt) via adapter
            plan = self._generate_research_plan(m3_state, needs, question)
            if plan:
                question['m3_plan'] = plan
            
            print(f"\n  Research Question: {question['formulation']}")
            print(f"  Investigation Strategy: {question['investigation_strategy']['type']}")
            
            # 5. Conduct investigation
            print(f"\n  Conducting investigation...")
            evidence = self._conduct_investigation(question, steps=20) or {}
            research_log['total_investigations'] += 1
            conclusion = evidence.get('conclusion') if isinstance(evidence, dict) else None
            
            # 6. Extract conclusion
            if conclusion:
                print(f"\n  Investigation Results:")
                print(f"    {conclusion.get('interpretation', 'No interpretation')}")
                
                # Record discovery
                if isinstance(conclusion, dict) and 'error' not in conclusion:
                    evidence_items = evidence.get('measurements', evidence.get('process_log', []))
                    discovery = {
                        'cycle': cycle + 1,
                        'need': primary_need,
                        'question': question['formulation'],
                        'finding': conclusion.get('interpretation', ''),
                        'evidence': evidence_items
                    }
                    research_log['discoveries'].append(discovery)
                    print(f"    [DISCOVERY RECORDED]")
                    try:
                        domain = str(topic).strip() if topic else str(primary_need.get('type', 'general')).strip()
                        if not domain:
                            domain = 'general'
                        investigation_parts = []
                        if isinstance(evidence_items, list):
                            for item in evidence_items[:3]:
                                if isinstance(item, dict):
                                    intent = str(item.get('intent', item.get('step', ''))).strip()
                                    result = str(item.get('result', item.get('value', ''))).strip().replace('\n', ' ')
                                    if len(result) > 180:
                                        result = result[:180] + '...'
                                    if intent and result:
                                        investigation_parts.append(f"{intent}: {result}")
                                    elif result:
                                        investigation_parts.append(result)
                                elif item is not None:
                                    txt = str(item).strip().replace('\n', ' ')
                                    if txt:
                                        investigation_parts.append(txt[:180] + ('...' if len(txt) > 180 else ''))
                        investigation_text = " | ".join(investigation_parts) if investigation_parts else "No detailed investigation log."
                        conclusion_text = str(conclusion.get('interpretation', 'No interpretation'))
                        if len(conclusion_text) > 240:
                            conclusion_text = conclusion_text[:240] + '...'
                        key_points = [
                            f"strategy={question.get('investigation_strategy', {}).get('type', 'unknown')}",
                            f"need_type={primary_need.get('type', 'unknown')}",
                            f"finding={conclusion_text}",
                        ]
                        research_content = "\n".join([
                            f"Need: {primary_need.get('description', primary_need.get('type', 'unknown'))}",
                            f"Question: {question.get('formulation', '')}",
                            f"Investigation: {investigation_text}",
                            f"Conclusion: {conclusion_text}",
                            f"Key points: {'; '.join(key_points)}",
                        ])
                        adapter = getattr(self, 'llm_adapter', None) or getattr(self, 'llm', None)
                        emb = None
                        if adapter is not None and hasattr(adapter, 'embed_text'):
                            emb_text = self._semantic_text_for_embedding(research_content)
                            emb = adapter.embed_text(emb_text, sys_identity="")
                        self._encode_semantic_memory_trace(
                            experience_name='research_discovery',
                            kind='research',
                            content=research_content,
                            embedding=emb,
                            tags=['research', domain],
                            extra_context={
                                'cycle': int(cycle + 1),
                                'question_id': str(question.get('id', '')),
                                'domain': domain,
                            },
                        )
                    except Exception:
                        pass
            else:
                print(f"\n  Investigation inconclusive")

            # Append a concise research report to the single creative file
            try:
                report_lines = [
                    f"연구 사이클 {cycle + 1}/{max_cycles}",
                    f"질문: {question.get('formulation')}",
                    f"전략: {question.get('investigation_strategy', {}).get('type')}",
                ]
                if question.get('m3_plan'):
                    report_lines.append(f"M3 계획: {question.get('m3_plan')}")
                if conclusion:
                    report_lines.append(
                        f"결론: {conclusion.get('interpretation', 'No interpretation')}"
                    )
                else:
                    report_lines.append("결론: 불충분")
                self._append_creative_output(
                    "\n".join(report_lines),
                    meta={'type': 'research', 'cycle': cycle + 1, 'm3_state': m3_state}
                )
            except Exception:
                pass
            
            # 7. Log cycle
            cycle_log = {
                'cycle': cycle + 1,
                'introspection': introspection,
                'm3_state': m3_state,
                'needs': needs,
                'question': question,
                'evidence': evidence,
                'timestamp': time.time()
            }
            research_log['cycles'].append(cycle_log)
            
            # 8. Continue consciousness process
            for _ in range(5):
                self._single_consciousness_step()
            
            # 9. Periodic checkpoint
            if (cycle + 1) % 20 == 0:
                try:
                    self._save_checkpoint()
                    print(f"\n  [Checkpoint saved]")
                except:
                    pass
        
        # Summary
        research_log['end_time'] = time.time()
        research_log['duration'] = research_log['end_time'] - research_log['start_time']
        
        print(f"\n{'='*70}")
        print(f"RESEARCH SUMMARY")
        print(f"{'='*70}")
        print(f"  Total cycles: {max_cycles}")
        print(f"  Needs detected: {research_log['total_needs_detected']}")
        print(f"  Questions formulated: {research_log['total_questions_formulated']}")
        print(f"  Investigations conducted: {research_log['total_investigations']}")
        print(f"  Discoveries made: {len(research_log['discoveries'])}")
        print(f"  Duration: {research_log['duration']:.2f}s")
        print(f"{'='*70}\n")
        
        # Save research log
        try:
            log_path = os.path.join(self.outdir, 'autonomous_research_log.json')
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(research_log, f, indent=2, default=str)
            print(f"Research log saved: {log_path}\n")
        except Exception as e:
            print(f"Failed to save research log: {e}\n")
        
        return research_log
    
    # ==================== AUTONOMOUS DATASET LEARNING ====================
    def _discover_datasets(self, dataset_root: str) -> List[Dict[str, Any]]:
        """Scan dataset folder and discover all learnable data files"""
        datasets = []
        
        def scan_recursive(path: str, category: str = ''):
            try:
                for entry in os.listdir(path):
                    full_path = os.path.join(path, entry)
                    if os.path.isdir(full_path):
                        new_category = f"{category}/{entry}" if category else entry
                        scan_recursive(full_path, new_category)
                    elif os.path.isfile(full_path):
                        ext = os.path.splitext(entry)[1].lower()
                        size = os.path.getsize(full_path)
                        
                        data_type = None
                        if ext in ['.txt', '.md', '.csv']:
                            data_type = 'text'
                        elif ext == '.json':
                            data_type = 'json'
                        elif ext == '.tsv':
                            data_type = 'tsv'
                        elif ext in ['.bin', '.vec', '.w2v']:
                            data_type = 'vector'
                        elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                            data_type = 'image'
                        
                        if data_type:
                            datasets.append({
                                'path': full_path,
                                'name': entry,
                                'category': category,
                                'type': data_type,
                                'size': size
                            })
            except Exception as e:
                print(f"[Error] Failed to scan directory {path}: {e}")
        
        scan_recursive(dataset_root)
        return datasets
    
    def _learn_text_data(self, filepath: str, max_bytes: int = 100000) -> np.ndarray:
        """Extract features from text file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_bytes)
            
            import hashlib
            words = content.lower().split()[:1000]
            features = np.zeros(512, dtype=np.float32)
            
            for word in words:
                h = int.from_bytes(hashlib.blake2b(word.encode('utf-8'), digest_size=8).digest(), 'little')
                idx = h % 512
                features[idx] += 1.0
            
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
        except Exception as e:
            print(f"[Error] _learn_text_data failed for {filepath}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def _learn_json_data(self, filepath: str) -> np.ndarray:
        """Extract features from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            features = []
            
            def extract_recursive(obj, depth=0):
                if depth > 3 or len(features) > 1000:
                    return
                
                if isinstance(obj, (int, float)):
                    features.append(float(obj))
                elif isinstance(obj, str):
                    h = hash(obj) % 1000
                    features.append(float(h) / 1000.0)
                elif isinstance(obj, list):
                    for item in obj[:50]:
                        extract_recursive(item, depth + 1)
                elif isinstance(obj, dict):
                    for v in list(obj.values())[:50]:
                        extract_recursive(v, depth + 1)
            
            extract_recursive(data)
            
            arr = np.array(features[:1000], dtype=np.float32)
            if arr.size == 0:
                return np.zeros(512, dtype=np.float32)
            
            result = np.zeros(512, dtype=np.float32)
            result[:min(arr.size, 512)] = arr[:min(arr.size, 512)]
            return result
        except Exception as e:
            print(f"[Error] _learn_json_data failed for {filepath}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def _learn_tsv_data(self, filepath: str, max_lines: int = None) -> np.ndarray:
        """Extract features from TSV file (knowledge graphs, news, etc.)"""
        try:
            import hashlib
            import json
            features = np.zeros(512, dtype=np.float32)
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if max_lines is not None and i >= max_lines:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    
                    # Knowledge graph triple format: subject \t relation \t object
                    if len(parts) == 3 and all(len(p.strip()) > 0 for p in parts):
                        # 트리플 구조: 각 요소를 별도로 해시
                        for j, part in enumerate(parts):
                            h = int.from_bytes(
                                hashlib.blake2b(
                                    f"{j}:{part}".encode('utf-8'), 
                                    digest_size=8
                                ).digest(), 
                                'little'
                            )
                            idx = h % 512
                            features[idx] += (3 - j) * 0.5  # subject > relation > object 가중치
                    
                    # MIND news format or other complex TSV
                    elif len(parts) > 3:
                        # 텍스트 필드 추출 (제목, 내용, 카테고리 등)
                        for j, part in enumerate(parts):
                            part = part.strip()
                            
                            # JSON 필드 파싱 시도
                            if part.startswith('[') or part.startswith('{'):
                                try:
                                    obj = json.loads(part)
                                    if isinstance(obj, list):
                                        for item in obj[:10]:  # 최대 10개 엔티티
                                            if isinstance(item, dict):
                                                for k, v in item.items():
                                                    if isinstance(v, str):
                                                        h = int.from_bytes(
                                                            hashlib.blake2b(
                                                                f"{k}:{v}".encode('utf-8'), 
                                                                digest_size=8
                                                            ).digest(), 
                                                            'little'
                                                        )
                                                        features[h % 512] += 0.3
                                except:
                                    pass
                            
                            # 일반 텍스트 필드
                            elif len(part) > 0:
                                # 단어 단위로 해시
                                words = part.split()[:50]  # 최대 50개 단어
                                for word in words:
                                    h = int.from_bytes(
                                        hashlib.blake2b(
                                            word.lower().encode('utf-8'), 
                                            digest_size=8
                                        ).digest(), 
                                        'little'
                                    )
                                    features[h % 512] += 0.1
                    
                    # 기타 형식: 모든 필드를 평등하게 해시
                    else:
                        for part in parts:
                            if len(part.strip()) > 0:
                                h = int.from_bytes(
                                    hashlib.blake2b(
                                        part.encode('utf-8'), 
                                        digest_size=8
                                    ).digest(), 
                                    'little'
                                )
                                features[h % 512] += 1.0
            
            # L2 정규화
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
        except Exception as e:
            print(f"[Error] _learn_tsv_data failed for {filepath}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def _learn_vector_data(self, filepath: str, max_vectors: int = 100) -> np.ndarray:
        """Extract features from vector embedding file"""
        try:
            vectors = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_vectors:
                        break
                    parts = line.strip().split()
                    if len(parts) > 1:
                        try:
                            vec = [float(x) for x in parts[1:]]
                            vectors.append(vec)
                        except:
                            continue
            
            if vectors:
                arr = np.array(vectors, dtype=np.float32).flatten()
                result = np.zeros(512, dtype=np.float32)
                result[:min(arr.size, 512)] = arr[:min(arr.size, 512)]
                return result
            else:
                return np.zeros(512, dtype=np.float32)
        except Exception as e:
            print(f"[Error] _learn_vector_data failed for {filepath}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def _learn_image_data(self, filepath: str) -> np.ndarray:
        """Extract features from image file"""
        try:
            from PIL import Image
            img = Image.open(filepath).convert('L')
            img = img.resize((32, 32))
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr.flatten()
        except Exception as e:
            print(f"[Error] _learn_image_data failed for {filepath}: {e}")
            return np.zeros(1024, dtype=np.float32)
    
    def _learn_legal_case(self, filepath: str, max_bytes: int = 50000) -> Dict[str, Any]:
        """법률 케이스 문서 학습 (Legal Case Corpus)"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_bytes)
            
            # 법률 용어 추출
            import re
            legal_terms = re.findall(r'\b[A-Z][a-z]+(:\s+[A-Z][a-z]+)*\b', content)
            case_numbers = re.findall(r'\b\d+\s+[A-Z]\.\w+\.\s+\d+\b', content)
            
            # 특징 벡터 생성
            import hashlib
            features = np.zeros(512, dtype=np.float32)
            for term in legal_terms[:200]:
                h = int.from_bytes(hashlib.blake2b(term.lower().encode(), digest_size=8).digest(), 'little')
                features[h % 512] += 1.0
            
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return {
                'features': features,
                'terms_count': len(legal_terms),
                'case_numbers': case_numbers[:5]
            }
        except Exception as e:
            print(f"[Error] _learn_legal_case failed for {filepath}: {e}")
            return {'features': np.zeros(512, dtype=np.float32), 'terms_count': 0}
    
    def _learn_word2vec_model(self, filepath: str) -> np.ndarray:
        """Word2Vec 모델/결과 학습"""
        try:
            # CSV 형식의 word2vec 결과 읽기
            import csv
            vectors = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= 100:  # 최대 100개 벡터
                        break
                    try:
                        vec = [float(x) for x in row if x.strip()]
                        if vec:
                            vectors.append(vec[:50])  # 최대 50 차원
                    except:
                        continue
            
            if vectors:
                arr = np.array(vectors, dtype=np.float32).flatten()
                result = np.zeros(512, dtype=np.float32)
                result[:min(arr.size, 512)] = arr[:min(arr.size, 512)]
                return result
            else:
                return np.zeros(512, dtype=np.float32)
        except Exception as e:
            print(f"[Error] _learn_word2vec_model failed for {filepath}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def _learn_entity_embedding(self, filepath: str, max_lines: int = None) -> np.ndarray:
        """엔티티 임베딩 학습 (.vec 파일)"""
        try:
            vectors = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if max_lines is not None and i >= max_lines:
                        break
                    parts = line.strip().split()
                    if len(parts) > 1:
                        try:
                            # 첫 번째는 엔티티명, 나머지는 벡터
                            vec = [float(x) for x in parts[1:]]
                            if vec:
                                vectors.append(vec[:50])
                        except:
                            continue
            
            if vectors:
                arr = np.array(vectors, dtype=np.float32).flatten()
                result = np.zeros(512, dtype=np.float32)
                result[:min(arr.size, 512)] = arr[:min(arr.size, 512)]
                return result
            else:
                return np.zeros(512, dtype=np.float32)
        except Exception as e:
            print(f"[Error] _learn_entity_embedding failed for {filepath}: {e}")
            return np.zeros(512, dtype=np.float32)
    
    def _feed_features_to_memory(self, features: np.ndarray, metadata: Dict[str, Any]):
        """Feed learned features into M3 memory systems"""
        try:
            features = np.asarray(features, dtype=np.float32).flatten()
            
            if hasattr(self, 'scope') and hasattr(self.scope, 'mem'):
                self.scope.mem.update(features)
            
            if hasattr(self, 'episodic_memory'):
                self.episodic_memory.store_episode(
                    context={'type': 'dataset_learning', **metadata},
                    content={'features': features, 'timestamp': time.time()},
                    salience=0.6
                )
            
            if hasattr(self, 'growing_som'):
                patch_size = min(len(features), 32)
                for i in range(0, len(features), patch_size):
                    patch = features[i:i+patch_size]
                    if len(patch) > 0:
                        self.growing_som.learn(patch, learning_rate=0.01)
        except Exception:
            pass
    
    def learn_from_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a single dataset file"""
        result = {
            'name': dataset_info['name'],
            'category': dataset_info['category'],
            'type': dataset_info['type'],
            'success': False,
            'features_learned': 0
        }
        
        try:
            dtype = dataset_info['type']
            path = dataset_info['path']
            name = dataset_info['name'].lower()
            
            # === 특화 데이터셋 처리 ===
            # Legal Case Corpus
            if 'legal' in name or 'case' in name:
                result_data = self._learn_legal_case(path)
                features = result_data.get('features', np.zeros(512, dtype=np.float32))
                result['legal_terms'] = result_data.get('terms_count', 0)
            
            # Word2Vec models/results
            elif 'word2vec' in name or name.endswith('.csv'):
                features = self._learn_word2vec_model(path)
            
            # Entity/Relation embeddings (.vec files)
            elif name.endswith('.vec') or 'embedding' in name:
                features = self._learn_entity_embedding(path)
            
            # MIND news TSV
            elif 'news.tsv' in name or 'behaviors.tsv' in name:
                features = self._learn_tsv_data(path)
            
            # Knowledge graph TSV
            elif 'edges_as_text' in name or 'edges_as_id' in name:
                features = self._learn_tsv_data(path)
            
            # 일반 데이터 타입별 처리
            elif dtype == 'text':
                features = self._learn_text_data(path)
            elif dtype == 'json':
                features = self._learn_json_data(path)
            elif dtype == 'tsv':
                features = self._learn_tsv_data(path)
            elif dtype == 'vector':
                features = self._learn_vector_data(path)
            elif dtype == 'image':
                features = self._learn_image_data(path)
            else:
                features = np.zeros(512, dtype=np.float32)
            
            if features is not None and features.size > 0:
                self._feed_features_to_memory(features, dataset_info)
                result['features_learned'] = int(features.size)
                result['success'] = True
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def run_autonomous_learning(self, dataset_root: str, max_iterations: int = -1, 
                               shuffle: bool = True, checkpoint_interval: int = 100):
        """
        Autonomous learning loop: discover and learn from all datasets
        
        Args:
            dataset_root: Path to dataset folder
            max_iterations: Max iterations (-1 for infinite)
            shuffle: Shuffle datasets before learning
            checkpoint_interval: Save checkpoint every N iterations
        """
        print(f"[M3 Autonomous Learning] Starting")
        print(f"  Dataset root: {dataset_root}")
        print(f"  Max iterations: {'infinite' if max_iterations < 0 else max_iterations}")
        
        datasets = self._discover_datasets(dataset_root)
        
        if not datasets:
            print(f"[M3] No datasets found in {dataset_root}")
            return
        
        print(f"[M3] Found {len(datasets)} datasets")
        
        by_type = {}
        for ds in datasets:
            t = ds['type']
            if t not in by_type:
                by_type[t] = 0
            by_type[t] += 1
        
        for dtype, count in by_type.items():
            print(f"  {dtype}: {count} files")
        
        if shuffle:
            import random
            random.shuffle(datasets)
        
        iteration = 0
        checkpoint_counter = 0
        learning_stats = {
            'total_learned': 0,
            'total_features': 0,
            'by_type': {},
            'successful': 0
        }
        
        try:
            while True:
                if max_iterations > 0 and iteration >= max_iterations:
                    break
                
                for dataset in datasets:
                    if max_iterations > 0 and iteration >= max_iterations:
                        break
                    
                    result = self.learn_from_dataset(dataset)
                    
                    if result['success']:
                        learning_stats['successful'] += 1
                        learning_stats['total_features'] += result['features_learned']
                        
                        dtype = result['type']
                        if dtype not in learning_stats['by_type']:
                            learning_stats['by_type'][dtype] = 0
                        learning_stats['by_type'][dtype] += 1
                    
                    learning_stats['total_learned'] += 1
                    
                    try:
                        self.step()
                    except Exception:
                        pass
                    
                    iteration += 1
                    checkpoint_counter += 1
                    
                    if checkpoint_counter >= checkpoint_interval:
                        print(f"[M3] Checkpoint at iteration {iteration}")
                        print(f"  Datasets learned: {learning_stats['total_learned']}")
                        print(f"  Total features: {learning_stats['total_features']}")
                        print(f"  Core step: {self.t}")
                        
                        try:
                            self.save_checkpoint(os.path.join(self.outdir, 'checkpoint.json'))
                        except Exception:
                            pass
                        
                        checkpoint_counter = 0
                    
                    if iteration % 10 == 0:
                        avg_features = learning_stats['total_features'] / max(1, learning_stats['total_learned'])
                        print(f"[M3] Progress {iteration}: learned={learning_stats['total_learned']}, "
                              f"avg_features={avg_features:.1f}, step={self.t}")
                
                if max_iterations < 0:
                    print(f"[M3] Completed epoch, total iterations: {iteration}")
                    if shuffle:
                        import random
                        random.shuffle(datasets)
                
        except KeyboardInterrupt:
            print(f"\n[M3] Learning interrupted by user at iteration {iteration}")
        
        print(f"\n[M3] Autonomous learning completed")
        print(f"  Total iterations: {iteration}")
        print(f"  Datasets learned: {learning_stats['total_learned']}")
        print(f"  Successful: {learning_stats['successful']}")
        print(f"  Total features: {learning_stats['total_features']}")
        print(f"  By type: {learning_stats['by_type']}")
        
        try:
            self.save_checkpoint(os.path.join(self.outdir, 'final_checkpoint.json'))
            print(f"  Final checkpoint saved to {self.outdir}")
        except Exception:
            pass
        
        return learning_stats
    # ==================== END AUTONOMOUS LEARNING ====================
    def _arc_ops_library(self, prob_id: str) -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
        # ops can close over problem-specific hints (palette mapping)
        inc = getattr(self, '_arc_cfg', {}).get('include_ops', {}) or {}
        ops: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
        if inc.get('id', True):
            ops.append(('id', self._arc_identity))
        if inc.get('rot', True):
            ops.extend([('rot90', self._arc_rot90), ('rot180', self._arc_rot180), ('rot270', self._arc_rot270)])
        if inc.get('flip', True):
            ops.extend([('flip_h', self._arc_flip_h), ('flip_v', self._arc_flip_v)])
        if inc.get('transpose', True):
            ops.append(('transpose', self._arc_transpose))
        if inc.get('remap_hist', True):
            ops.append(('remap_hist', (lambda g, _pid=prob_id: self._arc_remap_hist(_pid, g))))
        if inc.get('keep_largest', True):
            ops.append(('keep_largest', self._arc_keep_largest))
        if inc.get('crop_largest', True):
            ops.append(('crop_largest', self._arc_crop_largest))
        if inc.get('bbox_fill_outmode', True):
            ops.append(('bbox_fill_outmode', (lambda g, _pid=prob_id: self._arc_fill_bbox(g, self._arc_bbox_of_largest_region(g), int(self._arc_ctx.get(_pid, {}).get('out_mode', 0))))))
        # object-level ops
        if inc.get('obj_translate', True):
            ops.append(('move_tl', self._arc_move_largest_tl))
            ops.append(('center', self._arc_center_largest))
        if inc.get('obj_paint', True):
            ops.append(('paint_largest_outmode', self._arc_paint_largest_outmode))
            ops.append(('bbox_border_largest', self._arc_bbox_border_largest))
        # macros (global learned)
        if inc.get('macros', True) and getattr(self, '_arc_macros', None):
            for i, prog in enumerate(self._arc_macros):
                name = f"macro_m{i+1}"
                ops.append((name, (lambda g, _pid=prob_id, _prog=list(prog): self._arc_apply_program(_pid, _prog, g))))
        return ops

    def _arc_apply_program(self, prob_id: str, program: List[str], g: np.ndarray) -> np.ndarray:
        ops = dict(self._arc_ops_library(prob_id))
        out = g
        for name in program:
            out = ops.get(name, self._arc_identity)(out)
        return out

    def _arc_score_program(self, prob_id: str, program: List[str], train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float, float, float]:
        """Return (hard_accuracy, soft_avg, soft_min, soft_std) over train pairs.

        - hard_accuracy: fraction of pairs with exact grid match.
        - soft_avg: average per-pixel match ratio (0 if shape mismatch).
        - soft_min: minimum per-pixel match across pairs.
        - soft_std: standard deviation of per-pixel match across pairs.
        """
        if program is None:
            return 0.0, 0.0, 0.0, 0.0
        hard_correct = 0
        soft_vals: List[float] = []
        count = max(1, len(train_pairs))
        for tin, tout in train_pairs:
            pred = self._arc_apply_program(prob_id, program, tin)
            if pred.shape == tout.shape:
                if np.array_equal(pred, tout):
                    hard_correct += 1
                    soft_vals.append(1.0)
                else:
                    eq = float(np.mean((pred == tout))) if pred.size > 0 else 0.0
                    soft_vals.append(eq)
            else:
                soft_vals.append(0.0)
        hard = float(hard_correct) / count
        if soft_vals:
            soft_avg = float(np.mean(soft_vals))
            soft_min = float(np.min(soft_vals))
            soft_std = float(np.std(soft_vals)) if len(soft_vals) > 1 else 0.0
        else:
            soft_avg = soft_min = soft_std = 0.0
        return hard, soft_avg, soft_min, soft_std

    def _arc_search(self, prob: Dict[str, Any], max_time_per: float = 5.0, trace: bool = True, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> Tuple[Optional[List[str]], Dict[str, Any]]:
        import time as _time
        try:
            self._arc_current_id = prob['id']
        except Exception:
            pass
        train_pairs = prob['train']
        ops = self._arc_ops_library(prob['id'])
        # adaptive small budget (no premature stop; allow depth growth under time)
        H, W = train_pairs[0][0].shape
        colors = int(len(np.unique(train_pairs[0][0])))
        beam = 150 if (H*W <= 100 and colors <= 4) else 250
        # apply beam scale from research config
        try:
            beam = int(max(1, beam * float(getattr(self, '_arc_cfg', {}).get('beam_scale', 1.0))))
        except Exception:
            pass
        max_depth = 5 if H*W <= 100 else 6
        try:
            max_depth_cap = int(getattr(self, '_arc_cfg', {}).get('depth_cap', 10))
        except Exception:
            max_depth_cap = 10
        max_depth_cap = max(1, max_depth_cap)
        start = _time.perf_counter()
        cand = [([] , 0.0, 0.0, 0.0, 0.0)]  # (program, comp_score, hard, soft, len_pen)
        best = ([], -1e9, 0.0, 0.0, 0.0)
        round_idx = 0
        while _time.perf_counter() - start < max_time_per:
            # expand one depth level
            if round_idx >= max_depth:
                if max_depth < max_depth_cap:
                    max_depth += 1
                else:
                    break
            round_idx += 1
            newc = []
            for prog, comp_sc, s_h, s_soft, _lp in cand[:beam]:
                for name, _f in ops:
                    p2 = prog + [name]
                    s2_h, s2_soft_avg, s2_soft_min, s2_soft_std = self._arc_score_program(prob['id'], p2, train_pairs)
                    # composite objective with length penalty and consistency bonus
                    w = getattr(self, '_arc_cfg', {}).get('score_weights', {}) or {}
                    w_h, w_s, w_smin = float(w.get('w_hard', 1.0)), float(w.get('w_soft', 0.5)), float(w.get('w_soft_min', 0.25))
                    w_cons, w_len = float(w.get('w_consistency', 0.2)), float(w.get('w_len', 0.05))
                    len_pen = float(len(p2)) / 10.0
                    comp = (w_h * s2_h) + (w_s * s2_soft_avg) + (w_smin * s2_soft_min) + (w_cons * (1.0 - s2_soft_std)) - (w_len * len_pen)
                    newc.append((p2, comp, s2_h, s2_soft_avg, len_pen))
                    if trace and progress_cb and (len(newc) % 20 == 0):
                        tin0, tout0 = train_pairs[0]
                        pred0 = self._arc_apply_program(prob['id'], p2, tin0)
                        progress_cb({'type': 'arc_step', 'id': prob['id'], 'round': round_idx, 'program': p2, 'score': s2_h, 'soft': s2_soft_avg, 'preview_triplet': {'input': tin0.astype(int).tolist(), 'pred': pred0.astype(int).tolist(), 'target': tout0.astype(int).tolist()}})
                    # select by composite; tie-break on hard, then soft, then longer
                    b_prog, b_comp, b_h, b_soft, _ = best
                    if (comp > b_comp + 1e-9) or (abs(comp - b_comp) <= 1e-9 and ((s2_h > b_h + 1e-9) or (abs(s2_h - b_h) <= 1e-9 and ((s2_soft_avg > b_soft + 1e-9) or (abs(s2_soft_avg - b_soft) <= 1e-9 and len(p2) > len(b_prog)))))):
                        best = (p2, comp, s2_h, s2_soft_avg, len_pen)
                    # small time extension when improving
                        max_time_per = min(max_time_per * 1.15, 30.0)
            newc.sort(key=lambda x: (x[1], x[2], x[3], len(x[0])), reverse=True)
            cand = newc[:beam]
            if best[2] >= 1.0:
                break
            # widen beam if flat
            if cand and cand[0][2] <= best[2] and best[2] == 0.0:
                beam = min(beam + 50, 600)
        info = {'train_acc': float(best[2]), 'rounds': round_idx, 'soft_train': float(best[3])}
        return (best[0], info)

    # ============================== ARC Researcher ==============================
    def _bootstrap_ci(self, deltas: List[float], n: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        arr = np.asarray(deltas, dtype=np.float64)
        if arr.size == 0:
            return (0.0, 0.0)
        rng = np.random.default_rng(self.seed)
        means = []
        m = arr.size
        for _ in range(int(n)):
            idx = rng.integers(0, m, size=m)
            means.append(float(arr[idx].mean()))
        means.sort()
        lo = means[int(alpha/2 * (n-1))]
        hi = means[int((1 - alpha/2) * (n-1))]
        return (lo, hi)

    def run_research(self, arc_dir: str, steps: int = 5, sample_k: int = 100, per_prob_time: float = 60.0, bootstrap_samples: int = 1000, trace: bool = False, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        # 1) Sample problem files deterministically
        all_files = []
        for root, _dirs, names in os.walk(arc_dir):
            for name in sorted(names):
                if name.lower().endswith('.json'):
                    all_files.append(os.path.join(root, name))
        rng = np.random.default_rng(self.seed)
        if len(all_files) == 0:
            if progress_cb:
                progress_cb({'type': 'research_error', 'msg': 'no ARC files found'})
            return {'ok': False}
        files = list(all_files)
        rng.shuffle(files)
        files = files[:max(1, int(sample_k))]
        # 2) Baseline run (default config)
        self._arc_cfg['beam_scale'] = 1.0
        self._arc_cfg['depth_cap'] = 10
        self._arc_cfg['include_ops'] = {
            'id': True, 'rot': True, 'flip': True, 'transpose': True,
            'remap_hist': True, 'keep_largest': True, 'crop_largest': True, 'bbox_fill_outmode': True,
        }
        base = self.solve_arc_dir(arc_dir, out_dir=None, save_preds=False, trace=trace, max_time_per=per_prob_time, progress_cb=None, files=files, shuffle=False)
        base_map = {}
        for d in base.get('details', []):
            hard = float(d.get('train_acc', 0.0)); soft = float(d.get('soft_train', 0.0)); L = int(d.get('prog_len', 0))
            S = 1.0*hard + 0.5*soft - 0.05*(L/10.0)
            base_map[d['id']] = S
        # 3) Candidate actions
        actions = [
            {'name': 'beam_x1.5', 'beam_scale': 1.5},
            {'name': 'beam_x2.0', 'beam_scale': 2.0},
            {'name': 'depth_cap12', 'depth_cap': 12},
            {'name': 'rot_flip_only', 'include_ops': {'id': True,'rot': True,'flip': True,'transpose': False,'remap_hist': False,'keep_largest': False,'crop_largest': False,'bbox_fill_outmode': False,'obj_translate': False,'obj_paint': False,'macros': False}},
            {'name': 'obj_ops_on', 'include_ops': {'id': True,'rot': True,'flip': True,'transpose': True,'remap_hist': True,'keep_largest': True,'crop_largest': True,'bbox_fill_outmode': True,'obj_translate': True,'obj_paint': True,'macros': True}},
            {'name': 'macros_on', 'include_ops': {'macros': True}},
            {'name': 'macros_off', 'include_ops': {'macros': False}},
            {'name': 'len_penalty_up', 'score_weights': {'w_len': 0.1}},
            {'name': 'consistency_up', 'score_weights': {'w_consistency': 0.4}},
        ]
        history = []
        # UCB1 over actions
        nA = len(actions)
        counts = [0]*nA
        means = [0.0]*nA
        c_ucb = 1.2
        for step in range(int(steps)):
            # pick action by UCB
            t = step + 1
            idx = 0
            best_ucb = -1e9
            for i in range(nA):
                if counts[i] == 0:
                    u = 1e9
                else:
                    u = means[i] + c_ucb*np.sqrt(np.log(max(2, t)) / counts[i])
                if u > best_ucb:
                    best_ucb, idx = u, i
            a = actions[idx]
            # apply action config on top of defaults
            self._arc_cfg['beam_scale'] = float(a.get('beam_scale', self._arc_cfg['beam_scale']))
            self._arc_cfg['depth_cap'] = int(a.get('depth_cap', self._arc_cfg['depth_cap']))
            if 'include_ops' in a:
                # merge include ops partially
                inc = dict(self._arc_cfg['include_ops'])
                inc.update(a['include_ops'])
                self._arc_cfg['include_ops'] = inc
            if 'score_weights' in a:
                sw = dict(self._arc_cfg.get('score_weights', {}))
                sw.update(a['score_weights'])
                self._arc_cfg['score_weights'] = sw
            res = self.solve_arc_dir(arc_dir, out_dir=None, save_preds=False, trace=trace, max_time_per=per_prob_time, progress_cb=None, files=files, shuffle=False)
            treat_map = {}
            for d in res.get('details', []):
                hard = float(d.get('train_acc',0.0)); soft = float(d.get('soft_train',0.0)); L = int(d.get('prog_len',0))
                S = 1.0*hard + 0.5*soft - 0.05*(L/10.0)
                treat_map[d['id']] = S
            deltas = []
            for fid, s0 in base_map.items():
                s1 = treat_map.get(fid, 0.0)
                deltas.append(s1 - s0)
            lo, hi = self._bootstrap_ci(deltas, n=int(bootstrap_samples), alpha=0.05)
            adopted = (lo > 0.0)
            if adopted:
                base_map = treat_map
            reward = float(np.mean(deltas) if deltas else 0.0)
            # update UCB stats
            counts[idx] += 1
            means[idx] += (reward - means[idx]) / max(1, counts[idx])
            rec = {'step': step+1, 'action': a['name'], 'ci_low': lo, 'ci_high': hi, 'adopted': adopted, 'mean_delta': reward}
            history.append(rec)
            if progress_cb:
                progress_cb({'type': 'research_step', **rec})
        out = {'ok': True, 'history': history}
        try:
            os.makedirs(self.outdir, exist_ok=True)
            with open(os.path.join(self.outdir, 'research.json'), 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        if progress_cb:
            progress_cb({'type': 'research_done', 'steps': len(history)})
        return out
def build_parser():
    ap = argparse.ArgumentParser(description='M3 Consciousness System')
    ap.add_argument('--n', type=int, default=512, help='Number of nodes')
    ap.add_argument('--K', type=int, default=8, help='Number of categories')
    ap.add_argument('--seed', type=int, default=None, help='Random seed (None=random)')
    ap.add_argument('--outdir', type=str, default='docs&tests&data_sets/tests/logs', help='Output directory')
    ap.add_argument('--no-gui', action='store_true', help='Run without GUI (default: GUI enabled)')
    # ARC options (optional)
    ap.add_argument('--arc_dir', type=str, default=None, help='ARC problems folder (JSON files, recursive)')
    ap.add_argument('--arc_out', type=str, default='arc_preds', help='ARC predictions output folder')
    ap.add_argument('--arc_save', action='store_true', help='Save ARC predictions to --arc_out')
    ap.add_argument('--arc_trace', action='store_true', help='Enable ARC trace callbacks/logs')
    ap.add_argument('--arc_time_per', type=float, default=10.0, help='ARC soft time budget per problem (seconds)')
    ap.add_argument('--arc_require_solve', action='store_true', help='Retry a problem until solved or time cap reached')
    ap.add_argument('--arc_total_time_cap', type=float, default=60.0, help='Overall cap per problem when require_solve')
    return ap

def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.seed is None:
        args.seed = int(time.time() * 1000) % 2 ** 32
    
    # GUI mode check - default is GUI enabled unless --no-gui or --arc_dir is specified
    use_gui = not getattr(args, 'no_gui', False) and not args.arc_dir and _GUI_AVAILABLE
    if use_gui:
        try:
            # M3SimpleGUI is defined later in the file
            if 'M3SimpleGUI' not in globals():
                raise ImportError("M3SimpleGUI class not available (tkinter import may have failed)")
            system = M3ConsciousnessCore(n=args.n, K=args.K, seed=args.seed, max_iterations=None, outdir=args.outdir)
            
            # LLM Adapter 초기화 (module import; legacy file path no longer required)
            system.llm_adapter = None
            try:
                from llm_adapter import attach_llm_to_core
                attach_llm_to_core(system)
                print("✓ LLM Adapter initialized successfully")
            except ImportError as e:
                print(f"✗ LLM Adapter import failed: {e}")
                import traceback
                traceback.print_exc()
            except Exception as e:
                print(f"✗ LLM Adapter initialization failed: {e}")
                import traceback
                traceback.print_exc()
            
            gui = M3SimpleGUI(system)
            gui.mainloop()
            return
        except Exception as e:
            print(f"ERROR: Could not launch GUI: {e}")
            import traceback
            traceback.print_exc()
            return
    
    system = M3ConsciousnessCore(n=args.n, K=args.K, seed=args.seed, max_iterations=None, outdir=args.outdir)
    # If ARC folder provided, run ARC batch and exit
    if args.arc_dir:
        system.solve_arc_dir(
            args.arc_dir,
            out_dir=(args.arc_out if args.arc_save else None),
            save_preds=bool(args.arc_save),
            trace=bool(args.arc_trace),
            max_time_per=float(args.arc_time_per),
            require_solve=bool(args.arc_require_solve),
            total_time_cap=float(args.arc_total_time_cap),
        )
    else:
        system.run_autonomous()

# =================== SCOPE ENCODER + GUI HOOK (no fallbacks) ===================
# try:
#     from m3_scope import Scope
# except Exception as _e:
#     raise ImportError("m3_scope.py not found or failed to import; place it next to M3_CR.py") from _e

import numpy as _np
def _scope_compute_arousal(_state):
    import math as _math
    pem = _state['pred_err_map']
    td  = float(_state['td_error'])
    ign = float(_state['gw_ignition'])
    err_mean = float(_np.mean(pem)) if pem is not None else 0.0
    return 1.0/(1.0 + _math.exp(-(1.2*err_mean + 0.8*abs(td) + 1.0*ign)))

try:
    _EV = EvolutionVisualizer
    if not hasattr(_EV, "_scope_wired"):
        _EV._scope_wired = True
        _EV.__old_update__ = _EV.update
        def _update_scope(self, system_state):
            out = self.__old_update__(system_state)
            if not hasattr(self, "_scope"):
                self._scope = Scope()
            if not all(k in system_state for k in ('pred_err_map','td_error','gw_ignition')):
                # Drivers missing: do NOT fabricate data. Skip SCOPE encoding to preserve validity.
                print("[DEBUG] Missing scope drivers in system_state; skipping SCOPE encoding")
                # Ensure scope fields indicate no valid image produced
                self.scope_image = None
                self.scope_meta = None
                self.scope_arousal = None
                return out
            A = _scope_compute_arousal(system_state)
            u = _np.array(system_state.get('u_matrix', _np.zeros((32,32), dtype=_np.float32)), dtype=_np.float32)
            if _np.max(u) > _np.min(u):
                uu = (u - _np.min(u)) / (_np.max(u) - _np.min(u) + 1e-6)
            else:
                uu = u * 0.0
            try:
                import cv2 as _cv2
                frame = _cv2.resize(uu, (256,256), interpolation=_cv2.INTER_AREA)
            except Exception:
                # fallback plain resize
                frame = uu
            img, meta = self._scope.encode(frame, drivers={"arousal": A})
            self.scope_image = img
            self.scope_meta = meta
            self.scope_arousal = A
            return out
        _EV.update = _update_scope
except Exception as _e:
    # Do not break import if Scope is unavailable
    pass

# Optional GUI
_GUI_AVAILABLE = False
try:
    import tkinter as _tk
    from tkinter import scrolledtext as _scrolledtext
    import threading as _threading
    import time as _time
    import queue as _queue
    _GUI_AVAILABLE = True
except ImportError as e:
    print(f"GUI not available: {e}")
    _GUI_AVAILABLE = False

if _GUI_AVAILABLE:
    class M3SimpleGUI(_tk.Tk):
        def __init__(self, core: "M3ConsciousnessCore"):
            super().__init__()
            self.core = core
            self.running = False
            self.core_thread = None
            self._metric_queue = _queue.Queue()
            self._log_queue = _queue.Queue()
            
            self.title('M3 Consciousness')
            self.geometry('1000x700')
            self.configure(bg='#0a0a0a')
            
            BG = '#0a0a0a'
            BG_CARD = '#151515'
            TEXT = '#e0e0e0'
            TEXT_DIM = '#707070'
            ACCENT = '#00ff88'
            
            top = _tk.Frame(self, bg=BG, height=50)
            top.pack(fill=_tk.X, padx=10, pady=10)
            top.pack_propagate(False)
            
            self.start_btn = _tk.Button(top, text='▶ RUN', width=10, command=self._start, 
                                       bg=ACCENT, fg='#000', font=('Consolas', 10, 'bold'), 
                                       bd=0, cursor='hand2')
            self.start_btn.pack(side=_tk.LEFT, padx=5)
            
            self.stop_btn = _tk.Button(top, text='■ STOP', width=10, command=self._stop, 
                                      state=_tk.DISABLED, bg='#ff4444', fg='#000', 
                                      font=('Consolas', 10, 'bold'), bd=0, cursor='hand2')
            self.stop_btn.pack(side=_tk.LEFT, padx=5)
            
            self.status = _tk.Label(top, text='READY', bg=BG, fg='#ffaa00', 
                                   font=('Consolas', 12, 'bold'))
            self.status.pack(side=_tk.LEFT, padx=20)
            
            self.step_label = _tk.Label(top, text='T: 0', bg=BG, fg=TEXT_DIM, 
                                       font=('Consolas', 10))
            self.step_label.pack(side=_tk.LEFT, padx=10)
            
            # LLM 학습 버튼들 (항상 표시)
            _tk.Button(top, text='Save', width=8, command=self._save_llm,
                      bg='#0088ff', fg='#000', font=('Consolas', 9, 'bold'),
                      bd=0, cursor='hand2').pack(side=_tk.RIGHT, padx=5)
            
            _tk.Button(top, text='Train', width=8, command=self._train_llm,
                      bg='#ff8800', fg='#000', font=('Consolas', 9, 'bold'),
                      bd=0, cursor='hand2').pack(side=_tk.RIGHT, padx=5)
            
            main = _tk.Frame(self, bg=BG)
            main.pack(fill=_tk.BOTH, expand=True, padx=10, pady=(0,10))
            
            left = _tk.Frame(main, bg=BG_CARD, width=400)
            left.pack(side=_tk.LEFT, fill=_tk.BOTH, expand=True, padx=(0,5))
            
            _tk.Label(left, text='METRICS', bg=BG_CARD, fg=TEXT, 
                     font=('Consolas', 11, 'bold')).pack(pady=8)
            
            self.metrics = _scrolledtext.ScrolledText(left, height=30, bg='#101010', 
                                                     fg=TEXT, font=('Courier New', 9), 
                                                     bd=0, insertbackground=ACCENT)
            self.metrics.pack(fill=_tk.BOTH, expand=True, padx=8, pady=(0,8))
            
            right = _tk.Frame(main, bg=BG_CARD, width=550)
            right.pack(side=_tk.LEFT, fill=_tk.BOTH, expand=True, padx=(5,0))
            
            _tk.Label(right, text='CHAT', bg=BG_CARD, fg=TEXT, 
                     font=('Consolas', 11, 'bold')).pack(pady=8)
            
            self.chat = _scrolledtext.ScrolledText(right, height=20, bg='#101010', 
                                                  fg=TEXT, font=('Consolas', 9), 
                                                  bd=0, insertbackground=ACCENT)
            self.chat.pack(fill=_tk.BOTH, expand=True, padx=8, pady=(0,8))
            self.chat.tag_config('user', foreground='#00aaff', font=('Consolas', 9, 'bold'))
            self.chat.tag_config('assistant', foreground=ACCENT)
            
            chat_input = _tk.Frame(right, bg=BG_CARD)
            chat_input.pack(fill=_tk.X, padx=8, pady=(0,8))
            
            self.chat_var = _tk.StringVar()
            entry = _tk.Entry(chat_input, textvariable=self.chat_var, bg='#101010', 
                            fg=TEXT, font=('Consolas', 9), bd=1, insertbackground=ACCENT)
            entry.pack(side=_tk.LEFT, fill=_tk.X, expand=True, ipady=4)
            entry.bind('<Return>', lambda e: self._send_msg())
            
            _tk.Button(chat_input, text='SEND', command=self._send_msg, bg='#0088ff', 
                      fg='#fff', width=8, font=('Consolas', 9, 'bold'), bd=0, 
                      cursor='hand2').pack(side=_tk.LEFT, padx=(5,0))
            
            # LLM 컨트롤 버튼들 (항상 표시)
            llm_controls = _tk.Frame(right, bg=BG_CARD)
            llm_controls.pack(fill=_tk.X, padx=8, pady=(0,8))
            
            _tk.Label(llm_controls, text='LLM:', bg=BG_CARD, fg=TEXT_DIM,
                     font=('Consolas', 8)).pack(side=_tk.LEFT, padx=(0,5))
            
            _tk.Button(llm_controls, text='Train Chat', command=self._train_llm_chat,
                      bg='#ff8800', fg='#000', font=('Consolas', 8, 'bold'),
                      bd=0, cursor='hand2', width=10).pack(side=_tk.LEFT, padx=2)
            
            _tk.Button(llm_controls, text='Train Folder', command=self._train_llm_file,
                      bg='#ff6600', fg='#000', font=('Consolas', 8, 'bold'),
                      bd=0, cursor='hand2', width=11).pack(side=_tk.LEFT, padx=2)
            
            _tk.Button(llm_controls, text='DPO', command=self._train_llm_dpo,
                      bg='#ff4400', fg='#000', font=('Consolas', 8, 'bold'),
                      bd=0, cursor='hand2', width=8).pack(side=_tk.LEFT, padx=2)
            
            _tk.Button(llm_controls, text='Clear', command=self._clear_chat,
                      bg='#666666', fg='#fff', font=('Consolas', 8, 'bold'),
                      bd=0, cursor='hand2', width=8).pack(side=_tk.LEFT, padx=2)
            
            # 새로운 LLM 어댑터 기능 버튼들
            llm_advanced = _tk.Frame(right, bg=BG_CARD)
            llm_advanced.pack(fill=_tk.X, padx=8, pady=(0,8))
            
            _tk.Label(llm_advanced, text='Advanced:', bg=BG_CARD, fg=TEXT_DIM,
                     font=('Consolas', 8)).pack(side=_tk.LEFT, padx=(0,5))
            
            self.autonomy_btn = _tk.Button(llm_advanced, text='Autonomy Off', command=self._toggle_autonomy,
                                          bg='#00aa88', fg='#000', font=('Consolas', 8, 'bold'),
                                          bd=0, cursor='hand2', width=12)
            self.autonomy_btn.pack(side=_tk.LEFT, padx=2)

            _tk.Button(llm_advanced, text='Deep Sleep', command=self._trigger_sleep,
                      bg='#0000ff', fg='#fff', font=('Consolas', 8, 'bold'),
                      bd=0, cursor='hand2', width=12).pack(side=_tk.LEFT, padx=2)
            
            _tk.Button(llm_advanced, text='kNN Settings', command=self._knn_settings,
                      bg='#aa00aa', fg='#fff', font=('Consolas', 8, 'bold'),
                      bd=0, cursor='hand2', width=12).pack(side=_tk.LEFT, padx=2)
            
            self.m3_integration_btn = _tk.Button(llm_advanced, text='M3 Int Off', command=self._toggle_m3_integration,
                                                bg='#888800', fg='#000', font=('Consolas', 8, 'bold'),
                                                bd=0, cursor='hand2', width=12)
            self.m3_integration_btn.pack(side=_tk.LEFT, padx=2)
            
            _tk.Label(right, text='LOG', bg=BG_CARD, fg=TEXT_DIM, 
                     font=('Consolas', 9, 'bold')).pack(pady=(8,4))
            
            self.log = _scrolledtext.ScrolledText(right, height=8, bg='#050505', 
                                                 fg=TEXT_DIM, font=('Consolas', 8), bd=0)
            self.log.pack(fill=_tk.BOTH, expand=True, padx=8, pady=(0,8))
            
            self.protocol('WM_DELETE_WINDOW', self._on_close)
            
            # 초기 버튼 상태 설정
            self._update_button_states()
            
            # Show initial metrics
            self._show_initial_metrics()
            
            self.after(100, self._update_ui)
        
        def _show_initial_metrics(self):
            """Display initial metrics before RUN is pressed"""
            try:
                phi = self.core.phi_calculator.phi_history[-1] if self.core.phi_calculator.phi_history else 0.0
                m = {
                    't': self.core.t,
                    'phi': phi,
                    'meta': self.core.self_model.meta_awareness,
                    'conf': self.core.self_model.meta_confidence,
                    'unity': self.core.unified_subject.unity_score,
                    'energy': self.core.energy_ctrl.cognitive_energy,
                    'act': self.core.energy_ctrl.activation_level,
                }
                
                # Reward System Metrics
                if hasattr(self.core, 'rewards'):
                    rs = self.core.rewards
                    # Initial state: show 0.0 for drives as we haven't evaluated yet
                    m['drives'] = {k: 0.0 for k in rs.drives.keys()}
                    
                    if hasattr(rs, 'last_affect') and rs.last_affect is not None:
                        m['affect'] = {
                            'V': float(rs.last_affect[0]) if len(rs.last_affect) > 0 else 0.0,
                            'A': float(rs.last_affect[1]) if len(rs.last_affect) > 1 else 0.0,
                            'D': float(rs.last_affect[2]) if len(rs.last_affect) > 2 else 0.0
                        }
                
                # Legacy Qualia
                m['qualia'] = {
                    'arousal': self.core.qualia.arousal,
                    'valence': self.core.qualia.valence,
                    'entropy': self.core.qualia.entropy,
                    'engage': self.core.qualia.engagement,
                    'frust': self.core.qualia.frustration
                }
                
                ebar = int(m['energy'] / 100 * 20)
                abar = int(m['act'] * 20)
                energy_bar = '=' * ebar + '-' * (20 - ebar)
                active_bar = '=' * abar + '-' * (20 - abar)
                
                text = f"""
==========================================================
            CONSCIOUSNESS METRICS
==========================================================
  Phi (Phi)         {m['phi']:6.4f}
  Meta              {m['meta']:6.4f}
  Confidence        {m['conf']:6.4f}
  Unity             {m['unity']:6.4f}

            ENERGY & ACTIVATION
==========================================================
  Energy            [{energy_bar}] {m['energy']:5.1f}
  Activation        [{active_bar}] {m['act']:5.2f}
"""
                if 'drives' in m:
                    text += "\n            BIOLOGICAL DRIVES\n==========================================================\n"
                    for dname, dval in m['drives'].items():
                        bar_len = int(dval * 20)
                        bar = '#' * bar_len + '-' * (20 - bar_len)
                        text += f"  {dname:<15} [{bar}] {dval:5.2f}\n"

                if 'affect' in m:
                    text += "\n            AFFECT STATE (V-A-D)\n==========================================================\n"
                    text += f"  Valence           {m['affect']['V']:6.4f}\n"
                    text += f"  Arousal           {m['affect']['A']:6.4f}\n"
                    text += f"  Dominance         {m['affect']['D']:6.4f}\n"
                
                if 'qualia' in m:
                    q = m['qualia']
                    text += "\n            QUALIA STATE (Legacy)\n==========================================================\n"
                    text += f"  Arousal           {q['arousal']:6.4f}\n"
                    text += f"  Valence           {q['valence']:6.4f}\n"
                    text += f"  Entropy           {q['entropy']:6.4f}\n"
                    text += f"  Engagement        {q['engage']:6.4f}\n"
                    text += f"  Frustration       {q['frust']:6.4f}\n"
                
                text += "==========================================================\n"
                self.metrics.insert('1.0', text)
            except Exception as e:
                self.metrics.insert('1.0', f"Metrics loading...\n{e}")
        
        def _start(self):
            if self.running:
                return
            self.running = True
            self.start_btn.config(state=_tk.DISABLED)
            self.stop_btn.config(state=_tk.NORMAL)
            self.status.config(text='RUNNING', fg='#00ff88')
            self.core_thread = _threading.Thread(target=self._run_core, daemon=True)
            self.core_thread.start()
            self._log('M3 started')
        
        def _stop(self):
            self.running = False
            self.start_btn.config(state=_tk.NORMAL)
            self.stop_btn.config(state=_tk.DISABLED)
            self.status.config(text='STOPPED', fg='#ffaa00')
            self.core._save_checkpoint()
            self._log('M3 stopped')
        
        def _run_core(self):
            last_metric = 0
            while self.running:
                should_continue, intensity = self.core.energy_ctrl.should_continue()
                if not should_continue:
                    # Energy deadlock prevention: passive recovery while halted
                    try:
                        self.core.energy_ctrl.update_energy(0.0)
                    except Exception:
                        self.core.energy_ctrl.cognitive_energy += max(
                            0.5, self.core.energy_ctrl.recovery_rate_max * 0.3
                        )
                        self.core.energy_ctrl.cognitive_energy = min(
                            self.core.energy_ctrl.cognitive_energy,
                            self.core.energy_ctrl.energy_capacity
                        )
                    self.core.t += 1
                    _time.sleep(0.01)
                    continue
                
                self.core.energy_ctrl.internal_clock += 1
                self.core.world_state = self.core._get_current_world_state()
                goal = self.core.goal_gen.generate_goal(self.core.self_model, self.core.qualia, 
                                                       self.core.world_state, self.core.self_model.state_history)
                self.core.energy_ctrl.update_activation(self.core.qualia, self.core.self_model, goal)
                
                # --- Reward System Update (GUI Loop) ---
                if hasattr(self.core, 'rewards'):
                    viability_cost = 0.0
                    if hasattr(self.core.energy_ctrl, 'cognitive_energy'):
                        energy_ratio = self.core.energy_ctrl.cognitive_energy / max(1.0, self.core.energy_ctrl.energy_capacity)
                        viability_cost = max(0.0, 1.0 - energy_ratio)
                    
                    reward_ctx = {
                        "viability_cost": viability_cost,
                        "context": self.core.world_state,
                        "phi": self.core.phi_calculator.phi_history[-1] if self.core.phi_calculator.phi_history else 0.0,
                        "stability": self.core.world_state.get('stability', 0.5),
                        "t": self.core.t
                    }
                    # Evaluate and store for GUI
                    self.core._last_drive_scores = self.core.rewards.evaluate_all(self.core.h, reward_ctx)
                # ---------------------------------------
                
                if intensity < 0.2:
                    if self.core.t % 5 == 0:
                        self.core.generate_internal_events()
                    self.core.t += 1
                    _time.sleep(0.01)
                    continue
                
                if self.core.t % 100 == 0:
                    pert = self.core.rng.normal(0, 0.5, size=self.core.n)
                    self.core.h += pert
                    self.core.base_vec = np.abs(self.core.rng.normal(0, 0.5, size=self.core.K))
                    self.core.base_vec /= self.core.base_vec.sum()
                
                if self.core.t % 2 == 0:
                    self.core.generate_internal_events()
                
                if self.core.event_queue:
                    evt = self.core.select_most_important_event()
                    if evt:
                        self.core.process_event(evt, goal)
                
                action = self.core._decide_action(goal, intensity)
                delta, _ = self.core._execute_action(action)
                self.core._experience_qualia(delta, action)
                
                grounded = self.core.conceptual_space.ground_experience(self.core.qualia)
                self.core._submit_to_workspace(grounded, goal)
                contents = self.core.global_workspace.compete_for_consciousness()
                broadcast = self.core.global_workspace.broadcast(world_state=self.core.world_state)
                
                self.core.self_model.update_meta_awareness(contents)
                
                unified_exp = self.core.unified_subject.bind_experience(qualia=self.core.qualia, workspace_contents=contents, beliefs=self.core.self_model.state_history, goals=[goal], t=self.core.t)
                self.core.unified_subject.experience(unified_exp, 'unified_consciousness', is_intentional=goal is not None)

                # --- Autonomous Sleep Logic (Hybrid: Cognitive + Biological) ---
                # 1. Cognitive Decision (Goal-Driven): The GoalGenerator explicitly requests REST.
                # 2. Biological Reflex (Fail-safe): Energy is critically low and environment is safe.
                try:
                    energy_level = getattr(self.core.energy_ctrl, 'activation_level', 1.0)
                    stability = getattr(self.core.world_state, 'get', lambda k,d: d)('stability', 0.5)
                    
                    # Criteria A: Cognitive Decision
                    should_sleep_cognitive = (goal and goal.type == GoalType.REST)
                    
                    # Criteria B: Biological Reflex (Emergency Nap)
                    should_sleep_biological = (energy_level < 0.2 and stability > 0.6)

                    if should_sleep_cognitive or should_sleep_biological:
                        # Check debounce
                        now = _time.time()
                        last_sleep = getattr(self, '_last_auto_sleep', 0)
                        if now - last_sleep > 300: # Min 5 minutes between naps
                            reason = "Goal: REST" if should_sleep_cognitive else "Critical Fatigue"
                            self._log(f"💤 Auto-Sleep ({reason}). Consolidating Memories...")
                            self._trigger_sleep()
                            self._last_auto_sleep = now
                except Exception:
                    pass
                # -------------------------------------------------
                
                qvec = np.array([self.core.qualia.arousal, self.core.qualia.valence,
                               self.core.qualia.entropy, self.core.qualia.engagement,
                               self.core.qualia.frustration])
                self.core.growing_som.learn(qvec)
                
                self.core._reflect_and_learn(delta, goal)
                
                cost = self.core.energy_ctrl.compute_cognitive_cost(action)
                self.core.energy_ctrl.update_energy(cost)
                
                if self.core.t % 10 == 0:
                    self.core._log_state(delta, goal)
                
                if self.core.t % 10000 == 0:
                    self.core._save_checkpoint()
                
                # 더 자주 메트릭 업데이트 (10 스텝마다)
                if self.core.t % 10 == 0 and self.core.t != last_metric:
                    last_metric = self.core.t
                    self._push_metrics()
                
                self.core.t += 1
                _time.sleep(0.001)
            
            self.status.config(text='STOPPED', fg='#ffaa00')
        
        def _push_metrics(self):
            phi = self.core.phi_calculator.phi_history[-1] if self.core.phi_calculator.phi_history else 0.0
            m = {
                't': self.core.t,
                'phi': phi,
                'meta': self.core.self_model.meta_awareness,
                'conf': self.core.self_model.meta_confidence,
                'unity': self.core.unified_subject.unity_score,
                'energy': self.core.energy_ctrl.cognitive_energy,
                'act': self.core.energy_ctrl.activation_level,
            }
            
            # Reward System Metrics
            if hasattr(self.core, 'rewards'):
                rs = self.core.rewards
                # Use cached scores if available
                if hasattr(self.core, '_last_drive_scores'):
                    m['drives'] = self.core._last_drive_scores
                
                if hasattr(rs, 'last_affect') and rs.last_affect is not None:
                    m['affect'] = {
                        'V': float(rs.last_affect[0]) if len(rs.last_affect) > 0 else 0.0,
                        'A': float(rs.last_affect[1]) if len(rs.last_affect) > 1 else 0.0,
                        'D': float(rs.last_affect[2]) if len(rs.last_affect) > 2 else 0.0
                    }
            
            # Legacy Qualia (fallback)
            m['qualia'] = {
                'arousal': self.core.qualia.arousal,
                'valence': self.core.qualia.valence,
                'entropy': self.core.qualia.entropy,
                'engage': self.core.qualia.engagement,
                'frust': self.core.qualia.frustration
            }
            self._metric_queue.put(m)
        
        def _send_msg(self):
            msg = self.chat_var.get().strip()
            self.chat_var.set('')
            if not msg:
                return
            self.chat.insert(_tk.END, f'You: {msg}\n', 'user')
            self.chat.see(_tk.END)
            
            # Run generation in a separate thread to prevent GUI freezing
            def _process():
                try:
                    resp = self.core.handle_user_message(msg)
                    self.after(0, lambda: self._on_response(resp))
                except Exception as e:
                    self.after(0, lambda: self._log(f"Error: {e}"))
            
            _threading.Thread(target=_process, daemon=True).start()

        def _on_response(self, resp):
            self.chat.insert(_tk.END, f'M3: {resp}\n\n', 'assistant')
            self.chat.see(_tk.END)
        
        def _log(self, msg):
            formatted_msg = f'[{_time.strftime("%H:%M:%S")}] {msg}'
            self._log_queue.put(formatted_msg)
            print(formatted_msg)  # 콘솔에도 출력
        
        def _update_ui(self):
            if self.core:
                self.step_label.config(text=f'T: {self.core.t:,}')
            
            while not self._log_queue.empty():
                msg = self._log_queue.get_nowait()
                self.log.insert(_tk.END, f'{msg}\n')
                self.log.see(_tk.END)
            
            if not self._metric_queue.empty():
                m = self._metric_queue.get_nowait()
                self.metrics.delete('1.0', _tk.END)
                
                ebar = int(m['energy'] / 100 * 20)
                abar = int(m['act'] * 20)
                energy_bar = '=' * ebar + '-' * (20 - ebar)
                active_bar = '=' * abar + '-' * (20 - abar)
                
                text = f"""
==========================================================
            CONSCIOUSNESS METRICS
==========================================================
  Phi (Phi)         {m['phi']:6.4f}
  Meta              {m['meta']:6.4f}
  Confidence        {m['conf']:6.4f}
  Unity             {m['unity']:6.4f}

            ENERGY & ACTIVATION
==========================================================
  Energy            [{energy_bar}] {m['energy']:5.1f}
  Activation        [{active_bar}] {m['act']:5.2f}
"""
                if 'drives' in m:
                    text += "\n            BIOLOGICAL DRIVES\n==========================================================\n"
                    for dname, dval in m['drives'].items():
                        bar_len = int(dval * 20)
                        bar = '#' * bar_len + '-' * (20 - bar_len)
                        text += f"  {dname:<15} [{bar}] {dval:5.2f}\n"

                if 'affect' in m:
                    text += "\n            AFFECT STATE (V-A-D)\n==========================================================\n"
                    text += f"  Valence           {m['affect']['V']:6.4f}\n"
                    text += f"  Arousal           {m['affect']['A']:6.4f}\n"
                    text += f"  Dominance         {m['affect']['D']:6.4f}\n"
                
                # Legacy Qualia display (always show for now as fallback/comparison)
                if 'qualia' in m:
                    q = m['qualia']
                    text += "\n            QUALIA STATE (Legacy)\n==========================================================\n"
                    text += f"  Arousal           {q['arousal']:6.4f}\n"
                    text += f"  Valence           {q['valence']:6.4f}\n"
                    text += f"  Entropy           {q['entropy']:6.4f}\n"
                    text += f"  Engagement        {q['engage']:6.4f}\n"
                    text += f"  Frustration       {q['frust']:6.4f}\n"
                
                text += "==========================================================\n"
                self.metrics.insert('1.0', text)
            
            self.after(100, self._update_ui)
        
        def _train_llm_chat(self):
            """현재 채팅 대화로 LLM 학습"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                self._log('✗ LLM adapter not available - check console for errors')
                return
            
            def _train_thread():
                try:
                    self._log('Training LLM on chat history...')
                    if hasattr(self.core, '_chat_history') and len(self.core._chat_history) >= 2:
                        count = 0
                        # Convert deque to list for indexing
                        history = list(self.core._chat_history)
                        for i in range(0, len(history) - 1):
                            # Find user -> assistant pairs
                            user_msg = history[i]
                            asst_msg = history[i + 1]
                            
                            if user_msg.get('role') == 'user' and asst_msg.get('role') == 'assistant':
                                # Construct prompt exactly as in _generate_utterance
                                # Use context if available (up to 3 previous turns)
                                context_start = max(0, i - 4)
                                context_msgs = history[context_start:i+1] # Include current user msg
                                
                                prompt = ""
                                for msg in context_msgs:
                                    role = "User" if msg.get('role') == 'user' else "M3"
                                    text = msg.get('text', '')
                                    prompt += f"{role}: {text}\n"
                                prompt += "M3:"
                                
                                response = asst_msg.get('text', '')
                                
                                if prompt and response:
                                    adapter.train_on_example(prompt, response)
                                    count += 1
                                    
                        self._log(f'✓ Trained on {count} conversation pairs')
                    else:
                        self._log('No chat history to train on')
                except Exception as e:
                    self._log(f'✗ Training error: {e}')
                    import traceback
                    traceback.print_exc()
            
            _threading.Thread(target=_train_thread, daemon=True).start()
        
        def _train_llm_file(self):
            """폴더에서 LLM 학습"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                self._log('✗ LLM adapter not available - check console for errors')
                return
            
            # 폴더 선택 대화상자
            from tkinter import filedialog
            folder_path = filedialog.askdirectory(
                title='Select Training Data Folder',
                initialdir=self.core.outdir
            )
            
            if not folder_path:
                self._log('Folder selection cancelled')
                return
            
            def _train_thread():
                try:
                    self._log(f'Training LLM from folder: {os.path.basename(folder_path)}')
                    import glob
                    import json
                    import csv
                    
                    # 폴더 내 모든 학습 데이터 파일 찾기 (하위 폴더 포함)
                    files = []
                    supported_exts = ['.jsonl', '.json', '.csv', '.tsv', '.txt']
                    for root, dirs, fnames in os.walk(folder_path):
                        for fname in fnames:
                            if any(fname.lower().endswith(ext) for ext in supported_exts):
                                files.append(os.path.join(root, fname))
                    
                    if not files:
                        self._log(f'✗ No training files found in {folder_path}')
                        return
                    
                    self._log(f'Found {len(files)} file(s)')
                    total_count = 0
                    
                    # Check for Plastic Brain Adapter
                    adapter = getattr(self.core, 'llm_adapter', None)
                    is_plastic = hasattr(adapter, 'model') and type(adapter.model).__name__ == "M3PlasticPolicy"
                    is_plastic_wrapper = hasattr(adapter, 'learn') # Wrapper check
                    
                    if is_plastic or is_plastic_wrapper:
                        self._log("Detected 1.58-bit Plastic Brain. Using Hebbian Learning Mode.")
                        
                        for i, fpath in enumerate(files):
                            try:
                                # Simple text reading for plastic brain (for now)
                                # TODO: Handle structured JSONL specifically if needed
                                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                                    text_content = f.read()
                                
                                if len(text_content.strip()) < 10:
                                    continue
                                    
                                self._log(f"Processing ({i+1}/{len(files)}): {os.path.basename(fpath)}")
                                
                                # Call unified learn interface
                                # High arousal for explicit study material
                                if is_plastic_wrapper:
                                    adapter.learn(text_content, arousal=2.0)
                                else:
                                    # Direct model call fallback
                                    adapter.model.learn_from_text(text_content, arousal=2.0)
                                    
                                total_count += 1
                                
                                # Consolidate (Sleep) every 10 files to save memory and solidify
                                if total_count % 10 == 0:
                                    self._log("Micro-sleep consolidation...")
                                    if is_plastic_wrapper:
                                        adapter.model.sleep()
                                    else:
                                        adapter.model.sleep()
                                        
                            except Exception as e:
                                self._log(f"Error reading {fpath}: {e}")
                                
                        # Final consolidation
                        self._log("Final Sleep Consolidation...")
                        adapter.model.sleep()
                        self._log(f"✓ Hebbian Learning Complete. Processed {total_count} files.")
                        return

                    # === Conventional Training Fallback (Original Code) ===
                    try:
                        from tqdm import tqdm
                    except ImportError:
                        self._log("Installing tqdm...")
                        import subprocess, sys
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
                        from tqdm import tqdm
                    
                    # Batch training setup
                    batch_buffer = []
                    BATCH_SIZE = 8
                    epoch_losses = []
                    
                    def add_to_batch(prompt, response):
                        nonlocal total_count
                        # Basic validation
                        if not prompt or not response:
                            try:
                                self._log(f"[BATCH SKIP] {fname} empty prompt/response (prompt_len={len(str(prompt))}, resp_len={len(str(response))})")
                            except Exception:
                                pass
                            return

                        # Append and debug-log a short preview (avoid huge logs)
                        batch_buffer.append((prompt, response))
                        try:
                            p_preview = (str(prompt)[:160].replace('\n', ' '))
                            r_preview = (str(response)[:160].replace('\n', ' '))
                            # self._log(f"[BATCH ADD] {os.path.basename(file_path)} buffer={len(batch_buffer)}/{BATCH_SIZE} prompt={p_preview!r} -> resp={r_preview!r}")
                        except Exception:
                            pass

                        if len(batch_buffer) >= BATCH_SIZE:
                            try:
                                ret = adapter.train_batch(batch_buffer)
                                # adapter.train_batch returns (count, avg_loss) or count
                                if isinstance(ret, tuple):
                                    count, loss = ret
                                    total_count += count
                                    if loss > 0:
                                        epoch_losses.append(loss)
                                    # self._log(f"[BATCH SENT] {os.path.basename(file_path)} sent {len(batch_buffer)} -> trained {count}, loss={loss:.4f}")
                                elif isinstance(ret, int):
                                    total_count += ret
                                    self._log(f"[BATCH SENT] {os.path.basename(file_path)} sent {len(batch_buffer)} -> trained {ret}")
                                else:
                                    total_count += len(batch_buffer)
                                    self._log(f"[BATCH SENT] {os.path.basename(file_path)} sent {len(batch_buffer)} (adapter returned nothing)")
                            except Exception as e:
                                self._log(f"Error in batch training: {e}")
                            finally:
                                batch_buffer.clear()

                    # Legal Case Corpus: Map.txt 로드 (카테고리 매핑)
                    legal_map = {}
                    try:
                        map_path = os.path.join(folder_path, 'Map.txt')
                        if not os.path.exists(map_path):
                            # 하위 디렉토리에서 찾기
                            for root, dirs, fnames in os.walk(folder_path):
                                if 'Map.txt' in fnames:
                                    map_path = os.path.join(root, 'Map.txt')
                                    break
                        
                        if os.path.exists(map_path):
                            with open(map_path, 'r', encoding='utf-8', errors='ignore') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line or line.startswith('#'):
                                        continue
                                    parts = line.split(maxsplit=1)
                                    if len(parts) == 2:
                                        legal_map[parts[0]] = parts[1]
                            if legal_map:
                                self._log(f'Loaded legal category map: {len(legal_map)} categories')
                    except Exception as e:
                        pass  # Map.txt 없으면 무시

                    from contextlib import contextmanager

                    @contextmanager
                    def open_with_fallback(path):
                        """Try opening a text file with several encodings and yield a file object.
                        Logs which encoding succeeded. Falls back to latin-1 with replace if all fail.
                        """
                        encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
                        f = None
                        try:
                            for enc in encodings:
                                try:
                                    f = open(path, 'r', encoding=enc)
                                    # Test read
                                    f.read(1)
                                    f.seek(0)
                                    try:
                                        self._log(f'Opened {os.path.basename(path)} with encoding {enc}')
                                    except Exception:
                                        pass
                                    yield f
                                    return
                                except (UnicodeDecodeError, UnicodeError):
                                    if f:
                                        f.close()
                                        f = None
                                    continue
                                except Exception:
                                    if f:
                                        f.close()
                                    raise

                            # last resort: open with latin-1 and replace errors
                            f = open(path, 'r', encoding='latin-1', errors='replace')
                            try:
                                self._log(f'Opened {os.path.basename(path)} with fallback latin-1 (errors replaced)')
                            except Exception:
                                pass
                            yield f
                        finally:
                            if f:
                                f.close()

                    def process_one_file(file_path):
                        try:
                            fname = os.path.basename(file_path)
                            # 설정 파일 무시
                            if fname in ['hitproperties.json', 'hittypeproperties.json']:
                                return
                                
                            count = 0
                            error_count = 0
                            
                            # JSONL/JSON 파일 처리
                            if file_path.endswith('.jsonl') or file_path.endswith('.json'):
                                with open_with_fallback(file_path) as f:
                                    content = f.read().strip()
                                    
                                    # JSON 처리 로직 (재사용)
                                    def process_json_item(data):
                                        nonlocal count, error_count
                                        try:
                                            if isinstance(data, dict):
                                                # Debug: 첫 번째 아이템의 키 확인
                                                if count == 0 and error_count == 0:
                                                    self._log(f"Debug: First item keys in {fname}: {list(data.keys())}")

                                                # SLURP 데이터셋 (sentence -> intent/action)
                                                if 'sentence' in data and 'intent' in data:
                                                    prompt = f"Intent: {data['intent']}\nSentence: {data['sentence']}"
                                                    response = data.get('sentence_annotation', data['sentence'])
                                                    add_to_batch(prompt, response)
                                                    count += 1
                                                # S2ORC Metadata (title, abstract, journal, year)
                                                elif 'title' in data and ('abstract' in data or 'paperAbstract' in data):
                                                    title = data.get('title', '').strip()
                                                    abstract = data.get('abstract') or data.get('paperAbstract') or ''
                                                    abstract = abstract.strip()
                                                    journal = data.get('journal') or data.get('venue') or ''
                                                    year = data.get('year', '')
                                                    
                                                    if title:
                                                        meta = f"Journal: {journal} ({year})" if journal else ""
                                                        prompt = f"Title: {title}\n{meta}".strip()
                                                        response = f"Abstract: {abstract}" if abstract else f"Title: {title}"
                                                        add_to_batch(prompt, response)
                                                        count += 1

                                                # Multi-XScience (related_work + ref_abstract)
                                                elif 'related_work' in data and 'ref_abstract' in data:
                                                    # Target: Related Work section
                                                    target = data.get('related_work', '').strip()
                                                    
                                                    # Source: Current Abstract + Reference Abstracts
                                                    curr_abstract = data.get('abstract', '').strip()
                                                    refs = data.get('ref_abstract', {})
                                                    
                                                    ref_text = ""
                                                    for cite_id, cite_info in refs.items():
                                                        if isinstance(cite_info, dict):
                                                            abs_text = cite_info.get('abstract', '').strip()
                                                            if abs_text:
                                                                ref_text += f"[{cite_id}] {abs_text}\n"
                                                    
                                                    if target and (curr_abstract or ref_text):
                                                        prompt = f"Abstract: {curr_abstract}\n\nReferences:\n{ref_text}".strip()
                                                        add_to_batch(prompt, target)
                                                        count += 1

                                                # AAAI-21 (Acronym Disambiguation)
                                                elif 'acronym' in data and 'expansion' in data and 'tokens' in data:
                                                    acronym = data.get('acronym', '').strip()
                                                    expansion = data.get('expansion', '').strip()
                                                    tokens = data.get('tokens', [])
                                                    if isinstance(tokens, list):
                                                        tokens_str = " ".join(str(t) for t in tokens)
                                                        if acronym and expansion and tokens_str:
                                                            prompt = f"Expand acronym '{acronym}' in: {tokens_str}"
                                                            add_to_batch(prompt, expansion)
                                                            count += 1

                                                # ZEST / MTurk Templates Support
                                                # 1. Paraphrase Task
                                                elif 'paraphrase_q' in data and 'original_question' in data:
                                                    orig = data.get('original_question', '').strip()
                                                    paras = data.get('paraphrase_q', [])
                                                    if orig and isinstance(paras, list):
                                                        for p in paras:
                                                            if isinstance(p, str) and p.strip():
                                                                add_to_batch(f"Paraphrase: {orig}", p.strip())
                                                                count += 1
                                                
                                                # 2. Structure Task
                                                elif 'structure_example' in data and 'structure_final_question' in data:
                                                    ex = data.get('structure_example', '').strip()
                                                    final = data.get('structure_final_question', '').strip()
                                                    if ex and final:
                                                        add_to_batch(f"Restructure this question: {ex}", final)
                                                        count += 1

                                                # 3. Semantic Flips (Yes/No & Extraction)
                                                elif 'yes_no_example' in data and 'yes_no_good_example1' in data:
                                                    src = data.get('yes_no_example', '').strip()
                                                    tgt = data.get('yes_no_good_example1', '').strip()
                                                    if src and tgt:
                                                        add_to_batch(f"Convert to Yes/No question: {src}", tgt)
                                                        count += 1
                                                    
                                                    # Extraction part
                                                    src_ext = data.get('extraction_example', '').strip()
                                                    tgt_ext = data.get('extraction_good_example1', '').strip()
                                                    if src_ext and tgt_ext:
                                                        add_to_batch(f"Convert to extraction question: {src_ext}", tgt_ext)
                                                        count += 1

                                                # 4. Generic Question List (q_to_change)
                                                elif 'q_to_change' in data and isinstance(data['q_to_change'], list):
                                                    domain = data.get('domain', 'general')
                                                    for q in data['q_to_change']:
                                                        if isinstance(q, str) and q.strip():
                                                            # Self-supervised / Domain adaptation
                                                            add_to_batch(f"Generate a question about {domain}", q.strip())
                                                            count += 1

                                                # 일반적인 키 쌍 찾기
                                                elif 'prompt' in data and 'response' in data:
                                                    add_to_batch(data['prompt'], data['response'])
                                                    count += 1
                                                elif 'user' in data and 'assistant' in data:
                                                    add_to_batch(data['user'], data['assistant'])
                                                    count += 1
                                                elif 'input' in data and 'output' in data:
                                                    add_to_batch(data['input'], data['output'])
                                                    count += 1
                                                elif 'question' in data and 'answer' in data:
                                                    add_to_batch(data['question'], data['answer'])
                                                    count += 1
                                                # ZEST 형식: description + examples[{text, label}]
                                                elif 'description' in data and isinstance(data.get('examples'), list):
                                                    desc = str(data.get('description', '')).strip()
                                                    for ex in data.get('examples', []):
                                                        try:
                                                            if isinstance(ex, dict):
                                                                text = str(ex.get('text', '')).strip()
                                                                label = str(ex.get('label', '')).strip()
                                                                if desc and text and label:
                                                                    prompt = f"{desc}\nText: {text}"
                                                                    add_to_batch(prompt, label)
                                                                    count += 1
                                                        except:
                                                            continue
                                                # Chat 형식: messages/conversations
                                                elif 'messages' in data or 'conversations' in data:
                                                    msgs = data.get('messages') or data.get('conversations')
                                                    if isinstance(msgs, list) and msgs:
                                                        last_user = None
                                                        for m in msgs:
                                                            try:
                                                                role = str(m.get('role') or m.get('from') or '').lower()
                                                                content = str(m.get('content') or m.get('value') or '')
                                                            except:
                                                                continue
                                                            if role in ('user', 'human'):
                                                                last_user = content
                                                            elif role in ('assistant', 'bot', 'gpt') and last_user:
                                                                add_to_batch(last_user, content)
                                                                count += 1
                                                                last_user = None
                                                # Alpaca 형식: instruction + input + output
                                                elif 'instruction' in data and 'output' in data:
                                                    instruction = str(data.get('instruction', '')).strip()
                                                    inp = str(data.get('input', '')).strip()
                                                    output = str(data.get('output', '')).strip()
                                                    if instruction and output:
                                                        prompt = f"{instruction}\n{inp}" if inp else instruction
                                                        add_to_batch(prompt, output)
                                                        count += 1
                                                # ZEST MTurk Template 형식 (input_paragraph[], q_to_label[])
                                                elif 'input_paragraph' in data and 'q_to_label' in data:
                                                    inputs = data.get('input_paragraph')
                                                    questions = data.get('q_to_label')
                                                    if isinstance(inputs, list) and isinstance(questions, list):
                                                        for i in range(min(len(inputs), len(questions))):
                                                            try:
                                                                text = str(inputs[i]).strip()
                                                                q = str(questions[i]).strip()
                                                                if text and q:
                                                                    # 질문 생성 태스크로 학습
                                                                    add_to_batch(
                                                                        f"Generate a question for this text:\n{text}",
                                                                        q
                                                                    )
                                                                    count += 1
                                                            except:
                                                                continue
                                            # 리스트 형태 (2개 요소)
                                            elif isinstance(data, list) and len(data) == 2:
                                                add_to_batch(str(data[0]), str(data[1]))
                                                count += 1
                                        except Exception as e:
                                            if error_count < 3:
                                                self._log(f"Error processing JSON item in {fname}: {e}")
                                                error_count += 1

                                    # 1. 전체를 하나의 JSON으로 파싱 시도 (List or Dict)
                                    parsed_as_whole = False
                                    try:
                                        data_whole = json.loads(content)
                                        parsed_as_whole = True
                                        if isinstance(data_whole, list):
                                            for item in tqdm(data_whole, desc=f"JSON {fname}", leave=False):
                                                process_json_item(item)
                                        elif isinstance(data_whole, dict):
                                            process_json_item(data_whole)
                                    except json.JSONDecodeError:
                                        pass # JSONL 처리로 넘어감
                                    except Exception as e:
                                        if error_count < 3:
                                            self._log(f"Error parsing JSON file {fname}: {e}")
                                            error_count += 1

                                    # 2. JSONL 처리 (전체 파싱 실패 시)
                                    if not parsed_as_whole:
                                        lines = content.split('\n')
                                        for line in tqdm(lines, desc=f"JSONL {fname}", leave=False):
                                            try:
                                                if not line.strip():
                                                    continue
                                                data = json.loads(line.strip())
                                                process_json_item(data)
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing JSONL line in {fname}: {e}")
                                                    error_count += 1
                                                continue
                            
                            # TSV/TXT 파일 처리
                            elif file_path.endswith('.tsv') or file_path.endswith('.txt'):
                                with open_with_fallback(file_path) as f:
                                    # MIND news.tsv
                                    if 'news.tsv' in fname:
                                        for line in f:
                                            try:
                                                parts = line.strip().split('\t')
                                                if count == 0 and error_count == 0:
                                                    self._log(f"Debug: Parsing news.tsv line 1: {parts} (len={len(parts)})")
                                                
                                                if len(parts) >= 4:
                                                    # Category + Title -> Abstract
                                                    category = parts[1]
                                                    title = parts[3]
                                                    abstract = parts[4] if len(parts) > 4 else ""
                                                    prompt = f"Category: {category}\nTitle: {title}"
                                                    add_to_batch(prompt, abstract)
                                                    count += 1
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing news.tsv line: {e}")
                                                    error_count += 1
                                                continue
                                    
                                    # MIND behaviors.tsv (User History -> Next Click)
                                    elif 'behaviors.tsv' in fname:
                                        for line in f:
                                            try:
                                                parts = line.strip().split('\t')
                                                if count == 0 and error_count == 0:
                                                    self._log(f"Debug: Parsing behaviors.tsv line 1: {parts} (len={len(parts)})")

                                                if len(parts) >= 4:
                                                    history = parts[3]
                                                    impressions = parts[4] if len(parts) > 4 else ""
                                                    # 긍정적 클릭(1)만 추출
                                                    clicks = [x.split('-')[0] for x in impressions.split() if x.endswith('-1')]
                                                    if history and clicks:
                                                        prompt = f"User History: {history}"
                                                        response = f"Next Clicks: {' '.join(clicks)}"
                                                        add_to_batch(prompt, response)
                                                        count += 1
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing behaviors.tsv line: {e}")
                                                    error_count += 1
                                                continue

                                    # Knowledge Graph (Subject \t Relation \t Object)
                                    elif any(x in fname for x in ['train.txt', 'valid.txt', 'test.txt', 'kg.txt', 'train.tsv', 'valid.tsv', 'test.tsv', 'edges_as_text']):
                                        for line in f:
                                            try:
                                                parts = line.strip().split('\t')
                                                if count == 0 and error_count == 0:
                                                    self._log(f"Debug: Parsing KG line 1: {parts} (len={len(parts)})")

                                                if len(parts) == 3:
                                                    s, r, o = parts
                                                    # 양방향 학습
                                                    try:
                                                        add_to_batch(f"{s} {r}", o)
                                                        add_to_batch(f"{o} inverse_{r}", s)
                                                        count += 1
                                                        if count == 1:
                                                            self._log(f"Debug: Successfully trained first KG example in {fname}")
                                                    except Exception as train_err:
                                                        self._log(f"CRITICAL ERROR in train_on_example for {fname}: {train_err}")
                                                        import traceback
                                                        traceback.print_exc()
                                                        error_count += 1
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing KG line in {fname}: {e}")
                                                    error_count += 1
                                                continue
                                    
                                    # Legal Case Corpus (Map.txt는 위에서 처리됨, 여기서는 일반 텍스트)
                                    elif 'Map.txt' not in fname:
                                        try:
                                            content = f.read()
                                            # 파일명에서 카테고리 유추
                                            category = None
                                            for cat_id, cat_name in legal_map.items():
                                                if cat_id in fname:
                                                    category = cat_name
                                                    break
                                            
                                            if category:
                                                # 앞부분을 프롬프트로, 뒷부분을 응답으로 (간단한 방식)
                                                mid = len(content) // 2
                                                prompt = f"Legal Case ({category}):\n{content[:min(mid, 500)]}..."
                                                response = content[min(mid, 500):min(mid+500, len(content))]
                                                add_to_batch(prompt, response)
                                                count += 1
                                        except Exception as e:
                                            self._log(f"Error parsing Legal Case {fname}: {e}")
                                            pass
                            
                            # CSV 파일 처리
                            elif file_path.endswith('.csv'):
                                with open_with_fallback(file_path) as f:
                                    # 첫 줄 확인 (헤더 체크)
                                    first_line = f.readline().strip()
                                    f.seek(0)
                                    
                                    # 헤더가 있는 경우
                                    if any(k in first_line.lower() for k in ['prompt', 'user', 'input', 'question', 'response', 'assistant', 'output', 'answer']):
                                        reader = csv.DictReader(f)
                                        for row in tqdm(reader, desc=f"CSV {fname}", leave=False):
                                            try:
                                                prompt = row.get('prompt') or row.get('user') or row.get('input') or row.get('question')
                                                response = row.get('response') or row.get('assistant') or row.get('output') or row.get('answer')
                                                if prompt and response:
                                                    add_to_batch(prompt, response)
                                                    count += 1
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing CSV row in {fname}: {e}")
                                                    error_count += 1
                                                continue
                                    else:
                                        # 헤더 없는 경우 (쉼표로 구분된 데이터)
                                        reader = csv.reader(f)
                                        for row in tqdm(reader, desc=f"CSV {fname}", leave=False):
                                            try:
                                                # Word2vec 결과 형식 감지: token:score 형식 여러 개
                                                def _is_w2v_tok(x: str) -> bool:
                                                    x = str(x).strip()
                                                    if ':' not in x:
                                                        return False
                                                    parts = x.split(':', 1)
                                                    if len(parts) != 2:
                                                        return False
                                                    try:
                                                        float(parts[1])
                                                        return len(parts[0].strip()) > 0
                                                    except:
                                                        return False
                                                
                                                w2v_like = sum(1 for c in row if _is_w2v_tok(c)) >= max(2, int(0.6 * len(row)))
                                                
                                                if w2v_like:
                                                    # Word2vec 결과: token:score 쌍들
                                                    # 헤드 토큰과 유사 토큰으로 관련성 질문 생성
                                                    pairs = []
                                                    for c in row:
                                                        s = str(c).strip()
                                                        if not _is_w2v_tok(s):
                                                            continue
                                                        term, score = s.split(':', 1)
                                                        try:
                                                            pairs.append((term.strip(), float(score)))
                                                        except:
                                                            continue
                                                    
                                                    if len(pairs) >= 2:
                                                        head = pairs[0][0]
                                                        # top positives (next up to 3)
                                                        for term, _sc in pairs[1:4]:
                                                            prompt = f"Are the terms '{head}' and '{term}' semantically related (yes/no)"
                                                            add_to_batch(prompt, 'yes')
                                                            count += 1
                                                        # negatives: pick from tail
                                                        for term, _sc in pairs[-3:]:
                                                            if term == head:
                                                                continue
                                                            prompt = f"Are the terms '{head}' and '{term}' semantically related (yes/no)"
                                                            add_to_batch(prompt, 'no')
                                                            count += 1
                                                elif len(row) == 2:
                                                    add_to_batch(row[0], row[1])
                                                    count += 1
                                                elif len(row) == 3:
                                                    # 지식 그래프 트리플
                                                    subject, relation, obj = row[0], row[1], row[2]
                                                    prompt = f"What is the {relation.replace('_', ' ')} of {subject.replace('_', ' ')}"
                                                    response = obj.replace('_', ' ')
                                                    add_to_batch(prompt, response)
                                                    count += 1
                                                elif len(row) > 2:
                                                    # GoldenStandard 형식: 카테고리 + 동의어들
                                                    category = str(row[0]).strip()
                                                    for term in row[1:]:
                                                        t = str(term).strip()
                                                        if not t:
                                                            continue
                                                        # 카테고리 분류 학습
                                                        prompt = f"What category does '{t}' belong to"
                                                        add_to_batch(prompt, category)
                                                        count += 1
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing CSV row (no header) in {fname}: {e}")
                                                    error_count += 1
                                                continue
                            
                            # TSV 파일 처리
                            elif file_path.endswith('.tsv'):
                                with open_with_fallback(file_path) as f:
                                    # 첫 줄 확인 (헤더가 있는지 체크)
                                    first_line = f.readline().strip()
                                    f.seek(0)
                                    
                                    # 헤더가 있는 경우 (prompt, response 등의 컬럼명 포함)
                                    if '\t' in first_line and any(k in first_line.lower() for k in ['prompt', 'user', 'input', 'question', 'response', 'assistant']):
                                        reader = csv.DictReader(f, delimiter='\t')
                                        for row in tqdm(reader, desc=f"TSV {fname}", leave=False):
                                            try:
                                                prompt = row.get('prompt') or row.get('user') or row.get('input') or row.get('question')
                                                response = row.get('response') or row.get('assistant') or row.get('output') or row.get('answer')
                                                if prompt and response:
                                                    add_to_batch(prompt, response)
                                                    count += 1
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing TSV row in {fname}: {e}")
                                                    error_count += 1
                                                continue
                                    else:
                                        # 헤더 없는 지식 그래프 트리플 (subject relation object)
                                        for line in f:
                                            try:
                                                parts = line.strip().split('\t')
                                                if len(parts) >= 2:
                                                    # 2컬럼: prompt response
                                                    if len(parts) == 2:
                                                        add_to_batch(parts[0], parts[1])
                                                        count += 1
                                                    # 3컬럼: subject relation object -> 자연어 QA로 변환
                                                    elif len(parts) == 3:
                                                        subject, relation, obj = parts[0], parts[1], parts[2]
                                                        
                                                        # 관계명을 자연어로 변환
                                                        rel_clean = relation.replace('_', ' ').replace('.', ' ')
                                                        subj_clean = subject.replace('_', ' ').replace('.n.', ' (noun)').replace('.v.', ' (verb)')
                                                        obj_clean = obj.replace('_', ' ').replace('.n.', ' (noun)').replace('.v.', ' (verb)')
                                                        
                                                        # 다양한 질문 형식 생성
                                                        import random
                                                        templates = [
                                                            (f"What is the {rel_clean} of {subj_clean}", obj_clean),
                                                            (f"Tell me about the {rel_clean} relationship of {subj_clean}.", obj_clean),
                                                            (f"{subj_clean} has what {rel_clean}", obj_clean),
                                                            (f"Explain the {rel_clean} for {subj_clean}.", f"The {rel_clean} is {obj_clean}"),
                                                            (f"Define {subj_clean} in terms of {rel_clean}.", obj_clean),
                                                        ]
                                                        prompt, response = random.choice(templates)
                                                        add_to_batch(prompt, response)
                                                        count += 1
                                                    # MIND 뉴스 형식 (여러 컬럼)
                                                    elif len(parts) > 5:
                                                        # news.tsv: ID, category, subcategory, title, abstract, url, entities, ...
                                                        try:
                                                            title = parts[3] if len(parts) > 3 else ''
                                                            abstract = parts[4] if len(parts) > 4 else ''
                                                            category = parts[1] if len(parts) > 1 else ''
                                                            
                                                            if title and abstract:
                                                                # 제목 -> 요약
                                                                add_to_batch(
                                                                    f"Summarize this article: {title}",
                                                                    abstract
                                                                )
                                                                count += 1
                                                                
                                                                # 카테고리 분류
                                                                if category:
                                                                    add_to_batch(
                                                                        f"What category is this article: {title}",
                                                                        category
                                                                    )
                                                                    count += 1
                                                        except:
                                                            pass
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing TSV line (no header) in {fname}: {e}")
                                                    error_count += 1
                                                continue
                            
                            # TXT 파일 처리
                            elif file_path.endswith('.txt'):
                                # Legal Case Corpus 감지: preprocessed_cases/raw_cases 구조
                                norm_path = file_path.replace('\\', '/')
                                is_legal_case = ('preprocessed_cases' in norm_path or 'raw_cases' in norm_path) and legal_map
                                
                                if is_legal_case:
                                    # Legal Case: 디렉토리명이 카테고리 ID
                                    try:
                                        parts = norm_path.split('/')
                                        # preprocessed_cases/.../<category_id>/case*.txt 구조
                                        cat_id = None
                                        for i, part in enumerate(parts):
                                            if 'preprocessed_cases' in part or 'raw_cases' in part:
                                                if i + 1 < len(parts):
                                                    cat_id = parts[i + 1]
                                                break
                                        
                                        if cat_id and cat_id in legal_map:
                                            category = legal_map[cat_id]
                                            with open_with_fallback(file_path) as f:
                                                text = f.read().strip()
                                                if text:
                                                    # 법률 케이스 분류 학습
                                                    # 텍스트 앞부분만 사용 (너무 길면 truncate)
                                                    text_preview = text[:500] if len(text) > 500 else text
                                                    add_to_batch(
                                                        f"Classify this legal case: {text_preview}",
                                                        category
                                                    )
                                                    count += 1
                                    except Exception as e:
                                        self._log(f"Error parsing Legal Case TXT {fname}: {e}")
                                        pass
                                else:
                                    # 일반 TXT: 라인별 prompt/response 쌍
                                    with open_with_fallback(file_path) as f:
                                        lines = [l.strip() for l in f.readlines() if l.strip()]
                                        for i in tqdm(range(0, len(lines) - 1, 2), desc=f"TXT {fname}", leave=False):
                                            try:
                                                add_to_batch(lines[i], lines[i+1])
                                                count += 1
                                            except Exception as e:
                                                if error_count < 3:
                                                    self._log(f"Error parsing TXT pair in {fname}: {e}")
                                                    error_count += 1
                                                continue
                            
                            if count > 0:
                                self._log(f'  {fname}: {count} examples')
                        except Exception as e:
                            self._log(f'  ✗ Error in {os.path.basename(file_path)}: {e}')

                    import random
                    NUM_EPOCHS = 5
                    
                    for epoch in range(NUM_EPOCHS):
                        self._log(f"=== Starting Epoch {epoch+1}/{NUM_EPOCHS} ===")
                        random.shuffle(files)
                        epoch_losses.clear()
                        
                        for file_path in tqdm(files, desc=f"Epoch {epoch+1} Files", unit="file"):
                            process_one_file(file_path)
                        
                        # Process remaining batch at end of epoch
                        if batch_buffer:
                            try:
                                ret = adapter.train_batch(batch_buffer)
                                if isinstance(ret, tuple):
                                    c, l = ret
                                    total_count += c
                                    if l > 0: epoch_losses.append(l)
                                    self._log(f"[EPOCH FINAL] trained {c}, loss={l:.4f}")
                                elif isinstance(ret, int):
                                    total_count += ret
                                    self._log(f"[EPOCH FINAL] trained {ret}")
                                batch_buffer.clear()
                            except Exception as e:
                                self._log(f"Error in final batch: {e}")
                        
                        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                        self._log(f"=== Epoch {epoch+1} Completed. Avg Loss: {avg_epoch_loss:.4f} ===")

                    self._log(f'✓ Total trained: {total_count} examples from {len(files)} files')
                except Exception as e:
                    self._log(f'✗ Folder training error: {e}')
            
            _threading.Thread(target=_train_thread, daemon=True).start()
        
        def _train_llm_dpo(self):
            """DPO(Direct Preference Optimization) 학습"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                self._log('✗ LLM adapter not available - check console for errors')
                return
            
            def _train_thread():
                try:
                    self._log('Training LLM with DPO...')
                    result = adapter.train_dpo_from_dir(epochs=2, beta=0.1)
                    self._log(f"✓ DPO: loss={result.get('avg_loss', 0):.4f}, samples={result.get('num_samples', 0)}")
                except Exception as e:
                    self._log(f'✗ DPO training error: {e}')
            
            _threading.Thread(target=_train_thread, daemon=True).start()
        
        def _clear_chat(self):
            """채팅 히스토리 지우기"""
            self.chat.delete('1.0', _tk.END)
            if hasattr(self.core, '_chat_history'):
                self.core._chat_history.clear()
            self._log('Chat history cleared')
        
        def _train_llm(self):
            """LLM 대화 데이터로 학습 (레거시)"""
            self._train_llm_chat()
        
        def _clear_knn_memory(self, parent_win):
            """kNN 메모리 클리어"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                return
            
            try:
                knn = getattr(adapter, '_knn', None)
                if knn:
                    knn._keys.clear()
                    knn._vals.clear()
                    knn._access_counts.clear()
                    knn._total_queries = 0
                    self._log('✓ kNN memory cleared')
                    parent_win.destroy()
                else:
                    self._log('✗ kNN not available')
            except Exception as e:
                self._log(f'✗ Clear kNN error: {e}')
        
        def _reset_knn_tau(self, parent_win, new_tau):
            """kNN"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                return
            
            try:
                knn = getattr(adapter, '_knn', None)
                if knn:
                    knn.tau = float(new_tau)
                    self._log(f'✓ kNN tau reset to {new_tau}')
                    parent_win.destroy()
                else:
                    self._log('✗ kNN not available')
            except Exception as e:
                self._log(f'✗ Reset tau error: {e}')
        
        def _save_llm(self):
            """LLM 모델 저장"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                self._log('✗ LLM adapter not available - check console for errors')
                return
            
            def _save_thread():
                try:
                    save_path = os.path.join(self.core.outdir, 'llm_checkpoint.pt')
                    if hasattr(adapter, 'save_model'):
                        if getattr(adapter, '_hf_circuit_open', False):
                            self.after(0, lambda: self._log('⚠ Skipped LLM save: HF circuit breaker is open'))
                        else:
                            adapter.save_model(save_path)
                            self.after(0, lambda: self._log(f'✓ Model saved: {save_path}'))
                    elif hasattr(adapter, 'conv_policy'):
                        import torch
                        torch.save(adapter.conv_policy.model.state_dict(), save_path)
                        self.after(0, lambda: self._log(f'✓ Model saved (legacy): {save_path}'))
                    else:
                        self.after(0, lambda: self._log('✓ Model auto-saved (numpy policy)'))
                except Exception as e:
                    self.after(0, lambda: self._log(f'✗ Save error: {e}'))
            
            _threading.Thread(target=_save_thread, daemon=True).start()
        
        def _toggle_autonomy(self):
            """Autonomy loop 토글"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                self._log('✗ LLM adapter not available')
                return
            
            try:
                if hasattr(adapter, '_auto_running') and adapter._auto_running:
                    adapter.stop_autonomy_loop()
                    self.autonomy_btn.config(text='Autonomy Off', bg='#00aa88')
                    self._log('Autonomy loop stopped')
                else:
                    adapter.start_autonomy_loop()
                    self.autonomy_btn.config(text='Autonomy On', bg='#00ff88')
                    self._log('Autonomy loop started')
            except Exception as e:
                self._log(f'✗ Autonomy toggle error: {e}')
        
        def _knn_settings(self):
            """kNN 설정 창 열기"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                self._log('✗ LLM adapter not available')
                return
            
            # 설정 창 생성
            settings_win = _tk.Toplevel(self)
            settings_win.title('kNN-LM Settings')
            settings_win.geometry('400x300')
            settings_win.configure(bg='#0a0a0a')
            
            _tk.Label(settings_win, text='kNN-LM Configuration', bg='#0a0a0a', fg='#e0e0e0',
                     font=('Consolas', 12, 'bold')).pack(pady=10)
            
            # 현재 설정 표시
            try:
                knn = getattr(adapter, '_knn', None)
                if knn:
                    info_text = f"""Current Settings:
Tau: {knn.tau:.4f}
Max Items: {knn.max_items:,}
Key Dim: {knn.key_dim}
Total Queries: {knn._total_queries:,}
Items Stored: {len(knn._keys)}"""
                else:
                    info_text = "kNN not initialized"
                
                _tk.Label(settings_win, text=info_text, bg='#0a0a0a', fg='#707070',
                         font=('Consolas', 9), justify=_tk.LEFT).pack(pady=10)
            except Exception as e:
                _tk.Label(settings_win, text=f"Error getting settings: {e}", bg='#0a0a0a', fg='#ff4444',
                         font=('Consolas', 9)).pack(pady=10)
            
            # 설정 변경 버튼들
            btn_frame = _tk.Frame(settings_win, bg='#0a0a0a')
            btn_frame.pack(pady=20)
            
            _tk.Button(btn_frame, text='Clear kNN Memory', command=lambda: self._clear_knn_memory(settings_win),
                      bg='#ff4444', fg='#fff', font=('Consolas', 9, 'bold'),
                      bd=0, cursor='hand2').pack(side=_tk.LEFT, padx=5)
            
            _tk.Button(btn_frame, text='Reset Tau to 0.07', command=lambda: self._reset_knn_tau(settings_win, 0.07),
                      bg='#4444ff', fg='#fff', font=('Consolas', 9, 'bold'),
                      bd=0, cursor='hand2').pack(side=_tk.LEFT, padx=5)
            
            _tk.Button(btn_frame, text='Close', command=settings_win.destroy,
                      bg='#666666', fg='#fff', font=('Consolas', 9, 'bold'),
                      bd=0, cursor='hand2').pack(side=_tk.LEFT, padx=5)
        
        def _toggle_m3_integration(self):
            """M3 integration 토글"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                self._log('✗ LLM adapter not available')
                return
            
            try:
                if hasattr(adapter.model, 'use_m3_integration') and adapter.model.use_m3_integration:
                    adapter.model.use_m3_integration = False
                    self.m3_integration_btn.config(text='M3 Int Off', bg='#888800')
                    self._log('M3 integration disabled')
                else:
                    if hasattr(adapter.model, 'enable_m3_integration'):
                        adapter.model.enable_m3_integration()
                        self.m3_integration_btn.config(text='M3 Int On', bg='#ffff00')
                        self._log('M3 integration enabled')
                    else:
                        self._log('✗ M3 integration not supported')
            except Exception as e:
                self._log(f'✗ M3 integration toggle error: {e}')
        
        def _trigger_sleep(self):
            """Deep Sleep: Consolidate Traces -> Weights"""
            adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
            if adapter is None:
                self._log('✗ LLM adapter not available')
                return
            
            # Check for PlasticBrainPolicy (M3-Binary Brain)
            if hasattr(adapter, 'model') and hasattr(adapter.model, 'sleep'):
                try:
                    self._log('Initiating Deep Sleep (Memory Consolidation)...')
                    
                    def _sleep_task():
                        try:
                            adapter.model.sleep()
                            self.after(0, lambda: self._log('✓ Deep Sleep Complete: Memories Consolidated.'))
                        except Exception as e:
                            self.after(0, lambda: self._log(f'✗ Sleep Error: {e}'))
                    
                    # Run in thread to allow UI updates
                    import threading
                    threading.Thread(target=_sleep_task, daemon=True).start()
                    
                except Exception as e:
                    self._log(f'✗ Failed to start sleep cycle: {e}')
            else:
                self._log('✗ Deep Sleep requires PlasticBrainPolicy (1.58-bit)')

        def _update_button_states(self):
            """버튼 상태 업데이트"""
            try:
                adapter = getattr(self.core, 'llm_adapter', None) or getattr(self.core, 'llm', None)
                if adapter is None:
                    return
                
                # Autonomy 상태
                if hasattr(adapter, '_auto_running') and adapter._auto_running:
                    self.autonomy_btn.config(text='Autonomy On', bg='#00ff88')
                else:
                    self.autonomy_btn.config(text='Autonomy Off', bg='#00aa88')
                
                # M3 Integration 상태
                if hasattr(adapter.model, 'use_m3_integration') and adapter.model.use_m3_integration:
                    self.m3_integration_btn.config(text='M3 Int On', bg='#ffff00')
                else:
                    self.m3_integration_btn.config(text='M3 Int Off', bg='#888800')
                    
            except Exception:
                pass
        
        def _on_close(self):
            if self.running:
                self._stop()
                _time.sleep(0.5)
            self.destroy()
else:
    print("Running without GUI")
# ============================================================================

if __name__ == '__main__':
    main()

__all__ = [
    'Message',
    'SpanMeta',
    'MessageBus',
    'HebbianTrace',
    'RunningBaseline',
    'MetaController',
    'ConsciousnessBus',
    'M3ConsciousnessCore',
    'build_parser',
    'main',
]
