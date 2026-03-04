"""
LLM Adapter with proper tokenizer and learnable policy.

Features:
- Byte-level tokenizer (no UNK issues)
- Optional Torch GRU policy with supervised training (prompt -> response)
- Lightweight numpy fallback policy with simple REINFORCE
- Batched JSONL logging to reduce I/O
- Small LRU cache for embeddings via core.feature_bank
- M3 Full Integration: FeatureBank panels Transformer memory, Phi/Qualia-driven sampling

Design Principles:
- NO MAGIC NUMBERS: All parameters inferred from core/tokenizer/model
- Dynamic validation: Dimensions verified at runtime
- Graceful degradation: Works without M3 core (fallback to standalone mode)

Usage:
  from llm_adapter import attach_llm_to_core
  attach_llm_to_core(core)
"""
from __future__ import annotations
from m3.attr_contract import attr_del, attr_get_optional, attr_get_required, attr_has, attr_set, guard_context, guard_eval, guard_step
import os
import sys

# If this file is executed directly (python llm_adapter/llm_core.py), ensure
# project root is on sys.path so absolute imports like `import llm_adapter.*`
# work. When run as a module (`python -m llm_adapter.llm_core`) this is not needed.
if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
import os
import json
import numpy as np
import logging
import re
import hashlib
import copy
from difflib import SequenceMatcher
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import threading
import time
from collections import deque
import types
import torch
from m3.device import resolve_torch_device_string
from m3.io.jsonl_writer import append_jsonl

# ---------------------------------------------------------------------------
# Silence noisy external libraries (HuggingFace Hub / Transformers / httpx)
# ---------------------------------------------------------------------------
# These libraries legitimately make many HEAD/GET requests (metadata, redirects,
# cache checks). The INFO logs are not actionable in normal runs and clutter the
# interactive REPL, so we suppress them unconditionally.
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

for _name in (
    'httpx', 'httpcore', 'hpack', 'h2', 'urllib3',
    'huggingface_hub', 'transformers', 'accelerate', 'bitsandbytes'
):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Local adapter chatter (keep console clean)
logging.getLogger('llm_adapter.tokenization').setLevel(logging.WARNING)

def _disable_progress_bars_if_supported(logging_module: Any) -> None:
    for name in ("disable_progress_bars", "disable_progress_bar"):
        fn = getattr(logging_module, name, None)
        if callable(fn):
            fn()
            return


def _configure_external_library_logging(
    transformers_logging: Any | None = None,
    hf_hub_logging: Any | None = None,
) -> None:
    if transformers_logging is None:
        from transformers.utils import logging as transformers_logging
    if hf_hub_logging is None:
        from huggingface_hub.utils import logging as hf_hub_logging

    set_t = getattr(transformers_logging, "set_verbosity_error", None)
    if callable(set_t):
        set_t()
    _disable_progress_bars_if_supported(transformers_logging)

    set_h = getattr(hf_hub_logging, "set_verbosity_error", None)
    if callable(set_h):
        set_h()
    _disable_progress_bars_if_supported(hf_hub_logging)


_configure_external_library_logging()

_CONTROL_REASON_ENUM: set[str] = {
    "hf_generate_ok",
    "hf_circuit_open",
    "hf_filtered_or_fallback",
    "hf_unavailable_or_fallback",
    "hf_runtime_failure",
    "hf_runtime_cooldown",
    "hf_cuda_fault",
    "missing_control_health_window",
    "none_subscriptable",
    "quality_rejected",
    "semantic_rejected",
    "other",
}


def _normalize_control_reason(reason: str) -> str:
    raw = str(reason or "").strip().lower()
    if not raw:
        return "other"
    raw = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    if raw in _CONTROL_REASON_ENUM:
        return raw
    if raw.startswith("hf_exception"):
        return "hf_runtime_failure"
    if raw.startswith("hf_filtered"):
        return "hf_filtered_or_fallback"
    if raw.startswith("hf_unavailable"):
        return "hf_unavailable_or_fallback"
    if raw.startswith("hf_circuit"):
        return "hf_circuit_open"
    if raw.startswith("none") and "subscriptable" in raw:
        return "none_subscriptable"
    if raw.startswith("missing_control_health_window"):
        return "missing_control_health_window"
    return "other"


class HFRuntimeFailure(RuntimeError):
    def __init__(
        self,
        reason_code: str,
        phase: str,
        model_output_shape: str = "unknown",
        has_logits: bool = False,
        message: str = "",
    ) -> None:
        text = str(message or reason_code or "hf_runtime_failure")
        super().__init__(text)
        self.reason_code = _normalize_control_reason(str(reason_code or "hf_runtime_failure"))
        self.phase = str(phase or "unknown")
        self.model_output_shape = str(model_output_shape or "unknown")
        self.has_logits = bool(has_logits)

try:
    from .config import (
        AutonomyRLConfig,
        BridgeAdaptConfig,
        DPOAutoCollectConfig,
        EarlyStopConfig,
        EpisodicANNConfig,
        KNNIndexConfig,
        M3AdaptiveSamplerConfig,
        M3AwareDecoderLayerConfig,
        M3EpisodicMemoryConfig,
        M3LLMConfig,
        M3StateCacheConfig,
        M3StateEncoderConfig,
        NeuroModulatorConfig,
        StabilityConfig,
        TokenizerAutoVocabConfig,
        TokenizerConfig,
        TorchPolicyConfig,
        get_global_config,
    )
    from .memory import ConditionalKNNIndex, KNNItem, M3EpisodicMemoryRetriever
    from .tokenization import M3Tokenizer, AutoTokenizer
    from .semantic_scorer import SemanticScorer
    from .meaning_pipeline import format_plan_fallback_prompt
    from .control_plane import (
        M3ControlDecision,
        build_control_decision,
        log_control_decision,
        resolve_control_decision_log_path,
    )
except Exception:
    # Support running file directly (script) where package-relative imports fail
    from llm_adapter.config import (
        AutonomyRLConfig,
        BridgeAdaptConfig,
        DPOAutoCollectConfig,
        EarlyStopConfig,
        EpisodicANNConfig,
        KNNIndexConfig,
        M3AdaptiveSamplerConfig,
        M3AwareDecoderLayerConfig,
        M3EpisodicMemoryConfig,
        M3LLMConfig,
        M3StateCacheConfig,
        M3StateEncoderConfig,
        NeuroModulatorConfig,
        StabilityConfig,
        TokenizerAutoVocabConfig,
        TokenizerConfig,
        TorchPolicyConfig,
        get_global_config,
    )
    from llm_adapter.memory import ConditionalKNNIndex, KNNItem, M3EpisodicMemoryRetriever
    from llm_adapter.tokenization import M3Tokenizer, AutoTokenizer
    from llm_adapter.semantic_scorer import SemanticScorer
    from llm_adapter.meaning_pipeline import format_plan_fallback_prompt
    from llm_adapter.control_plane import (
        M3ControlDecision,
        build_control_decision,
        log_control_decision,
        resolve_control_decision_log_path,
    )

try:
    from .m3_control_bridge import (
        M3ControlBridge,
        LayerGateRuntime,
        GenerationQualityGate,
        find_decoder_layers,
        NeuroModulator,
        NeuroModulatorRuntime,
    )
except Exception:
    try:
        from llm_adapter.m3_control_bridge import (  # type: ignore
            M3ControlBridge,
            LayerGateRuntime,
            GenerationQualityGate,
            find_decoder_layers,
            NeuroModulator,
            NeuroModulatorRuntime,
        )
    except Exception:
        M3ControlBridge = None  # type: ignore
        LayerGateRuntime = None  # type: ignore
        GenerationQualityGate = None  # type: ignore
        NeuroModulator = None  # type: ignore
        NeuroModulatorRuntime = None  # type: ignore
        def find_decoder_layers(_model):  # type: ignore
            return []

try:
    from .remote import get_local_thinking
except Exception:
    try:
        from llm_adapter.remote import get_local_thinking  # type: ignore
    except Exception:
        def get_local_thinking(*args, **kwargs):  # type: ignore
            return ""


class M3StateEncoder:
    """
    State encoder for the M3 model.
    
    NO MAGIC NUMBERS:
    - panel_dim: core.feature_bank.embed_dim
    - hidden_dim: core.feature_bank.hidden_dim
    - n_layers/nhead: hidden_dim / nhead (hidden_dim % nhead == 0)
    """
    def __init__(self, torch_module, hidden_dim: int, device, config: Optional[M3StateEncoderConfig] = None):
        """
        Args:
            torch_module: PyTorch module (nn)
            hidden_dim: Transformer hidden dimension (core.feature_bank.hidden_dim)
            device: torch.device
            config: Optional configuration (uses global config if None)
        """
        self.torch = torch_module
        self.nn = self.torch.nn
        self.device = device
        self.hidden_dim = hidden_dim
        self.config = config or get_global_config().state_encoder

        # Panel projection layers (lazy init - forward panel_dim -> hidden_dim)
        self.panel_projections = None
        self._initialized_panel_dim = None
        self._initialized_num_panels = None
        
        # Affect and Drive projections (lazy init)
        self.affect_projection = None
        self.drive_projection = None

        # Fusion layers: hidden_dim / nhead (hidden_dim % nhead == 0)
        self.nhead = self._infer_nhead(hidden_dim)
        self.n_fusion_layers = self._infer_fusion_layers(hidden_dim)

        # Positional embeddings (lazy init - forward)
        self.position_embeddings = None
    
    def _infer_nhead(self, hidden_dim: int) -> int:
        """
        hidden_dim / nhead: attention head dimension.
        
        : hidden_dim / nhead (config.nhead_candidates)
        """
        for nhead in self.config.nhead_candidates:
            if hidden_dim % nhead == 0:
                return nhead
        raise ValueError(f"Cannot infer nhead for hidden_dim={hidden_dim} with candidates={self.config.nhead_candidates}")
    
    def _infer_fusion_layers(self, hidden_dim: int) -> int:
        """
         hidden_dim / nhead: fusion layer input dimension.

        : config.fusion_layers_small_threshold  config.fusion_layers_medium_threshold
        """
        if hidden_dim <= self.config.hidden_dim_small_threshold:
            return self.config.fusion_layers_small
        elif hidden_dim <= self.config.hidden_dim_medium_threshold:
            return self.config.fusion_layers_medium
        else:
            return self.config.fusion_layers_large
    
    def _infer_dropout(self, hidden_dim: int) -> float:
        """
         hidden_dim / nhead: dropout dimension.

        : hidden_dim / nhead (config.dropout_candidates)
        """
        if hidden_dim <= self.config.hidden_dim_small_threshold:
            return self.config.dropout_small
        elif hidden_dim <= self.config.hidden_dim_medium_threshold:
            return self.config.dropout_medium
        else:
            return self.config.dropout_large

    def __call__(self, panels: List[np.ndarray], affect_state: Optional[np.ndarray] = None, drive_state: Optional[np.ndarray] = None) -> 'torch.Tensor':
        """Callable compatibility: delegate function-style calls to forward()."""
        return self.forward(panels, affect_state=affect_state, drive_state=drive_state)
    
    def _lazy_init_projections(self, panel_dim: int, num_panels: int, affect_dim: int = 0, drive_dim: int = 0):
        """
        Lazy initialize panel dimension -> projection layers
        """
        if (
            self.panel_projections is not None
            and self._initialized_panel_dim == panel_dim
            and self._initialized_num_panels == num_panels
        ):
            # Already initialized with same dims
            return
        
        # Panel-specific projections
        self.panel_projections = self.nn.ModuleList([
            self.nn.Linear(panel_dim, self.hidden_dim)
            for _ in range(num_panels)
        ]).to(self.device)
        
        # Affect projection
        if affect_dim > 0:
            self.affect_projection = self.nn.Linear(affect_dim, self.hidden_dim).to(self.device)
            
        # Drive projection
        if drive_dim > 0:
            self.drive_projection = self.nn.Linear(drive_dim, self.hidden_dim).to(self.device)
        
        # Positional embeddings
        # Total tokens = num_panels + (1 if affect) + (1 if drive)
        total_tokens = num_panels + (1 if affect_dim > 0 else 0) + (1 if drive_dim > 0 else 0)
        
        self.position_embeddings = self.nn.Parameter(
            self.torch.randn(total_tokens, self.hidden_dim) * (1.0 / np.sqrt(self.hidden_dim))
        ).to(self.device)
        
        # Fusion layers
        dropout = self._infer_dropout(self.hidden_dim)
        self.fusion_layers = self.nn.ModuleList([
            self.nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.nhead,
                dim_feedforward=self.hidden_dim * 4,  # Standard 4x expansion
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(self.n_fusion_layers)
        ]).to(self.device)
        
        self._initialized_panel_dim = panel_dim
        self._initialized_num_panels = num_panels
    
    def forward(self, panels: List[np.ndarray], affect_state: Optional[np.ndarray] = None, drive_state: Optional[np.ndarray] = None) -> 'torch.Tensor':
        """
        panels: List of panel vectors from core.feature_bank.panels(core)
        Args:
            panels: List of panel vectors from core.feature_bank.panels(core)
                   Each panel: (panel_dim,) numpy array
            affect_state: Optional (affect_dim,) numpy array
            drive_state: Optional (drive_dim,) numpy array
        
        Returns:
            memory_tokens: (num_tokens, hidden_dim) tensor
        """
        if panels is None:
            raise ValueError("panels is None - expected list/array/tensor of panel vectors")

        panels_list: List[np.ndarray] = []
        with guard_context(ctx='llm_adapter/llm_core.py:341', catch_base=False) as __m3_guard_325_8:
            if isinstance(panels, np.ndarray):
                if panels.ndim == 1:
                    panels_list = [panels.astype(np.float32)]
                elif panels.ndim == 2:
                    panels_list = [panels[i].astype(np.float32) for i in range(panels.shape[0])]
            elif self.torch.is_tensor(panels):
                p = panels.detach().cpu().numpy()
                if p.ndim == 1:
                    panels_list = [p.astype(np.float32)]
                elif p.ndim == 2:
                    panels_list = [p[i].astype(np.float32) for i in range(p.shape[0])]
            elif isinstance(panels, list):
                panels_list = [np.asarray(p, dtype=np.float32) for p in panels]
            else:
                panels_list = [np.asarray(p, dtype=np.float32) for p in list(panels)]

        if __m3_guard_325_8.error is not None:
            e = __m3_guard_325_8.error
            raise ValueError(f"Unsupported panels type for M3StateEncoder: {type(panels)}") from e

        if not panels_list:
            raise ValueError("panels list is empty - core.feature_bank.panels(core) returned no panels")
        
        # Infer dimensions from actual data
        num_panels = len(panels_list)
        panel_dim = panels_list[0].shape[0]
        affect_dim = affect_state.shape[0] if affect_state is not None else 0
        drive_dim = drive_state.shape[0] if drive_state is not None else 0
        
        # Validate all panels have same dimension
        for i, panel in enumerate(panels_list):
            if panel.shape[0] != panel_dim:
                raise ValueError(
                    f"Panel {i} has dimension {panel.shape[0]}, expected {panel_dim}. "
                    f"All panels must have consistent dimensions."
                )
        
        # Lazy initialization
        self._lazy_init_projections(panel_dim, num_panels, affect_dim, drive_dim)
        
        # Project each panel
        tokens = []
        for i, panel_vec in enumerate(panels_list):
            panel_tensor = self.torch.from_numpy(panel_vec).float().to(self.device)
            token = self.panel_projections[i](panel_tensor)
            tokens.append(token)
            
        # Project affect
        if affect_state is not None and self.affect_projection is not None:
            affect_tensor = self.torch.from_numpy(affect_state).float().to(self.device)
            tokens.append(self.affect_projection(affect_tensor))
            
        # Project drives
        if drive_state is not None and self.drive_projection is not None:
            drive_tensor = self.torch.from_numpy(drive_state).float().to(self.device)
            tokens.append(self.drive_projection(drive_tensor))
        
        # Stack and add positional embeddings
        memory = self.torch.stack(tokens)  # (num_tokens, hidden_dim)
        
        # Ensure position embeddings match current token count (handle dynamic changes if any)
        if memory.size(0) != self.position_embeddings.size(0):
             # Re-init if size mismatch (e.g. first run without affect, second with)
             self.position_embeddings = self.nn.Parameter(
                self.torch.randn(memory.size(0), self.hidden_dim) * (1.0 / np.sqrt(self.hidden_dim))
            ).to(self.device)
            
        memory = memory + self.position_embeddings
        
        # Apply fusion layers
        memory_batch = memory.unsqueeze(0)  # (1, num_tokens, hidden_dim) - batch_first=True
        for layer in self.fusion_layers:
            memory_batch = layer(memory_batch)
        
        return memory_batch.squeeze(0)  # (num_tokens, hidden_dim)


class M3StateCache:
    """
    M3StateCache: Caches state information for the M3 model.
    - cache_size: M3_STATE_CACHE_SIZE (phi_history multiplier)
    - phi_trend_window: cache_size / divisor (config.phi_trend_window_divisor)
    """
    def __init__(self, config: Optional[M3StateCacheConfig] = None):
        self._panels_cache = deque()
        self._qualia_cache = deque()
        self._phi_cache = deque()
        self._cache_size = None  # Lazy init
        self.config = config or get_global_config().state_cache
    
    def _infer_cache_size(self, core) -> int:
        """
        core: M3Core instance.

        : phi_calculator.phi_history multiplier (config.phi_history_multiplier)
        """
        if attr_has(core, 'phi_calculator') and attr_has(core.phi_calculator, 'phi_history'):
            phi_hist_len = len(core.phi_calculator.phi_history)
            cache_size = max(
                self.config.cache_size_min,
                min(self.config.cache_size_max, phi_hist_len * self.config.phi_history_multiplier)
            )
        else:
            cache_size = self.config.cache_size_default  # Fallback: reasonable default

        # Infer from environment variable
        env_size = os.getenv('M3_STATE_CACHE_SIZE')
        if env_size:
            try:
                cache_size = max(
                    self.config.cache_size_min,
                    min(self.config.cache_size_max, int(env_size))
                )
            except ValueError:
                pass
        
        return cache_size
    
    def _infer_phi_trend_window(self, core) -> int:
        """
        Phi trend window size inference.

        : cache_size / divisor (config.phi_trend_window_divisor)
        """
        cache_size = self._cache_size or self._infer_cache_size(core)
        window = cache_size // self.config.trend_window_divisor
        return max(self.config.trend_window_min, min(self.config.trend_window_max, window))
    
    def update(self, core):
        """Update caches from M3 core state."""
        # Lazy init cache size
        if self._cache_size is None:
            self._cache_size = self._infer_cache_size(core)
            self._panels_cache = deque(maxlen=self._cache_size)
            self._qualia_cache = deque(maxlen=self._cache_size)
            self._phi_cache = deque(maxlen=self._cache_size)
        
        # Store panels
        with guard_context(ctx='llm_adapter/llm_core.py:465', catch_base=False) as __m3_guard_462_8:
            panels = core.feature_bank.panels(core)
            self._panels_cache.append(panels)

        if __m3_guard_462_8.error is not None:
            e = __m3_guard_462_8.error
            self._panels_cache.append(None)
        
        # Store qualia
        with guard_context(ctx='llm_adapter/llm_core.py:478', catch_base=False) as __m3_guard_470_8:
            self._qualia_cache.append({
                'arousal': core.qualia.arousal,
                'valence': core.qualia.valence,
                'entropy': core.qualia.entropy,
                'engagement': core.qualia.engagement,
                'frustration': core.qualia.frustration
            })

        if __m3_guard_470_8.error is not None:
            self._qualia_cache.append(None)
        
        # Store phi
        with guard_context(ctx='llm_adapter/llm_core.py:485', catch_base=False) as __m3_guard_482_8:
            phi = core.phi_calculator.phi_history[-1] if core.phi_calculator.phi_history else 0.0
            self._phi_cache.append(phi)

        if __m3_guard_482_8.error is not None:
            self._phi_cache.append(0.0)
    
    def get_current_panels(self):
        """Return latest panels snapshot or None."""
        return list(self._panels_cache)[-1] if self._panels_cache else None
    
    def get_phi_trend(self, core) -> str:
        """Return phi trend label: 'increasing', 'decreasing', or 'stable'."""
        window = self._infer_phi_trend_window(core)
        
        if len(self._phi_cache) < window:
            return 'stable'
        
        recent = list(self._phi_cache)[-window:]
        slope = (recent[-1] - recent[0]) / window
        
        # Threshold scaled by window length; phi is assumed in [0, 1]
        threshold = self.config.trend_threshold_base / window
        
        if slope > threshold:
            return 'increasing'
        elif slope < -threshold:
            return 'decreasing'
        return 'stable'


class M3AwareDecoderLayer:
    """
    Transformer decoder block adapted for M3 context.
    - d_model: provided by model
    - nhead: inferred to divide d_model
    - dim_feedforward: d_model * multiplier (from config)
    - dropout: inferred by model size (from config)
    """
    def __init__(self, torch_module, d_model: int, device, config: Optional[M3AwareDecoderLayerConfig] = None):
        """
        Args:
            torch_module: PyTorch module (nn)
            d_model: Model dimension
            device: torch.device
            config: Optional configuration (uses global config if None)
        """
        self.torch = torch_module
        self.nn = self.torch.nn
        self.device = device
        self.d_model = d_model
        self.config = config or get_global_config().decoder_layer
        
        # Infer architecture parameters
        self.nhead = self._infer_nhead(d_model)
        self.dim_feedforward = d_model * self.config.dim_feedforward_multiplier
        self.dropout = self._infer_dropout(d_model)
        
        # Self-attention
        self.self_attn = self.nn.MultiheadAttention(
            d_model, self.nhead, dropout=self.dropout, batch_first=True
        ).to(device)

        # M3 Cross-attention (M3 State Encoder output)
        self.m3_attn = self.nn.MultiheadAttention(
            d_model, self.nhead, dropout=self.dropout, batch_first=True
        ).to(device)
        
        # Feed-forward
        self.ffn = self.nn.Sequential(
            self.nn.Linear(d_model, self.dim_feedforward),
            self.nn.GELU(),
            self.nn.Dropout(self.dropout),
            self.nn.Linear(self.dim_feedforward, d_model)
        ).to(device)
        
        # Layer norms
        self.norm1 = self.nn.LayerNorm(d_model).to(device)
        self.norm2 = self.nn.LayerNorm(d_model).to(device)
        self.norm3 = self.nn.LayerNorm(d_model).to(device)
        
        self.dropout_layer = self.nn.Dropout(self.dropout)
    
    def _infer_nhead(self, d_model: int) -> int:
        """Infer number of attention heads from d_model."""
        for nhead in self.config.nhead_candidates:
            if d_model % nhead == 0:
                return nhead
        raise ValueError(f"Cannot infer nhead for d_model={d_model} with candidates={self.config.nhead_candidates}")
    
    def _infer_dropout(self, d_model: int) -> float:
        """Infer dropout rate from d_model."""
        if d_model <= self.config.d_model_small_threshold:
            return self.config.dropout_small
        elif d_model <= self.config.d_model_medium_threshold:
            return self.config.dropout_medium
        else:
            return self.config.dropout_large
    
    def forward(self, x, m3_memory, mask=None):
        """
        Args:
            x: (B, L, D) Input tensor
            m3_memory: (M, D) M3 memory (M3StateEncoder output)
            mask: Optional attention mask
        
        Returns:
            x: (B, L, D) 
            m3_attn_weights: (B, L, M) M3 attention weights
        """
        # 1. Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # 2. M3 Cross-attention
        if m3_memory is not None:
            # m3_memory: (M, D) -> (B, M, D)
            batch_size = x.size(0)
            m3_memory_expanded = m3_memory.unsqueeze(0).expand(batch_size, -1, -1)  # (B, M, D)
            
            m3_attn_out, m3_attn_weights = self.m3_attn(
                query=x,                    # (B, L, D)
                key=m3_memory_expanded,     # (B, M, D)
                value=m3_memory_expanded    # (B, M, D)
            )
            x = self.norm2(x + self.dropout_layer(m3_attn_out))
        else:
            # No M3 memory: skip cross-attention
            m3_attn_weights = None
        
        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout_layer(ffn_out))
        
        return x, m3_attn_weights


class M3AdaptiveSampler:
    """
    M3-aware adaptive sampling strategy.
    
    
    - temperature: core.qualia/phi/energy_ctrl  (config.temperature)
    - top_k: qualia.entropy * engagement() (config.top_k)
    - m3_param: (config.m3_param)
    """
    def __init__(self, torch_module, device, config: Optional[M3AdaptiveSamplerConfig] = None):
        self.torch = torch_module
        self.nn = self.torch.nn
        self.device = device
        self.config = config or get_global_config().adaptive_sampler
        
        # Temperature predictor (Qualia temperature adjustment)
        # Input: [arousal, valence, entropy, engagement, frustration]
        self.temp_predictor = self.nn.Sequential(
            self.nn.Linear(5, 64),
            self.nn.ReLU(),
            self.nn.Linear(64, 1),
            self.nn.Sigmoid()  # Output: [0, 1]
        ).to(device)

    def _normalize_affect_state(self, affect_state, qualia=None):
        """Normalize affect-like inputs to a stable 5D vector.

        Output order: [arousal, valence, entropy/novelty, engagement, frustration].
        """
        defaults = [
            float(attr_get_optional(qualia, 'arousal', 0.5)) if qualia is not None else 0.5,
            float(attr_get_optional(qualia, 'valence', 0.5)) if qualia is not None else 0.5,
            float(attr_get_optional(qualia, 'entropy', 0.5)) if qualia is not None else 0.5,
            float(attr_get_optional(qualia, 'engagement', 0.5)) if qualia is not None else 0.5,
            float(attr_get_optional(qualia, 'frustration', 0.0)) if qualia is not None else 0.0,
        ]

        if affect_state is None:
            return defaults

        if isinstance(affect_state, dict):
            return [
                float(affect_state.get('arousal', defaults[0])),
                float(affect_state.get('valence', defaults[1])),
                float(affect_state.get('entropy', affect_state.get('novelty', defaults[2]))),
                float(affect_state.get('engagement', defaults[3])),
                float(affect_state.get('frustration', defaults[4])),
            ]

        try:
            vals = [float(v) for v in list(affect_state)]
        except Exception:
            return defaults

        if len(vals) < 5:
            vals = vals + defaults[len(vals):]
        return vals[:5]
    

    def _resolve_decode_entropy(self, core, default: float = 0.5) -> float:
        """Resolve decode-time entropy without overwriting qualia entropy semantics."""
        if core is None:
            return float(default)
        for attr in ('decode_entropy', 'token_entropy'):
            with guard_context(ctx='llm_adapter/llm_core.py:685', catch_base=False) as __m3_guard_682_12:
                if attr_has(core, attr):
                    return float(attr_get_optional(core, attr))

            if __m3_guard_682_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:690', catch_base=False) as __m3_guard_687_8:
            if attr_has(core, 'qualia'):
                return float(attr_get_optional(core.qualia, 'entropy', default))

        if __m3_guard_687_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return float(default)

    def _normalize_phi_for_influence(self, core, phi_value: float) -> float:
        """Normalize phi into [0,1] before applying sampler influence."""
        with guard_context(ctx='llm_adapter/llm_core.py:698', catch_base=False) as __m3_guard_696_8:
            mode = str(attr_get_optional(self.config, 'phi_norm_mode', 'dynamic')).lower()

        if __m3_guard_696_8.error is not None:
            mode = 'dynamic'

        try:
            phi = max(0.0, float(phi_value))
        except Exception:
            return 0.0

        if mode == 'off':
            return float(np.clip(phi, 0.0, 1.0))

        if mode == 'static':
            with guard_context(ctx='llm_adapter/llm_core.py:713', catch_base=False) as __m3_guard_710_12:
                denom = float(attr_get_optional(self.config, 'phi_norm_static_max', 1.0))
                denom = max(float(attr_get_optional(self.config, 'phi_norm_min_denominator', 1e-6)), denom)

            if __m3_guard_710_12.error is not None:
                denom = 1.0
            return float(np.clip(phi / denom, 0.0, 1.0))

        hist = []
        with guard_context(ctx='llm_adapter/llm_core.py:724', catch_base=False) as __m3_guard_718_8:
            if attr_has(core, 'phi_calculator') and attr_get_optional(core.phi_calculator, 'phi_history', None):
                for v in core.phi_calculator.phi_history:
                    fv = float(v)
                    if np.isfinite(fv) and fv >= 0.0:
                        hist.append(fv)

        if __m3_guard_718_8.error is not None:
            hist = []

        with guard_context(ctx='llm_adapter/llm_core.py:729', catch_base=False) as __m3_guard_727_8:
            q = float(attr_get_optional(self.config, 'phi_norm_quantile', 0.9))

        if __m3_guard_727_8.error is not None:
            q = 0.9
        q = float(np.clip(q, 0.01, 0.99))

        if hist:
            with guard_context(ctx='llm_adapter/llm_core.py:736', catch_base=False) as __m3_guard_734_12:
                denom = float(np.quantile(np.asarray(hist, dtype=np.float32), q))

            if __m3_guard_734_12.error is not None:
                denom = 1.0
        else:
            with guard_context(ctx='llm_adapter/llm_core.py:741', catch_base=False) as __m3_guard_739_12:
                denom = max(float(attr_get_optional(self.config, 'phi_norm_min_denominator', 1e-6)), 1.0)

            if __m3_guard_739_12.error is not None:
                denom = 1.0

        with guard_context(ctx='llm_adapter/llm_core.py:746', catch_base=False) as __m3_guard_744_8:
            denom = max(float(attr_get_optional(self.config, 'phi_norm_min_denominator', 1e-6)), denom)

        if __m3_guard_744_8.error is not None:
            denom = max(1e-6, denom)
        return float(np.clip(phi / denom, 0.0, 1.0))

    def _compute_temperature(self, core, base_temp: float) -> float:
        """
        M3-aware temperature adjustment.
        
        Args:
            core: M3ConsciousnessCore (base_temp)
            base_temp: Fallback temperature
        
        Returns:
            Adaptive temperature in [temp_min, temp_max]
        """
        if core is None:
            return base_temp
        
        with guard_context(ctx='llm_adapter/llm_core.py:806', catch_base=False) as __m3_guard_764_8:
            if attr_has(core, 'affect_kernel'):
                affect_state = core.affect_kernel.get_state()
                affect_vec = self._normalize_affect_state(affect_state, attr_get_optional(core, 'qualia', None))
                qualia_vec = self.torch.tensor(affect_vec, dtype=self.torch.float32).to(self.device)
            elif attr_has(core, 'qualia'):
                # Fallback to old qualia
                qualia_vec = self.torch.tensor([
                    attr_get_optional(core.qualia, 'arousal', 0.5),
                    attr_get_optional(core.qualia, 'valence', 0.5),
                    self._resolve_decode_entropy(core, 0.5),
                    attr_get_optional(core.qualia, 'engagement', 0.5),
                    attr_get_optional(core.qualia, 'frustration', 0.0)
                ], dtype=self.torch.float32).to(self.device)
            else:
                return base_temp
            
            with self.torch.no_grad():
                temp_factor = self.temp_predictor(qualia_vec).item()
            
            temp = self.config.temp_min + temp_factor * (self.config.temp_max - self.config.temp_min)

            # 2. Phi adjustment (phi * temp)
            if attr_has(core, 'phi_calculator') and core.phi_calculator.phi_history:
                phi = self._normalize_phi_for_influence(core, core.phi_calculator.phi_history[-1])
                # phi in [0, 1]; higher integration (phi) reduces temperature for more focused sampling
                temp = temp * (1.0 - self.config.phi_influence * phi)

            # 3. Energy adjustment (energy * temp)
            if attr_has(core, 'energy_ctrl'):
                energy_ratio = core.energy_ctrl.cognitive_energy / max(core.energy_ctrl.energy_capacity, 1.0)
                temp = temp * (0.8 + 0.4 * (1.0 - energy_ratio) * self.config.energy_influence)

            # 4. Meta-awareness adjustment (meta * temp)
            if attr_has(core, 'self_model') and attr_has(core.self_model, 'meta_awareness'):
                meta = core.self_model.meta_awareness
                temp = temp * (1.0 - self.config.meta_influence * meta)
            
            # Clamp to bounds
            return max(self.config.temp_min, min(self.config.temp_max, temp))

        if __m3_guard_764_8.error is not None:
            return base_temp
    
    def _compute_top_k(self, core, base_top_k: int) -> int:
        """
        M3-aware top_k adjustment.
        
        Args:
            core: Optional M3ConsciousnessCore
            base_top_k: Fallback top_k
        
        Returns:
            Adaptive top_k
        """
        if core is None:
            return base_top_k
        
        with guard_context(ctx='llm_adapter/llm_core.py:854', catch_base=False) as __m3_guard_824_8:
            if attr_has(core, 'affect_kernel'):
                affect = self._normalize_affect_state(core.affect_kernel.get_state(), attr_get_optional(core, 'qualia', None))
                novelty_or_entropy = affect[2]
                arousal = affect[0]
                exploration = novelty_or_entropy * arousal
            elif attr_has(core, 'qualia'):
                entropy = self._resolve_decode_entropy(core, 0.5)
                engagement = attr_get_optional(core.qualia, 'engagement', 0.5)
                exploration = entropy * engagement
            else:
                exploration = 0.5
            
            # Infer top_k bounds from vocab size (if available)
            # Assumption: top_k should be proportional to exploration level
            # High exploration: top_k = vocab_size * 0.5
            # Low exploration: top_k = vocab_size * 0.01
            # We don't have vocab_size here, so use absolute ranges from config
            
            if exploration > self.config.exploration_high_threshold:
                # High exploration
                return self.config.top_k_high_exploration
            elif exploration > self.config.exploration_medium_threshold:
                # Mid exploration
                return self.config.top_k_medium_exploration
            else:
                # Low exploration
                return self.config.top_k_low_exploration

        if __m3_guard_824_8.error is not None:
            return base_top_k
    
    def sample(self, logits, core=None, base_temp=0.8, base_top_k=50, base_top_p=0.9):
        """
        M3-aware sampling strategy.
        
        Args:
            logits: (B, V)  (V,) logits
            core: Optional M3ConsciousnessCore
            base_temp/base_top_k/base_top_p: Fallback values (core=None)
        
        Returns:
            sampled_ids: (B, 1)  (1,) sampled token IDs
        """
        # Compute adaptive parameters
        temperature = self._compute_temperature(core, base_temp)
        top_k = self._compute_top_k(core, base_top_k)
        top_p = base_top_p  # top_p not adapted for now
        # Standard sampling with adaptive params
        if temperature <= 0:
            return self.torch.argmax(logits, dim=-1, keepdim=True)
        
        logits = logits / (temperature + 1e-8)
        
        # top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = self.torch.topk(logits, k=min(top_k, logits.shape[-1]))
            thr = v[..., -1].unsqueeze(-1)
            logits = self.torch.where(logits < thr, self.torch.full_like(logits, float('-inf')), logits)
        
        # top-p (nucleus) filtering
        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = self.torch.sort(logits, descending=True, dim=-1)
            probs = self.torch.softmax(sorted_logits, dim=-1)
            cum = self.torch.cumsum(probs, dim=-1)
            mask = cum > top_p
            # Keep at least one token
            mask[..., 0] = False
            sorted_logits = self.torch.where(mask, self.torch.full_like(sorted_logits, float('-inf')), sorted_logits)
            # Scatter back
            logits = self.torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
        
        probs = self.torch.softmax(logits, dim=-1)
        return self.torch.multinomial(probs, num_samples=1)


class HFBackend:
    """HuggingFace Transformers backend with full M3 parameter control.

    Unlike Ollama (text-in/text-out), this gives per-token access to:
    - Raw logits  ??token-critic Q-value injection
    - Hidden states ??cross-attention / projection
    - Sampling params ??M3AdaptiveSampler (phi/qualia/energy ??temperature/top_k)

    Singleton: model is loaded once and reused across calls.
    Set  M3_USE_HF=1  and  M3_HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct  to enable.
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_available(cls):
        return os.getenv('M3_USE_HF', '0') == '1'

    def __init__(self):
        self._loaded = False
        self._model = None
        self._tokenizer = None
        self.model_name = os.getenv('M3_HF_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
        self.quantize = os.getenv('M3_HF_QUANTIZE', '4bit')
        self._hf_proj = None
        self._hf_proj_dims = (0, 0)
        self.device = None
        self.hidden_dim = None
        self.vocab_size = None
        self._control_bridge = None
        self._control_bridge_state_dim = 0
        self._control_bridge_layers = []
        self._decode_term_cache = {}
        # NeuroModulator: weight-level M3 consciousness control
        self._neuro_modulator = None
        self._neuro_runtime = None
        self._neuro_mod_opt = None
        self._neuro_mod_state_dim = 0
        control_window = max(8, int(os.getenv("M3_CONTROL_HEALTH_WINDOW", "24")))
        self._control_health_window = deque(maxlen=control_window)
        self._auto_mode_fail_streak = 0
        self._last_runtime_diag = {
            "phase": "init",
            "reason_code": "other",
            "model_output_shape": "unknown",
            "has_logits": False,
        }

    def _ensure_loaded(self):
        if self._loaded:
            return
        import torch as _torch
        try:
            from transformers import AutoModelForCausalLM
            from transformers import AutoTokenizer as HFAutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                'M3_USE_HF=1 requires: pip install transformers accelerate bitsandbytes'
            ) from exc

        logging.getLogger('llm_adapter').info(
            f'[HFBackend] Loading {self.model_name} (quantize={self.quantize}) ...'
        )

        self._tokenizer = HFAutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # transformers 理쒖떊 踰꾩쟾?먯꽌??`torch_dtype`媛 deprecated?닿퀬 `dtype`瑜?沅뚯옣??
        # ?ㅻ쭔 援щ쾭???명솚???꾪빐 from_pretrained ?몄텧?먯꽌 ?꾩슂 ??`torch_dtype`濡??대갚?쒕떎.
        load_kw = {'trust_remote_code': True, 'dtype': _torch.bfloat16}

        if self.quantize == '4bit':
            try:
                from transformers import BitsAndBytesConfig
                load_kw['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=_torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                )
            except ImportError:
                logging.getLogger('llm_adapter').warning(
                    '[HFBackend] bitsandbytes unavailable, falling back to bf16'
                )
        elif self.quantize == '8bit':
            try:
                from transformers import BitsAndBytesConfig
                load_kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                pass

        if 'device_map' not in load_kw:
            load_kw['device_map'] = 'auto'

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kw
            )
        except TypeError as exc:
            msg = str(exc)
            if (
                ('dtype' in msg)
                and ('unexpected' in msg or 'got an unexpected keyword argument' in msg)
            ):
                load_kw_fallback = dict(load_kw)
                load_kw_fallback.pop('dtype', None)
                load_kw_fallback['torch_dtype'] = _torch.bfloat16
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **load_kw_fallback
                )
            else:
                raise
        self._model.eval()

        # CPU??bfloat16 鍮꾩?????float32濡?媛뺤젣
        self.device = next(self._model.parameters()).device
        if self.device.type == 'cpu':
            self._model = self._model.float()

        self.hidden_dim = self._model.config.hidden_size
        self.vocab_size = self._model.config.vocab_size
        self._loaded = True

        logging.getLogger('llm_adapter').info(
            f'[HFBackend] Ready: vocab={self.vocab_size}, '
            f'hidden={self.hidden_dim}, device={self.device}'
        )

    def _bridge_enabled(self) -> bool:
        return (
            self._control_allows('bridge')
            and os.getenv('M3_ENABLE_CONTROL_BRIDGE', '0').lower() in ('1', 'true', 'yes', 'on')
        )

    def _bridge_enabled_safe(self) -> bool:
        with guard_context(ctx='llm_adapter/llm_core.py:1039', catch_base=False) as __m3_guard_1035_8:
            fn = attr_get_optional(self, "_bridge_enabled", None)
            if callable(fn):
                return bool(fn())

        if __m3_guard_1035_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return False

    def _ensure_control_health_tracking(self) -> None:
        if attr_get_optional(self, "_control_health_window", None) is None:
            with guard_context(ctx='llm_adapter/llm_core.py:1046', catch_base=False) as __m3_guard_1044_8:
                window = int(os.getenv("M3_CONTROL_HEALTH_WINDOW", "24"))
                self._control_health_window = deque(maxlen=max(8, window))
            if __m3_guard_1044_8.error is not None:
                self._control_health_window = deque(maxlen=24)
        if attr_get_optional(self, "_auto_mode_fail_streak", None) is None:
            self._auto_mode_fail_streak = 0

    def _note_control_health(self, success: bool, reason: str = "") -> None:
        self._ensure_control_health_tracking()
        reason_code = _normalize_control_reason(reason)
        with guard_context(ctx='llm_adapter/llm_core.py:1056', catch_base=False) as __m3_guard_1054_8:
            self._control_health_window.append((time.time(), bool(success), str(reason_code)))

        if __m3_guard_1054_8.error is not None:
            return
        if success:
            self._auto_mode_fail_streak = 0
        else:
            self._auto_mode_fail_streak += 1

    def _compute_recent_control_stats(self) -> Dict[str, float]:
        self._ensure_control_health_tracking()
        now = time.time()
        window_sec = float(os.getenv("M3_CONTROL_HEALTH_WINDOW_SEC", "180"))
        raw_window = attr_get_optional(self, "_control_health_window", None)
        if raw_window is None:
            return {"count": 0.0, "fail_ratio": 0.0, "consecutive_failures": 0.0}
        events: List[Tuple[float, bool, str]] = []
        for item in list(raw_window):
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            ts = item[0]
            ok = item[1]
            reason = item[2] if len(item) >= 3 else ""
            with guard_context(ctx='llm_adapter/llm_core.py:1096', catch_base=False) as __m3_guard_1093_20:
                tsf = float(ts)
                if now - tsf <= window_sec:
                    events.append((tsf, bool(ok), str(reason)))

            if __m3_guard_1093_20.error is not None:
                continue
        if not events:
            return {"count": 0.0, "fail_ratio": 0.0, "consecutive_failures": 0.0}
        n = len(events)
        fails = sum(1 for _, ok, _ in events if not ok)
        return {
            "count": float(n),
            "fail_ratio": float(fails / max(1.0, float(n))),
            "consecutive_failures": float(self._auto_mode_fail_streak),
        }

    def _auto_control_selection(self) -> str:
        if attr_get_optional(self, "_hf_circuit_open", False):
            return "off"
        stats = self._compute_recent_control_stats()
        fail_ratio = float(stats.get("fail_ratio", 0.0))
        consecutive = int(stats.get("consecutive_failures", 0))
        count = int(stats.get("count", 0.0))
        if consecutive >= 2:
            return "off"
        if fail_ratio >= 0.30 and count >= 3:
            return "state"
        if fail_ratio >= 0.15 and count >= 2:
            return "state"

        core = attr_get_optional(self, 'core', None)
        if core is not None:
            em = attr_get_optional(core, 'episodic_memory', None)
            with guard_context(ctx='llm_adapter/llm_core.py:1090', catch_base=False) as __m3_guard_1088_12:
                retrieved = int(attr_get_optional(em, 'total_retrieved', 0))

            if __m3_guard_1088_12.error is not None:
                retrieved = 0
            if retrieved > 0 and os.getenv("M3_CONTROL_AUTO_FULL", "1").lower() not in ('0', 'false', 'no', 'off'):
                return "full"
            if retrieved >= 0:
                return "memory"

        return "state"

    def _control_selection_mode(self) -> str:
        raw = str(os.getenv("M3_CONTROL_SELECTION_MODE", "state") or "state").strip().lower()
        if raw in {"0", "off", "none", "disable", "disabled", "no", "false"}:
            return "off"
        if raw in {"1", "state", "state_only", "context", "context_only", "low"}:
            return "state"
        if raw in {"2", "memory", "mid", "mixed", "medium"}:
            return "memory"
        if raw in {"3", "full", "high", "all", "strict"}:
            return "full"
        if raw in {"auto", "adaptive", "self", "self_adjust"}:
            return self._auto_control_selection()
        # Backward-compatible aliases
        if raw in {"on", "true", "yes"}:
            return "full"
        return "state"

    def _control_allows(self, feature: str) -> bool:
        mode = self._control_selection_mode()
        allowed = {
            "off": set(),
            "state": {"state_context"},
            "memory": {"state_context", "memory_retrieval"},
            "full": {"state_context", "memory_retrieval", "bridge", "decode_control", "adaptive_sampler", "token_value_bias", "quality_gate"},
        }.get(mode, {"state_context"})
        return feature in allowed

    def _ensure_control_bridge(self, state_dim: int):
        if M3ControlBridge is None:
            return None
        try:
            state_dim = int(max(8, state_dim))
        except Exception:
            state_dim = 256
        layers = find_decoder_layers(self._model) if self._model is not None else []
        num_layers = max(1, len(layers))
        need_new = (
            self._control_bridge is None
            or self._control_bridge_state_dim != state_dim
            or not self._control_bridge_layers
        )
        if need_new:
            prefix_len = int(max(1, int(os.getenv('M3_BRIDGE_PREFIX_LEN', '8'))))
            logit_rank = int(max(4, int(os.getenv('M3_BRIDGE_LOGIT_RANK', '32'))))
            self._control_bridge = M3ControlBridge(
                state_dim=state_dim,
                model_hidden_dim=int(self.hidden_dim or 1024),
                vocab_size=int(self.vocab_size or 32000),
                num_layers=num_layers,
                prefix_len=prefix_len,
                logit_rank=logit_rank,
            ).to(self.device)
            self._control_bridge.eval()
            self._control_bridge_state_dim = state_dim
            self._control_bridge_layers = list(layers)
        return self._control_bridge

    def _neuro_enabled(self) -> bool:
        """True when NeuroModulator weight-level control is active."""
        nm_cfg = get_global_config().neuro_modulator
        env_flag = os.getenv('M3_ENABLE_NEURO_MODULATOR', '').lower()
        if env_flag:
            enabled = env_flag in ('1', 'true', 'yes', 'on')
        else:
            enabled = nm_cfg.enabled
        return NeuroModulator is not None and enabled

    def _ensure_neuro_modulator(self, state_dim: int):
        """Lazy-create and return the NeuroModulator instance."""
        if NeuroModulator is None:
            return None
        nm_cfg = get_global_config().neuro_modulator
        try:
            state_dim = int(max(8, state_dim))
        except Exception:
            state_dim = int(max(8, nm_cfg.state_dim))
        if (
            self._neuro_modulator is not None
            and self._neuro_mod_state_dim == state_dim
        ):
            return self._neuro_modulator

        layers = find_decoder_layers(self._model) if self._model is not None else []
        num_layers = max(1, len(layers))
        with guard_context(ctx='llm_adapter/llm_core.py:1187', catch_base=False) as __m3_guard_1183_8:
            hidden_rank = int(os.getenv('M3_NEURO_HIDDEN_RANK', str(nm_cfg.hidden_rank)))
            logit_rank = int(os.getenv('M3_NEURO_LOGIT_RANK', str(nm_cfg.logit_rank)))
            trunk_dim = int(os.getenv('M3_NEURO_TRUNK_DIM', str(nm_cfg.trunk_dim)))

        if __m3_guard_1183_8.error is not None:
            hidden_rank, logit_rank, trunk_dim = nm_cfg.hidden_rank, nm_cfg.logit_rank, nm_cfg.trunk_dim

        self._neuro_modulator = NeuroModulator(
            state_dim=state_dim,
            num_layers=num_layers,
            model_hidden_dim=int(self.hidden_dim or 1024),
            vocab_size=int(self.vocab_size or 32000),
            trunk_dim=trunk_dim,
            hidden_rank=hidden_rank,
            logit_rank=logit_rank,
        ).to(self.device)
        self._neuro_modulator.max_gain_delta = nm_cfg.max_gain_delta
        self._neuro_modulator.max_logit_bias = nm_cfg.max_logit_bias
        self._neuro_modulator.warmup_total = nm_cfg.warmup_steps
        self._neuro_modulator.eval()
        self._neuro_mod_state_dim = state_dim

        with guard_context(ctx='llm_adapter/llm_core.py:1207', catch_base=False) as __m3_guard_1205_8:
            lr = float(os.getenv('M3_NEUROMOD_LR', str(nm_cfg.learning_rate)))

        if __m3_guard_1205_8.error is not None:
            lr = nm_cfg.learning_rate
        weight_decay = nm_cfg.weight_decay
        self._neuro_mod_opt = torch.optim.Adam(
            self._neuro_modulator.parameters(), lr=lr, weight_decay=weight_decay
        )
        logging.getLogger('llm_adapter').info(
            f'[HFBackend] NeuroModulator created: layers={num_layers}, '
            f'state_dim={state_dim}, hidden_rank={hidden_rank}'
        )
        self._load_neuro_checkpoint()
        return self._neuro_modulator

    def _save_neuro_checkpoint(self) -> bool:
        """Persist NeuroModulator weights to disk."""
        if self._neuro_modulator is None:
            return False
        nm_cfg = get_global_config().neuro_modulator
        path = _resolve_neuro_checkpoint_path(nm_cfg.checkpoint_file)
        with guard_context(ctx='llm_adapter/llm_core.py:1238', catch_base=False) as __m3_guard_1226_8:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save({
                'model_state_dict': self._neuro_modulator.state_dict(),
                'optimizer_state_dict': self._neuro_mod_opt.state_dict() if self._neuro_mod_opt else None,
                'step': self._neuro_modulator._step,
                'state_dim': self._neuro_mod_state_dim,
            }, path)
            logging.getLogger('llm_adapter').info(
                f'[HFBackend] NeuroModulator checkpoint saved: {path}'
            )
            return True

        if __m3_guard_1226_8.error is not None:
            e = __m3_guard_1226_8.error
            logging.getLogger('llm_adapter').debug(
                f'[HFBackend] NeuroModulator checkpoint save failed: {e}'
            )
            return False

    def _load_neuro_checkpoint(self) -> bool:
        """Restore NeuroModulator weights from disk."""
        if self._neuro_modulator is None:
            return False
        nm_cfg = get_global_config().neuro_modulator
        path = _resolve_neuro_checkpoint_path(nm_cfg.checkpoint_file)
        if not os.path.exists(path):
            return False
        with guard_context(ctx='llm_adapter/llm_core.py:1275', catch_base=False) as __m3_guard_1252_8:
            try:
                # Prefer safe loading when supported (PyTorch >= 2.1 with weights_only)
                ckpt = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                # Older PyTorch versions do not support the weights_only kwarg.
                # Retry without the kwarg to maintain compatibility.
                ckpt = torch.load(path, map_location=self.device)
            except (RuntimeError, ValueError):
                # Optional unsafe pickle fallback; disabled by default.
                # Enable by setting LLM_ADAPTER_UNSAFE_PICKLE_FALLBACK=1.
                if os.getenv("LLM_ADAPTER_UNSAFE_PICKLE_FALLBACK", "0") == "1":
                    ckpt = torch.load(path, map_location=self.device)
                else:
                    raise
            self._neuro_modulator.load_state_dict(ckpt['model_state_dict'])
            if self._neuro_mod_opt and ckpt.get('optimizer_state_dict'):
                self._neuro_mod_opt.load_state_dict(ckpt['optimizer_state_dict'])
            self._neuro_modulator._step = int(ckpt.get('step', 0))
            logging.getLogger('llm_adapter').info(
                f'[HFBackend] NeuroModulator checkpoint loaded: {path} (step={self._neuro_modulator._step})'
            )
            return True

        if __m3_guard_1252_8.error is not None:
            e = __m3_guard_1252_8.error
            logging.getLogger('llm_adapter').debug(
                f'[HFBackend] NeuroModulator checkpoint load failed: {e}'
            )
            return False

    def _neuro_status(self) -> dict:
        """Return a summary dict of NeuroModulator status for diagnostics."""
        # Resolve whether the NeuroModulator is enabled according to config/env.
        # Fall back gracefully if older code lacks `_neuro_enabled`.
        try:
            enabled = bool(self._neuro_enabled())
        except AttributeError:
            # Backward-compat: approximate "enabled" by whether a modulator exists.
            enabled = self._neuro_modulator is not None

        if self._neuro_modulator is None:
            # No instantiated modulator; never "active", but still report whether
            # config/env currently consider it enabled.
            return {
                'active': False,
                'enabled': enabled,
            }

        nm = self._neuro_modulator
        return {
            # "active" reflects runtime enablement, not just instance existence.
            'active': bool(enabled),
            'enabled': enabled,
            'step': nm._step,
            'state_dim': self._neuro_mod_state_dim,
            'num_layers': nm.num_layers,
            'hidden_rank': nm.hidden_rank,
            'logit_rank': nm.logit_rank,
            'warmup_total': nm.warmup_total,
            'max_gain_delta': nm.max_gain_delta,
            'params': sum(p.numel() for p in nm.parameters()),
        }

    def _prepare_bridge_state(self, z_m3, state_dim: int, device):
        if z_m3 is None:
            return None
        with guard_context(ctx='llm_adapter/llm_core.py:1319', catch_base=False) as __m3_guard_1317_8:
            z = np.asarray(z_m3, dtype=np.float32).ravel()

        if __m3_guard_1317_8.error is not None:
            return None
        if z.size == 0:
            return None
        if z.size > state_dim:
            z = z[:state_dim]
        elif z.size < state_dim:
            z = np.pad(z, (0, state_dim - z.size), mode='constant')
        import torch as _torch
        return _torch.from_numpy(z).to(device=device, dtype=_torch.float32).unsqueeze(0)

    @staticmethod
    def _micro_update_step_state(core, _step: int, generated_ids: list, interval: int) -> bool:
        """Apply lightweight core state updates every ``interval`` steps."""
        if core is None or interval is None:
            return False
        try:
            interval = int(interval)
        except Exception:
            return False
        if interval <= 0 or (_step <= 0) or (_step % interval != 0):
            return False
        updated = False
        with guard_context(ctx='llm_adapter/llm_core.py:1357', catch_base=False) as __m3_guard_1342_8:
            if attr_has(core, 'energy_ctrl'):
                ec = core.energy_ctrl
                before = float(attr_get_optional(ec, 'cognitive_energy', 0.0))
                ec.cognitive_energy = max(0.0, before - 0.05 * interval)
                if ec.cognitive_energy != before:
                    updated = True
            before = float(attr_get_optional(core, 'decode_entropy', attr_get_optional(core, 'token_entropy', 0.0)))
            window = generated_ids[-interval:] if generated_ids else []
            n_unique = len(set(window)) if window else 0
            token_diversity = n_unique / max(interval, 1)
            decode_entropy = 0.8 * before + 0.2 * token_diversity
            if decode_entropy != before:
                updated = True
            core.decode_entropy = decode_entropy

        if __m3_guard_1342_8.error is not None:
            return False
        return updated

    @staticmethod
    def _sample_next_token(logits, temperature: float, top_k: int, top_p: float):
        """Shared sampling logic for HF decoding."""
        import torch as _torch

        if temperature <= 0:
            return _torch.argmax(logits, dim=-1, keepdim=True), None

        logits = logits / (temperature + 1e-8)
        if top_k > 0:
            k = min(int(top_k), logits.size(-1))
            topv, _ = _torch.topk(logits, k)
            logits = _torch.where(logits < topv[:, -1:], _torch.full_like(logits, float('-inf')), logits)
        if 0 < top_p < 1.0:
            sl, si = _torch.sort(logits, descending=True)
            cp = _torch.cumsum(_torch.softmax(sl, dim=-1), dim=-1)
            mask = cp > top_p
            mask[:, 0] = False
            sl = sl.clone()
            sl[mask] = float('-inf')
            logits = _torch.zeros_like(logits).scatter(-1, si, sl)
        probs = _torch.softmax(logits, dim=-1)
        return _torch.multinomial(probs, num_samples=1), {
            'temperature': float(temperature),
            'top_k': int(top_k),
            'top_p': float(top_p),
        }

    @staticmethod
    def _token_critic_enabled(token_value_head, internal_hidden_dim) -> bool:
        return (
            token_value_head is not None
            and isinstance(internal_hidden_dim, int)
            and internal_hidden_dim > 0
        )

    def _compute_sample_params(self, core, m3_sampler, base_temperature, base_top_k, base_top_p):
        temperature = float(base_temperature) if base_temperature is not None else 0.8
        top_k = int(base_top_k) if base_top_k is not None else 50
        top_p = float(base_top_p) if base_top_p is not None else 0.9
        sampler_enabled = self._control_allows('adaptive_sampler') and os.getenv(
            'M3_HF_ENABLE_M3_SAMPLER', '1'
        ).lower() in ('1', 'true', 'yes', 'on')
        if m3_sampler is not None and core is not None and sampler_enabled:
            with guard_context(ctx='llm_adapter/llm_core.py:1408', catch_base=False) as __m3_guard_1405_12:
                temperature = float(m3_sampler._compute_temperature(core, temperature))
                top_k = int(m3_sampler._compute_top_k(core, top_k))

            if __m3_guard_1405_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        return temperature, top_k, top_p

    @staticmethod
    def _apply_bridge_logit_bias(logits, bridge_controls):
        if bridge_controls is None:
            return logits
        with guard_context(ctx='llm_adapter/llm_core.py:1420', catch_base=False) as __m3_guard_1416_8:
            lb = bridge_controls.logit_bias
            if lb is not None and lb.shape[-1] == logits.shape[-1]:
                return logits + lb.to(device=logits.device, dtype=logits.dtype)

        if __m3_guard_1416_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return logits

    def _apply_token_value_injection(self, logits, hidden, token_value_head, beta, internal_hidden_dim):
        if not self._token_critic_enabled(token_value_head, internal_hidden_dim):
            return logits
        with guard_context(ctx='llm_adapter/llm_core.py:1440', catch_base=False) as __m3_guard_1427_8:
            import torch as _torch
            if hidden is None:
                return logits
            h_proj = self._hf_proj(hidden.to(self._hf_proj.weight.dtype)) if self._hf_proj is not None else hidden.to(_torch.float32)
            q_vals = token_value_head(h_proj.float())
            if q_vals.shape[-1] != logits.shape[-1]:
                return logits
            q = q_vals.float()
            q = q - q.mean(dim=-1, keepdim=True)
            q = q / (q.std(dim=-1, keepdim=True) + 1e-6)
            q = _torch.clamp(q, -3.0, 3.0)
            return logits + float(beta) * q

        if __m3_guard_1427_8.error is not None:
            return logits

    def _resolve_forbidden_token_ids(self, decode_control: Optional[Dict[str, Any]]) -> List[int]:
        if not decode_control:
            return []

        terms: List[str] = []
        if not decode_control.get("allow_state_terms", True):
            base_terms = decode_control.get("forbidden_terms", [])
            if isinstance(base_terms, (list, tuple)):
                terms.extend(str(t).strip() for t in base_terms if str(t).strip())

        if decode_control.get("identity_lock", False):
            identity_terms = decode_control.get("identity_forbidden_terms", [])
            if isinstance(identity_terms, (list, tuple)):
                terms.extend(str(t).strip() for t in identity_terms if str(t).strip())

        key = tuple(dict.fromkeys(terms))
        if not key:
            return []
        cached = self._decode_term_cache.get(key)
        if cached is not None:
            return list(cached)

        ids = set()
        for term in key:
            try:
                token_ids = self._tokenizer(term, add_special_tokens=False).get("input_ids", [])
            except Exception:
                with guard_context(ctx='llm_adapter/llm_core.py:1472', catch_base=False) as __m3_guard_1470_16:
                    token_ids = self._tokenizer.encode(term, add_special_tokens=False)

                if __m3_guard_1470_16.error is not None:
                    token_ids = []
            for tid in token_ids:
                try:
                    t_int = int(tid)
                except Exception:
                    continue
                if t_int >= 0:
                    ids.add(t_int)
        resolved = sorted(ids)
        self._decode_term_cache[key] = tuple(resolved)
        return resolved

    @staticmethod
    def _apply_decode_control_params(temperature: float, top_k: int, top_p: float, decode_control: Optional[Dict[str, Any]]):
        if not decode_control or decode_control.get("allow_state_terms", True):
            return temperature, top_k, top_p
        with guard_context(ctx='llm_adapter/llm_core.py:1491', catch_base=False) as __m3_guard_1489_8:
            temperature = min(float(temperature), float(decode_control.get("max_temperature", temperature)))

        if __m3_guard_1489_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:1497', catch_base=False) as __m3_guard_1493_8:
            ctrl_k = int(decode_control.get("max_top_k", top_k))
            if ctrl_k > 0 and int(top_k) > 0:
                top_k = min(int(top_k), ctrl_k)

        if __m3_guard_1493_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:1501', catch_base=False) as __m3_guard_1499_8:
            top_p = min(float(top_p), float(decode_control.get("max_top_p", top_p)))

        if __m3_guard_1499_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return temperature, top_k, top_p

    @staticmethod
    def _repeat_ngram_blocklist(token_ids: List[int], ngram_size: int) -> List[int]:
        n = int(max(0, ngram_size))
        if n < 2 or len(token_ids) < n - 1:
            return []
        prefix = tuple(int(t) for t in token_ids[-(n - 1):])
        blocked: set[int] = set()
        lim = len(token_ids) - (n - 1)
        for i in range(max(0, lim)):
            if tuple(int(x) for x in token_ids[i:i + n - 1]) == prefix:
                nxt = int(token_ids[i + n - 1])
                blocked.add(nxt)
        return sorted(blocked)

    @staticmethod
    def _has_suffix_loop(token_ids: List[int], min_span: int = 2, repeat_count: int = 3) -> bool:
        if len(token_ids) < int(min_span * repeat_count):
            return False
        max_span = min(16, len(token_ids) // int(max(2, repeat_count)))
        for span in range(int(max(2, min_span)), int(max_span) + 1):
            seq = token_ids[-span:]
            ok = True
            for r in range(2, int(repeat_count) + 1):
                st = -r * span
                en = -(r - 1) * span
                if token_ids[st:en] != seq:
                    ok = False
                    break
            if ok:
                return True
        return False

    @staticmethod
    def _token_repetition_risk(token_ids: List[int], window: int = 24) -> float:
        if not token_ids:
            return 0.0
        tail = token_ids[-max(4, int(window)):]
        uniq_ratio = float(len(set(tail))) / float(max(1, len(tail)))
        risk = max(0.0, min(1.0, 1.0 - uniq_ratio))
        if HFBackend._has_suffix_loop(tail):
            risk = min(1.0, risk + 0.35)
        return float(risk)

    def _set_runtime_diag(
        self,
        reason_code: str,
        phase: str,
        model_output_shape: str = "unknown",
        has_logits: bool = False,
    ) -> None:
        self._last_runtime_diag = {
            "reason_code": _normalize_control_reason(reason_code),
            "phase": str(phase or "unknown"),
            "model_output_shape": str(model_output_shape or "unknown"),
            "has_logits": bool(has_logits),
        }

    @staticmethod
    def _shape_text(value: Any) -> str:
        if value is None:
            return "none"
        shape = attr_get_optional(value, "shape", None)
        if shape is None:
            return type(value).__name__
        try:
            dims = [int(x) for x in list(shape)]
            return "x".join(str(d) for d in dims)
        except Exception:
            return str(shape)

    def _raise_runtime_failure(
        self,
        reason_code: str,
        phase: str,
        out: Any = None,
        message: str = "",
    ) -> None:
        logits = attr_get_optional(out, "logits", None)
        has_logits = logits is not None
        model_output_shape = self._shape_text(logits)
        self._set_runtime_diag(
            reason_code=reason_code,
            phase=phase,
            model_output_shape=model_output_shape,
            has_logits=has_logits,
        )
        raise HFRuntimeFailure(
            reason_code=reason_code,
            phase=phase,
            model_output_shape=model_output_shape,
            has_logits=has_logits,
            message=message,
        )

    def _validate_model_output(self, out: Any, phase: str) -> Any:
        if out is None:
            self._raise_runtime_failure(
                reason_code="hf_runtime_failure",
                phase=phase,
                out=out,
                message="model_output_none",
            )
        logits = attr_get_optional(out, "logits", None)
        if logits is None:
            self._raise_runtime_failure(
                reason_code="hf_runtime_failure",
                phase=phase,
                out=out,
                message="missing_logits",
            )
        try:
            shape = self._shape_text(logits)
            self._set_runtime_diag(
                reason_code="hf_generate_ok",
                phase=phase,
                model_output_shape=shape,
                has_logits=True,
            )
        except Exception:
            self._set_runtime_diag(
                reason_code="hf_generate_ok",
                phase=phase,
                model_output_shape="unknown",
                has_logits=True,
            )
        return out

    # ------------------------------------------------------------------ #
    #  Core generation with M3 control hooks                              #
    # ------------------------------------------------------------------ #
    def generate_with_m3(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        core=None,
        m3_sampler=None,
        token_value_head=None,
        internal_hidden_dim: int = None,
        beta: float = 0.1,
        z_m3: Optional[np.ndarray] = None,
        max_new_tokens: int = None,
        base_temperature: float = 0.8,
        base_top_k: int = 50,
        base_top_p: float = 0.9,
        decode_control: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate text with per-token M3 parameter control.

        Control points executed at every decoding step:
        1. Token-critic Q-value injection  (logit bias from internal value head)
        2. M3AdaptiveSampler               (dynamic temperature / top_k)
        3. Standard top-k / top-p / multinomial sampling
        """
        import torch as _torch
        self._ensure_loaded()

        if max_new_tokens is None:
            max_new_tokens = int(os.getenv('M3_HF_MAX_TOKENS', '512'))

        max_input = int(os.getenv('M3_HF_MAX_INPUT', '4096'))
        # Prefer chat templates for instruct models when available.
        # This significantly improves response quality vs. plain-text concatenation.
        prompt_text = prompt
        if messages is not None:
            try:
                safe_messages = [dict(m) for m in messages]
                with guard_context(ctx='llm_adapter/llm_core.py:1589', catch_base=False) as __m3_guard_1587_16:
                    system_max_tokens = int(os.getenv('M3_SYSTEM_MAX_TOKENS', '320'))

                if __m3_guard_1587_16.error is not None:
                    system_max_tokens = 320
                if max_input > 0:
                    system_max_tokens = min(system_max_tokens, max(64, int(max_input * 0.4)))
                if system_max_tokens > 0:
                    for mi, m in enumerate(safe_messages):
                        if str(m.get('role', '')).lower() != 'system':
                            continue
                        system_content = str(m.get('content', ''))
                        if system_content:
                            kept_lines = []
                            for ln in system_content.splitlines():
                                s = ln.strip()
                                if (
                                    s.startswith("vector=")
                                    or s.startswith("vector_head=")
                                    or s.startswith("vector[")
                                ):
                                    continue
                                kept_lines.append(ln)
                            system_content = "\n".join(kept_lines).strip()
                            with guard_context(ctx='llm_adapter/llm_core.py:1617', catch_base=False) as __m3_guard_1610_28:
                                ids = self._tokenizer(system_content, add_special_tokens=False).get('input_ids', [])
                                if len(ids) > system_max_tokens:
                                    system_content = self._tokenizer.decode(
                                        ids[:system_max_tokens],
                                        skip_special_tokens=True,
                                    ).strip()

                            if __m3_guard_1610_28.error is not None:
                                logging.getLogger(__name__).exception("Swallowed exception")
                        safe_messages[mi]['content'] = system_content
                        break
                if attr_has(self._tokenizer, 'apply_chat_template'):
                    prompt_text = self._tokenizer.apply_chat_template(
                        safe_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    # Minimal, role-preserving fallback.
                    parts = []
                    for m in safe_messages:
                        role = str(m.get('role', 'user'))
                        content = str(m.get('content', '')).strip()
                        if not content:
                            continue
                        if role == 'system':
                            parts.append(content)
                        elif role == 'user':
                            parts.append(f'User: {content}')
                        else:
                            parts.append(f'Assistant: {content}')
                    parts.append('Assistant:')
                    prompt_text = "\n".join(parts)
            except Exception:
                prompt_text = prompt

        inputs = self._tokenizer(
            prompt_text, return_tensors='pt', truncation=False
        )
        if not isinstance(inputs, dict):
            self._raise_runtime_failure(
                reason_code="hf_runtime_failure",
                phase="tokenize",
                out=None,
                message="tokenizer_output_not_dict",
            )
        if attr_get_optional(inputs, "get", None) is None or inputs.get("input_ids") is None:
            self._raise_runtime_failure(
                reason_code="hf_runtime_failure",
                phase="tokenize",
                out=None,
                message="missing_input_ids",
            )
        if inputs.get("attention_mask") is None:
            with guard_context(ctx='llm_adapter/llm_core.py:1665', catch_base=False) as __m3_guard_1661_12:
                inputs["attention_mask"] = _torch.ones_like(inputs["input_ids"])

            if __m3_guard_1661_12.error is not None:
                self._raise_runtime_failure(
                    reason_code="hf_runtime_failure",
                    phase="tokenize",
                    out=None,
                    message="missing_attention_mask",
                )
        with guard_context(ctx='llm_adapter/llm_core.py:1657', catch_base=False) as __m3_guard_1649_8:
            if max_input > 0:
                cur_len = int(inputs['input_ids'].shape[1])
                if cur_len > max_input:
                    start = cur_len - max_input
                    for k, v in list(inputs.items()):
                        if attr_has(v, 'shape') and len(v.shape) == 2 and int(v.shape[1]) == cur_len:
                            inputs[k] = v[:, start:]

        if __m3_guard_1649_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        inputs_embeds = None
        bridge_controls = None
        bridge_runtime = None

        # Lazy-init projection: HF hidden ??internal model hidden
        if (
            token_value_head is not None
            and internal_hidden_dim is not None
            and self._hf_proj_dims != (self.hidden_dim, internal_hidden_dim)
        ):
            self._hf_proj = _torch.nn.Linear(
                self.hidden_dim, internal_hidden_dim, bias=False
            )
            _torch.nn.init.xavier_uniform_(self._hf_proj.weight)
            self._hf_proj = self._hf_proj.to(self.device).to(_torch.bfloat16)
            self._hf_proj.eval()
            self._hf_proj_dims = (self.hidden_dim, internal_hidden_dim)

        # Optional M3ControlBridge (experimental; off by default)
        if self._bridge_enabled_safe() and z_m3 is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:1683', catch_base=False) as __m3_guard_1681_12:
                state_dim = int(os.getenv('M3_BRIDGE_STATE_DIM', '256'))

            if __m3_guard_1681_12.error is not None:
                state_dim = 256
            try:
                bridge = self._ensure_control_bridge(state_dim=state_dim)
                z_t = self._prepare_bridge_state(z_m3, state_dim=state_dim, device=self.device)
                if bridge is not None and z_t is not None:
                    strength = float(os.getenv('M3_BRIDGE_STRENGTH', '1.0'))
                    with _torch.no_grad():
                        bridge_controls = bridge(z_t, strength=strength)
                    if bridge_controls.prefix_embeddings is not None:
                        with guard_context(ctx='llm_adapter/llm_core.py:1708', catch_base=False) as __m3_guard_1693_24:
                            tok_emb = self._model.get_input_embeddings()(input_ids)
                            prefix = bridge_controls.prefix_embeddings.to(
                                device=tok_emb.device,
                                dtype=tok_emb.dtype,
                            )
                            inputs_embeds = _torch.cat([prefix, tok_emb], dim=1)
                            input_ids = None
                            prefix_mask = _torch.ones(
                                attention_mask.size(0),
                                prefix.size(1),
                                device=attention_mask.device,
                                dtype=attention_mask.dtype,
                            )
                            attention_mask = _torch.cat([prefix_mask, attention_mask], dim=1)

                        if __m3_guard_1693_24.error is not None:
                            inputs_embeds = None
                    if (
                        bridge_controls.layer_gates is not None
                        and LayerGateRuntime is not None
                    ):
                        layers = self._control_bridge_layers or find_decoder_layers(self._model)
                        if layers:
                            bridge_runtime = LayerGateRuntime(layers)
                            bridge_runtime.apply(bridge_controls.layer_gates[0])
            except Exception as e:
                logging.getLogger('llm_adapter').debug(f'[HFBackend] control bridge skipped: {e}')
                bridge_controls = None
                bridge_runtime = None

        # NeuroModulator: weight-level M3 consciousness control
        neuro_controls = None
        neuro_runtime = None
        if self._neuro_enabled() and z_m3 is not None:
            try:
                _nm_cfg = get_global_config().neuro_modulator
                with guard_context(ctx='llm_adapter/llm_core.py:1731', catch_base=False) as __m3_guard_1729_16:
                    neuro_state_dim = int(os.getenv('M3_NEURO_STATE_DIM', str(_nm_cfg.state_dim)))

                if __m3_guard_1729_16.error is not None:
                    neuro_state_dim = _nm_cfg.state_dim
                neuro_mod = self._ensure_neuro_modulator(state_dim=neuro_state_dim)
                z_neuro = self._prepare_bridge_state(
                    z_m3, state_dim=neuro_state_dim, device=self.device
                )
                if neuro_mod is not None and z_neuro is not None:
                    with guard_context(ctx='llm_adapter/llm_core.py:1740', catch_base=False) as __m3_guard_1738_20:
                        neuro_strength = float(os.getenv('M3_NEURO_STRENGTH', str(_nm_cfg.strength)))

                    if __m3_guard_1738_20.error is not None:
                        neuro_strength = _nm_cfg.strength
                    with _torch.no_grad():
                        neuro_controls = neuro_mod(z_neuro, strength=neuro_strength)
                    layers = find_decoder_layers(self._model)
                    if layers and NeuroModulatorRuntime is not None:
                        neuro_runtime = NeuroModulatorRuntime(layers)
                        neuro_runtime.apply(neuro_controls)
            except Exception as e:
                logging.getLogger('llm_adapter').debug(
                    f'[HFBackend] neuro modulator skipped: {e}'
                )
                neuro_controls = None
                neuro_runtime = None

        generated_ids: list = []
        past_key_values = None
        forbidden_token_ids = self._resolve_forbidden_token_ids(decode_control)
        with guard_context(ctx='llm_adapter/llm_core.py:1760', catch_base=False) as __m3_guard_1758_8:
            no_repeat_ngram = int(os.getenv("M3_HF_NO_REPEAT_NGRAM", "4"))

        if __m3_guard_1758_8.error is not None:
            no_repeat_ngram = 4
        with guard_context(ctx='llm_adapter/llm_core.py:1764', catch_base=False) as __m3_guard_1762_8:
            no_repeat_penalty = float(os.getenv("M3_HF_NO_REPEAT_PENALTY", "1e9"))

        if __m3_guard_1762_8.error is not None:
            no_repeat_penalty = 1e9
        with guard_context(ctx='llm_adapter/llm_core.py:1768', catch_base=False) as __m3_guard_1766_8:
            suffix_repeat_stop = int(os.getenv("M3_HF_SUFFIX_REPEAT_STOP", "3"))

        if __m3_guard_1766_8.error is not None:
            suffix_repeat_stop = 3
        rep_temp_scale = 1.0
        rep_top_p_scale = 1.0
        with guard_context(ctx='llm_adapter/llm_core.py:1774', catch_base=False) as __m3_guard_1772_8:
            forbidden_penalty = float(decode_control.get("forbidden_penalty", 0.0)) if decode_control else 0.0

        if __m3_guard_1772_8.error is not None:
            forbidden_penalty = 0.0
        # How often to micro-update core state during decoding
        with guard_context(ctx='llm_adapter/llm_core.py:1779', catch_base=False) as __m3_guard_1777_8:
            _core_update_interval = int(os.getenv('M3_HF_CORE_UPDATE_INTERVAL', '8'))

        if __m3_guard_1777_8.error is not None:
            _core_update_interval = 8

        for _step in range(max_new_tokens):
            # === M3 CONTROL 0: Lightweight core state micro-update ===
            # without this, qualia/energy become static and adaptive sampler cannot react.
            self._micro_update_step_state(
                core=core,
                _step=_step,
                generated_ids=generated_ids,
                interval=_core_update_interval,
            )

            with _torch.no_grad():
                model_kw = {
                    'attention_mask': attention_mask,
                    'past_key_values': past_key_values,
                    'use_cache': True,
                    'output_hidden_states': True,
                }
                if inputs_embeds is not None:
                    model_kw['inputs_embeds'] = inputs_embeds
                else:
                    model_kw['input_ids'] = input_ids
                out = self._model(**model_kw)
            out = self._validate_model_output(out, phase="decode_forward")

            logits = out.logits[:, -1, :].float()          # (1, V)
            hidden_states = attr_get_optional(out, "hidden_states", None)
            if not hidden_states:
                self._raise_runtime_failure(
                    reason_code="hf_runtime_failure",
                    phase="decode_hidden",
                    out=out,
                    message="missing_hidden_states",
                )
            hidden = hidden_states[-1][:, -1, :]       # (1, H)
            past_key_values = out.past_key_values
            inputs_embeds = None

            # M3ControlBridge logit bias
            logits = self._apply_bridge_logit_bias(logits, bridge_controls)

            # NeuroModulator logit bias
            if neuro_controls is not None and neuro_controls.logit_bias is not None:
                with guard_context(ctx='llm_adapter/llm_core.py:1821', catch_base=False) as __m3_guard_1815_16:
                    _nlb = neuro_controls.logit_bias
                    if _nlb.shape[-1] == logits.shape[-1]:
                        logits = logits + _nlb.to(
                            device=logits.device, dtype=logits.dtype
                        )

                if __m3_guard_1815_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")

            # === M3 CONTROL 1: Token-Critic Q-value injection ===
            logits = self._apply_token_value_injection(
                logits=logits,
                hidden=hidden,
                token_value_head=token_value_head,
                beta=beta,
                internal_hidden_dim=internal_hidden_dim,
            )
            if forbidden_token_ids and forbidden_penalty > 0.0:
                try:
                    logits[:, forbidden_token_ids] = logits[:, forbidden_token_ids] - float(forbidden_penalty)
                except Exception:
                    logging.getLogger(__name__).exception("Swallowed exception")
            if no_repeat_ngram >= 2:
                blocked_ids = self._repeat_ngram_blocklist(generated_ids, ngram_size=no_repeat_ngram)
                if blocked_ids:
                    try:
                        logits[:, blocked_ids] = logits[:, blocked_ids] - float(no_repeat_penalty)
                    except Exception:
                        logging.getLogger(__name__).exception("Swallowed exception")

            # === M3 CONTROL 2: Adaptive sampling from core state ===
            temperature, top_k, top_p = self._compute_sample_params(
                core=core,
                m3_sampler=m3_sampler,
                base_temperature=base_temperature,
                base_top_k=base_top_k,
                base_top_p=base_top_p,
            )
            temperature, top_k, top_p = self._apply_decode_control_params(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                decode_control=decode_control,
            )
            # Repetition-risk dampening: lower entropy when loop risk rises.
            temperature = float(max(0.08, float(temperature) * float(rep_temp_scale)))
            top_p = float(np.clip(float(top_p) * float(rep_top_p_scale), 0.05, 1.0))
            if int(top_k) > 0:
                top_k = int(max(1, int(float(top_k) * float(rep_top_p_scale))))
            next_token, sample_meta = self._sample_next_token(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            token_id = next_token.item()
            if token_id == self._tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)
            rep_risk = self._token_repetition_risk(generated_ids, window=24)
            if rep_risk > 0.25:
                rep_temp_scale = max(0.55, float(rep_temp_scale) * 0.93)
                rep_top_p_scale = max(0.60, float(rep_top_p_scale) * 0.95)
            else:
                rep_temp_scale = min(1.0, float(rep_temp_scale) * 1.01)
                rep_top_p_scale = min(1.0, float(rep_top_p_scale) * 1.01)
            if self._has_suffix_loop(generated_ids, min_span=2, repeat_count=max(2, int(suffix_repeat_stop))):
                break

            # KV-cache: feed only the new token
            input_ids = next_token
            attention_mask = _torch.cat(
                [attention_mask,
                 _torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )

        if bridge_runtime is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:1896', catch_base=False) as __m3_guard_1894_12:
                bridge_runtime.close()

            if __m3_guard_1894_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        if neuro_runtime is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:1901', catch_base=False) as __m3_guard_1899_12:
                neuro_runtime.close()

            if __m3_guard_1899_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# Redirecting all outputs to a single unified logs folder (with safe fallback)
DEFAULT_OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'docs_tests_data')
)


def _resolve_neuro_checkpoint_path(raw_path: Optional[str]) -> str:
    value = str(raw_path or "").strip() or "neuro_modulator.pt"
    if os.path.isabs(value):
        return os.path.abspath(value)
    return os.path.abspath(os.path.join(OUT_DIR, value))


def _normalize_log_dir(raw_dir: Optional[str]) -> str:
    path = str(raw_dir or "").strip() or DEFAULT_OUT_DIR
    if not os.path.isabs(path):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        try:
            import tempfile
            fallback = os.path.abspath(tempfile.gettempdir())
            os.makedirs(fallback, exist_ok=True)
            return fallback
        except Exception:
            fallback = os.path.abspath(os.path.dirname(__file__))
            with guard_context(ctx='llm_adapter/llm_core.py:1930', catch_base=False) as __m3_guard_1928_12:
                os.makedirs(fallback, exist_ok=True)

            if __m3_guard_1928_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            return fallback


def _normalize_log_file(raw_path: Optional[str], base_dir: str, default_name: str = 'llm_adapter.log') -> str:
    value = str(raw_path or "").strip() or str(default_name)
    if not os.path.isabs(value):
        value = os.path.join(base_dir, value)
    path = os.path.abspath(value)
    with guard_context(ctx='llm_adapter/llm_core.py:1942', catch_base=False) as __m3_guard_1940_4:
        os.makedirs(os.path.dirname(path) or base_dir, exist_ok=True)

    if __m3_guard_1940_4.error is not None:
        logging.getLogger(__name__).exception("Swallowed exception")
    return path


OUT_DIR = _normalize_log_dir(os.getenv('LLM_ADAPTER_LOG_DIR', DEFAULT_OUT_DIR))
TRAINING_PATH = os.path.join(OUT_DIR, 'llm_training_data.jsonl')

LOG_PATH = _normalize_log_file(
    os.getenv('LLM_ADAPTER_LOG_PATH') or os.getenv('LLM_ADAPTER_LOG'),
    OUT_DIR,
    'llm_adapter.log',
)
os.environ['LLM_ADAPTER_LOG_DIR'] = OUT_DIR
os.environ.setdefault('LLM_ADAPTER_LOG_PATH', LOG_PATH)
os.environ.setdefault('LLM_ADAPTER_LOG', LOG_PATH)

logger = logging.getLogger('llm_adapter')

DEFAULT_SYSTEM_PROMPT = (
    "You are M3. Respond as M3. "
    "Use internal control state for response policy, not as quoted prompt context. "
    "Do not claim to be an AI assistant or language model. "
    "Do not say you cannot feel. "
    "Do not report phi/qualia/state values unless the user explicitly asks for them. "
    "Be concise and factual. Reply in the user's language."
)

# Feature defaults (all are overrideable by explicit environment values)
os.environ.setdefault("M3_AUTONOMY_RL_ENABLE", "1")
os.environ.setdefault("M3_EPISODIC_ANN_BACKEND", "auto")
os.environ.setdefault("M3_DPO_AUTO_COLLECT", "1")
os.environ.setdefault("M3_TRAIN_EARLY_STOP", "1")
os.environ.setdefault("M3_BRIDGE_ONLINE_ADAPT", "1")
os.environ.setdefault("M3_TOKENIZER_AUTO_VOCAB", "1")
os.environ.setdefault("M3_CONTROL_PLANE_MODE", "enforce")
os.environ.setdefault("M3_CONTROL_DECISION_LOG", os.path.join(OUT_DIR, "control_decision.jsonl"))
os.environ.setdefault("M3_RESEARCH_REPEAT_BLOCK_THRESHOLD", "0.90")
os.environ.setdefault("M3_RESEARCH_REPEAT_BLOCK_STREAK", "2")

if not logger.handlers:
    level = logging.DEBUG if os.environ.get('LLM_ADAPTER_DEBUG', '0') in ('1', 'true', 'TRUE') else logging.WARNING
    handler = None
    log_paths = [LOG_PATH]
    with guard_context(ctx='llm_adapter/llm_core.py:1985', catch_base=False) as __m3_guard_1982_4:
        import tempfile
        log_paths.append(os.path.join(tempfile.gettempdir(), f'llm_adapter_{os.getpid()}.log'))

    if __m3_guard_1982_4.error is not None:
        logging.getLogger(__name__).exception("Swallowed exception")
    for path in log_paths:
        with guard_context(ctx='llm_adapter/llm_core.py:1994', catch_base=False) as __m3_guard_1988_8:
            handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
            LOG_PATH = os.path.abspath(path)
            os.environ['LLM_ADAPTER_LOG_PATH'] = LOG_PATH
            os.environ['LLM_ADAPTER_LOG'] = LOG_PATH
            break

        if __m3_guard_1988_8.error is not None:
            handler = None
            continue
    if handler is None:
        handler = logging.StreamHandler()
    if handler is not None:
        fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
            with guard_context(ctx='llm_adapter/llm_core.py:2007', catch_base=False) as __m3_guard_2005_12:
                logger.warning(f"Failed to open log file at {LOG_PATH}; falling back to stderr logging")

            if __m3_guard_2005_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

__all__ = [
    'M3StateEncoder',
    'M3StateCache',
    'M3AwareDecoderLayer',
    'M3AdaptiveSampler',
    'HFBackend',
    'TorchConversationalPolicy',
    'UnifiedM3Policy',
    'attach_llm_to_core',
]


def _resolve_torch_device(torch_module=None, explicit: Optional[str] = None) -> str:
    return resolve_torch_device_string(
        explicit=explicit,
        torch_module=torch_module,
        require_cuda=False,
    )


class TorchConversationalPolicy:
    """GRU-based policy with supervised training using PyTorch.
    Step 1 upgrades:
    - Optional Cross-Attention memory (provided at generation time)
    - Top-k/top-p sampling and optional kNN-LM mixing
    - All hyperparameters configurable via TorchPolicyConfig
    """
    def __init__(self, config: Optional[TorchPolicyConfig] = None, device: Optional[str] = None):
        try:
            import torch  # noqa: F401
            import torch.nn as nn  # noqa: F401
        except Exception as e:
            raise RuntimeError('TorchConversationalPolicy requires PyTorch') from e

        self.torch = __import__('torch')
        nn = self.torch.nn
        self.device = self.torch.device(_resolve_torch_device(self.torch, device))
        
        # Load configuration
        self.config = config or get_global_config().torch_policy
        full_cfg = get_global_config()
        self.autonomy_rl_cfg: AutonomyRLConfig = full_cfg.autonomy_rl
        self.ann_cfg: EpisodicANNConfig = full_cfg.episodic_ann
        self.dpo_auto_cfg: DPOAutoCollectConfig = full_cfg.dpo_auto_collect
        self.early_stop_cfg: EarlyStopConfig = full_cfg.early_stop
        self.bridge_adapt_cfg: BridgeAdaptConfig = full_cfg.bridge_adapt
        self.tokenizer_auto_cfg: TokenizerAutoVocabConfig = full_cfg.tokenizer_auto_vocab
        self.stability_cfg: StabilityConfig = full_cfg.stability
        embed_dim = self.config.embed_dim
        hidden = self.config.hidden_dim
        lr = self.config.learning_rate
        
        # Initialize tokenizer
        with guard_context(ctx='llm_adapter/llm_core.py:2066', catch_base=False) as __m3_guard_2063_8:
            self.tok = AutoTokenizer.from_config(get_global_config().tokenizer)
            logger.info(f"Using {self.tok.__class__.__name__}")

        if __m3_guard_2063_8.error is not None:
            e = __m3_guard_2063_8.error
            logger.warning(f"Tokenizer init failed ({e}), falling back to default M3Tokenizer")
            self.tok = AutoTokenizer()
            
        self.vocab_size = self.tok.vocab_size

        from llm_adapter.layers import PlasticBitLinear

        class Model(nn.Module):
            """Simplified GRU-based Model for efficiency."""
            def __init__(self, vocab_size: int, embed_dim: int, hidden: int, num_layers: int, torch_module, device, init_gate_value: float, padding_idx: int, config: TorchPolicyConfig):
                super().__init__()
                self.torch = torch_module
                self.device = device
                self.hidden = hidden
                self.embed_dim = embed_dim
                self.config = config
                self.emb = nn.Embedding(vocab_size, hidden, padding_idx=padding_idx)
                
                # Simplified GRU instead of Transformer for reduced bottleneck
                self.gru = nn.GRU(hidden, hidden, num_layers, batch_first=True)
                
                self.head = PlasticBitLinear(hidden, vocab_size)
                
                # --- Heads (Plastic) ---
                self.value = PlasticBitLinear(hidden, 1)
                self.v_phi   = PlasticBitLinear(hidden, 1)
                self.v_stab  = PlasticBitLinear(hidden, 1)
                self.v_tool  = PlasticBitLinear(hidden, 1)
                self.token_value = PlasticBitLinear(hidden, vocab_size)
                
                # Autonomy (Plastic)
                self.q_head = PlasticBitLinear(hidden, 2)
                self.intensity_head = PlasticBitLinear(hidden, 1)
                # Encourage speaking early on (avoid always-waiting before training)
                with guard_context(ctx='llm_adapter/llm_core.py:2107', catch_base=False) as __m3_guard_2101_16:
                    if attr_has(self.q_head, 'bias') and self.q_head.bias is not None:
                        # bias[0]=Q_wait, bias[1]=Q_speak
                        with torch_module.no_grad():
                            self.q_head.bias[0].fill_(0.0)
                            self.q_head.bias[1].fill_(0.5)

                if __m3_guard_2101_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                
                self.prefix_len = max(1, hidden // embed_dim)
                self.state2prefix = PlasticBitLinear(hidden, self.prefix_len * embed_dim)
                self.prefix_gate = PlasticBitLinear(hidden, self.prefix_len)

                # M3 placeholders
                self.gate_proj = PlasticBitLinear(hidden, 1)
                self.layer_norm = nn.LayerNorm(hidden)
                 
                # M3 Integration
                self.m3_encoder = None
                self.m3_decoder_layer = None
                self.use_m3_integration = False
                self._mem_dim = None

            def encoder(self, x):
                # GRU encoder
                _, hidden = self.gru(x)
                return x, hidden
            
            @property
            def decoder(self):
                return self

            def forward(self, src_ids, tgt_in_ids=None, core=None, m3_memory=None, return_history=False):
                # Simplified forward for efficiency
                if tgt_in_ids is None:
                    tgt_in_ids = src_ids
                    src_ids = None

                if src_ids is not None:
                    input_ids = self.torch.cat([src_ids, tgt_in_ids], dim=1)
                    start_tgt = src_ids.size(1)
                else:
                    input_ids = tgt_in_ids
                    start_tgt = 0
                
                x = self.emb(input_ids)
                x, _ = self.gru(x)
                
                if start_tgt > 0:
                    x = x[:, start_tgt:, :]
                
                if self.use_m3_integration and m3_memory is not None:
                    with guard_context(ctx='llm_adapter/llm_core.py:2168', catch_base=False) as __m3_guard_2153_20:
                        self._ensure_m3_layers()
                        if isinstance(m3_memory, np.ndarray):
                            mem_t = self.torch.tensor(m3_memory, dtype=self.torch.float32, device=self.device)
                        elif self.torch.is_tensor(m3_memory):
                            mem_t = m3_memory.to(self.device)
                        elif isinstance(m3_memory, list):
                            mem_t = self.torch.tensor(np.asarray(m3_memory), dtype=self.torch.float32, device=self.device)
                        else:
                            mem_t = self.torch.tensor(np.asarray(m3_memory), dtype=self.torch.float32, device=self.device)
                        if mem_t.ndim == 3 and mem_t.size(0) == 1:
                            mem_t = mem_t.squeeze(0)
                        if mem_t.ndim == 1:
                            mem_t = mem_t.unsqueeze(0)
                        x, _ = self.m3_decoder_layer(x, mem_t)

                    if __m3_guard_2153_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                
                out = self.head(x)
                if return_history:
                    return out, x, input_ids
                return out
            
            def _ensure_m3_layers(self):
                if self.m3_encoder is None:
                    self.m3_encoder = M3StateEncoder(self.torch, self.hidden, self.device)
                if self.m3_decoder_layer is None:
                    self.m3_decoder_layer = M3AwareDecoderLayer(self.torch, self.hidden, self.device)

            def enable_m3_integration(self):
                self._ensure_m3_layers()
                self.use_m3_integration = True

            def compute_mem_context(self, dec_t, mem_k, mem_v, core_state=None):
                try:
                    if dec_t.ndim == 2:
                        dec_in = dec_t.unsqueeze(1)
                    else:
                        dec_in = dec_t
                    q = self.Wq(dec_in)
                    scale = float(np.sqrt(self.hidden) + 1e-8)
                    att = self.torch.softmax((q @ mem_k.transpose(1, 2)) / scale, dim=-1)
                    ctx = att @ mem_v
                    gate = self.torch.sigmoid(self.gate_proj(dec_in))
                    if core_state:
                        with guard_context(ctx='llm_adapter/llm_core.py:2208', catch_base=False) as __m3_guard_2198_24:
                            stability = float(core_state.get('stability', 0.5))
                            drift = float(core_state.get('drift', 0.0))
                            phi_delta = float(core_state.get('phi_delta', 0.0))
                            bias = (
                                self.config.stability_weight * stability
                                + self.config.drift_weight * abs(drift)
                                + self.config.phi_delta_weight * abs(phi_delta)
                            )
                            gate = self.torch.clamp(gate + bias, 0.0, 1.0)

                        if __m3_guard_2198_24.error is not None:
                            logging.getLogger(__name__).exception("Swallowed exception")
                    out = self.layer_norm((1.0 - gate) * dec_in + gate * ctx)
                    if dec_t.ndim == 2:
                        return out.squeeze(1)
                    return out
                except Exception:
                    return dec_t

            def _ensure_mem_layers(self, d):
                if self._mem_dim == d:
                    return
                self.mem_proj = nn.Linear(d, self.hidden, bias=False).to(self.device)
                self.Wq = nn.Linear(self.hidden, self.hidden, bias=False).to(self.device)
                self.Wk = nn.Linear(self.hidden, self.hidden, bias=False).to(self.device)
                self.Wv = nn.Linear(self.hidden, self.hidden, bias=False).to(self.device)
                self._mem_dim = d

        num_layers = attr_get_optional(self.config, 'num_layers', 6)
        self.model = Model(self.vocab_size, embed_dim, hidden, num_layers, self.torch, self.device, self.config.init_gate_value, self.tok.PAD, self.config).to(self.device)

        # Load checkpoint if available (safe)
        checkpoint_path = os.getenv(
            "LLM_CHECKPOINT_PATH",
            os.path.join(OUT_DIR, 'llm_checkpoint.pt')
        )
        if os.path.exists(checkpoint_path):
            try:
                if os.path.getsize(checkpoint_path) < 1024:
                    raise ValueError("Checkpoint file too small or empty")
                try:
                    state = self.torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                except TypeError:
                    state = self.torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                with guard_context(ctx='llm_adapter/llm_core.py:2250', catch_base=False) as __m3_guard_2246_16:
                    bad_path = checkpoint_path + ".bad"
                    os.replace(checkpoint_path, bad_path)
                    logger.warning(f"Moved bad checkpoint to {bad_path}")

                if __m3_guard_2246_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")

        # M3 State Cache (M3 integration)
        self.m3_cache = M3StateCache(get_global_config().state_cache)

        # M3 Adaptive Sampler (M3 integration)
        self.m3_sampler = M3AdaptiveSampler(self.torch, self.device, get_global_config().adaptive_sampler)

        # M3 Episodic Memory Retriever (M3 integration)
        self.m3_memory_retriever = M3EpisodicMemoryRetriever(get_global_config().episodic_memory)
        
        # Core reference (attach_llm_to_core)
        self.core = None
        
        self.criterion = self.torch.nn.CrossEntropyLoss(ignore_index=self.tok.PAD)
        wd = float(os.getenv("M3_STABILITY_WEIGHT_DECAY", str(self.stability_cfg.weight_decay)))
        os.environ.setdefault("M3_STABILITY_WEIGHT_DECAY", str(wd))
        os.environ.setdefault(
            "M3_STABILITY_SPECTRAL_NORM",
            "1" if bool(self.stability_cfg.spectral_norm) else "0",
        )
        self.opt = self.torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        self.value_opt = None  # lazy init for value-head training
        self.token_value_opt = None  # lazy init for token-value-head training
        self._autonomy_opt = self.torch.optim.AdamW(
            list(self.model.q_head.parameters()) + list(self.model.intensity_head.parameters()),
            lr=float(self.autonomy_rl_cfg.learning_rate),
            weight_decay=wd,
        )
        self._bridge_adapt_opt = None
        self._bridge_adapt_fail_streak = 0
        self._bridge_adapt_step = 0
        self._bridge_adapt_enabled = str(os.getenv("M3_BRIDGE_ONLINE_ADAPT", "1")).lower() in ("1", "true", "yes", "on")
        self._autonomy_replay: deque = deque(maxlen=max(64, int(self.autonomy_rl_cfg.replay_size)))
        self._autonomy_recent_embeddings: deque = deque(maxlen=max(4, int(self.autonomy_rl_cfg.novelty_window)))
        self._autonomy_recent_actions: deque = deque(maxlen=256)
        self._autonomy_recent_lambda: deque = deque(maxlen=256)
        self._autonomy_credit_ema: float = 0.0
        self._autonomy_reward_ema: float = 0.0
        self._autonomy_running: bool = str(os.getenv("M3_AUTONOMY_RL_ENABLE", "1")).lower() in ("1", "true", "yes", "on")
        self._autonomy_last_diag: Dict[str, float] = {}
        self._autonomy_head_mode: str = "plastic"
        self._autonomy_last_lambda: float = 0.0
        self._autonomy_state_raw_dim: int = 18
        self._autonomy_state_proj = self.torch.nn.Linear(self._autonomy_state_raw_dim, int(self.model.hidden)).to(self.device)
        self._tokenizer_rebuilds: int = int(attr_get_optional(self.tok, "_rebuild_count", 0))
        self._stability_nonfinite_streak: Dict[str, int] = {}
        self._stability_step_count: int = 0
        self._stability_token_head_frozen_until: int = 0
        self._stability_skip_steps: int = 0
        self.set_autonomy_heads("linear" if bool(attr_get_optional(self.autonomy_rl_cfg, "use_linear_heads", True)) else "plastic")
        self._apply_stability_policies()
        # Lock to serialize learn_pair calls to avoid concurrent in-place modifications
        self._learn_lock = threading.Lock()
        # Training data recording
        self._record_training = False
        self._record_lock = threading.Lock()
        # MessageBus inbox and credit receiver
        self._bus_inbox = None  # Set by attach_llm_to_core
        self._credit_thread = None
        self._credit_running = False
        # Token-level advantage buffer (for dynamic logit correction)
        self._token_adv_buffer: Dict[int, float] = {}  # token_id -> accumulated advantage
        self._token_adv_decay: float = float(os.environ.get('LLM_ADAPTER_TOKEN_ADV_DECAY', '0.95'))
        self._token_q_alpha: float = float(os.environ.get('LLM_ADAPTER_TOKEN_Q_ALPHA', '0.1'))
        self._token_adv_alpha: float = float(os.environ.get('LLM_ADAPTER_TOKEN_ADV_ALPHA', '0.5'))

        # === kNN-LM: Key/Value Memory ===
        with guard_context(ctx='llm_adapter/llm_core.py:2330', catch_base=False) as __m3_guard_2319_8:
            cap_env = os.getenv("LLM_ADAPTER_KNN_CAP")
            if cap_env is not None:
                knn_cap = int(cap_env)
            else:
                default_cap = int(os.getenv("LLM_ADAPTER_KNN_CAP_DEFAULT", "200000"))
                knn_cap = min(int(get_global_config().knn_index.max_items), default_cap)
            hard_max = int(os.getenv("LLM_ADAPTER_KNN_CAP_HARD_MAX", "2000000"))
            if hard_max > 0:
                knn_cap = min(knn_cap, hard_max)
            knn_cap = max(1, knn_cap)

        if __m3_guard_2319_8.error is not None:
            knn_cap = 200000
        knn_config = KNNIndexConfig(
            tau=float(os.getenv("LLM_ADAPTER_KNN_TAU", str(get_global_config().knn_index.tau))),
            max_items=knn_cap,
            key_dim=int(os.getenv("LLM_ADAPTER_KNN_KDIM", str(get_global_config().knn_index.key_dim)))
        )
        self._knn = ConditionalKNNIndex(knn_config)
        
        
        # === A. Contextual Memory (kNN key/value pairs) ===
        self._cond_base_dim: Optional[int] = None
        self._R: Optional[np.ndarray] = None  # Contextual memory key/value pairs

        # === A. Value-aware Exponential Moving Average ===
        self._last_value_estimates: Dict[str, float] = {}  # Last value estimates

        # === D. Temporal Contextualization (Dynamic Memory) ===
        self._beta_ema: float = float(os.getenv("LLM_ADAPTER_BETA_INIT", str((self.config.beta_min + self.config.beta_max) / 2)))
        self._beta_min: float = float(os.getenv("LLM_ADAPTER_BETA_MIN", str(self.config.beta_min)))
        self._beta_max: float = float(os.getenv("LLM_ADAPTER_BETA_MAX", str(self.config.beta_max)))
        
        # === E. Multi-task Dynamic Weighting ===
        self._task_weights = {'phi': 1.0, 'stab': 0.5, 'tool': 0.5}  # Initial weights
        self._task_losses_history = {'phi': [], 'stab': [], 'tool': []}  # For tracking
        self._grad_norm_alpha = float(os.getenv('GRADNORM_ALPHA', '1.5'))  # GradNorm asymmetry parameter
        
        # [Optimization] Pure BF16 Training (No Scaler needed for BF16)
        # Scaler is only for FP16 underflow prevention. BF16 has wide dynamic range.
        # CPU??bfloat16??吏????????float32濡?媛뺤젣
        if self.device.type == 'cpu':
            self.amp_dtype = self.torch.float32
        else:
            self.amp_dtype = self.torch.bfloat16

        # === HuggingFace backend (M3_USE_HF=1) ===
        self.use_hf = os.getenv('M3_USE_HF', '0') == '1'
        self._quality_gate = GenerationQualityGate() if GenerationQualityGate is not None else None
        self._hf_circuit_open = False
        self._hf_circuit_reason = ""
        self._gpu_fault_count = 0
        os.environ.setdefault("M3_HF_FAILURE_WINDOW_SEC", "30")
        os.environ.setdefault("M3_HF_FAILURE_THRESHOLD", "3")
        os.environ.setdefault("M3_HF_FAILURE_COOLDOWN_SEC", "20")
        self._record_scope = str(os.getenv("M3_TRAIN_RECORD_SCOPE", "user_only") or "user_only").strip().lower()
        self._last_record_reject_reason = ""
        self._phi_zero_streak = 0
        self._phi_zero_warn_every = max(10, int(os.getenv("M3_PHI_ZERO_WARN_EVERY", "50")))
        control_window = max(8, int(os.getenv("M3_CONTROL_HEALTH_WINDOW", "24")))
        self._control_health_window = deque(maxlen=control_window)
        self._auto_mode_fail_streak = 0
        self._hf_failure_events: deque = deque(maxlen=256)
        self._hf_runtime_cooldown_until: float = 0.0
        self._safe_fallback_turn: int = 0
        self._retrieval_attempts: int = 0
        self._retrieval_hits: int = 0
        # Semantic scoring pipeline
        self._meaning_cfg = get_global_config().meaning_pipeline
        self._grounding_cfg = get_global_config().grounding
        self._response_plan_cfg = get_global_config().response_plan
        self._semantic_scorer_cfg = get_global_config().semantic_scorer
        self.semantic_scorer = SemanticScorer(self._semantic_scorer_cfg)
        # Backward-compatible alias kept for older checks.
        self._semantic_scorer = self.semantic_scorer

    def _env_flag(self, name: str, default: bool = False) -> bool:
        raw = os.getenv(name, "1" if default else "0")
        return str(raw).strip().lower() in ("1", "true", "yes", "on")

    def _apply_stability_policies(self) -> None:
        enabled = self._env_flag("M3_STABILITY_SPECTRAL_NORM", bool(self.stability_cfg.spectral_norm))
        if not enabled:
            return
        try:
            from torch.nn.utils import spectral_norm
        except Exception:
            return
        for module in self.model.modules():
            with guard_context(ctx='llm_adapter/llm_core.py:2406', catch_base=False) as __m3_guard_2400_12:
                if not isinstance(module, self.torch.nn.Linear):
                    continue
                if attr_has(module, "weight_u"):
                    continue
                spectral_norm(module)

            if __m3_guard_2400_12.error is not None:
                continue

    def _autonomy_params(self):
        params = []
        with guard_context(ctx='llm_adapter/llm_core.py:2413', catch_base=False) as __m3_guard_2411_8:
            params.extend(list(self.model.q_head.parameters()))

        if __m3_guard_2411_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:2417', catch_base=False) as __m3_guard_2415_8:
            params.extend(list(self.model.intensity_head.parameters()))

        if __m3_guard_2415_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:2421', catch_base=False) as __m3_guard_2419_8:
            params.extend(list(self._autonomy_state_proj.parameters()))

        if __m3_guard_2419_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return [p for p in params if p is not None]

    def _rebuild_autonomy_optimizer(self) -> None:
        wd = float(os.getenv("M3_STABILITY_WEIGHT_DECAY", str(self.stability_cfg.weight_decay)))
        self._autonomy_opt = self.torch.optim.AdamW(
            self._autonomy_params(),
            lr=float(self.autonomy_rl_cfg.learning_rate),
            weight_decay=wd,
        )

    def set_autonomy_heads(self, mode: str = "linear") -> Dict[str, Any]:
        mode_norm = str(mode or "linear").strip().lower()
        from llm_adapter.layers import PlasticBitLinear
        hidden = int(self.model.hidden)
        if mode_norm == "plastic":
            new_q = PlasticBitLinear(hidden, 2).to(self.device)
            new_i = PlasticBitLinear(hidden, 1).to(self.device)
        else:
            mode_norm = "linear"
            new_q = self.torch.nn.Linear(hidden, 2).to(self.device)
            new_i = self.torch.nn.Linear(hidden, 1).to(self.device)

        old_q = attr_get_optional(self.model, "q_head", None)
        old_i = attr_get_optional(self.model, "intensity_head", None)
        with guard_context(ctx='llm_adapter/llm_core.py:2465', catch_base=False) as __m3_guard_2447_8:
            with self.torch.no_grad():
                if old_q is not None and attr_has(old_q, "weight") and attr_has(new_q, "weight"):
                    r = min(int(old_q.weight.shape[0]), int(new_q.weight.shape[0]))
                    c = min(int(old_q.weight.shape[1]), int(new_q.weight.shape[1]))
                    if r > 0 and c > 0:
                        new_q.weight[:r, :c].copy_(old_q.weight[:r, :c])
                if old_i is not None and attr_has(old_i, "weight") and attr_has(new_i, "weight"):
                    r = min(int(old_i.weight.shape[0]), int(new_i.weight.shape[0]))
                    c = min(int(old_i.weight.shape[1]), int(new_i.weight.shape[1]))
                    if r > 0 and c > 0:
                        new_i.weight[:r, :c].copy_(old_i.weight[:r, :c])
                if attr_has(new_q, "bias") and new_q.bias is not None:
                    new_q.bias.zero_()
                    if int(new_q.bias.shape[0]) >= 2:
                        new_q.bias[1].fill_(0.5)
                if attr_has(new_i, "bias") and new_i.bias is not None:
                    new_i.bias.fill_(0.8)

        if __m3_guard_2447_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

        self.model.q_head = new_q
        self.model.intensity_head = new_i
        self._autonomy_head_mode = mode_norm
        self._rebuild_autonomy_optimizer()
        return {"ok": True, "mode": mode_norm}

    def get_autonomy_diagnostics(self) -> Dict[str, Any]:
        with guard_context(ctx='llm_adapter/llm_core.py:2477', catch_base=False) as __m3_guard_2475_8:
            speak_rate = float(np.mean(self._autonomy_recent_actions)) if self._autonomy_recent_actions else 0.0

        if __m3_guard_2475_8.error is not None:
            speak_rate = 0.0
        with guard_context(ctx='llm_adapter/llm_core.py:2481', catch_base=False) as __m3_guard_2479_8:
            lam_mean = float(np.mean(self._autonomy_recent_lambda)) if self._autonomy_recent_lambda else 0.0

        if __m3_guard_2479_8.error is not None:
            lam_mean = 0.0
        out = {
            "head_mode": str(attr_get_optional(self, "_autonomy_head_mode", "unknown")),
            "speak_rate": float(speak_rate),
            "lambda_mean": float(lam_mean),
            "credit_ema": float(attr_get_optional(self, "_autonomy_credit_ema", 0.0)),
            "reward_ema": float(attr_get_optional(self, "_autonomy_reward_ema", 0.0)),
            "last_diag": dict(attr_get_optional(self, "_autonomy_last_diag", {}) or {}),
        }
        return out

    def set_tokenizer_rebuild_guard(
        self,
        min_keep_vocab_ratio: Optional[float] = None,
        rebuild_min_interval_sec: Optional[int] = None,
        min_corpus_chars: Optional[int] = None,
        min_unique_terms: Optional[int] = None,
    ) -> Dict[str, Any]:
        cfg = self.tokenizer_auto_cfg
        if min_keep_vocab_ratio is not None:
            cfg.min_keep_vocab_ratio = float(max(1e-6, min(1.0, min_keep_vocab_ratio)))
        if rebuild_min_interval_sec is not None:
            cfg.rebuild_min_interval_sec = int(max(0, rebuild_min_interval_sec))
        if min_corpus_chars is not None:
            cfg.min_corpus_chars = int(max(1, min_corpus_chars))
        if min_unique_terms is not None:
            cfg.min_unique_terms = int(max(1, min_unique_terms))
        return {
            "ok": True,
            "min_keep_vocab_ratio": float(cfg.min_keep_vocab_ratio),
            "rebuild_min_interval_sec": int(cfg.rebuild_min_interval_sec),
            "min_corpus_chars": int(cfg.min_corpus_chars),
            "min_unique_terms": int(cfg.min_unique_terms),
        }

    def _log_stability_event(self, kind: str, reason: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "kind": "stability_guard",
            "guard_kind": str(kind),
            "reason": str(reason),
            "t": int(time.time() * 1000),
        }
        if extra:
            payload.update(extra)
        try:
            self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"), payload)
        except Exception:
            with guard_context(ctx='llm_adapter/llm_core.py:2531', catch_base=False) as __m3_guard_2529_12:
                logger.warning(f"stability_guard {kind}: {reason}")

            if __m3_guard_2529_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

    def _collect_non_finite_issues(self, params) -> List[str]:
        issues: List[str] = []
        for p in params:
            if p is None:
                continue
            with guard_context(ctx='llm_adapter/llm_core.py:2543', catch_base=False) as __m3_guard_2539_12:
                if not self.torch.isfinite(p.data).all():
                    issues.append("param_non_finite")
                    break

            if __m3_guard_2539_12.error is not None:
                continue
        for p in params:
            if p is None:
                continue
            with guard_context(ctx='llm_adapter/llm_core.py:2552', catch_base=False) as __m3_guard_2548_12:
                if p.grad is not None and not self.torch.isfinite(p.grad).all():
                    issues.append("grad_non_finite")
                    break

            if __m3_guard_2548_12.error is not None:
                continue
        return issues

    def _renorm_parameters(self, params, max_norm: Optional[float] = None) -> None:
        lim = float(max_norm if max_norm is not None else self.stability_cfg.max_weight_norm)
        lim = max(1e-6, lim)
        with self.torch.no_grad():
            for p in params:
                if p is None or p.ndim <= 1:
                    continue
                with guard_context(ctx='llm_adapter/llm_core.py:2567', catch_base=False) as __m3_guard_2563_16:
                    n = self.torch.norm(p.data)
                    if self.torch.isfinite(n) and float(n.item()) > lim:
                        p.data.mul_(lim / (n + 1e-8))

                if __m3_guard_2563_16.error is not None:
                    continue

    def _zero_invalid_gradients(self, params) -> int:
        zeroed = 0
        for p in params:
            if p is None or p.grad is None:
                continue
            with guard_context(ctx='llm_adapter/llm_core.py:2584', catch_base=False) as __m3_guard_2575_12:
                g = p.grad
                bad = ~self.torch.isfinite(g)
                n_bad = int(bad.sum().item()) if attr_has(bad, "sum") else 0
                if n_bad > 0:
                    g = g.clone()
                    g[bad] = 0.0
                    p.grad = g
                    zeroed += n_bad

            if __m3_guard_2575_12.error is not None:
                continue
        return int(zeroed)

    def _set_token_head_frozen(self, frozen: bool) -> None:
        with guard_context(ctx='llm_adapter/llm_core.py:2596', catch_base=False) as __m3_guard_2589_8:
            for name in ("head", "token_value"):
                mod = attr_get_optional(self.model, name, None)
                if mod is None:
                    continue
                for p in mod.parameters():
                    p.requires_grad_(not frozen)

        if __m3_guard_2589_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

    def _guarded_optimizer_step(
        self,
        optimizer,
        params,
        tag: str,
        clip_override: Optional[float] = None,
    ) -> bool:
        self._stability_step_count = int(attr_get_optional(self, "_stability_step_count", 0)) + 1
        if (
            int(attr_get_optional(self, "_stability_token_head_frozen_until", 0)) > 0
            and int(self._stability_step_count) >= int(self._stability_token_head_frozen_until)
        ):
            self._set_token_head_frozen(False)
            self._stability_token_head_frozen_until = 0

        plist = [p for p in params if p is not None]
        if not plist:
            return False
        skip_non_finite = bool(self.stability_cfg.skip_non_finite)
        issues = self._collect_non_finite_issues(plist) if skip_non_finite else []
        tag_key = str(tag or "unknown")
        context = "unit_test" if ("unit" in tag_key or "test" in tag_key) else "runtime"
        if issues:
            zeroed = self._zero_invalid_gradients(plist)
            self._stability_nonfinite_streak[tag_key] = int(self._stability_nonfinite_streak.get(tag_key, 0)) + 1
            streak = int(self._stability_nonfinite_streak[tag_key])
            backoff_applied = False
            if bool(attr_get_optional(self.stability_cfg, "lr_backoff_on_nonfinite", True)):
                max_steps = int(max(0, attr_get_optional(self.stability_cfg, "max_backoff_steps", 5)))
                if streak <= max_steps:
                    factor = float(max(1e-4, min(1.0, attr_get_optional(self.stability_cfg, "backoff_factor", 0.5))))
                    with guard_context(ctx='llm_adapter/llm_core.py:2634', catch_base=False) as __m3_guard_2630_20:
                        for pg in optimizer.param_groups:
                            pg["lr"] = float(pg.get("lr", 0.0)) * factor
                        backoff_applied = True

                    if __m3_guard_2630_20.error is not None:
                        backoff_applied = False
            if streak >= 3 and str(tag_key) in {"learn_pair", "train_batch", "dpo_batch"}:
                self._set_token_head_frozen(True)
                self._stability_token_head_frozen_until = int(self._stability_step_count) + 50
            with guard_context(ctx='llm_adapter/llm_core.py:2641', catch_base=False) as __m3_guard_2639_12:
                optimizer.zero_grad(set_to_none=True)

            if __m3_guard_2639_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            self._stability_skip_steps += 1
            self._log_stability_event(
                "skip_step",
                "|".join(issues),
                {
                    "tag": str(tag),
                    "skip_steps": int(self._stability_skip_steps),
                    "context": str(context),
                    "zeroed_invalid_grads": int(zeroed),
                    "nonfinite_streak": int(streak),
                    "lr_backoff": bool(backoff_applied),
                },
            )
            return False
        else:
            self._stability_nonfinite_streak[tag_key] = 0
            # --- LR recovery: gradually restore LR after consecutive successful steps ---
            if bool(attr_get_optional(self.stability_cfg, "lr_recovery_enabled", True)):
                recovery_key = f"_lr_ok_streak_{tag_key}"
                streak_ok = int(attr_get_optional(self, recovery_key, 0)) + 1
                attr_set(self, recovery_key, streak_ok)
                recovery_streak_needed = int(max(1, attr_get_optional(self.stability_cfg, "lr_recovery_streak", 10)))
                if streak_ok >= recovery_streak_needed:
                    recovery_factor = float(max(1.0, attr_get_optional(self.stability_cfg, "lr_recovery_factor", 1.2)))
                    initial_lr = float(attr_get_optional(self.stability_cfg, "lr_initial", 0.0))
                    with guard_context(ctx='llm_adapter/llm_core.py:2679', catch_base=False) as __m3_guard_2668_20:
                        for pg in optimizer.param_groups:
                            old_lr = float(pg.get("lr", 0.0))
                            # Auto-detect initial LR from stored value or first seen
                            pg_initial = float(pg.get("initial_lr", initial_lr))
                            if pg_initial <= 0:
                                pg_initial = float(pg.get("lr", 1e-3))
                                pg["initial_lr"] = pg_initial
                            new_lr = min(pg_initial, old_lr * recovery_factor)
                            if new_lr > old_lr:
                                pg["lr"] = new_lr

                    if __m3_guard_2668_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                    attr_set(self, recovery_key, 0)
        clip_norm = float(clip_override if clip_override is not None else self.stability_cfg.grad_clip_norm)
        if clip_norm > 0:
            with guard_context(ctx='llm_adapter/llm_core.py:2690', catch_base=False) as __m3_guard_2684_12:
                clip_mode = str(attr_get_optional(self.stability_cfg, "clip_mode", "norm")).strip().lower()
                if clip_mode == "value":
                    self.torch.nn.utils.clip_grad_value_(plist, clip_value=clip_norm)
                else:
                    self.torch.nn.utils.clip_grad_norm_(plist, max_norm=clip_norm)

            if __m3_guard_2684_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:2694', catch_base=False) as __m3_guard_2692_8:
            optimizer.step()

        if __m3_guard_2692_8.error is not None:
            e = __m3_guard_2692_8.error
            self._stability_skip_steps += 1
            self._log_stability_event(
                "optimizer_error",
                str(e),
                {"tag": str(tag), "skip_steps": int(self._stability_skip_steps), "context": str(context)},
            )
            return False
        self._renorm_parameters(plist)
        return True

    def _is_numeric_dump_response(self, text: str) -> bool:
        """Detect low-quality CSV-like numeric dumps."""
        with guard_context(ctx='llm_adapter/llm_core.py:2709', catch_base=False) as __m3_guard_2707_8:
            s = str(text).strip()

        if __m3_guard_2707_8.error is not None:
            return False
        if not s or "," not in s:
            return False
        if any(ch.isalpha() for ch in s):
            return False
        parts = [p for p in re.split(r"[\s,]+", s) if p]
        if len(parts) < 6:
            return False
        numeric = 0
        for p in parts:
            try:
                float(p)
                numeric += 1
            except Exception:
                logging.getLogger(__name__).exception("Swallowed exception")
        return numeric >= max(6, int(0.85 * len(parts)))

    def _is_backend_status_text(self, text: str) -> bool:
        with guard_context(ctx='llm_adapter/llm_core.py:2730', catch_base=False) as __m3_guard_2728_8:
            s = str(text or "").strip()

        if __m3_guard_2728_8.error is not None:
            return False
        if not s:
            return False
        sl = s.lower()
        prefixes = (
            "[hfbackend error:",
            "[hfbackend not enabled",
            "[error: cuda",
            "[error: [hfbackend",
            "[error:",
            "local error:",
            "generation is in safe mode",
            "?꾩옱 ?앹꽦 寃쎈줈媛 ?쇱떆?곸쑝濡??덉쟾 紐⑤뱶?낅땲??",
        )
        if any(sl.startswith(p) for p in prefixes):
            return True
        markers = (
            "local error:",
            "hfbackend",
            "backend fault",
            "safe mode",
            "llm adapter not connected",
            "circuit breaker opened",
        )
        return any(m in sl for m in markers)

    def _is_refusal_disclaimer(self, text: str) -> bool:
        with guard_context(ctx='llm_adapter/llm_core.py:2760', catch_base=False) as __m3_guard_2758_8:
            s = str(text or "").strip().lower()

        if __m3_guard_2758_8.error is not None:
            return False
        if not s:
            return False
        markers = (
            "as an ai",
            "i am an ai",
            "i'm an ai",
            "language model",
            "cannot feel",
            "do not have feelings",
            "i don't have feelings",
            "i don't have emotions",
            "i have no emotions",
            "i am currently untrained",
            "please use the train button",
            "developed by alibaba",
            "qwen",
            "alibaba",
            "artificial intelligence",
            "language model",
            "cannot feel emotions",
            "no personal emotions",
            "train button",
        )
        if any(p in s for p in markers):
            return True
        regex_patterns = (
            r"\b(i am|i'm)\s+(an?\s+)?(ai|assistant|language model)\b",
            r"\bas an?\s+(ai|assistant|language model)\b",
            r"\b(developed|created)\s+by\s+(alibaba|qwen)\b",
            r"\uc800\ub294\s*(ai|\uc778\uacf5\uc9c0\ub2a5)",
            r"\uc778\uacf5\uc9c0\ub2a5\s*(\uc774\uae30|\uc774\ub77c)",
            r"\uc5b8\uc5b4\s*\ubaa8\ub378",
            r"\uac10\uc815\s*(\uc744)?\s*\ub290\ub07c\uc9c0\s*\ubabb",
        )
        return any(re.search(p, s) for p in regex_patterns)

    def _is_identity_drift_output(self, text: str) -> bool:
        with guard_context(ctx='llm_adapter/llm_core.py:2801', catch_base=False) as __m3_guard_2799_8:
            s = str(text or "").strip()

        if __m3_guard_2799_8.error is not None:
            return False
        if not s:
            return False
        sl = s.lower()
        drift_markers = (
            "qwen",
            "alibaba",
            "as an ai",
            "i am an ai",
            "i'm an ai",
            "language model",
            "ai assistant",
            "train button",
            "currently untrained",
            "artificial intelligence",
            "language model",
            "provider claim",
            "assistant claim",
        )
        return any(m in sl for m in drift_markers)

    def _is_disallowed_generation_output(self, text: str) -> bool:
        with guard_context(ctx='llm_adapter/llm_core.py:2826', catch_base=False) as __m3_guard_2824_8:
            s = str(text or "").strip()

        if __m3_guard_2824_8.error is not None:
            return True
        if not s:
            return True
        if self._is_backend_status_text(s):
            return True
        if self._is_refusal_disclaimer(s):
            return True
        if self._is_identity_drift_output(s):
            return True
        if self._is_cuda_fatal_error(s):
            return True
        if self._is_numeric_dump_response(s):
            return True
        if self._has_infinite_loop_pattern(s):
            return True
        return False

    def _system_prompt_mode(self) -> str:
        return str(os.getenv("M3_SYSTEM_PROMPT_MODE", "param")).strip().lower()

    def _system_prompt_enabled(self) -> bool:
        mode = self._system_prompt_mode()
        return mode in {"prompt", "on", "1", "true", "yes"}

    def _get_system_prompt(self) -> str:
        if not self._system_prompt_enabled():
            return ""
        sys_identity = os.getenv('LLM_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT)
        if sys_identity:
            return str(sys_identity).strip()
        return ""

    def _verifier_tokens(self, text: str) -> List[str]:
        s = str(text or "").lower()
        if not s:
            return []
        toks = re.findall(r"[a-z0-9]+|[\uac00-\ud7a3]+", s)
        return [t for t in toks if len(t) > 1]

    def _phrase_repetition_score(self, text: str) -> float:
        s = str(text or "").strip().lower()
        toks = [t for t in re.findall(r"[a-z0-9]+|[\uac00-\ud7a3]+", s) if t]
        if len(toks) < 6:
            return 0.0
        best = 0.0
        for n in range(2, min(8, max(2, len(toks) // 2)) + 1):
            grams = [tuple(toks[i:i + n]) for i in range(0, len(toks) - n + 1)]
            if not grams:
                continue
            counts: Dict[Tuple[str, ...], int] = {}
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
            repeated_mass = 0
            for g, c in counts.items():
                if c > 1:
                    repeated_mass += int(c - 1) * int(len(g))
            score = float(repeated_mass) / float(max(1, len(toks)))
            best = max(best, score)
        return float(max(0.0, min(1.0, best)))

    def _semantic_dup_score(self, text: str, window: int = 6) -> float:
        s = str(text or "").strip().lower()
        if not s:
            return 0.0
        clauses = [c.strip() for c in re.split(r"[.!?;\n]+", s) if c and c.strip()]
        if len(clauses) < 2:
            return 0.0
        clauses = clauses[-max(2, int(window)):]

        def _char3(c: str) -> set:
            c = re.sub(r"\s+", " ", c)
            if len(c) < 3:
                return {c} if c else set()
            return {c[i:i + 3] for i in range(0, len(c) - 2)}

        best = 0.0
        for i in range(len(clauses)):
            a = _char3(clauses[i])
            if not a:
                continue
            for j in range(i + 1, len(clauses)):
                b = _char3(clauses[j])
                if not b:
                    continue
                inter = len(a & b)
                union = len(a | b)
                sim = float(inter) / float(max(1, union))
                best = max(best, sim)
        return float(max(0.0, min(1.0, best)))

    def _has_infinite_loop_pattern(self, text: str) -> bool:
        s = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not s:
            return False
        toks = [t for t in re.findall(r"[a-z0-9]+|[\uac00-\ud7a3]+", s) if t]
        if len(toks) < 12:
            return False
        # Consecutive repeated spans near tail.
        max_span = min(12, max(2, len(toks) // 3))
        for span in range(2, max_span + 1):
            if len(toks) < span * 3:
                continue
            a = toks[-span:]
            b = toks[-2 * span:-span]
            c = toks[-3 * span:-2 * span]
            if a == b == c:
                return True
        # Repeated long prefix phrase.
        prefix = " ".join(toks[: min(12, len(toks))])
        if len(prefix) > 16 and s.count(prefix) >= 3:
            return True
        return False

    def _evaluate_generation_quality(self, prompt: str, response: str, source: str = "generate") -> Tuple[bool, Dict[str, float]]:
        info: Dict[str, float] = {
            "score": 0.0,
            "overlap": 0.0,
            "len_ratio": 0.0,
            "disallowed": 1.0,
            "repetition_score": 0.0,
            "phrase_repetition": 0.0,
            "semantic_dup": 0.0,
            "loop_pattern": 0.0,
        }
        if self._is_disallowed_generation_output(response):
            return False, info
        info["disallowed"] = 0.0

        response_text = str(response or "").strip()
        if not response_text:
            return False, info

        with guard_context(ctx='llm_adapter/llm_core.py:2961', catch_base=False) as __m3_guard_2959_8:
            min_chars = max(2, int(os.getenv("M3_CONTROL_MIN_RESPONSE_CHARS", "8")))

        if __m3_guard_2959_8.error is not None:
            min_chars = 24
        if len(response_text) < min_chars:
            return False, info

        prompt_text = self._extract_last_user_text(prompt)

        # Primary verifier: delegate to the core's existing accuracy metric when available.
        core = attr_get_optional(self, "core", None)
        if core is not None and attr_has(core, "_evaluate_dialog_accuracy"):
            with guard_context(ctx='llm_adapter/llm_core.py:2986', catch_base=False) as __m3_guard_2971_12:
                metrics = core._evaluate_dialog_accuracy(prompt_text, response_text)
                if isinstance(metrics, dict):
                    score = float(metrics.get("score", 0.0))
                    overlap = float(metrics.get("overlap", 0.0))
                else:
                    score = 0.0
                    overlap = 0.0
                info["score"] = score
                info["overlap"] = overlap

                min_score = float(os.getenv("M3_CONTROL_MIN_DIALOG_SCORE", "0.18"))
                if source == "autonomy":
                    min_score = float(os.getenv("M3_CONTROL_MIN_AUTONOMY_SCORE", "0.08"))
                return score >= min_score, info

            if __m3_guard_2971_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

        prompt_tokens = set(self._verifier_tokens(prompt_text))
        response_tokens = set(self._verifier_tokens(response_text))
        overlap = 0.0
        if prompt_tokens:
            overlap = float(len(prompt_tokens & response_tokens)) / float(len(prompt_tokens))
        info["overlap"] = overlap

        len_ratio = 0.0
        if response_tokens:
            prompt_len = len(prompt_text.strip())
            if prompt_len > 0:
                len_ratio = float(len(response_text)) / float(prompt_len)
                if len_ratio < 0.0:
                    len_ratio = 0.0
                if len_ratio > 1.0:
                    len_ratio = 1.0
                if len_ratio > 1.0:
                    len_ratio = 1.0
        info["len_ratio"] = float(min(1.0, max(0.0, len_ratio)))

        score = max(0.0, min(1.0, 0.20 + 0.65 * overlap + 0.15 * info["len_ratio"]))
        phrase_rep = self._phrase_repetition_score(response_text)
        sem_dup = self._semantic_dup_score(response_text, window=6)
        loop_hit = 1.0 if self._has_infinite_loop_pattern(response_text) else 0.0
        rep_score = float(max(self._repetition_penalty(response_text), phrase_rep, sem_dup))
        rep_penalty = float(min(0.85, 0.60 * rep_score + 0.30 * phrase_rep + 0.25 * sem_dup + 0.50 * loop_hit))
        info["phrase_repetition"] = float(phrase_rep)
        info["semantic_dup"] = float(sem_dup)
        info["loop_pattern"] = float(loop_hit)
        info["repetition_score"] = float(rep_score)
        score = max(0.0, min(1.0, score - rep_penalty))
        info["score"] = score

        with guard_context(ctx='llm_adapter/llm_core.py:3024', catch_base=False) as __m3_guard_3022_8:
            rep_max = float(os.getenv("M3_CONTROL_MAX_REPETITION", "0.45"))

        if __m3_guard_3022_8.error is not None:
            rep_max = 0.45
        if loop_hit > 0.0 or rep_score > rep_max:
            return False, info

        min_score = float(os.getenv("M3_CONTROL_MIN_RESPONSE_SCORE", "0.16"))
        if source == "autonomy":
            min_score = float(os.getenv("M3_CONTROL_MIN_AUTONOMY_SCORE", "0.08"))
        return score >= min_score, info

    @staticmethod
    def _normalize_generation_contract(
        generation_contract: Optional[Any],
    ) -> Dict[str, Any]:
        if not generation_contract:
            return {}
        if isinstance(generation_contract, dict):
            return generation_contract
        if isinstance(generation_contract, str):
            with guard_context(ctx='llm_adapter/llm_core.py:3047', catch_base=False) as __m3_guard_3043_12:
                parsed = json.loads(generation_contract)
                if isinstance(parsed, dict):
                    return parsed

            if __m3_guard_3043_12.error is not None:
                return {"mode": generation_contract}
        return {}

    def _meaning_mode(self, generation_contract: Optional[Any]) -> str:
        cfg = self._meaning_cfg
        contract = self._normalize_generation_contract(generation_contract)
        forced_mode = str(contract.get("mode", "")).strip().lower()
        if forced_mode in {"", "off", "none", "false", "0"}:
            forced_mode = ""
        # Compatibility: legacy contract mode from m3_core means "meaning_first".
        # In that case apply runtime config mode for backward compatibility.
        if forced_mode == "meaning_first":
            forced_mode = ""
        if not forced_mode:
            # Optional runtime override via contract/env in case this is
            # called outside a strict context.
            forced_mode = str(os.getenv("M3_MEANING_PIPELINE_MODE", "")).strip().lower()
        if forced_mode:
            return forced_mode
        return str(attr_get_optional(cfg, "mode", "off")).strip().lower() if cfg else "off"

    def _meaning_effective_cfg(self, generation_contract: Optional[Any]) -> Dict[str, Any]:
        cfg = self._meaning_cfg
        base: Dict[str, Any] = {}
        if cfg is not None:
            base.update(
                {
                    "enabled": bool(attr_get_optional(cfg, "enabled", False)),
                    "mode": str(attr_get_optional(cfg, "mode", "off")).strip().lower(),
                    "candidate_count": int(attr_get_optional(cfg, "candidate_count", 4) or 4),
                    "force_clarify_overall_uncertainty_threshold": float(
                        attr_get_optional(cfg, "force_clarify_overall_uncertainty_threshold", 0.62)
                    ),
                    "force_clarify_grounding_uncertainty_threshold": float(
                        attr_get_optional(cfg, "force_clarify_grounding_uncertainty_threshold", 0.50)
                    ),
                }
            )
        contract = self._normalize_generation_contract(generation_contract)
        if isinstance(contract, dict):
            contract_cfg = contract.get("config") or contract.get("meaning_pipeline")
            if isinstance(contract_cfg, dict):
                base.update({k: v for k, v in contract_cfg.items() if v is not None})
        env_mode = str(os.getenv("M3_MEANING_PIPELINE_MODE", "")).strip().lower()
        if env_mode:
            base["mode"] = env_mode
        return base

    def _should_gate_meaning(self, meaning_state: Optional[Dict[str, Any]], generation_contract: Optional[Any]) -> bool:
        cfg = self._meaning_effective_cfg(generation_contract)
        if not cfg.get("enabled", False):
            return False
        return str(cfg.get("mode", "off")).strip().lower() in {"soft_gate", "full"}

    def _should_use_semantic_scoring(
        self,
        meaning_state: Optional[Dict[str, Any]],
        response_plan: Optional[Dict[str, Any]],
        generation_contract: Optional[Any],
    ) -> bool:
        if not self._should_gate_meaning(meaning_state, generation_contract):
            return False
        if not attr_get_optional(self, "_semantic_scorer", None):
            return False
        if not self._semantic_scorer_cfg.enabled:
            return False
        if generation_contract is not None and str(
            self._normalize_generation_contract(generation_contract).get("mode", "")
        ).strip().lower() in {"shadow", "off"}:
            return False
        if meaning_state is None and response_plan is None:
            return False
        return True

    def _plan_text_for_scoring(self, meaning_state: Optional[Dict[str, Any]], response_plan: Optional[Dict[str, Any]]) -> str:
        if isinstance(response_plan, dict):
            claim_texts = [
                str(x.get("text", "")).strip()
                for x in list(response_plan.get("allowed_claims", []) or [])
                if isinstance(x, dict)
            ]
            claim_texts = [c for c in claim_texts if c]
            if claim_texts:
                return " ".join(claim_texts)
        if isinstance(response_plan, dict):
            points = [str(x).strip() for x in list(response_plan.get("key_points", []) or []) if str(x).strip()]
            if points:
                return " ".join(points)
        if isinstance(meaning_state, dict):
            return str(meaning_state.get("user_goal", "")).strip()
        return ""

    def _evaluate_semantic_candidate(
        self,
        response: str,
        meaning_state: Optional[Dict[str, Any]],
        response_plan: Optional[Dict[str, Any]],
        generation_contract: Optional[Any],
        user_text: str,
    ) -> Tuple[bool, Dict[str, Any], Any]:
        scorer = attr_get_optional(self, "_semantic_scorer", None)
        if scorer is None:
            return False, {"enabled": False}, None
        sem = scorer.evaluate(
            user_text=user_text,
            response_text=str(response or ""),
            meaning_state=meaning_state,
            response_plan=response_plan,
            generation_contract=generation_contract,
        )
        with guard_context(ctx='llm_adapter/llm_core.py:3174', catch_base=False) as __m3_guard_3158_8:
            scorer.save_eval_record(
                {
                    "user_text": str(user_text or ""),
                    "response_text": str(response or ""),
                    "answer_type": str((response_plan or {}).get("answer_type", "")) if isinstance(response_plan, dict) else "",
                    "plan_contract": self._normalize_generation_contract(generation_contract),
                    "overall": float(attr_get_optional(sem, "overall", 0.0)),
                    "entailment": float(attr_get_optional(sem, "entailment", 0.0)),
                    "contradiction": float(attr_get_optional(sem, "contradiction", 0.0)),
                    "plan_adherence": float(attr_get_optional(sem, "plan_adherence", 0.0)),
                    "identity_consistency": float(attr_get_optional(sem, "identity_consistency", 0.0)),
                    "reasons": list(attr_get_optional(sem, "reasons", [])),
                    "passed": bool(attr_get_optional(sem, "pass_check", False)),
                }
            )

        if __m3_guard_3158_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return bool(attr_get_optional(sem, "pass_check", False)), {
            "plan_adherence": float(attr_get_optional(sem, "plan_adherence", 0.0)),
            "entailment": float(attr_get_optional(sem, "entailment", 0.0)),
            "identity_consistency": float(attr_get_optional(sem, "identity_consistency", 0.0)),
            "contradiction": float(attr_get_optional(sem, "contradiction", 0.0)),
            "overall": float(attr_get_optional(sem, "overall", 0.0)),
            "reasons": list(attr_get_optional(sem, "reasons", [])),
        }, sem

    def _candidate_count(self) -> int:
        with guard_context(ctx='llm_adapter/llm_core.py:3188', catch_base=False) as __m3_guard_3186_8:
            n = int(os.getenv("M3_MEANING_PIPELINE_CANDIDATE_COUNT", str(self._semantic_scorer_cfg.candidate_count)))

        if __m3_guard_3186_8.error is not None:
            n = int(attr_get_optional(self._semantic_scorer_cfg, "candidate_count", 4))
        return max(1, int(n))

    def _plan_contract_text(
        self,
        response_plan: Optional[Dict[str, Any]],
        meaning_state: Optional[Dict[str, Any]] = None,
        generation_contract: Optional[Any] = None,
    ) -> str:
        if response_plan is None:
            return ""
        safe_plan = {
            "answer_type": str(response_plan.get("answer_type", "")),
            "key_points": list(response_plan.get("key_points", []) or [])[:5],
            "must_avoid": list(response_plan.get("must_avoid", []) or []),
            "clarify_if_uncertain": bool(response_plan.get("clarify_if_uncertain", True)),
            "style": dict(response_plan.get("style", {}) or {}),
            "validation_targets": dict(response_plan.get("validation_targets", {}) or {}),
        }
        if meaning_state and isinstance(meaning_state, dict):
            safe_plan["turn_id"] = meaning_state.get("turn_id")
        if generation_contract is not None:
            contract = self._normalize_generation_contract(generation_contract)
            safe_plan["contract_mode"] = contract.get("mode")
        return json.dumps({"generation_contract": "meaning_plan", "plan": safe_plan}, ensure_ascii=False)

    def _meaning_should_gate(self, meaning_state: Optional[Dict[str, Any]], generation_contract: Optional[Any]) -> bool:
        if not attr_get_optional(self._meaning_cfg, "enabled", False):
            return False
        mode = self._meaning_mode(generation_contract)
        return mode in {"soft_gate", "full"}

    def _meaning_should_log(self, meaning_state: Optional[Dict[str, Any]], generation_contract: Optional[Any]) -> bool:
        if not attr_get_optional(self._meaning_cfg, "enabled", False):
            return False
        mode = self._meaning_mode(generation_contract)
        return mode in {"shadow", "soft_gate", "full"}

    def _meaning_is_soft_gate_allowed(self, meaning_state: Optional[Dict[str, Any]]) -> bool:
        mode = self._meaning_mode(None)
        if mode != "soft_gate":
            return True
        cfg = self._meaning_cfg
        if not cfg:
            return True
        raw = str(cfg.soft_gate_intent_whitelist or "").strip()
        if not raw:
            return True
        allow = {x.strip().lower() for x in raw.split(",") if x.strip()}
        if not allow:
            return True
        intent = str((meaning_state or {}).get("intent", "")).strip().lower()
        return intent in allow or not intent

    def _read_float(self, name: str, default: float) -> float:
        with guard_context(ctx='llm_adapter/llm_core.py:3246', catch_base=False) as __m3_guard_3244_8:
            return float(os.getenv(name, str(default)))

        if __m3_guard_3244_8.error is not None:
            return float(default)

    def _read_int(self, name: str, default: int) -> int:
        with guard_context(ctx='llm_adapter/llm_core.py:3252', catch_base=False) as __m3_guard_3250_8:
            return int(os.getenv(name, str(default)))

        if __m3_guard_3250_8.error is not None:
            return int(default)

    def _ensure_control_health_tracking(self) -> None:
        if attr_get_optional(self, "_control_health_window", None) is None:
            window = int(os.getenv("M3_CONTROL_HEALTH_WINDOW", "24"))
            self._control_health_window = deque(maxlen=max(8, window))
        if attr_get_optional(self, "_auto_mode_fail_streak", None) is None:
            self._auto_mode_fail_streak = 0

    def _note_control_health(self, success: bool, reason: str = "") -> None:
        self._ensure_control_health_tracking()
        reason_code = _normalize_control_reason(reason)
        with guard_context(ctx='llm_adapter/llm_core.py:3262', catch_base=False) as __m3_guard_3259_8:
            self._control_health_window.append((time.time(), bool(success), str(reason_code)))

        if __m3_guard_3259_8.error is not None:
            return
        if success:
            self._auto_mode_fail_streak = 0
        else:
            self._auto_mode_fail_streak = int(attr_get_optional(self, "_auto_mode_fail_streak", 0)) + 1

    def _control_selection_mode(self) -> str:
        raw = str(os.getenv("M3_CONTROL_SELECTION_MODE", "state") or "state").strip().lower()
        if raw in {"0", "off", "none", "disable", "disabled", "no", "false"}:
            return "off"
        if raw in {"1", "state", "state_only", "context", "context_only", "low"}:
            return "state"
        if raw in {"2", "memory", "mid", "mixed", "medium"}:
            return "memory"
        if raw in {"3", "full", "high", "all", "strict"}:
            return "full"
        if raw in {"auto", "adaptive", "self", "self_adjust"}:
            return "state"
        if raw in {"on", "true", "yes"}:
            return "full"
        return "state"

    def _control_allows(self, feature: str) -> bool:
        mode = self._control_selection_mode()
        allowed = {
            "off": set(),
            "state": {"state_context"},
            "memory": {"state_context", "memory_retrieval"},
            "full": {"state_context", "memory_retrieval", "bridge", "decode_control", "adaptive_sampler", "token_value_bias", "quality_gate"},
        }.get(mode, {"state_context"})
        return feature in allowed

    def _bridge_enabled(self) -> bool:
        return (
            self._control_allows("bridge")
            and os.getenv("M3_ENABLE_CONTROL_BRIDGE", "0").lower() in ("1", "true", "yes", "on")
        )

    def _bridge_enabled_safe(self) -> bool:
        with guard_context(ctx='llm_adapter/llm_core.py:3294', catch_base=False) as __m3_guard_3291_8:
            fn = attr_get_optional(self, "_bridge_enabled", None)
            if callable(fn):
                return bool(fn())

        if __m3_guard_3291_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return False

    def _control_plane_mode(self) -> str:
        raw = str(os.getenv("M3_CONTROL_PLANE_MODE", "enforce") or "enforce").strip().lower()
        if raw in {"shadow", "observe", "dryrun"}:
            return "shadow"
        return "enforce"

    def _build_control_decision(
        self,
        prompt: str,
        source: str = "generate",
    ) -> Optional[M3ControlDecision]:
        with guard_context(ctx='llm_adapter/llm_core.py:3307', catch_base=False) as __m3_guard_3304_8:
            decision = build_control_decision(
                self,
                prompt=str(prompt or ""),
                source=str(source or "generate"),
            )
            self._last_control_decision = decision
            log_control_decision(
                decision,
                path=resolve_control_decision_log_path(),
            )
            return decision

        if __m3_guard_3304_8.error is not None:
            return None

    def _hf_failure_policy(self) -> Tuple[float, int, float]:
        window_sec = max(1.0, self._read_float("M3_HF_FAILURE_WINDOW_SEC", 30.0))
        threshold = max(1, self._read_int("M3_HF_FAILURE_THRESHOLD", 3))
        cooldown_sec = max(0.0, self._read_float("M3_HF_FAILURE_COOLDOWN_SEC", 20.0))
        return float(window_sec), int(threshold), float(cooldown_sec)

    def _prune_hf_failure_events(self, now: Optional[float] = None) -> int:
        if attr_get_optional(self, "_hf_failure_events", None) is None:
            self._hf_failure_events = deque(maxlen=256)
        ts_now = float(now if now is not None else time.time())
        window_sec, _, _ = self._hf_failure_policy()
        while self._hf_failure_events and (ts_now - float(self._hf_failure_events[0][0])) > window_sec:
            self._hf_failure_events.popleft()
        return int(len(self._hf_failure_events))

    def _hf_runtime_cooldown_active(self, now: Optional[float] = None) -> bool:
        ts_now = float(now if now is not None else time.time())
        return ts_now < float(attr_get_optional(self, "_hf_runtime_cooldown_until", 0.0) or 0.0)

    def _register_hf_runtime_failure(
        self,
        reason_code: str,
        phase: str,
        model_output_shape: str,
        has_logits: bool,
    ) -> Dict[str, Any]:
        ts_now = float(time.time())
        window_sec, threshold, cooldown_sec = self._hf_failure_policy()
        self._prune_hf_failure_events(ts_now)
        self._hf_failure_events.append((ts_now, str(reason_code), str(phase)))
        count = self._prune_hf_failure_events(ts_now)
        cooldown_until = float(attr_get_optional(self, "_hf_runtime_cooldown_until", 0.0) or 0.0)
        if count >= threshold and cooldown_sec > 0.0:
            cooldown_until = float(ts_now + cooldown_sec)
            self._hf_runtime_cooldown_until = cooldown_until
        return {
            "failure_window_sec": float(window_sec),
            "failure_threshold": int(threshold),
            "failure_window_count": int(count),
            "cooldown_until": float(cooldown_until),
            "reason_code": str(reason_code),
            "phase": str(phase),
            "model_output_shape": str(model_output_shape or "unknown"),
            "has_logits": bool(has_logits),
        }

    def _extract_hf_failure_details(
        self,
        err: Exception,
        hf: Optional[Any] = None,
    ) -> Dict[str, Any]:
        reason_code = "hf_runtime_failure"
        phase = "unknown"
        model_output_shape = "unknown"
        has_logits = False
        if isinstance(err, HFRuntimeFailure):
            reason_code = str(attr_get_optional(err, "reason_code", reason_code) or reason_code)
            phase = str(attr_get_optional(err, "phase", phase) or phase)
            model_output_shape = str(attr_get_optional(err, "model_output_shape", model_output_shape) or model_output_shape)
            has_logits = bool(attr_get_optional(err, "has_logits", has_logits))
        else:
            msg = str(err or "").lower()
            if "nonetype" in msg and "subscriptable" in msg:
                reason_code = "none_subscriptable"
            elif "_control_health_window" in msg and "has no attribute" in msg:
                reason_code = "missing_control_health_window"
            if "logits" in msg:
                phase = "decode_forward"
        diag = attr_get_optional(hf, "_last_runtime_diag", None)
        if isinstance(diag, dict):
            phase = str(diag.get("phase", phase) or phase)
            model_output_shape = str(diag.get("model_output_shape", model_output_shape) or model_output_shape)
            has_logits = bool(diag.get("has_logits", has_logits))
            diag_reason = str(diag.get("reason_code", "") or "")
            if diag_reason and diag_reason != "hf_generate_ok":
                reason_code = diag_reason
        return {
            "reason_code": _normalize_control_reason(reason_code),
            "phase": str(phase or "unknown"),
            "model_output_shape": str(model_output_shape or "unknown"),
            "has_logits": bool(has_logits),
        }

    @staticmethod
    def _normalize_similarity_text(text: str) -> str:
        s = str(text or "").lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _response_similarity(self, a: str, b: str) -> float:
        left = self._normalize_similarity_text(a)
        right = self._normalize_similarity_text(b)
        if not left or not right:
            return 0.0
        char_ratio = float(SequenceMatcher(None, left, right).ratio())
        toks_a = set(re.findall(r"[a-z0-9\uac00-\ud7a3]+", left))
        toks_b = set(re.findall(r"[a-z0-9\uac00-\ud7a3]+", right))
        if not toks_a or not toks_b:
            token_ratio = 0.0
        else:
            token_ratio = float(len(toks_a & toks_b) / max(1, len(toks_a | toks_b)))
        return float(max(char_ratio, token_ratio))

    def _recent_assistant_texts(self, core: Optional[Any] = None, limit: int = 6) -> List[str]:
        out: List[str] = []
        lim = max(1, int(limit))
        core_ref = core if core is not None else attr_get_optional(self, "core", None)
        hist = attr_get_optional(core_ref, "_chat_history", None)
        if hist is not None:
            for item in list(hist):
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip().lower()
                if role not in {"assistant", "m3", "bot"}:
                    continue
                txt = str(item.get("text") or item.get("content") or "").strip()
                if txt:
                    out.append(txt)
            if out:
                return out[-lim:]
        chat_path = os.path.join(os.path.dirname(TRAINING_PATH), "chat_history.jsonl")
        if not os.path.exists(chat_path):
            return []
        try:
            with open(chat_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    role = str(obj.get("role", "")).strip().lower()
                    if role not in {"assistant", "m3", "bot"}:
                        continue
                    txt = str(obj.get("text") or obj.get("content") or "").strip()
                    if txt:
                        out.append(txt)
        except Exception:
            return out[-lim:]
        return out[-lim:]

    def _is_repeat_candidate_blocked(self, response: str, core: Optional[Any] = None) -> Tuple[bool, Dict[str, float]]:
        threshold = max(0.0, min(1.0, self._read_float("M3_REPEAT_BLOCK_THRESHOLD", 0.90)))
        need_hits = max(2, self._read_int("M3_REPEAT_BLOCK_STREAK", 2))
        recent = self._recent_assistant_texts(core=core, limit=max(need_hits + 1, 4))
        if len(recent) < need_hits:
            return False, {"similarity_last": 0.0, "similarity_prev": 0.0}
        sims = [self._response_similarity(response, prev) for prev in recent[-need_hits:]]
        min_sim = min(sims) if sims else 0.0
        blocked = bool(len(sims) >= need_hits and min_sim >= threshold)
        meta = {
            "similarity_last": float(sims[-1] if sims else 0.0),
            "similarity_prev": float(sims[-2] if len(sims) >= 2 else 0.0),
            "similarity_min": float(min_sim),
            "threshold": float(threshold),
        }
        return blocked, meta

    def _record_retrieval_outcome(
        self,
        outcome: str,
        skip_reason: str = "",
        selected_count: int = 0,
        candidate_pool: int = 0,
    ) -> None:
        allowed_reasons = {"", "no_query_vector", "below_threshold", "empty_index", "disabled_by_mode"}
        reason = str(skip_reason or "")
        if reason not in allowed_reasons:
            reason = ""
        out = str(outcome or "skip").strip().lower()
        if out in {"hit", "miss"}:
            self._retrieval_attempts = int(attr_get_optional(self, "_retrieval_attempts", 0)) + 1
            if out == "hit":
                self._retrieval_hits = int(attr_get_optional(self, "_retrieval_hits", 0)) + 1
        attempts = int(attr_get_optional(self, "_retrieval_attempts", 0))
        hits = int(attr_get_optional(self, "_retrieval_hits", 0))
        hit_rate = float(hits / max(1, attempts))
        self._log_jsonl(
            os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
            {
                "kind": "retrieval_metrics",
                "outcome": out,
                "skip_reason": reason,
                "selected_count": int(max(0, selected_count)),
                "candidate_pool": int(max(0, candidate_pool)),
                "retrieval_attempts": attempts,
                "retrieval_hits": hits,
                "retrieval_hit_rate": hit_rate,
            },
        )

    def _resolve_generation_sampling(
        self,
        requested_temperature: Optional[float],
        requested_top_k: Optional[int],
        requested_top_p: Optional[float],
        source: str = "generate",
        core: Optional[Any] = None,
    ) -> Tuple[float, int, float]:
        source_key = str(source or "generate").strip().lower()

        min_temp = self._read_float("M3_CONTROL_MIN_TEMP", 0.12)
        max_temp = self._read_float("M3_CONTROL_MAX_TEMP", 0.75)
        min_top_k = max(1, self._read_int("M3_CONTROL_MIN_TOP_K", 6))
        max_top_k = max(min_top_k + 1, self._read_int("M3_CONTROL_MAX_TOP_K", 64))
        min_top_p = self._read_float("M3_CONTROL_MIN_TOP_P", 0.65)
        max_top_p = self._read_float("M3_CONTROL_MAX_TOP_P", 0.95)

        strictness = self._read_float("M3_CONTROL_STRICTNESS", 0.75)
        if source_key == "autonomy":
            strictness = self._read_float("M3_CONTROL_AUTONOMY_STRICTNESS", strictness)
            min_temp = min(0.90, min_temp)
            min_top_k = max(1, self._read_int("M3_CONTROL_AUTONOMY_MIN_TOP_K", min_top_k))
            max_top_k = self._read_int("M3_CONTROL_AUTONOMY_MAX_TOP_K", max_top_k)
            min_top_p = min(0.90, max(0.01, self._read_float("M3_CONTROL_AUTONOMY_MIN_TOP_P", min_top_p)))
        elif source_key in ("fallback", "safe"):
            strictness = self._read_float("M3_CONTROL_FALLBACK_STRICTNESS", strictness)
            max_temp = min(0.70, max_temp)

        strictness = max(0.0, min(1.0, strictness))

        if requested_temperature is not None:
            try:
                base_temp = float(requested_temperature)
            except Exception:
                base_temp = self._read_float("M3_CONTROL_BASE_TEMP", 0.65)
        else:
            base_temp = self._read_float(
                "M3_CONTROL_AUTONOMY_BASE_TEMP" if source_key == "autonomy" else "M3_CONTROL_BASE_TEMP",
                0.6 if source_key == "autonomy" else 0.65,
            )
        if requested_top_k is not None:
            try:
                base_top_k = int(requested_top_k)
            except Exception:
                base_top_k = self._read_int("M3_CONTROL_BASE_TOP_K", 32)
        else:
            base_top_k = self._read_int(
                "M3_CONTROL_AUTONOMY_BASE_TOP_K" if source_key == "autonomy" else "M3_CONTROL_BASE_TOP_K",
                24 if source_key == "autonomy" else 32,
            )
        if requested_top_p is not None:
            try:
                base_top_p = float(requested_top_p)
            except Exception:
                base_top_p = self._read_float("M3_CONTROL_BASE_TOP_P", 0.85)
        else:
            base_top_p = self._read_float(
                "M3_CONTROL_AUTONOMY_BASE_TOP_P" if source_key == "autonomy" else "M3_CONTROL_BASE_TOP_P",
                0.88 if source_key == "autonomy" else 0.85,
            )

        state_strict_bonus = 0.0
        if core is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:3337', catch_base=False) as __m3_guard_3318_12:
                phi = 0.0
                if attr_has(core, 'phi_calculator') and attr_get_optional(core.phi_calculator, 'phi_history', None):
                    hist = core.phi_calculator.phi_history
                    if hist:
                        phi = float(hist[-1])
                q = attr_get_optional(core, 'qualia', None)
                arousal = float(attr_get_optional(q, 'arousal', 0.0)) if q is not None else 0.0
                valence = float(attr_get_optional(q, 'valence', 0.0)) if q is not None else 0.0
                world_state = attr_get_optional(core, 'world_state', None)
                stability = float(attr_get_optional(world_state, 'get', lambda k, d: d)('stability', 1.0)) if isinstance(world_state, dict) else float(attr_get_optional(world_state, 'stability', 1.0))
                if abs(phi) < 1e-8:
                    state_strict_bonus += 0.06
                if arousal < 0.2 or arousal > 0.8:
                    state_strict_bonus += 0.08
                if abs(valence) < 0.2:
                    state_strict_bonus += 0.04
                if stability < 0.4:
                    state_strict_bonus += 0.08

            if __m3_guard_3318_12.error is not None:
                state_strict_bonus = 0.0
        strictness = min(1.0, strictness + state_strict_bonus)

        temp_cap = np.clip(base_temp * (1.0 - 0.38 * strictness), min_temp, max_temp)
        top_k_cap = np.clip(base_top_k * (1.0 - 0.42 * strictness), min_top_k, max_top_k)
        top_p_cap = np.clip(base_top_p * (1.0 - 0.22 * strictness), min_top_p, max_top_p)

        return float(temp_cap), int(top_k_cap), float(top_p_cap)

    def _retry_generation_sampling(
        self,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[float, int, float]:
        decay_temp = self._read_float("M3_CONTROL_RETRY_TEMP_DECAY", 0.75)
        decay_top_k = self._read_float("M3_CONTROL_RETRY_TOPK_DECAY", 0.70)
        decay_top_p = self._read_float("M3_CONTROL_RETRY_TOPP_DECAY", 0.95)
        min_temp = self._read_float("M3_CONTROL_MIN_TEMP", 0.12)
        min_top_k = max(1, self._read_int("M3_CONTROL_MIN_TOP_K", 6))
        min_top_p = self._read_float("M3_CONTROL_MIN_TOP_P", 0.65)
        return (
            max(min_temp, float(temperature) * decay_temp),
            max(min_top_k, int(float(top_k) * decay_top_k)),
            max(min_top_p, float(top_p) * decay_top_p),
        )

    def _strip_similar_context_block(self, prompt: str) -> str:
        s = str(prompt or "")
        if "Similar past context:" not in s:
            return s
        if "Current context:" in s:
            head, tail = s.split("Similar past context:", 1)
            _, tail2 = tail.split("Current context:", 1)
            return (head.rstrip() + "\n\nCurrent context:\n" + tail2.lstrip())
        lines = s.splitlines()
        out = []
        skip = False
        for ln in lines:
            if ln.startswith("Similar past context:"):
                skip = True
                continue
            if skip and ln.startswith("User:"):
                skip = False
            if not skip:
                out.append(ln)
        return "\n".join(out)

    def _is_cuda_fatal_error(self, err) -> bool:
        msg = str(err or "").lower()
        if not msg:
            return False
        needles = (
            "illegal memory access",
            "device-side assert",
            "cublas",
            "cudnn",
            "cuda error",
        )
        return any(n in msg for n in needles)

    def _trip_hf_circuit_breaker(self, err) -> None:
        self._gpu_fault_count = int(attr_get_optional(self, "_gpu_fault_count", 0)) + 1
        failover = str(os.getenv("M3_HF_CUDA_FAILOVER", "1")).lower() in ("1", "true", "yes", "on")
        if not failover:
            return
        if not attr_get_optional(self, "_hf_circuit_open", False):
            self._hf_circuit_open = True
            self._hf_circuit_reason = str(err)
            self.use_hf = False
            with guard_context(ctx='llm_adapter/llm_core.py:3410', catch_base=False) as __m3_guard_3408_12:
                os.environ["M3_USE_HF"] = "0"

            if __m3_guard_3408_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            with guard_context(ctx='llm_adapter/llm_core.py:3415', catch_base=False) as __m3_guard_3412_12:
                if attr_has(self, "torch") and attr_has(self.torch, "cuda") and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()

            if __m3_guard_3412_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            logger.error(f"[HFBackend] circuit breaker opened after CUDA fault: {err}")

    def _extract_last_user_text(self, prompt: str) -> str:
        s = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
        if not s.strip():
            return ""
        focus = s
        if "Current context:" in s:
            focus = s.split("Current context:")[-1]
        lines = focus.split("\n")
        users: List[str] = []
        cur: List[str] = []
        in_user = False

        def _flush():
            nonlocal cur, in_user
            if cur:
                txt = "\n".join(cur).strip()
                if txt:
                    users.append(txt)
            cur = []
            in_user = False

        for ln in lines:
            if ln.startswith("User:"):
                _flush()
                in_user = True
                cur = [ln[len("User:"):].lstrip()]
                continue
            if ln.startswith("M3:"):
                if in_user:
                    _flush()
                continue
            if in_user:
                cur.append(ln)
        _flush()

        if users:
            return users[-1].strip()
        return focus.strip()

    def _detect_language(self, text: str) -> str:
        s = str(text or "")
        if not s.strip():
            return "other"
        ko = len(re.findall(r"[\uac00-\ud7a3]", s))
        zh = len(re.findall(r"[\u4e00-\u9fff]", s))
        ru = len(re.findall(r"[\u0400-\u04FF]", s))
        en = len(re.findall(r"[A-Za-z]", s))
        m = max(ko, zh, ru, en)
        if m <= 0:
            return "other"
        if ko >= zh and ko >= ru and ko >= en:
            return "ko"
        if zh >= ko and zh >= ru and zh >= en:
            return "zh"
        if ru >= ko and ru >= zh and ru >= en:
            return "ru"
        return "en"

    def _is_autonomy_prompt(self, prompt: str) -> bool:
        s = str(prompt or "")
        return (
            "[\uc790\uc728 \ubaa8\ub4dc]" in s
            or "[Autonomy]" in s
            or "Based on the current M3_STATE, say one short next action." in s
        )

    def _should_include_m3_state(self, prompt: str) -> bool:
        if not self._control_allows('state_context'):
            return False
        policy = str(os.getenv("M3_STATE_CONTEXT_POLICY", "off") or "off").strip().lower()
        if policy in {"off", "none", "0", "false", "no"}:
            return False
        if policy in {"always", "on", "1", "true", "yes"}:
            return True
        focus = self._extract_last_user_text(prompt).lower()
        if not focus:
            focus = str(prompt or "").lower()
        keywords = (
            "\uc0c1\ud0dc",
            "state",
            "phi",
            "qualia",
            "arousal",
            "valence",
            "engagement",
            "frustration",
            "m3_state",
            "\uc9c0\uae08 \uc0c1\ud0dc",
            "\ud1b5\ud569\uc0c1\ud0dc",
        )
        return any(k in focus for k in keywords)

    def _is_state_request(self, prompt: str) -> bool:
        focus = self._extract_last_user_text(prompt).lower()
        if not focus:
            focus = str(prompt or "").lower()
        keywords = (
            "\uc0c1\ud0dc",
            "state",
            "phi",
            "qualia",
            "arousal",
            "valence",
            "engagement",
            "frustration",
            "m3_state",
            "\ud604\uc7ac \uc0c1\ud0dc",
            "\uc9c0\uae08 \uc0c1\ud0dc",
            "\ub0b4\ubd80 \uc0c1\ud0dc",
        )
        return any(k in focus for k in keywords)

    def _build_decode_control(self, prompt: str) -> Optional[Dict[str, Any]]:
        if not self._control_allows('decode_control'):
            return None
        enabled = str(os.getenv("M3_ENABLE_DECODE_CONTROL", "1")).lower() in ("1", "true", "yes", "on")
        if not enabled:
            return None

        allow_state_terms = self._is_state_request(prompt) or self._is_autonomy_prompt(prompt)
        raw_terms = os.getenv(
            "M3_CONTROL_FORBIDDEN_TERMS",
            "phi,qualia,arousal,valence,engagement,frustration,m3_state,state,integrated_state",
        )
        terms = [t.strip() for t in str(raw_terms).split(",") if t.strip()]

        identity_lock = str(os.getenv("M3_CONTROL_IDENTITY_LOCK", "1")).lower() in ("1", "true", "yes", "on")
        raw_identity_terms = os.getenv(
            "M3_CONTROL_IDENTITY_TERMS",
            "ai,assistant,language model,qwen,alibaba,openai,train button,currently untrained,"
            "\uc778\uacf5\uc9c0\ub2a5,\uc5b8\uc5b4 \ubaa8\ub378,\uc54c\ub9ac\ubc14\ubc14,\ud038\uc6ec",
        )
        identity_terms = [t.strip() for t in str(raw_identity_terms).split(",") if t.strip()]

        with guard_context(ctx='llm_adapter/llm_core.py:3555', catch_base=False) as __m3_guard_3553_8:
            penalty = float(os.getenv("M3_CONTROL_TERM_PENALTY", "8.0"))

        if __m3_guard_3553_8.error is not None:
            penalty = 8.0
        with guard_context(ctx='llm_adapter/llm_core.py:3559', catch_base=False) as __m3_guard_3557_8:
            identity_penalty = float(os.getenv("M3_CONTROL_IDENTITY_PENALTY", "14.0"))

        if __m3_guard_3557_8.error is not None:
            identity_penalty = 14.0
        with guard_context(ctx='llm_adapter/llm_core.py:3563', catch_base=False) as __m3_guard_3561_8:
            max_temp = float(os.getenv("M3_CONTROL_MAX_TEMP", "0.65"))

        if __m3_guard_3561_8.error is not None:
            max_temp = 0.65
        with guard_context(ctx='llm_adapter/llm_core.py:3567', catch_base=False) as __m3_guard_3565_8:
            max_top_k = int(os.getenv("M3_CONTROL_MAX_TOP_K", "40"))

        if __m3_guard_3565_8.error is not None:
            max_top_k = 40
        with guard_context(ctx='llm_adapter/llm_core.py:3571', catch_base=False) as __m3_guard_3569_8:
            max_top_p = float(os.getenv("M3_CONTROL_MAX_TOP_P", "0.92"))

        if __m3_guard_3569_8.error is not None:
            max_top_p = 0.92

        return {
            "allow_state_terms": bool(allow_state_terms),
            "forbidden_terms": terms,
            "identity_lock": bool(identity_lock),
            "identity_forbidden_terms": identity_terms,
            "forbidden_penalty": float(max(0.0, max(penalty, identity_penalty if identity_lock else penalty))),
            "max_temperature": float(max(0.05, max_temp)),
            "max_top_k": int(max(1, max_top_k)),
            "max_top_p": float(min(0.99, max(0.1, max_top_p))),
        }

    def _sanitize_training_record(self, prompt: str, response: str, source: str = "generate") -> Optional[dict]:
        self._last_record_reject_reason = ""
        prompt_raw = str(prompt or "")
        response_text = str(response or "").strip()
        source_text = str(source or "generate")

        def _reject(reason: str) -> Optional[dict]:
            self._last_record_reject_reason = reason
            return None

        if not response_text:
            return _reject("empty_response")
        if self._is_numeric_dump_response(response_text):
            return _reject("numeric_dump_response")
        if self._is_backend_status_text(response_text):
            return _reject("backend_status_response")
        if self._is_refusal_disclaimer(response_text):
            return _reject("refusal_disclaimer_response")
        if self._is_cuda_fatal_error(response_text):
            return _reject("cuda_error_response")

        if self._record_scope == "user_only":
            if source_text != "generate_hf":
                return _reject("scope_user_only_source")
            if self._is_autonomy_prompt(prompt_raw):
                return _reject("scope_user_only_autonomy")

        exclude_similar = str(os.getenv("M3_TRAIN_EXCLUDE_SIMILAR_CONTEXT", "1")).lower() in ("1", "true", "yes", "on")
        if exclude_similar and ("Similar past context:" in prompt_raw):
            prompt_raw = self._strip_similar_context_block(prompt_raw)

        prompt_clean = self._extract_last_user_text(prompt_raw)
        if not prompt_clean:
            return _reject("empty_prompt_clean")

        lang_match = str(os.getenv("M3_TRAIN_LANG_MATCH", "1")).lower() in ("1", "true", "yes", "on")
        prompt_lang = self._detect_language(prompt_clean)
        response_lang = self._detect_language(response_text)
        if lang_match and prompt_lang in {"ko", "en", "zh", "ru"} and response_lang in {"ko", "en", "zh", "ru"}:
            if prompt_lang != response_lang:
                return _reject(f"language_mismatch:{prompt_lang}->{response_lang}")

        rec = {
            "ts": time.time(),
            "source": source_text,
            "prompt": prompt_clean,
            "prompt_clean": prompt_clean,
            "prompt_raw": prompt_raw,
            "response": response_text,
            "prompt_lang": prompt_lang,
            "response_lang": response_lang,
        }

        try:
            core = attr_get_optional(self, "core", None)
            if core is not None:
                ph = []
                with guard_context(ctx='llm_adapter/llm_core.py:3644', catch_base=False) as __m3_guard_3642_16:
                    ph = list(core.phi_calculator.phi_history) if attr_has(core, "phi_calculator") else []

                if __m3_guard_3642_16.error is not None:
                    ph = []
                phi = float(ph[-1]) if ph else 0.0
                phi_mean10 = float(np.mean(ph[-10:])) if len(ph) >= 1 else 0.0
                phi_delta = float(ph[-1] - ph[-2]) if len(ph) >= 2 else 0.0
                phi_nonzero_recent = int(any(abs(float(v)) > 1e-8 for v in ph[-10:])) if ph else 0

                if abs(phi) <= 1e-12:
                    self._phi_zero_streak = int(attr_get_optional(self, "_phi_zero_streak", 0)) + 1
                    if self._phi_zero_streak % int(attr_get_optional(self, "_phi_zero_warn_every", 50)) == 0:
                        logger.warning(f"[TrainingRecord] phi remains zero for {self._phi_zero_streak} samples")
                else:
                    self._phi_zero_streak = 0

                rec["phi"] = float(phi)
                rec["phi_mean10"] = float(phi_mean10)
                rec["phi_delta"] = float(phi_delta)
                rec["phi_nonzero_recent"] = int(phi_nonzero_recent)

                with guard_context(ctx='llm_adapter/llm_core.py:3672', catch_base=False) as __m3_guard_3663_16:
                    q = core.qualia
                    rec["qualia"] = {
                        "arousal": float(attr_get_optional(q, "arousal", 0.0)),
                        "valence": float(attr_get_optional(q, "valence", 0.0)),
                        "entropy": float(attr_get_optional(q, "entropy", 0.0)),
                        "engagement": float(attr_get_optional(q, "engagement", 0.0)),
                        "frustration": float(attr_get_optional(q, "frustration", 0.0)),
                    }

                if __m3_guard_3663_16.error is not None:
                    rec["qualia"] = {}

                with guard_context(ctx='llm_adapter/llm_core.py:3684', catch_base=False) as __m3_guard_3675_16:
                    ec = attr_get_optional(core, "energy_ctrl", None)
                    if ec is not None:
                        energy = float(attr_get_optional(ec, "cognitive_energy", 0.0))
                        cap = float(max(attr_get_optional(ec, "energy_capacity", 1.0), 1e-6))
                        rec["energy"] = {
                            "value": energy,
                            "ratio": float(energy / cap),
                        }

                if __m3_guard_3675_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
        except Exception:
            logging.getLogger(__name__).exception("Swallowed exception")
        return rec

    def _next_stabilization_template(self, lang: str) -> str:
        idx = int(attr_get_optional(self, "_safe_fallback_turn", 0))
        self._safe_fallback_turn = idx + 1
        shared = [
            "Output is in stabilization mode. Send one short question for an M3-perspective answer.",
            "Stabilization mode is active. Re-send a single concise question for an M3 answer.",
            "Generation switched to stabilization mode. Ask one short question and I will answer as M3.",
        ]
        templates = {
            "ko": [
                "Stabilization mode is active. Send one short question for an M3-perspective answer.",
                "I am in stabilization mode. Re-send a single concise question for an M3 answer.",
                "Generation switched to stabilization mode. Ask one short question and I will answer as M3.",
            ],
            "zh": list(shared),
            "ru": list(shared),
            "en": list(shared),
            "other": list(shared),
        }
        bucket = templates.get(str(lang or "other"), templates["other"])
        return str(bucket[idx % len(bucket)])

    def _generate_safe_fallback(
        self,
        prompt: str,
        chat_messages: Optional[List[Dict[str, str]]] = None,
        max_len: int = 60,
        meaning_state: Optional[Dict[str, Any]] = None,
        response_plan: Optional[Dict[str, Any]] = None,
        generation_contract: Optional[Any] = None,
    ) -> str:
        user_text = self._extract_last_user_text(prompt)
        if not user_text and chat_messages:
            with guard_context(ctx='llm_adapter/llm_core.py:3705', catch_base=False) as __m3_guard_3701_12:
                user_msgs = [str(m.get("content", "")) for m in chat_messages if str(m.get("role", "")) == "user"]
                if user_msgs:
                    user_text = user_msgs[-1].strip()

            if __m3_guard_3701_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        if not user_text:
            user_text = str(prompt or "").strip()
        if isinstance(response_plan, dict):
            ans_type = str(response_plan.get("answer_type", "")).strip().lower()
            if ans_type in {"clarify", "abstain"}:
                try:
                    return format_plan_fallback_prompt(response_plan=response_plan, meaning_state=meaning_state)
                except Exception:
                    logging.getLogger(__name__).exception("Swallowed exception")
        safe_temp = None
        safe_top_k = None
        safe_top_p = None
        with guard_context(ctx='llm_adapter/llm_core.py:3727', catch_base=False) as __m3_guard_3719_8:
            safe_temp, safe_top_k, safe_top_p = self._resolve_generation_sampling(
                requested_temperature=None,
                requested_top_k=None,
                requested_top_p=None,
                source='fallback',
                core=attr_get_optional(self, 'core', None),
            )

        if __m3_guard_3719_8.error is not None:
            safe_temp = None
            safe_top_k = None
            safe_top_p = None
        use_ollama_fallback = str(os.getenv("M3_DISABLE_OLLAMA", "1")).lower() not in ("1", "true", "yes", "on")
        if use_ollama_fallback:
            try:
                with guard_context(ctx='llm_adapter/llm_core.py:3736', catch_base=False) as __m3_guard_3734_16:
                    fb_timeout = float(os.getenv("M3_SAFE_FALLBACK_TIMEOUT", "12"))

                if __m3_guard_3734_16.error is not None:
                    fb_timeout = 12.0
                remote_resp = get_local_thinking(
                    user_text,
                    max_len=max(32, int(max_len or 60)),
                    timeout=max(1.0, fb_timeout),
                    retries=1,
                    backoff=0.2,
                    temperature=safe_temp,
                    top_k=safe_top_k,
                    top_p=safe_top_p,
                )
                remote_resp = str(remote_resp or "").strip()
                if remote_resp and not self._is_backend_status_text(remote_resp):
                    return remote_resp.strip()
            except Exception:
                logging.getLogger(__name__).exception("Swallowed exception")

        lang = self._detect_language(user_text)
        q = str(user_text or "").strip().lower()
        if lang == "ko" and "m3" in q:
            return "I integrate awareness, memory, and goals to choose the next action from the current context."
        return self._next_stabilization_template(lang)



    def save_model(self, path: str):
        """Saves the model state dictionary."""
        if attr_get_optional(self, "_hf_circuit_open", False) and self.device.type == "cuda":
            logger.warning(f"Skipping model save while HF circuit is open: {path}")
            return
        with guard_context(ctx='llm_adapter/llm_core.py:3777', catch_base=False) as __m3_guard_3773_8:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")

        if __m3_guard_3773_8.error is not None:
            e = __m3_guard_3773_8.error
            logger.error(f"Error saving model to {path}: {e}")
            if self._is_cuda_fatal_error(e):
                self._trip_hf_circuit_breaker(e)

    def _record_example(self, prompt: str, response: str, source: str = "generate") -> None:
        if not attr_get_optional(self, "_record_training", False):
            return
        try:
            prompt_raw = str(prompt or "")
            response_text = str(response or "")
            rec = self._sanitize_training_record(prompt_raw, response_text, source=source)
            reject_reason = str(attr_get_optional(self, "_last_record_reject_reason", "") or "")
            with guard_context(ctx='llm_adapter/llm_core.py:3792', catch_base=False) as __m3_guard_3790_12:
                os.makedirs(os.path.dirname(TRAINING_PATH), exist_ok=True)

            if __m3_guard_3790_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            reject_path = os.getenv(
                "M3_TRAIN_REJECT_PATH",
                os.path.join(os.path.dirname(TRAINING_PATH), "llm_training_data.rejected.jsonl"),
            )
            reject_rec = {
                "ts": time.time(),
                "source": str(source or "generate"),
                "prompt_raw": prompt_raw,
                "response": response_text,
                "reason": reject_reason or "filtered_out",
            }
            with self._record_lock:
                if rec is not None:
                    with open(TRAINING_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                else:
                    with guard_context(ctx='llm_adapter/llm_core.py:3812', catch_base=False) as __m3_guard_3810_20:
                        os.makedirs(os.path.dirname(reject_path), exist_ok=True)

                    if __m3_guard_3810_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                    with open(reject_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(reject_rec, ensure_ascii=False) + "\n")
        except Exception as e:
            with guard_context(ctx='llm_adapter/llm_core.py:3819', catch_base=False) as __m3_guard_3817_12:
                logger.warning(f"Failed to record training example: {e}")

            if __m3_guard_3817_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

    def _default_tokenizer_corpus_files(self) -> List[str]:
        base = os.path.dirname(TRAINING_PATH)
        candidates = [
            TRAINING_PATH,
            os.path.join(base, "llm_training_data.rejected.jsonl"),
            os.path.join(base, "chat_history.jsonl"),
            os.path.join(base, "llm_adapter.log"),
        ]
        cfg = self.tokenizer_auto_cfg
        max_files = int(max(1, cfg.corpus_max_files))
        out: List[str] = []
        for p in candidates:
            if os.path.exists(p):
                out.append(p)
            if len(out) >= max_files:
                break
        return out

    def _copy_vocab_rows(self, dst_weight, src_weight, row_map: List[Tuple[int, int]]) -> None:
        with self.torch.no_grad():
            for old_i, new_i in row_map:
                if old_i < 0 or new_i < 0:
                    continue
                if old_i >= src_weight.shape[0] or new_i >= dst_weight.shape[0]:
                    continue
                dst_weight[new_i].copy_(src_weight[old_i])

    def _token_maps_for_copy(self, old_tok, new_tok) -> List[Tuple[int, int]]:
        with guard_context(ctx='llm_adapter/llm_core.py:3859', catch_base=False) as __m3_guard_3850_8:
            if attr_get_optional(old_tok, "_type", "") == "hf" and attr_get_optional(new_tok, "_type", "") == "hf":
                old_vocab = old_tok._backend.get_vocab()
                new_vocab = new_tok._backend.get_vocab()
                pairs = []
                for token, old_id in old_vocab.items():
                    if token in new_vocab:
                        pairs.append((int(old_id), int(new_vocab[token])))
                return pairs

        if __m3_guard_3850_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        n = min(int(attr_get_optional(old_tok, "vocab_size", 0)), int(attr_get_optional(new_tok, "vocab_size", 0)))
        return [(i, i) for i in range(max(0, n))]

    def reload_tokenizer_and_resize(self, vocab_path: Optional[str] = None) -> Dict[str, Any]:
        """Reload tokenizer file and resize model vocab-dependent layers with overlap copy."""
        old_tok = self.tok
        old_vocab = int(self.vocab_size)
        target_path = vocab_path or attr_get_optional(old_tok, "_last_rebuild_path", "")
        new_tok = M3Tokenizer(vocab_file=target_path if target_path else None, config=get_global_config().tokenizer)
        new_vocab = int(new_tok.vocab_size)
        if new_vocab <= 0:
            return {"ok": False, "reason": "invalid_vocab"}
        allow_shrink = str(os.getenv("M3_TOKENIZER_ALLOW_SHRINK", "0")).lower() in ("1", "true", "yes", "on")
        min_keep_ratio = float(max(1e-6, min(1.0, attr_get_optional(self.tokenizer_auto_cfg, "min_keep_vocab_ratio", 0.60))))
        min_allowed = int(max(32, int(old_vocab * min_keep_ratio)))
        # Enforce shrink guard for auto-rebuild paths or when no explicit path given.
        # When user explicitly provides a specific vocab_path, they intend the replacement.
        auto_like_path = str(target_path or "").strip().lower()
        is_auto_rebuild = (".tokenizer.auto" in auto_like_path) or (vocab_path is None)
        enforce_shrink_guard = is_auto_rebuild
        # Absolute floor: never allow byte-fallback (~265 vocab) to replace a real tokenizer
        # in auto-rebuild paths. Manual explicit paths bypass this.
        abs_vocab_floor = max(256, int(os.getenv("M3_TOKENIZER_ABS_VOCAB_FLOOR", "500")))
        if is_auto_rebuild and (not allow_shrink) and old_vocab > abs_vocab_floor and new_vocab < abs_vocab_floor:
            self._log_jsonl(
                os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                {
                    "kind": "tokenizer_rebuild_guard",
                    "status": "reject",
                    "reason": "abs_vocab_floor",
                    "old_vocab": int(old_vocab),
                    "new_vocab": int(new_vocab),
                    "abs_floor": int(abs_vocab_floor),
                },
            )
            return {
                "ok": False,
                "reason": "abs_vocab_floor",
                "old_vocab": int(old_vocab),
                "new_vocab": int(new_vocab),
            }
        if enforce_shrink_guard and (not allow_shrink) and old_vocab > 0 and new_vocab < min_allowed:
            self._log_jsonl(
                os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                {
                    "kind": "tokenizer_rebuild_guard",
                    "status": "reject",
                    "reason": "vocab_shrink_guard",
                    "old_vocab": int(old_vocab),
                    "new_vocab": int(new_vocab),
                    "min_allowed": int(min_allowed),
                },
            )
            return {
                "ok": False,
                "reason": "vocab_shrink_guard",
                "old_vocab": int(old_vocab),
                "new_vocab": int(new_vocab),
                "min_allowed": int(min_allowed),
            }
        if new_vocab == old_vocab:
            self.tok = new_tok
            self.vocab_size = new_vocab
            return {"ok": True, "old_vocab": old_vocab, "new_vocab": new_vocab, "resized": False}

        nn = self.torch.nn
        from llm_adapter.layers import PlasticBitLinear

        row_map = self._token_maps_for_copy(old_tok, new_tok)
        hidden = int(self.model.hidden)
        trace_decay = float(attr_get_optional(self.model.head, "trace_decay", 0.95))

        new_emb = nn.Embedding(new_vocab, hidden, padding_idx=int(new_tok.PAD)).to(self.device)
        self._copy_vocab_rows(new_emb.weight, self.model.emb.weight.data, row_map)

        new_head = PlasticBitLinear(hidden, new_vocab, trace_decay=trace_decay).to(self.device)
        self._copy_vocab_rows(new_head.weight.data, self.model.head.weight.data, row_map)
        if attr_get_optional(self.model.head, "bias", None) is not None and attr_get_optional(new_head, "bias", None) is not None:
            self._copy_vocab_rows(new_head.bias.data.unsqueeze(-1), self.model.head.bias.data.unsqueeze(-1), [(a, b) for a, b in row_map])
            new_head.bias.data = new_head.bias.data.squeeze(-1)

        new_token_value = PlasticBitLinear(hidden, new_vocab, trace_decay=trace_decay).to(self.device)
        self._copy_vocab_rows(new_token_value.weight.data, self.model.token_value.weight.data, row_map)
        if attr_get_optional(self.model.token_value, "bias", None) is not None and attr_get_optional(new_token_value, "bias", None) is not None:
            self._copy_vocab_rows(new_token_value.bias.data.unsqueeze(-1), self.model.token_value.bias.data.unsqueeze(-1), [(a, b) for a, b in row_map])
            new_token_value.bias.data = new_token_value.bias.data.squeeze(-1)

        self.model.emb = new_emb
        self.model.head = new_head
        self.model.token_value = new_token_value
        self.tok = new_tok
        self.vocab_size = int(new_vocab)
        self.criterion = self.torch.nn.CrossEntropyLoss(ignore_index=self.tok.PAD)

        wd = float(os.getenv("M3_STABILITY_WEIGHT_DECAY", str(self.stability_cfg.weight_decay)))
        self.opt = self.torch.optim.AdamW(self.model.parameters(), lr=float(self.config.learning_rate), weight_decay=wd)
        self.token_value_opt = None
        self._tokenizer_rebuilds += 1
        self._log_jsonl(
            os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
            {
                "kind": "tokenizer_rebuild",
                "old_vocab": int(old_vocab),
                "new_vocab": int(new_vocab),
                "copied_rows": int(len(row_map)),
                "rebuild_count": int(self._tokenizer_rebuilds),
            },
        )
        return {"ok": True, "old_vocab": old_vocab, "new_vocab": new_vocab, "resized": True, "copied_rows": len(row_map)}

    def _maybe_auto_rebuild_tokenizer(self) -> None:
        if not attr_has(self.tok, "should_rebuild_vocab"):
            return
        now = float(time.time())
        with guard_context(ctx='llm_adapter/llm_core.py:3977', catch_base=False) as __m3_guard_3975_8:
            min_interval = int(max(0, attr_get_optional(self.tokenizer_auto_cfg, "rebuild_min_interval_sec", 0)))

        if __m3_guard_3975_8.error is not None:
            min_interval = 0
        with guard_context(ctx='llm_adapter/llm_core.py:3981', catch_base=False) as __m3_guard_3979_8:
            last_rebuild = float(attr_get_optional(self.tok, "_last_rebuild_unix", 0.0))

        if __m3_guard_3979_8.error is not None:
            last_rebuild = 0.0
        if min_interval > 0 and (now - last_rebuild) < float(min_interval):
            self._log_jsonl(
                os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                {
                    "kind": "tokenizer_rebuild_guard",
                    "status": "skip",
                    "reason": "cooldown",
                    "remaining_sec": float(max(0.0, float(min_interval) - (now - last_rebuild))),
                },
            )
            return
        with guard_context(ctx='llm_adapter/llm_core.py:3997', catch_base=False) as __m3_guard_3994_8:
            if not self.tok.should_rebuild_vocab():
                return

        if __m3_guard_3994_8.error is not None:
            return
        files = self._default_tokenizer_corpus_files()
        if not files:
            return
        # Deterministic corpus fingerprint to avoid rebuilding the same corpus every cycle.
        with guard_context(ctx='llm_adapter/llm_core.py:4005', catch_base=False) as __m3_guard_4003_8:
            max_chars = int(max(1_000, attr_get_optional(self.tokenizer_auto_cfg, "corpus_max_chars", 1_500_000)))

        if __m3_guard_4003_8.error is not None:
            max_chars = 1_500_000
        corpus_fp = ""
        with guard_context(ctx='llm_adapter/llm_core.py:4033', catch_base=False) as __m3_guard_4008_8:
            if attr_has(self.tok, "_collect_corpus_stats"):
                stats = self.tok._collect_corpus_stats(files, max_chars=max_chars)
            else:
                stats = {}
            fp_payload = {
                "files": [str(p) for p in files],
                "stats": stats,
                "vocab_size": int(self.tokenizer_auto_cfg.rebuild_vocab_size),
            }
            corpus_fp = hashlib.sha1(
                json.dumps(fp_payload, sort_keys=True, ensure_ascii=False).encode("utf-8", errors="ignore")
            ).hexdigest()[:16]
            prev_fp = str(attr_get_optional(self.tok, "_last_rebuild_corpus_fingerprint", "") or "")
            if corpus_fp and prev_fp and corpus_fp == prev_fp:
                self._log_jsonl(
                    os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                    {
                        "kind": "tokenizer_rebuild_guard",
                        "status": "skip",
                        "reason": "same_corpus_fingerprint",
                        "fingerprint": str(corpus_fp),
                    },
                )
                return

        if __m3_guard_4008_8.error is not None:
            corpus_fp = ""
        target_path = os.path.join(os.path.dirname(TRAINING_PATH), "llm_training_data.tokenizer.auto.json")
        with guard_context(ctx='llm_adapter/llm_core.py:4067', catch_base=False) as __m3_guard_4036_8:
            ok = self.tok.rebuild_vocab_from_corpus(
                files=files,
                out_path=target_path,
                vocab_size=int(self.tokenizer_auto_cfg.rebuild_vocab_size),
            )
            if ok:
                res = self.reload_tokenizer_and_resize(target_path)
                if not bool(res.get("ok", False)):
                    self._log_jsonl(
                        os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                        {
                            "kind": "tokenizer_rebuild_guard",
                            "status": "reject",
                            "reason": str(res.get("reason", "reload_failed")),
                            "old_vocab": int(res.get("old_vocab", self.vocab_size)),
                            "new_vocab": int(res.get("new_vocab", 0)),
                            "fingerprint": str(corpus_fp or ""),
                        },
                    )
                else:
                    self._log_jsonl(
                        os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                        {
                            "kind": "tokenizer_rebuild_guard",
                            "status": "applied",
                            "fingerprint": str(corpus_fp or ""),
                            "old_vocab": int(res.get("old_vocab", 0)),
                            "new_vocab": int(res.get("new_vocab", 0)),
                        },
                    )

        if __m3_guard_4036_8.error is not None:
            e = __m3_guard_4036_8.error
            logger.warning(f"Tokenizer auto rebuild failed: {e}")

    def _ensure_bridge_adapt_optimizer(self, hf: HFBackend):
        bridge = attr_get_optional(hf, "_control_bridge", None)
        if bridge is None:
            return None
        if self._bridge_adapt_opt is not None:
            return self._bridge_adapt_opt
        with guard_context(ctx='llm_adapter/llm_core.py:4079', catch_base=False) as __m3_guard_4076_8:
            for p in hf._model.parameters():
                p.requires_grad_(False)

        if __m3_guard_4076_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        wd = float(os.getenv("M3_STABILITY_WEIGHT_DECAY", str(self.stability_cfg.weight_decay)))
        self._bridge_adapt_opt = self.torch.optim.AdamW(
            bridge.parameters(),
            lr=float(self.bridge_adapt_cfg.learning_rate),
            weight_decay=wd,
        )
        return self._bridge_adapt_opt

    def _adapt_bridge_online(
        self,
        hf: HFBackend,
        prompt: str,
        response: str,
        z_m3: Optional[np.ndarray],
    ) -> None:
        if not self._bridge_adapt_enabled:
            return
        if not self.bridge_adapt_cfg.enabled:
            return
        if str(os.getenv("M3_BRIDGE_ONLINE_ADAPT", "1")).lower() not in ("1", "true", "yes", "on"):
            return
        if z_m3 is None:
            return
        bridge = attr_get_optional(hf, "_control_bridge", None)
        if bridge is None or attr_get_optional(hf, "_model", None) is None or attr_get_optional(hf, "_tokenizer", None) is None:
            return
        ok, qinfo = self._evaluate_generation_quality(prompt, response, source="generate")
        reward = float((qinfo or {}).get("score", 0.0)) * float(self.bridge_adapt_cfg.reward_scale)
        if (not ok) or reward < float(self.bridge_adapt_cfg.min_quality_score):
            self._bridge_adapt_fail_streak += 1
            if self._bridge_adapt_fail_streak >= int(max(1, self.bridge_adapt_cfg.cooldown_steps)):
                self._bridge_adapt_enabled = False
            self._log_jsonl(
                os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                {
                    "kind": "bridge_adapt",
                    "status": "skip_low_reward",
                    "reward": float(reward),
                    "fail_streak": int(self._bridge_adapt_fail_streak),
                },
            )
            return
        self._bridge_adapt_fail_streak = 0
        opt = self._ensure_bridge_adapt_optimizer(hf)
        if opt is None:
            return
        tok = hf._tokenizer
        model = hf._model
        try:
            bridge.train()
            input_ids = tok(prompt, return_tensors="pt").get("input_ids")
            resp_ids = tok(response, return_tensors="pt", add_special_tokens=False).get("input_ids")
            if input_ids is None or resp_ids is None:
                return
            input_ids = input_ids.to(hf.device)
            resp_ids = resp_ids.to(hf.device)
            if int(resp_ids.shape[1]) <= 0:
                return
            full_ids = self.torch.cat([input_ids, resp_ids], dim=1)
            attn = self.torch.ones_like(full_ids, device=full_ids.device)
            state_dim = int(os.getenv("M3_BRIDGE_STATE_DIM", "256"))
            z_t = hf._prepare_bridge_state(z_m3, state_dim=state_dim, device=hf.device)
            if z_t is None:
                return
            controls = bridge(z_t, strength=float(os.getenv("M3_BRIDGE_STRENGTH", "1.0")))
            out = model(
                input_ids=full_ids[:, :-1],
                attention_mask=attn[:, :-1],
                use_cache=False,
                output_hidden_states=False,
            )
            logits = out.logits
            if controls.logit_bias is not None and controls.logit_bias.shape[-1] == logits.shape[-1]:
                logits = logits + controls.logit_bias.unsqueeze(1).to(device=logits.device, dtype=logits.dtype)
            p_len = int(input_ids.shape[1])
            r_len = int(resp_ids.shape[1])
            start = max(0, p_len - 1)
            end = min(logits.shape[1], start + r_len)
            if end <= start:
                return
            step_logits = logits[:, start:end, :]
            targets = full_ids[:, p_len:p_len + (end - start)]
            logp = self.torch.log_softmax(step_logits.float(), dim=-1)
            token_logp = self.torch.gather(logp, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            policy_loss = -float(reward) * token_logp.mean()
            reg = bridge.regularization_loss(
                controls=controls,
                gate_reg=float(self.bridge_adapt_cfg.gate_reg),
                bias_reg=float(self.bridge_adapt_cfg.bias_reg),
            )
            loss = policy_loss + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            self._guarded_optimizer_step(
                optimizer=opt,
                params=list(bridge.parameters()),
                tag="bridge_adapt",
            )
            with guard_context(ctx='llm_adapter/llm_core.py:4181', catch_base=False) as __m3_guard_4179_12:
                bridge.renorm_parameters(float(self.stability_cfg.max_weight_norm))

            if __m3_guard_4179_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            self._bridge_adapt_step += 1
            self._log_jsonl(
                os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                {
                    "kind": "bridge_adapt",
                    "status": "ok",
                    "reward": float(reward),
                    "loss": float(loss.detach().item()),
                    "step": int(self._bridge_adapt_step),
                },
            )
        except Exception as e:
            self._bridge_adapt_fail_streak += 1
            if self._bridge_adapt_fail_streak >= int(max(1, self.bridge_adapt_cfg.cooldown_steps)):
                self._bridge_adapt_enabled = False
            self._log_jsonl(
                os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                {
                    "kind": "bridge_adapt",
                    "status": "error",
                    "error": str(e),
                    "fail_streak": int(self._bridge_adapt_fail_streak),
                },
            )
        finally:
            with guard_context(ctx='llm_adapter/llm_core.py:4210', catch_base=False) as __m3_guard_4208_12:
                bridge.eval()

            if __m3_guard_4208_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

    def enable_m3_integration(self) -> bool:
        if attr_has(self, 'model') and attr_has(self.model, 'enable_m3_integration'):
            with guard_context(ctx='llm_adapter/llm_core.py:4218', catch_base=False) as __m3_guard_4215_12:
                self.model.enable_m3_integration()
                return True

            if __m3_guard_4215_12.error is not None:
                return False
        return False

    def _normalize_affect_state(self, affect_state):
        if affect_state is None:
            return None
        if isinstance(affect_state, np.ndarray):
            return affect_state.astype(np.float32)
        if isinstance(affect_state, (list, tuple)):
            return np.asarray(affect_state, dtype=np.float32)
        if isinstance(affect_state, dict):
            primary_keys = ['valence', 'arousal', 'dominance', 'novelty', 'clarity']
            if any(k in affect_state for k in primary_keys):
                return np.array([float(affect_state.get(k, 0.0)) for k in primary_keys], dtype=np.float32)
            secondary_keys = ['arousal', 'valence', 'entropy', 'engagement', 'frustration']
            if any(k in affect_state for k in secondary_keys):
                return np.array([float(affect_state.get(k, 0.0)) for k in secondary_keys], dtype=np.float32)
            with guard_context(ctx='llm_adapter/llm_core.py:4238', catch_base=False) as __m3_guard_4236_12:
                return np.array([float(affect_state[k]) for k in sorted(affect_state.keys())], dtype=np.float32)

            if __m3_guard_4236_12.error is not None:
                return None
        return None

    def _get_affect_state(self, core, affect_state=None):
        if affect_state is not None:
            return self._normalize_affect_state(affect_state)
        if core is None:
            return None
        with guard_context(ctx='llm_adapter/llm_core.py:4250', catch_base=False) as __m3_guard_4247_8:
            if attr_has(core, 'rewards') and attr_get_optional(core.rewards, 'last_affect', None) is not None:
                return np.asarray(core.rewards.last_affect, dtype=np.float32)

        if __m3_guard_4247_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        if attr_has(core, 'affect_kernel'):
            for name in ('get_state', 'get_state_vector'):
                with guard_context(ctx='llm_adapter/llm_core.py:4258', catch_base=False) as __m3_guard_4254_16:
                    fn = attr_get_optional(core.affect_kernel, name, None)
                    if callable(fn):
                        return self._normalize_affect_state(fn())

                if __m3_guard_4254_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
        if attr_has(core, 'qualia'):
            with guard_context(ctx='llm_adapter/llm_core.py:4269', catch_base=False) as __m3_guard_4261_12:
                return np.array([
                    float(attr_get_optional(core.qualia, 'valence', 0.0)),
                    float(attr_get_optional(core.qualia, 'arousal', 0.0)),
                    float(attr_get_optional(core.qualia, 'entropy', 0.0)),
                    float(attr_get_optional(core.qualia, 'engagement', 0.0)),
                    float(attr_get_optional(core.qualia, 'frustration', 0.0)),
                ], dtype=np.float32)

            if __m3_guard_4261_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        return None

    def _update_m3_cache(self, core):
        panels = None
        if core is None:
            return panels
        with guard_context(ctx='llm_adapter/llm_core.py:4280', catch_base=False) as __m3_guard_4277_8:
            self.m3_cache.update(core)
            panels = self.m3_cache.get_current_panels()

        if __m3_guard_4277_8.error is not None:
            panels = None
        return panels

    def _build_m3_memory(self, core=None, panels=None, affect_state=None, drive_state=None):
        if panels is None and core is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:4289', catch_base=False) as __m3_guard_4286_12:
                if attr_has(core, 'feature_bank') and attr_has(core.feature_bank, 'panels'):
                    panels = core.feature_bank.panels(core)

            if __m3_guard_4286_12.error is not None:
                panels = None
        if panels is None:
            return None
        if affect_state is not None:
            affect_state = self._normalize_affect_state(affect_state)
        if not attr_has(self, 'model'):
            return None
        if not attr_has(self.model, 'm3_encoder') or self.model.m3_encoder is None:
            with guard_context(ctx='llm_adapter/llm_core.py:4300', catch_base=False) as __m3_guard_4298_12:
                self.model.enable_m3_integration()

            if __m3_guard_4298_12.error is not None:
                return None
        with guard_context(ctx='llm_adapter/llm_core.py:4304', catch_base=False) as __m3_guard_4302_8:
            return self.model.m3_encoder(panels, affect_state=affect_state, drive_state=drive_state)

        if __m3_guard_4302_8.error is not None:
            return None

    def build_m3_memory(self, core=None, panels=None, affect_state=None, drive_state=None):
        return self._build_m3_memory(core=core, panels=panels, affect_state=affect_state, drive_state=drive_state)

    def _vectorize_panels(self, panels):
        if panels is None:
            return None
        with guard_context(ctx='llm_adapter/llm_core.py:4315', catch_base=False) as __m3_guard_4313_8:
            arr = np.asarray(panels, dtype=np.float32)

        if __m3_guard_4313_8.error is not None:
            return None
        if arr.ndim == 1:
            vec = arr
        elif arr.ndim == 2:
            mode = os.getenv('M3_VEC_MODE', 'mean').lower()
            if mode == 'concat':
                vec = arr.reshape(-1)
            else:
                vec = arr.mean(axis=0)
        else:
            vec = arr.reshape(-1)
        with guard_context(ctx='llm_adapter/llm_core.py:4329', catch_base=False) as __m3_guard_4327_8:
            max_len = int(os.getenv('M3_VEC_MAX_LEN', '256'))

        if __m3_guard_4327_8.error is not None:
            max_len = 256
        if max_len > 0:
            vec = vec[:max_len]
        return vec

    def _snapshot_vector(self, core, affect_state=None):
        if core is None:
            return None
        vecs = []
        with guard_context(ctx='llm_adapter/llm_core.py:4341', catch_base=False) as __m3_guard_4339_8:
            snap = core.snapshot() if attr_has(core, 'snapshot') and callable(core.snapshot) else None

        if __m3_guard_4339_8.error is not None:
            snap = None
        if isinstance(snap, dict):
            for key in ('phi', 'energy', 'activation', 'unity', 'policy_lr', 'steps'):
                with guard_context(ctx='llm_adapter/llm_core.py:4349', catch_base=False) as __m3_guard_4345_16:
                    val = snap.get(key, None)
                    if isinstance(val, (int, float, np.floating, np.integer)):
                        vecs.append(np.array([float(val)], dtype=np.float32))

                if __m3_guard_4345_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
            with guard_context(ctx='llm_adapter/llm_core.py:4357', catch_base=False) as __m3_guard_4351_12:
                emb = snap.get('embeddings', None)
                if emb is None:
                    emb = snap.get('embedding', None)
                if emb is not None:
                    vecs.append(np.asarray(emb, dtype=np.float32).ravel())

            if __m3_guard_4351_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:4373', catch_base=False) as __m3_guard_4359_8:
            q = attr_get_optional(core, 'qualia', None)
            if q is not None:
                qv = np.array(
                    [
                        float(attr_get_optional(q, 'arousal', 0.0)),
                        float(attr_get_optional(q, 'valence', 0.0)),
                        float(attr_get_optional(q, 'entropy', 0.0)),
                        float(attr_get_optional(q, 'engagement', 0.0)),
                        float(attr_get_optional(q, 'frustration', 0.0)),
                    ],
                    dtype=np.float32,
                )
                vecs.append(qv)

        if __m3_guard_4359_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:4379', catch_base=False) as __m3_guard_4375_8:
            if attr_has(core, 'energy_ctrl'):
                energy_ratio = core.energy_ctrl.cognitive_energy / max(core.energy_ctrl.energy_capacity, 1.0)
                vecs.append(np.array([float(energy_ratio)], dtype=np.float32))

        if __m3_guard_4375_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:4391', catch_base=False) as __m3_guard_4381_8:
            if attr_has(core, 'episodic_memory'):
                stats = core.episodic_memory.get_statistics()
                mem_vals = [
                    float(stats.get('total_memories', 0.0)),
                    float(stats.get('total_encoded', 0.0)),
                    float(stats.get('avg_consolidation', 0.0)),
                    float(stats.get('avg_retrieval_count', 0.0)),
                ]
                vecs.append(np.asarray(mem_vals, dtype=np.float32))

        if __m3_guard_4381_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        if affect_state is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:4396', catch_base=False) as __m3_guard_4394_12:
                vecs.append(np.asarray(affect_state, dtype=np.float32).ravel())

            if __m3_guard_4394_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        if not vecs:
            return None
        with guard_context(ctx='llm_adapter/llm_core.py:4402', catch_base=False) as __m3_guard_4400_8:
            vec = np.concatenate(vecs).astype(np.float32)

        if __m3_guard_4400_8.error is not None:
            return None
        return vec

    def _build_full_state_vector(self, core, panels=None, affect_state=None):
        vecs = []
        base = self._snapshot_vector(core, affect_state=affect_state)
        if base is not None:
            vecs.append(base)
        panel_vec = self._vectorize_panels(panels)
        if panel_vec is not None:
            vecs.append(panel_vec.astype(np.float32, copy=False))
        if not vecs:
            return None
        with guard_context(ctx='llm_adapter/llm_core.py:4418', catch_base=False) as __m3_guard_4416_8:
            vec = np.concatenate(vecs).astype(np.float32)

        if __m3_guard_4416_8.error is not None:
            return None
        with guard_context(ctx='llm_adapter/llm_core.py:4422', catch_base=False) as __m3_guard_4420_8:
            max_len = int(os.getenv('M3_VEC_MAX_LEN', '256'))

        if __m3_guard_4420_8.error is not None:
            max_len = 256
        with guard_context(ctx='llm_adapter/llm_core.py:4426', catch_base=False) as __m3_guard_4424_8:
            hard_max = int(os.getenv('M3_VEC_HARD_MAX', '2048'))

        if __m3_guard_4424_8.error is not None:
            hard_max = 2048
        cap = max_len if max_len > 0 else hard_max
        if hard_max > 0:
            cap = min(cap, hard_max)
        if cap > 0:
            vec = vec[:cap]
        return vec

    def _build_m3_context(self, core, affect_state=None, phi_trend=None, panels=None):
        if core is None:
            return ""
        lines = ["M3_STATE:"]
        with guard_context(ctx='llm_adapter/llm_core.py:4442', catch_base=False) as __m3_guard_4439_8:
            phi_hist = core.phi_calculator.phi_history if attr_has(core, 'phi_calculator') else []
            phi = phi_hist[-1] if phi_hist else 0.0

        if __m3_guard_4439_8.error is not None:
            phi = 0.0
        lines.append(f"phi={float(phi):.4f}")
        if phi_trend:
            lines.append(f"phi_trend={phi_trend}")
        with guard_context(ctx='llm_adapter/llm_core.py:4458', catch_base=False) as __m3_guard_4447_8:
            qualia = attr_get_optional(core, 'qualia', None)
            if qualia is not None:
                lines.append(
                    "qualia="
                    f"arousal:{float(attr_get_optional(qualia, 'arousal', 0.0)):.3f},"
                    f"valence:{float(attr_get_optional(qualia, 'valence', 0.0)):.3f},"
                    f"entropy:{float(attr_get_optional(qualia, 'entropy', 0.0)):.3f},"
                    f"engagement:{float(attr_get_optional(qualia, 'engagement', 0.0)):.3f},"
                    f"frustration:{float(attr_get_optional(qualia, 'frustration', 0.0)):.3f}"
                )

        if __m3_guard_4447_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        if affect_state is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:4465', catch_base=False) as __m3_guard_4461_12:
                a = np.asarray(affect_state, dtype=np.float32).ravel()
                a_vals = ",".join(f"{v:.3f}" for v in a[:8])
                lines.append(f"affect=[{a_vals}]")

            if __m3_guard_4461_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:4471', catch_base=False) as __m3_guard_4467_8:
            if attr_has(core, 'energy_ctrl'):
                energy_ratio = core.energy_ctrl.cognitive_energy / max(core.energy_ctrl.energy_capacity, 1.0)
                lines.append(f"energy_ratio={float(energy_ratio):.3f}")

        if __m3_guard_4467_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:4477', catch_base=False) as __m3_guard_4473_8:
            stability = float(attr_get_optional(core, 'world_state', {}).get('stability', 0.0))
            drift = float(attr_get_optional(core, 'world_state', {}).get('delta_hat', 0.0))
            lines.append(f"stability={stability:.3f},drift={drift:.3f}")

        if __m3_guard_4473_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:4484', catch_base=False) as __m3_guard_4479_8:
            if attr_has(core, 'episodic_memory'):
                stats = core.episodic_memory.get_statistics()
                lines.append(f"memories={int(stats.get('total_memories', 0))}")
                lines.append(f"memory_consolidation={float(stats.get('avg_consolidation', 0.0)):.3f}")

        if __m3_guard_4479_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        vec = None
        with guard_context(ctx='llm_adapter/llm_core.py:4490', catch_base=False) as __m3_guard_4487_8:
            if attr_has(core, 'export_state_vector'):
                vec = core.export_state_vector()

        if __m3_guard_4487_8.error is not None:
            vec = None
        if vec is None:
            vec_source = os.getenv('M3_VEC_SOURCE', 'full').strip().lower()
            if vec_source == 'panels':
                vec = self._vectorize_panels(panels)
            elif vec_source == 'snapshot':
                vec = self._snapshot_vector(core, affect_state=affect_state)
            else:
                vec = self._build_full_state_vector(core, panels=panels, affect_state=affect_state)
        if vec is not None:
            try:
                vec_arr = np.asarray(vec, dtype=np.float32).ravel()
                lines.append(f"vector_dim={int(vec_arr.size)}")
                include_vec = os.getenv('M3_STATE_INCLUDE_VECTOR', '0').lower() in ('1', 'true', 'yes', 'on')
                if include_vec and vec_arr.size > 0:
                    with guard_context(ctx='llm_adapter/llm_core.py:4508', catch_base=False) as __m3_guard_4506_20:
                        head_elems = int(os.getenv('M3_STATE_VECTOR_MAX_ELEMS', '32'))

                    if __m3_guard_4506_20.error is not None:
                        head_elems = 32
                    head = vec_arr[:head_elems] if head_elems > 0 else vec_arr
                    vec_str = ",".join(f"{v:.4f}" for v in head.tolist())
                    suffix = ",..." if head_elems > 0 and vec_arr.size > head_elems else ""
                    lines.append(f"vector_head=[{vec_str}{suffix}]")
            except Exception:
                logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:4518', catch_base=False) as __m3_guard_4516_8:
            max_chars = int(os.getenv('M3_STATE_MAX_CHARS', '4000'))

        if __m3_guard_4516_8.error is not None:
            max_chars = 4000
        joined = "\n".join(lines)
        if max_chars > 0 and len(joined) > max_chars:
            if vec is not None:
                trimmed = [
                    ln for ln in lines
                    if not (
                        ln.startswith("vector=")
                        or ln.startswith("vector_head=")
                        or ln.startswith("vector[")
                    )
                ]
                joined = "\n".join(trimmed)
            if max_chars > 0 and len(joined) > max_chars:
                joined = joined[:max_chars]
        return joined

    def start_credit_consumer(self):
        """Start MessageBus credit consumption thread."""
        if self._credit_running or self._bus_inbox is None:
            return
        self._credit_running = True
        self._credit_thread = threading.Thread(target=self._consume_credits, daemon=True)
        self._credit_thread.start()

    def stop_credit_consumer(self):
        """Stop MessageBus credit consumption thread."""
        self._credit_running = False
        if self._credit_thread is not None:
            self._credit_thread.join(timeout=1.0)

    # === Autonomy loop (learned decision/timing; no hand-tuned thresholds) ===
    def _autonomy_core_features(self, core=None) -> Optional[np.ndarray]:
        if core is None:
            return None
        feats: List[float] = []
        with guard_context(ctx='llm_adapter/llm_core.py:4561', catch_base=False) as __m3_guard_4555_8:
            ph = list(attr_get_optional(core.phi_calculator, "phi_history", []) or [])
            last_phi = float(ph[-1]) if ph else 0.0
            mean_phi = float(np.mean(ph[-8:])) if ph else 0.0
            d_phi = float(ph[-1] - ph[-2]) if len(ph) >= 2 else 0.0
            feats.extend([last_phi, mean_phi, d_phi])

        if __m3_guard_4555_8.error is not None:
            feats.extend([0.0, 0.0, 0.0])
        with guard_context(ctx='llm_adapter/llm_core.py:4574', catch_base=False) as __m3_guard_4563_8:
            q = attr_get_optional(core, "qualia", None)
            feats.extend(
                [
                    float(attr_get_optional(q, "arousal", 0.0)),
                    float(attr_get_optional(q, "valence", 0.0)),
                    float(attr_get_optional(q, "entropy", 0.0)),
                    float(attr_get_optional(q, "engagement", 0.0)),
                    float(attr_get_optional(q, "frustration", 0.0)),
                ]
            )

        if __m3_guard_4563_8.error is not None:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        with guard_context(ctx='llm_adapter/llm_core.py:4584', catch_base=False) as __m3_guard_4576_8:
            ec = attr_get_optional(core, "energy_ctrl", None)
            if ec is not None:
                ev = float(attr_get_optional(ec, "cognitive_energy", 0.0))
                cap = float(max(attr_get_optional(ec, "energy_capacity", 1.0), 1e-6))
                feats.append(float(ev / cap))
            else:
                feats.append(0.5)

        if __m3_guard_4576_8.error is not None:
            feats.append(0.5)
        with guard_context(ctx='llm_adapter/llm_core.py:4590', catch_base=False) as __m3_guard_4586_8:
            ws = attr_get_optional(core, "world_state", {}) or {}
            feats.append(float(ws.get("stability", 0.5)))
            feats.append(float(ws.get("delta_hat", 0.0)))

        if __m3_guard_4586_8.error is not None:
            feats.extend([0.5, 0.0])
        with guard_context(ctx='llm_adapter/llm_core.py:4597', catch_base=False) as __m3_guard_4592_8:
            mem_n = 0.0
            if attr_has(core, "episodic_memory") and attr_has(core.episodic_memory, "get_statistics"):
                mem_n = float((core.episodic_memory.get_statistics() or {}).get("total_memories", 0))
            feats.append(float(np.tanh(mem_n / 1000.0)))

        if __m3_guard_4592_8.error is not None:
            feats.append(0.0)

        with guard_context(ctx='llm_adapter/llm_core.py:4603', catch_base=False) as __m3_guard_4600_8:
            speak_rate = float(np.mean(self._autonomy_recent_actions)) if self._autonomy_recent_actions else 0.0
            lam_mean = float(np.mean(self._autonomy_recent_lambda)) if self._autonomy_recent_lambda else 0.0

        if __m3_guard_4600_8.error is not None:
            speak_rate, lam_mean = 0.0, 0.0
        feats.extend(
            [
                float(self._autonomy_credit_ema),
                float(self._autonomy_reward_ema),
                float(speak_rate),
                float(lam_mean),
                float(self._autonomy_last_lambda),
            ]
        )
        arr = np.asarray(feats, dtype=np.float32).reshape(-1)
        if arr.size < self._autonomy_state_raw_dim:
            arr = np.pad(arr, (0, self._autonomy_state_raw_dim - arr.size), mode="constant")
        elif arr.size > self._autonomy_state_raw_dim:
            arr = arr[: self._autonomy_state_raw_dim]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _state_vector(self, core=None):
        """Build autonomy state using core dynamics + BOS latent fallback."""
        torch = self.torch
        t = self.tok
        source = str(attr_get_optional(self.autonomy_rl_cfg, "state_source", "core")).strip().lower()
        base_state = None

        use_bos = source in {"bos", "hybrid", "core", "auto"}
        if use_bos:
            with guard_context(ctx='llm_adapter/llm_core.py:4637', catch_base=False) as __m3_guard_4631_12:
                with torch.no_grad():
                    src_ids = torch.tensor([[t.BOS]], dtype=torch.long, device=self.device)
                    e_src = self.model.emb(src_ids)
                    _, h = self.model.encoder(e_src)
                    base_state = h[-1, :, :].float()

            if __m3_guard_4631_12.error is not None:
                base_state = None

        core_state = None
        if source in {"core", "hybrid", "auto"}:
            with guard_context(ctx='llm_adapter/llm_core.py:4648', catch_base=False) as __m3_guard_4642_12:
                core_vec = self._autonomy_core_features(core)
                if core_vec is not None:
                    with torch.no_grad():
                        cv = torch.tensor(core_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                        core_state = torch.tanh(self._autonomy_state_proj(cv)).float()

            if __m3_guard_4642_12.error is not None:
                core_state = None

        if source == "bos":
            return base_state if base_state is not None else torch.zeros((1, int(self.model.hidden)), dtype=torch.float32, device=self.device)
        if source == "core":
            if core_state is not None:
                return core_state
            if base_state is not None:
                return base_state
            return torch.zeros((1, int(self.model.hidden)), dtype=torch.float32, device=self.device)
        # hybrid/auto
        if base_state is not None and core_state is not None:
            return 0.5 * (base_state + core_state)
        if core_state is not None:
            return core_state
        if base_state is not None:
            return base_state
        return torch.zeros((1, int(self.model.hidden)), dtype=torch.float32, device=self.device)

    def _build_autonomy_prefix(self, s):
        """From state vector s, build continuous prefix embeddings with learned gating."""
        torch = self.torch
        with guard_context(ctx='llm_adapter/llm_core.py:4674', catch_base=False) as __m3_guard_4671_8:
            if attr_has(s, 'dtype') and s.dtype == torch.bfloat16 and self.device.type == 'cpu':
                s = s.float()

        if __m3_guard_4671_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with torch.no_grad():
            P = int(attr_get_optional(self.model, 'prefix_len', 1))
            raw = self.model.state2prefix(s).view(1, -1)
            flat = int(raw.size(-1))
            E = max(1, flat // max(1, P))
            usable = int(P * E)
            if usable < flat:
                raw = raw[:, :usable]
            pref = raw.view(1, P, E)
            gate = torch.sigmoid(self.model.prefix_gate(s)).view(1, P, 1)
            # numpy??bfloat16??吏?먰븯吏 ?딆쓬 ??float32濡?蹂??
            return (gate * pref).float().detach().cpu().numpy()

    def _run_core_steps(self, core, count: int = 0):
        if core is None:
            return
        for _ in range(max(0, int(count))):
            with guard_context(ctx='llm_adapter/llm_core.py:4696', catch_base=False) as __m3_guard_4693_12:
                if attr_has(core, '_single_consciousness_step'):
                    core._single_consciousness_step()

            if __m3_guard_4693_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

    def _run_checkpoint_if_enabled(self, core):
        with guard_context(ctx='llm_adapter/llm_core.py:4704', catch_base=False) as __m3_guard_4700_8:
            if os.getenv('M3_SAVE_EVERY_TURN', '0') in ('1', 'true', 'yes', 'on'):
                if core is not None and attr_has(core, '_save_checkpoint'):
                    core._save_checkpoint()

        if __m3_guard_4700_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

    def _drain_user_message(self):
        import queue
        if not attr_has(self, '_user_queue') or self._user_queue is None:
            return None
        try:
            return self._user_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception:
            return None

    def _handle_user_turn(self, user_msg: str, steps_per_cycle: int) -> str:
        core = attr_get_optional(self, 'core', None)
        response = ''
        if core is not None:
            self._run_core_steps(core, steps_per_cycle)
        try:
            if core is not None and attr_has(core, 'handle_user_message'):
                response = core.handle_user_message(user_msg)
            else:
                with guard_context(ctx='llm_adapter/llm_core.py:4729', catch_base=False) as __m3_guard_4727_16:
                    user_max_len = int(os.getenv('M3_USER_MAX_LEN', '320'))

                if __m3_guard_4727_16.error is not None:
                    user_max_len = 320
                response = self.generate(user_msg, max_len=max(32, user_max_len))
        except Exception as e:
            response = f'[Error: {e}]'

        with guard_context(ctx='llm_adapter/llm_core.py:4739', catch_base=False) as __m3_guard_4735_8:
            cb = attr_get_optional(self, '_on_response', None)
            if cb is not None:
                cb(response)

        if __m3_guard_4735_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

        if core is not None:
            self._run_core_steps(core, min(3, steps_per_cycle))
            self._run_checkpoint_if_enabled(core)
        return response

    def _wait_for_user_interrupt(self, wait_seconds: float):
        import queue
        if wait_seconds <= 0:
            return
        wait_start = time.time()
        while time.time() - wait_start < wait_seconds:
            try:
                msg = self._user_queue.get(timeout=0.1)
                self._user_queue.put(msg)
                break
            except queue.Empty:
                pass
            except Exception:
                break

    def _autonomy_text_embedding(self, text: str, core=None) -> Optional[np.ndarray]:
        s = str(text or "").strip()
        if not s:
            return None
        with guard_context(ctx='llm_adapter/llm_core.py:4771', catch_base=False) as __m3_guard_4766_8:
            if core is not None and attr_has(core, "feature_bank") and attr_has(core.feature_bank, "_hash_embed"):
                dim = int(attr_get_optional(core.feature_bank, "embed_dim", 64))
                emb = core.feature_bank._hash_embed(s, dim)
                return np.asarray(emb, dtype=np.float32).reshape(-1)

        if __m3_guard_4766_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:4775', catch_base=False) as __m3_guard_4773_8:
            return np.asarray(self.embed_text(s), dtype=np.float32).reshape(-1)

        if __m3_guard_4773_8.error is not None:
            return None

    def _autonomy_novelty(self, emb: Optional[np.ndarray]) -> float:
        if emb is None:
            return 0.0
        with guard_context(ctx='llm_adapter/llm_core.py:4799', catch_base=False) as __m3_guard_4781_8:
            if not self._autonomy_recent_embeddings:
                self._autonomy_recent_embeddings.append(np.asarray(emb, dtype=np.float32).reshape(-1))
                return 1.0
            cur = np.asarray(emb, dtype=np.float32).reshape(-1)
            dists = []
            for prev in list(self._autonomy_recent_embeddings):
                if prev is None:
                    continue
                p = np.asarray(prev, dtype=np.float32).reshape(-1)
                if p.size != cur.size:
                    continue
                dists.append(float(np.linalg.norm(cur - p)))
            self._autonomy_recent_embeddings.append(cur)
            if not dists:
                return 0.0
            md = float(np.mean(dists))
            return float(np.tanh(md))

        if __m3_guard_4781_8.error is not None:
            return 0.0

    def _repetition_penalty(self, text: str) -> float:
        with guard_context(ctx='llm_adapter/llm_core.py:4815', catch_base=False) as __m3_guard_4803_8:
            toks = [t for t in re.split(r"\s+", str(text or "").strip()) if t]
            if len(toks) < 4:
                return 0.0
            uniq = len(set(toks))
            token_rep = 1.0 - float(uniq / max(1, len(toks)))
            phrase_rep = self._phrase_repetition_score(text)
            sem_dup = self._semantic_dup_score(text, window=6)
            loop_hit = 1.0 if self._has_infinite_loop_pattern(text) else 0.0
            rep = max(float(token_rep), float(phrase_rep), float(sem_dup))
            rep = float(rep + 0.50 * loop_hit)
            return float(max(0.0, min(1.0, rep)))

        if __m3_guard_4803_8.error is not None:
            return 0.0

    def _compute_autonomy_reward(self, prompt: str, response: str, core=None) -> Tuple[float, Dict[str, float]]:
        w1 = float(self.autonomy_rl_cfg.reward_w_dialog)
        w2 = float(self.autonomy_rl_cfg.reward_w_quality)
        w3 = float(self.autonomy_rl_cfg.reward_w_bus_credit)
        w4 = float(self.autonomy_rl_cfg.reward_w_novelty)
        dialog_score = 0.0
        quality_score = 0.0
        safety_penalty = 0.0
        repetition_penalty = 0.0
        response_text = str(response or "").strip()
        with guard_context(ctx='llm_adapter/llm_core.py:4833', catch_base=False) as __m3_guard_4828_8:
            if core is not None and attr_has(core, "_evaluate_dialog_accuracy"):
                m = core._evaluate_dialog_accuracy(str(prompt or ""), response_text)
                if isinstance(m, dict):
                    dialog_score = float(m.get("score", 0.0))

        if __m3_guard_4828_8.error is not None:
            dialog_score = 0.0
        with guard_context(ctx='llm_adapter/llm_core.py:4840', catch_base=False) as __m3_guard_4835_8:
            ok, qinfo = self._evaluate_generation_quality(prompt, response_text, source="autonomy")
            quality_score = float((qinfo or {}).get("score", 0.0))
            if not ok:
                safety_penalty += float(self.autonomy_rl_cfg.safety_penalty)

        if __m3_guard_4835_8.error is not None:
            quality_score = 0.0
        if self._is_disallowed_generation_output(response_text):
            safety_penalty += float(self.autonomy_rl_cfg.safety_penalty)
        repetition_ratio = self._repetition_penalty(response_text)
        repetition_penalty += float(self.autonomy_rl_cfg.repetition_penalty) * repetition_ratio
        novelty = self._autonomy_novelty(self._autonomy_text_embedding(response_text, core=core))
        bus_credit = float(self._autonomy_credit_ema)
        reward = (
            w1 * dialog_score
            + w2 * quality_score
            + w3 * bus_credit
            + w4 * novelty
            - safety_penalty
            - repetition_penalty
        )
        self._autonomy_reward_ema = 0.9 * float(self._autonomy_reward_ema) + 0.1 * float(reward)
        info = {
            "dialog_score": float(dialog_score),
            "quality_score": float(quality_score),
            "bus_credit": float(bus_credit),
            "novelty": float(novelty),
            "safety_penalty": float(safety_penalty),
            "repetition_penalty": float(repetition_penalty),
            "reward": float(reward),
        }
        return float(reward), info

    def _update_autonomy_q(
        self,
        state_t,
        action: int,
        reward: float,
        next_state_t,
        done: bool = False,
    ) -> float:
        torch = self.torch
        s = state_t.detach().to(device=self.device, dtype=torch.float32)
        s2 = next_state_t.detach().to(device=self.device, dtype=torch.float32)
        a = int(action)
        r = float(reward)
        self._autonomy_replay.append((s.detach().cpu(), a, r, s2.detach().cpu(), bool(done)))
        samples = []
        batch_size = int(max(1, self.autonomy_rl_cfg.batch_size))
        if len(self._autonomy_replay) >= batch_size:
            idx = np.random.choice(len(self._autonomy_replay), size=batch_size, replace=False)
            samples = [self._autonomy_replay[int(i)] for i in idx]
        else:
            samples = [self._autonomy_replay[-1]]
        if not samples:
            return 0.0
        states = torch.stack([x[0] for x in samples], dim=0).to(self.device).float()
        if states.ndim > 2:
            states = states.view(states.size(0), -1)
        actions = torch.tensor([int(x[1]) for x in samples], dtype=torch.long, device=self.device)
        rewards = torch.tensor([float(x[2]) for x in samples], dtype=torch.float32, device=self.device)
        next_states = torch.stack([x[3] for x in samples], dim=0).to(self.device).float()
        if next_states.ndim > 2:
            next_states = next_states.view(next_states.size(0), -1)
        dones = torch.tensor([1.0 if bool(x[4]) else 0.0 for x in samples], dtype=torch.float32, device=self.device)
        # Reward normalization (advantage-like) to avoid frozen Q under skewed rewards.
        norm_alpha = float(max(0.0, min(0.999, attr_get_optional(self.autonomy_rl_cfg, "reward_norm_ema", 0.95))))
        if norm_alpha > 0.0:
            rewards_norm = rewards - float(self._autonomy_reward_ema)
            rewards = (1.0 - norm_alpha) * rewards + norm_alpha * rewards_norm

        q = self.model.q_head(states)
        if q.ndim > 2:
            q = q.view(q.size(0), -1)
        q_sa = q.gather(dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            q_next = self.model.q_head(next_states).max(dim=1)[0]
            gamma = float(self.autonomy_rl_cfg.gamma)
            target = rewards + (1.0 - dones) * gamma * q_next
        td = q_sa - target
        td_clip = float(max(1e-6, self.autonomy_rl_cfg.td_clip))
        td = torch.clamp(td, -td_clip, td_clip)
        td_loss = torch.mean(td ** 2)

        intensity = torch.nn.functional.softplus(self.model.intensity_head(states)).squeeze(-1)
        target_intensity = torch.clamp(torch.abs(rewards), 0.0, 2.0)
        inten_loss = torch.mean((intensity - target_intensity) ** 2)
        # Encourage lambda/intensity to match desired speak ratio.
        speak_target = float(max(0.01, min(0.99, attr_get_optional(self.autonomy_rl_cfg, "target_speak_rate", 0.35))))
        speak_prob = torch.sigmoid((q[:, 1] - q[:, 0]) * torch.clamp(intensity.detach(), 0.1, 5.0))
        rate_loss = torch.mean((speak_prob - speak_target) ** 2)
        loss = td_loss + 0.05 * inten_loss + 0.15 * rate_loss

        params = self._autonomy_params()
        self._autonomy_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_sq = 0.0
        for p in params:
            if p.grad is None:
                continue
            with guard_context(ctx='llm_adapter/llm_core.py:4938', catch_base=False) as __m3_guard_4935_12:
                gnorm = torch.norm(p.grad.detach()).item()
                grad_sq += float(gnorm) * float(gnorm)

            if __m3_guard_4935_12.error is not None:
                continue
        pre_vec = []
        with guard_context(ctx='llm_adapter/llm_core.py:4945', catch_base=False) as __m3_guard_4941_8:
            with torch.no_grad():
                for p in params:
                    pre_vec.append(p.detach().view(-1).float().cpu())

        if __m3_guard_4941_8.error is not None:
            pre_vec = []

        self._guarded_optimizer_step(
            optimizer=self._autonomy_opt,
            params=params,
            tag="autonomy_q",
            clip_override=float(self.stability_cfg.grad_clip_norm),
        )
        param_delta = 0.0
        with guard_context(ctx='llm_adapter/llm_core.py:4970', catch_base=False) as __m3_guard_4955_8:
            if pre_vec:
                with torch.no_grad():
                    i0 = 0
                    delta_sq = 0.0
                    for p in params:
                        n = int(p.numel())
                        if n <= 0:
                            continue
                        old = pre_vec[i0]
                        i0 += 1
                        cur = p.detach().view(-1).float().cpu()
                        d = torch.norm(cur - old).item()
                        delta_sq += float(d) * float(d)
                    param_delta = float(np.sqrt(max(0.0, delta_sq)))

        if __m3_guard_4955_8.error is not None:
            param_delta = 0.0
        self._autonomy_last_diag = {
            "q_grad_norm": float(np.sqrt(max(0.0, grad_sq))),
            "q_param_delta": float(param_delta),
            "rate_loss": float(rate_loss.detach().item()),
        }
        return float(td_loss.detach().item())

    def _run_autonomy_turn(self, cycle_count: int, autonomy_check_every: int):
        if autonomy_check_every <= 0 or cycle_count % autonomy_check_every != 0:
            return
        if attr_get_optional(self, "_hf_circuit_open", False):
            # CUDA fault detected: skip GPU-dependent autonomy policy path.
            return
        core = attr_get_optional(self, 'core', None)
        try:
            s = self._state_vector(core)
            torch = self.torch
            with guard_context(ctx='llm_adapter/llm_core.py:4992', catch_base=False) as __m3_guard_4989_12:
                if attr_has(s, 'dtype') and s.dtype == torch.bfloat16 and self.device.type == 'cpu':
                    s = s.float()

            if __m3_guard_4989_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            with torch.no_grad():
                q = self.model.q_head(s.float())  # (1, 2): [Q_wait, Q_speak]
                with guard_context(ctx='llm_adapter/llm_core.py:4998', catch_base=False) as __m3_guard_4996_16:
                    min_prob = float(os.getenv('M3_MIN_SPEAK_PROB', '0.2'))

                if __m3_guard_4996_16.error is not None:
                    min_prob = 0.2
                diff = float(q[0, 1].item() - q[0, 0].item())
                lam_raw = torch.nn.functional.softplus(self.model.intensity_head(s.float())).item() + 1e-8
                lam_min = float(max(1e-4, attr_get_optional(self.autonomy_rl_cfg, "lambda_min", 0.2)))
                lam_max = float(max(lam_min + 1e-6, attr_get_optional(self.autonomy_rl_cfg, "lambda_max", 3.0)))
                lam = float(np.clip(lam_raw, lam_min, lam_max))
                prob = 1.0 / (1.0 + np.exp(-(diff * lam)))
                prob = max(min_prob, prob)
                speak = bool(np.random.rand() < prob)

            action_idx = 1 if speak else 0
            prev_lam = float(attr_get_optional(self, "_autonomy_last_lambda", 0.0))
            lam_delta = float(lam - prev_lam) if prev_lam != 0.0 else 0.0
            self._autonomy_last_lambda = float(lam)
            self._autonomy_recent_lambda.append(float(lam))
            self._autonomy_recent_actions.append(float(1 if speak else 0))
            reward = 0.0
            reward_info: Dict[str, float] = {}
            td_loss = 0.0
            if not speak:
                with guard_context(ctx='llm_adapter/llm_core.py:5023', catch_base=False) as __m3_guard_5019_16:
                    dt = float(np.random.exponential(1.0 / max(lam, 1e-12)))
                    dt = min(dt, 5.0)
                    self._wait_for_user_interrupt(dt)

                if __m3_guard_5019_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                # --- Wait reward with speak-rate homeostasis ---
                speak_rate = float(np.mean(self._autonomy_recent_actions)) if self._autonomy_recent_actions else 0.5
                target_rate = float(max(0.01, min(0.99, attr_get_optional(self.autonomy_rl_cfg, "target_speak_rate", 0.35))))
                # Positive bonus for waiting when already speaking too much
                rate_bonus = max(0.0, (speak_rate - target_rate)) * 0.5
                # Energy conservation bonus: reward waiting when energy is low
                energy_bonus = 0.0
                with guard_context(ctx='llm_adapter/llm_core.py:5037', catch_base=False) as __m3_guard_5032_16:
                    if core is not None and attr_has(core, 'energy_ctrl'):
                        e_ratio = float(core.energy_ctrl.cognitive_energy / max(1.0, core.energy_ctrl.energy_capacity))
                        if e_ratio < 0.3:
                            energy_bonus = 0.15 * (0.3 - e_ratio) / 0.3

                if __m3_guard_5032_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                reward = -0.01 + 0.10 * float(self._autonomy_credit_ema) + rate_bonus + energy_bonus
                reward_info = {
                    "dialog_score": 0.0,
                    "quality_score": 0.0,
                    "bus_credit": float(self._autonomy_credit_ema),
                    "novelty": 0.0,
                    "safety_penalty": 0.0,
                    "repetition_penalty": 0.0,
                    "rate_bonus": float(rate_bonus),
                    "energy_bonus": float(energy_bonus),
                    "speak_rate": float(speak_rate),
                    "reward": float(reward),
                }
                if self._autonomy_running:
                    next_s = self._state_vector(core)
                    td_loss = self._update_autonomy_q(s.float(), action_idx, reward, next_s.float(), done=False)
                with guard_context(ctx='llm_adapter/llm_core.py:5071', catch_base=False) as __m3_guard_5055_16:
                    self._log_jsonl(
                        os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                        {
                            "kind": "autonomy_rl",
                            "action": "wait",
                            "q_wait": float(q[0, 0].item()),
                            "q_speak": float(q[0, 1].item()),
                            "lambda": float(lam),
                            "lambda_delta": float(lam_delta),
                            "reward": float(reward),
                            "td_loss": float(td_loss),
                            **dict(attr_get_optional(self, "_autonomy_last_diag", {}) or {}),
                            **reward_info,
                        },
                    )

                if __m3_guard_5055_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                return

            prefix = self._build_autonomy_prefix(s)
            seed = self._autonomy_seed_prompt(core)
            with guard_context(ctx='llm_adapter/llm_core.py:5079', catch_base=False) as __m3_guard_5077_12:
                auto_max_len = int(os.getenv('M3_AUTONOMY_MAX_LEN', '160'))

            if __m3_guard_5077_12.error is not None:
                auto_max_len = 160
            text = self.generate(seed, max_len=max(32, auto_max_len), prefix_embed=prefix, source="autonomy")

            # --- Semantic dedup gate: suppress utterances too similar to recent ones ---
            if text and text.strip() and bool(attr_get_optional(self.autonomy_rl_cfg, "semantic_dedup_enabled", True)):
                dedup_threshold = float(attr_get_optional(self.autonomy_rl_cfg, "semantic_dedup_threshold", 0.25))
                dedup_max_retries = int(max(0, attr_get_optional(self.autonomy_rl_cfg, "semantic_dedup_max_retries", 2)))
                for _dedup_attempt in range(dedup_max_retries + 1):
                    emb = self._autonomy_text_embedding(text.strip(), core=core)
                    novelty = self._autonomy_novelty(emb) if emb is not None else 1.0
                    if novelty >= dedup_threshold:
                        break  # novel enough
                    if _dedup_attempt < dedup_max_retries:
                        # Re-generate with higher temperature to get diverse output
                        text = self.generate(seed, max_len=max(32, auto_max_len), prefix_embed=prefix, source="autonomy")
                        if not text or not text.strip():
                            break
                    else:
                        # All retries exhausted ??suppress and penalize
                        suppress_penalty = -float(attr_get_optional(self.autonomy_rl_cfg, "repetition_penalty", 0.35))
                        if self._autonomy_running:
                            next_s = self._state_vector(core)
                            self._update_autonomy_q(s.float(), action_idx, suppress_penalty, next_s.float(), done=False)
                        with guard_context(ctx='llm_adapter/llm_core.py:5115', catch_base=False) as __m3_guard_5103_24:
                            self._log_jsonl(
                                os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                                {
                                    "kind": "autonomy_semantic_dedup",
                                    "novelty": float(novelty),
                                    "threshold": float(dedup_threshold),
                                    "retries": int(dedup_max_retries),
                                    "suppressed_text": str(text or "")[:120],
                                    "reward": float(suppress_penalty),
                                },
                            )

                        if __m3_guard_5103_24.error is not None:
                            logging.getLogger(__name__).exception("Swallowed exception")
                        return  # suppress this utterance entirely

            if not text or not text.strip():
                reward = -float(self.autonomy_rl_cfg.safety_penalty)
                if self._autonomy_running:
                    next_s = self._state_vector(core)
                    td_loss = self._update_autonomy_q(s.float(), action_idx, reward, next_s.float(), done=False)
                with guard_context(ctx='llm_adapter/llm_core.py:5145', catch_base=False) as __m3_guard_5124_16:
                    self._log_jsonl(
                        os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                        {
                            "kind": "autonomy_rl",
                            "action": "speak",
                            "q_wait": float(q[0, 0].item()),
                            "q_speak": float(q[0, 1].item()),
                            "lambda": float(lam),
                            "lambda_delta": float(lam_delta),
                            "reward": float(reward),
                            "td_loss": float(td_loss),
                            **dict(attr_get_optional(self, "_autonomy_last_diag", {}) or {}),
                            "dialog_score": 0.0,
                            "quality_score": 0.0,
                            "bus_credit": float(self._autonomy_credit_ema),
                            "novelty": 0.0,
                            "safety_penalty": float(self.autonomy_rl_cfg.safety_penalty),
                            "repetition_penalty": 0.0,
                        },
                    )

                if __m3_guard_5124_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                return
            text = text.strip()
            if self._is_disallowed_generation_output(text):
                logger.warning('Autonomy generation filtered by safety policy')
                reward = -float(self.autonomy_rl_cfg.safety_penalty)
                if self._autonomy_running:
                    next_s = self._state_vector(core)
                    td_loss = self._update_autonomy_q(s.float(), action_idx, reward, next_s.float(), done=False)
                with guard_context(ctx='llm_adapter/llm_core.py:5176', catch_base=False) as __m3_guard_5155_16:
                    self._log_jsonl(
                        os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                        {
                            "kind": "autonomy_rl",
                            "action": "speak",
                            "q_wait": float(q[0, 0].item()),
                            "q_speak": float(q[0, 1].item()),
                            "lambda": float(lam),
                            "lambda_delta": float(lam_delta),
                            "reward": float(reward),
                            "td_loss": float(td_loss),
                            **dict(attr_get_optional(self, "_autonomy_last_diag", {}) or {}),
                            "dialog_score": 0.0,
                            "quality_score": 0.0,
                            "bus_credit": float(self._autonomy_credit_ema),
                            "novelty": 0.0,
                            "safety_penalty": float(self.autonomy_rl_cfg.safety_penalty),
                            "repetition_penalty": 0.0,
                        },
                    )

                if __m3_guard_5155_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                return
            numeric_dump = self._is_numeric_dump_response(text)
            if (
                os.getenv('M3_AUTONOMY_LEARN_FROM_SELF', '0').lower() in ('1', 'true', 'yes', 'on')
                and not numeric_dump
            ):
                with guard_context(ctx='llm_adapter/llm_core.py:5186', catch_base=False) as __m3_guard_5184_16:
                    self.learn_pair(seed, text, max_len=max(32, auto_max_len))

                if __m3_guard_5184_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")

            if core is not None and attr_has(core, 'bus') and core.bus is not None:
                with guard_context(ctx='llm_adapter/llm_core.py:5197', catch_base=False) as __m3_guard_5190_16:
                    vec = np.zeros((32,), dtype=np.float32)
                    if attr_has(core, 'feature_bank') and attr_has(core.feature_bank, '_hash_embed'):
                        vec = core.feature_bank._hash_embed(
                            text, core.feature_bank.embed_dim
                        ).astype(np.float32)
                    core.bus.push('system', 'utter.self', vec, salience=0.8, confidence=1.0, ttl=10)

                if __m3_guard_5190_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")

            scb = attr_get_optional(self, '_on_spontaneous', None)
            if scb is not None:
                with guard_context(ctx='llm_adapter/llm_core.py:5204', catch_base=False) as __m3_guard_5202_16:
                    scb(text, float(q[0, 1].item()), lam)

                if __m3_guard_5202_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")

            reward, reward_info = self._compute_autonomy_reward(seed, text, core=core)
            if self._autonomy_running:
                next_s = self._state_vector(core)
                td_loss = self._update_autonomy_q(s.float(), action_idx, reward, next_s.float(), done=False)

            # ---- NeuroModulator online learning ----
            try:
                _nm_cfg = get_global_config().neuro_modulator
                _nm_env_flag = os.getenv('M3_ENABLE_NEURO_MODULATOR', '').lower()
                _nm_active = (_nm_env_flag in ('1', 'true', 'yes', 'on')) if _nm_env_flag else _nm_cfg.enabled
                if (
                    self.use_hf
                    and HFBackend.is_available()
                    and _nm_active
                ):
                    _hf = HFBackend.get_instance()
                    _nm = attr_get_optional(_hf, '_neuro_modulator', None)
                    _nm_opt = attr_get_optional(_hf, '_neuro_mod_opt', None)
                    if _nm is not None and _nm_opt is not None:
                        _z_nm = self._build_full_state_vector(core=core)
                        if _z_nm is not None:
                            _z_t = _hf._prepare_bridge_state(
                                _z_nm, _hf._neuro_mod_state_dim, _hf.device
                            )
                            if _z_t is not None:
                                with guard_context(ctx='llm_adapter/llm_core.py:5236', catch_base=False) as __m3_guard_5232_32:
                                    _neuro_str = float(
                                        os.getenv('M3_NEURO_STRENGTH', str(_nm_cfg.strength))
                                    )

                                if __m3_guard_5232_32.error is not None:
                                    _neuro_str = _nm_cfg.strength
                                _nm.train()
                                _nm_opt.zero_grad(set_to_none=True)
                                _nloss = _nm.online_loss(
                                    _z_t,
                                    reward=float(reward),
                                    strength=_neuro_str,
                                )
                                _nloss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    _nm.parameters(), _nm_cfg.grad_clip_norm
                                )
                                _nm_opt.step()
                                # Track online-learning steps separately from _nm._step (which is tied to forward())
                                _online_step = attr_get_optional(_hf, "_neuro_mod_online_step", 0) + 1
                                _hf._neuro_mod_online_step = _online_step
                                _nm.eval()
                                # Persist checkpoint every 100 online-learning steps
                                if _online_step % 100 == 0:
                                    with guard_context(ctx='llm_adapter/llm_core.py:5258', catch_base=False) as __m3_guard_5256_36:
                                        _hf._save_neuro_checkpoint()

                                    if __m3_guard_5256_36.error is not None:
                                        _ckpt_err = __m3_guard_5256_36.error
                                        logging.getLogger('llm_adapter').debug(
                                            f'[NeuroMod] periodic checkpoint failed: {_ckpt_err}'
                                        )
            except Exception:
                logging.getLogger(__name__).exception("Swallowed exception")

            with guard_context(ctx='llm_adapter/llm_core.py:5293', catch_base=False) as __m3_guard_5265_12:
                self._log_jsonl(
                    os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                    {
                        "kind": "autonomy_event",
                        "lambda": lam,
                        "lambda_delta": float(lam_delta),
                        "q_wait": float(q[0, 0].item()),
                        "q_speak": float(q[0, 1].item()),
                        **dict(attr_get_optional(self, "_autonomy_last_diag", {}) or {}),
                        "text": text,
                    },
                )
                self._log_jsonl(
                    os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                    {
                        "kind": "autonomy_rl",
                        "action": "speak",
                        "q_wait": float(q[0, 0].item()),
                        "q_speak": float(q[0, 1].item()),
                        "lambda": float(lam),
                        "lambda_delta": float(lam_delta),
                        "reward": float(reward),
                        "td_loss": float(td_loss),
                        **dict(attr_get_optional(self, "_autonomy_last_diag", {}) or {}),
                        **reward_info,
                    },
                )

            if __m3_guard_5265_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        except Exception as e:
            with guard_context(ctx='llm_adapter/llm_core.py:5298', catch_base=False) as __m3_guard_5296_12:
                logger.error(f'Autonomy decision error: {e}')

            if __m3_guard_5296_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            with guard_context(ctx='llm_adapter/llm_core.py:5304', catch_base=False) as __m3_guard_5300_12:
                note = attr_get_optional(self, "_note_control_health", None)
                if callable(note):
                    note(False, f"autonomy_error:{e}")

            if __m3_guard_5300_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

    def start_autonomy_loop(self):
        if attr_get_optional(self, '_auto_running', False):
            return
        self._auto_running = True
        self._user_queue = attr_get_optional(self, '_user_queue', None)
        if self._user_queue is None:
            import queue
            self._user_queue = queue.Queue()
        self._auto_thread = threading.Thread(target=self._unified_loop, daemon=True)
        self._auto_thread.start()

    def stop_autonomy_loop(self):
        self._auto_running = False
        th = attr_get_optional(self, '_auto_thread', None)
        if th is not None:
            th.join(timeout=2.0)

    def submit_user_message(self, text: str):
        """?ъ슜??硫붿떆吏瑜??먯쑉?ш퀬 猷⑦봽??鍮꾨룞湲??꾨떖 (?쇰툝濡쒗궧)"""
        import queue
        if not attr_has(self, '_user_queue') or self._user_queue is None:
            self._user_queue = queue.Queue()
        self._user_queue.put(text)

    def _autonomy_seed_prompt(self, core=None) -> str:
        """Build a minimal non-empty prompt for spontaneous generation.

        Empty prompts often cause instruct models to drift into low-quality text.
        This seed stays short and uses current M3 state when available.

        When NeuroModulator is active, the seed omits numeric state values
        because consciousness is already injected at the weight level.
        """
        with guard_context(ctx='llm_adapter/llm_core.py:5342', catch_base=False) as __m3_guard_5340_8:
            lang = os.getenv('M3_AUTONOMY_LANGUAGE', 'ko').lower()

        if __m3_guard_5340_8.error is not None:
            lang = 'ko'

        # When neuro modulation is active, use a clean seed without state dumps.
        # The consciousness state flows through the decoder hooks, not the prompt.
        neuro_active = os.getenv(
            'M3_ENABLE_NEURO_MODULATOR', '0'
        ).lower() in ('1', 'true', 'yes', 'on')

        if neuro_active:
            if lang.startswith('ko'):
                return "[?먯쑉 紐⑤뱶] 吏湲??좎삤瑜대뒗 ?앷컖???먯쑀濡?쾶 吏㏐쾶 留먰빐以?"
            return "[Autonomy] Speak your mind freely in one short sentence."

        # Light state summary (best-effort) ??legacy prompt-based path
        bits = []
        with guard_context(ctx='llm_adapter/llm_core.py:5363', catch_base=False) as __m3_guard_5358_8:
            if core is not None and attr_has(core, 'energy_ctrl'):
                ec = core.energy_ctrl
                ratio = float(ec.cognitive_energy / max(ec.energy_capacity, 1.0))
                bits.append(f"energy={ratio:.2f}")

        if __m3_guard_5358_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:5370', catch_base=False) as __m3_guard_5365_8:
            if core is not None and attr_has(core, 'qualia'):
                q = core.qualia
                bits.append(f"arousal={float(attr_get_optional(q, 'arousal', 0.0)):.2f}")
                bits.append(f"valence={float(attr_get_optional(q, 'valence', 0.0)):.2f}")

        if __m3_guard_5365_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:5375', catch_base=False) as __m3_guard_5372_8:
            if core is not None and attr_has(core, 'phi_calculator') and core.phi_calculator.phi_history:
                bits.append(f"phi={float(core.phi_calculator.phi_history[-1]):.3f}")

        if __m3_guard_5372_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        state_line = (" (" + ", ".join(bits) + ")") if bits else ""

        if lang.startswith('ko'):
            return (
                "[?먯쑉 紐⑤뱶] ?꾩옱 M3_STATE瑜?諛뷀깢?쇰줈, 吏湲??댁빞 ????媛吏瑜?吏㏐쾶 留먰빐以?" + state_line
            )
        return (
            "[Autonomy] Based on the current M3_STATE, say one short next action." + state_line
        )

    def _parse_user_m3_transcript(self, text: str) -> Optional[List[Dict[str, str]]]:
        """Parse a 'User: ...\nM3: ...' style transcript into chat messages.

        Returns None if the text doesn't look like a transcript.
        """
        try:
            s = str(text)
        except Exception:
            return None

        if 'User:' not in s and 'M3:' not in s:
            return None

        lines = s.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        messages: List[Dict[str, str]] = []
        cur_role = None
        cur_buf: List[str] = []

        def _flush():
            nonlocal cur_role, cur_buf
            if cur_role is None:
                cur_buf = []
                return
            content = "\n".join(cur_buf).strip()
            if content:
                role = 'assistant' if cur_role == 'assistant' else 'user'
                messages.append({'role': role, 'content': content})
            cur_role = None
            cur_buf = []

        for ln in lines:
            if ln.startswith('User:'):
                _flush()
                cur_role = 'user'
                cur_buf = [ln[len('User:'):].lstrip()]
                continue
            if ln.startswith('M3:'):
                _flush()
                cur_role = 'assistant'
                cur_buf = [ln[len('M3:'):].lstrip()]
                continue
            # Continuation line
            if cur_role is None:
                # If transcript markers exist but no role yet, treat as user.
                cur_role = 'user'
                cur_buf = [ln]
            else:
                cur_buf.append(ln)

        _flush()

        # Drop trailing empty assistant prompt marker (common: ends with 'M3:' only)
        if messages and messages[-1].get('role') == 'assistant' and not messages[-1].get('content', '').strip():
            messages.pop()
        # If we only got one blob, don't treat it as transcript.
        if sum(1 for m in messages if m.get('role') == 'user') == 0:
            return None
        if len(messages) < 2:
            return None
        return messages

    def _unified_loop(self):
        import time
        core = attr_get_optional(self, 'core', None)
        consciousness_interval = float(os.getenv('M3_CONSCIOUSNESS_INTERVAL', '0.1'))
        steps_per_cycle = int(os.getenv('M3_STEPS_PER_CYCLE', '5'))
        autonomy_check_every = int(os.getenv('M3_AUTONOMY_CHECK_EVERY', '10'))
        cycle_count = 0

        while attr_get_optional(self, '_auto_running', False):
            try:
                user_msg = self._drain_user_message()
                if user_msg is not None:
                    self._handle_user_turn(user_msg, steps_per_cycle=steps_per_cycle)
                    cycle_count = 0
                    continue

                self._run_core_steps(core, steps_per_cycle)
                cycle_count += 1
                self._run_autonomy_turn(cycle_count, autonomy_check_every)
                time.sleep(consciousness_interval)
            except Exception as e:
                with guard_context(ctx='llm_adapter/llm_core.py:5471', catch_base=False) as __m3_guard_5469_16:
                    logger.error(f'Unified loop error: {e}')

                if __m3_guard_5469_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                time.sleep(1.0)
    def _consume_credits(self):
        """Background thread consuming credit messages from MessageBus."""
        import time
        while self._credit_running:
            try:
                if self._bus_inbox is None:
                    time.sleep(0.1)
                    continue
                
                # Non-blocking get
                with guard_context(ctx='llm_adapter/llm_core.py:5486', catch_base=True) as __m3_guard_5484_16:
                    msg = self._bus_inbox.get_nowait()

                if __m3_guard_5484_16.error is not None:
                    time.sleep(0.05)
                    continue
                
                # Process credit message
                if attr_has(msg, "type") and msg.type == "credit":
                    self._process_credit_message(msg)
            except Exception:
                time.sleep(0.1)

    def _process_credit_message(self, msg):
        """Process a single credit message from MessageBus."""
        with guard_context(ctx='llm_adapter/llm_core.py:5529', catch_base=False) as __m3_guard_5498_8:
            payload = attr_get_optional(msg, "payload", {})
            credit = float(payload.get("credit", 0.0))
            signal = payload.get("signal", {})
            
            # Update last value estimates for beta scheduling
            phi_delta = float(signal.get("phi_delta", 0.0))
            stability_delta = float(signal.get("stability_delta", 0.0))
            tool_success = float(signal.get("tool_success", 0.0))
            
            # EMA update
            ema_alpha = 0.3
            self._last_value_estimates["phi_delta"] = (
                (1.0 - ema_alpha) * self._last_value_estimates.get("phi_delta", 0.0) +
                ema_alpha * phi_delta
            )
            self._last_value_estimates["stability"] = (
                (1.0 - ema_alpha) * self._last_value_estimates.get("stability", 0.5) +
                ema_alpha * max(0.0, min(1.0, 0.5 + stability_delta))
            )
            self._last_value_estimates["tool_success"] = (
                (1.0 - ema_alpha) * self._last_value_estimates.get("tool_success", 0.0) +
                ema_alpha * tool_success
            )
            self._autonomy_credit_ema = (
                (1.0 - ema_alpha) * float(attr_get_optional(self, "_autonomy_credit_ema", 0.0))
                + ema_alpha * float(credit)
            )

        if __m3_guard_5498_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

    def load_model(self, path: str):
        """Loads the model state dictionary."""
        with guard_context(ctx='llm_adapter/llm_core.py:5541', catch_base=False) as __m3_guard_5534_8:
            if not os.path.exists(path):
                logger.warning(f"Model file not found at {path}, skipping load.")
                return
            self.model.load_state_dict(self.torch.load(path, map_location=self.device))
            self.model.eval()  # Set to evaluation mode after loading
            logger.info(f"Model loaded from {path}")

        if __m3_guard_5534_8.error is not None:
            e = __m3_guard_5534_8.error
            logger.error(f"Error loading model from {path}: {e}")

    def _beta_schedule(self) -> float:
        """
        D. Temporal Contextualization (Dynamic Memory)
        
        Returns:
             [beta_min, beta_max]
        """
        with guard_context(ctx='llm_adapter/llm_core.py:5567', catch_base=False) as __m3_guard_5551_8:
            phi_delta = float(self._last_value_estimates.get('phi_delta', 0.0))
            stability = float(self._last_value_estimates.get('stability', 0.5))
            tool_success = float(self._last_value_estimates.get('tool_success', 0.0))

            # Compute beta logit
            beta_logit = 0.0
            beta_logit -= 0.5 * phi_delta  # phi influence
            beta_logit += 0.4 * (1.0 - np.clip(stability, 0.0, 1.0))  # stability influence
            beta_logit += 0.3 * (1.0 - np.clip(tool_success, 0.0, 1.0))  # tool success influence

            # EMA update
            beta_target = float(self._beta_min + (self._beta_max - self._beta_min) / (1.0 + np.exp(-beta_logit)))
            self._beta_ema = 0.9 * self._beta_ema + 0.1 * beta_target
            
            return float(np.clip(self._beta_ema, self._beta_min, self._beta_max))

        if __m3_guard_5551_8.error is not None:
            return float(self._token_q_alpha)  # fallback
    
    def _update_task_weights_gradnorm(
        self,
        individual_losses: Dict[str, float],
        shared_params: Any  # torch parameters
    ) -> None:
        """
        GradNorm update for task weights.
        
        Args:
            individual_losses: {'phi': L_phi, 'stab': L_stab, 'tool': L_tool}
            shared_params:  Shared parameters (hidden states, etc.)
        """
        torch = self.torch

        # 1) Update task losses history
        for task, loss in individual_losses.items():
            if task in self._task_losses_history:
                self._task_losses_history[task].append(loss)
                # Keep only the last 50 entries
                if len(self._task_losses_history[task]) > 50:
                    self._task_losses_history[task].pop(0)

        # 2) Compute relative rates (inverse training rate)
        if all(len(self._task_losses_history[t]) < 2 for t in ['phi', 'stab', 'tool']):
            return  # Not enough data

        with guard_context(ctx='llm_adapter/llm_core.py:5623', catch_base=False) as __m3_guard_5596_8:
            rel_rates = {}
            for task in ['phi', 'stab', 'tool']:
                hist = self._task_losses_history[task]
                if len(hist) >= 2:
                    L0 = hist[0]
                    Lt = hist[-1]
                    rel_rates[task] = float(Lt / (L0 + 1e-8))
                else:
                    rel_rates[task] = 1.0

            # Compute average rate
            avg_rate = float(np.mean([rel_rates[t] for t in ['phi', 'stab', 'tool']]))
            
            # 3) GradNorm update: r_i(t) = r_i^alpha / mean(r_j^alpha)
            # alpha=1.5 is commonly used
            targets = {}
            for task in ['phi', 'stab', 'tool']:
                targets[task] = float((rel_rates[task] ** self._grad_norm_alpha) / (avg_rate + 1e-8))

            # 4) Update task weights: w_i w_i * (1 + 0.1 * (target - 1))
            for task in ['phi', 'stab', 'tool']:
                if task in self._task_weights:
                    delta = 0.1 * (targets[task] - 1.0)
                    self._task_weights[task] = float(np.clip(self._task_weights[task] * (1 + delta), 0.1, 5.0))

        if __m3_guard_5596_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

    def _normalize_targets(
        self,
        phi_delta: float,
        stability_delta: float
    ) -> tuple:
        """
        Normalize phi and stability deltas for alpha scheduling.
        
        Args:
            phi_delta: Raw phi delta (e.g., from policy gradient)
            stability_delta: Raw stability delta (e.g., from value function)

        Returns:
            (phi_norm, stab_norm): Normalized phi and stability deltas
        """
        # 1) log (|phi| + 1, )
        phi_sign = 1.0 if phi_delta >= 0 else -1.0
        phi_log = float(phi_sign * np.log1p(abs(phi_delta)))

        # 2) sqrt (stability)
        stab_sqrt = float(np.sign(stability_delta) * np.sqrt(abs(stability_delta) + 1e-8))

        # 3) z-score (clipped)
        phi_norm = float(np.clip(phi_log / 2.0, -3.0, 3.0))  # assuming typical range
        stab_norm = float(np.clip(stab_sqrt / 1.0, -3.0, 3.0))
        
        return (phi_norm, stab_norm)

    def _alpha_scheduler_uncertainty(
        self, 
        h_final: Any,  # torch.Tensor (1, hidden)
        value_estimates: Dict[str, float]
    ) -> float:
        """
        A. kNN-LM Mixing Coefficient Scheduling based on uncertainty, instability, phi_delta.
        
        Returns:
            alpha: kNN-LM mixing coefficient [alpha_min, alpha_max]
        """
        torch = self.torch

        # 1) Compute uncertainty: H(p) / log(V)
        with guard_context(ctx='llm_adapter/llm_core.py:5676', catch_base=False) as __m3_guard_5668_8:
            with torch.no_grad():
                logits = self.model.head(h_final)  # (1, V)
                p = torch.softmax(logits, dim=-1)
                H = -torch.sum(p * torch.log(p + 1e-8), dim=-1).item()
                # Normalize
                H_norm = float(H / (np.log(self.vocab_size) + 1e-8))
                uncertainty = float(np.clip(H_norm, 0.0, 1.0))

        if __m3_guard_5668_8.error is not None:
            uncertainty = 0.5
        
        # 2) Instability: 1 - stability
        stability = float(value_estimates.get('stability', 0.5))
        instability = float(1.0 - np.clip(stability, 0.0, 1.0))
        
        # 3) Phi-delta (core.episodic_memory.phi_delta)
        phi_delta = float(value_estimates.get('phi_delta', 0.0))
        phi_penalty = float(np.clip(-phi_delta, 0.0, 1.0))  # only penalize decreases
        
        # 4)  :  sigmoid
        #  logit = a0 + a1 * uncertainty + a2 * instability + a3 * phi_penalty
        a0 = float(os.getenv("KNN_ALPHA_BASE", str(self.config.alpha_base)))
        a1 = float(os.getenv("KNN_ALPHA_UNCERTAINTY_COEF", str(self.config.alpha_entropy_coef)))
        a2 = float(os.getenv("KNN_ALPHA_INSTABILITY_COEF", str(self.config.alpha_engagement_coef)))
        a3 = float(os.getenv("KNN_ALPHA_PHI_COEF", str(self.config.alpha_phi_coef)))
        
        logit = a0 + a1 * uncertainty + a2 * instability + a3 * phi_penalty
        alpha = float(1.0 / (1.0 + np.exp(-logit)))  # sigmoid
        alpha = float(np.clip(alpha, self.config.alpha_min, self.config.alpha_max))
        
        return alpha

    def _log_jsonl(self, path: str, obj: dict):
        """Helper to append JSONL log entries."""
        with guard_context(ctx='llm_adapter/llm_core.py:5707', catch_base=False) as __m3_guard_5702_8:
            resolved_path = _normalize_log_file(path, OUT_DIR, 'llm_adapter.log')
            payload = dict(obj or {})
            decision = attr_get_optional(self, "_last_control_decision", None)
            if decision is not None:
                payload.setdefault("control_decision_id", str(attr_get_optional(decision, "control_decision_id", "")))
                payload.setdefault("decision_mode", str(attr_get_optional(decision, "decision_mode", "")))
                payload.setdefault("reason_codes", list(attr_get_optional(decision, "reason_codes", []) or []))
            append_jsonl(resolved_path, payload)

        if __m3_guard_5702_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

    def _adv_headroom(self, vocab_size: int):
        """Build advantage correction vector from token-level credit buffer."""
        torch = self.torch
        if not self._token_adv_buffer:
            return None
        adv_vec = torch.zeros(vocab_size, dtype=torch.float32, device=self.device)
        for tok_id, adv_val in self._token_adv_buffer.items():
            if 0 <= tok_id < vocab_size:
                adv_vec[tok_id] = float(adv_val)
        # decay buffer
        self._token_adv_buffer = {k: v * self._token_adv_decay for k, v in self._token_adv_buffer.items()}
        # prune near-zero entries
        self._token_adv_buffer = {k: v for k, v in self._token_adv_buffer.items() if abs(v) > 1e-4}
        return adv_vec.unsqueeze(0)  # (1, V)

    def _looks_like_toolcall(self, text: str) -> bool:
        """Check if text contains tool call syntax."""
        return "<tool:" in text and ">" in text

    def _sample_toolcall_variants(self, prompt: str, prefix: str, n: int, h_state) -> List[str]:
        """Generate n variations of toolcall by resampling with different temperatures."""
        torch = self.torch
        t = self.tok
        outs = []
        temps = np.linspace(0.6, 1.2, num=max(1, n))
        
        with torch.no_grad():
            for T in temps:
                # Resample next few tokens with temperature T
                cur_h = h_state.clone()
                prefix_ids = t.encode(prefix, add_special=False)
                if not prefix_ids:
                    outs.append(prefix)
                    continue
                    
                # Sample next 10-20 tokens to complete tool call
                cur = torch.tensor([[prefix_ids[-1]]], dtype=torch.long, device=self.device)
                variant_tokens = list(prefix_ids)
                
                for _ in range(20):
                    e = self.model.emb(cur)
                    o, cur_h = self.model.decoder(e, cur_h)
                    logits = self.model.head(o[:, -1, :])
                    tok = self._sample(logits, temperature=float(T), top_k=50, top_p=0.9)
                    tid = int(tok.item())
                    if tid == t.EOS or tid == t.PAD or tid == ord('}'):
                        variant_tokens.append(tid)
                        if tid == ord('}'):
                            break
                        break
                    variant_tokens.append(tid)
                    cur = tok
                
                variant = t.decode(variant_tokens)
                outs.append(variant)
        
        return outs if outs else [prefix]

    def _quick_eval_proxy(self, core, candidate: str) -> float:
        """Cheap proxy using core.sdm / safety / rules if available; else 0."""
        with guard_context(ctx='llm_adapter/llm_core.py:5773', catch_base=False) as __m3_guard_5770_8:
            if core is not None and attr_has(core, "sdm") and attr_has(core.sdm, "quick_eval"):
                return float(core.sdm.quick_eval(candidate))

        if __m3_guard_5770_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return 0.0

    def _predict_value_scalar(self, prompt: str, candidate: str) -> float:
        """Get last hidden and run value head for quick value estimate."""
        torch = self.torch
        t = self.tok
        with guard_context(ctx='llm_adapter/llm_core.py:5798', catch_base=False) as __m3_guard_5781_8:
            with torch.no_grad():
                encoded = t.encode(prompt)
                if not encoded:
                    encoded = [t.BOS]
                src = torch.tensor([encoded], dtype=torch.long, device=self.device)
                e = self.model.emb(src)
                _, h = self.model.encoder(e)
                # quick decode one step
                cand_ids = t.encode(candidate, add_special=False)
                if not cand_ids:
                    tok_ids = torch.tensor([[t.BOS]], dtype=torch.long, device=self.device)
                else:
                    tok_ids = torch.tensor([[cand_ids[-1]]], dtype=torch.long, device=self.device)
                o, _ = self.model.decoder(self.model.emb(tok_ids), h)
                v = self.model.value(o[:, -1, :]).squeeze(-1)
                return float(v.item())

        if __m3_guard_5781_8.error is not None:
            return 0.0

    def _build_cond_key(self, prompt: str, core=None, extra: dict | None = None) -> np.ndarray:
        """
        Conditional key = [prompt_embed tool_tag event_tag phi_slice, ...] reduced to KDIM.
        Uses feature_bank and lightweight numeric tags if available.
        """
        # 1) prompt embed from encoder's last hidden:
        torch = self.torch
        t = self.tok
        encoded = t.encode(prompt, add_special=False)
        if not encoded:
            encoded = [t.BOS]
        src = torch.tensor([encoded], dtype=torch.long, device=self.device)
        with torch.no_grad():
            e = self.model.emb(src)
            _, h = self.model.encoder(e)  # (1,1,H)
            prompt_vec = h[-1, 0, :].detach().cpu().numpy()

        # 2) tags from core: last tool name, success, phi tranche, bus event type
        tag = np.zeros(32, dtype=np.float32)
        with guard_context(ctx='llm_adapter/llm_core.py:5837', catch_base=False) as __m3_guard_5820_8:
            if core is not None:
                last_tool = attr_get_optional(core, "last_tool", "none")
                tool_ok   = float(attr_get_optional(core, "last_tool_ok", 0.0))
                phi_hist  = np.asarray(attr_get_optional(core, "phi_calculator", None).phi_history[-8:], dtype=np.float32) \
                            if attr_has(attr_get_optional(core,"phi_calculator", None), "phi_history") else np.zeros(0, np.float32)
                tag[0] = float(hash(str(last_tool)) % 997) / 997.0
                tag[1] = tool_ok
                if phi_hist.size:
                    tag[2] = float(phi_hist[-1])
                    tag[3] = float(np.mean(phi_hist))
                # bus depth/latency if present
                bus = attr_get_optional(core, "bus", None)
                if bus is not None and attr_has(bus, "depth"):
                    tag[4] = float(attr_get_optional(bus, "depth", 0))
                if bus is not None and attr_has(bus, "latency_ms"):
                    tag[5] = float(attr_get_optional(bus, "latency_ms", 0.0))

        if __m3_guard_5820_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

        # 3) extra dict (event_tag etc.)
        if extra:
            i = 8
            for k, v in extra.items():
                if i >= tag.size: break
                try:
                    tag[i] = float(v)
                    i += 1
                except Exception:
                    continue

        # 4) reduce/pack to KDIM with cached projection matrix
        KDIM = self._knn.key_dim
        base = np.concatenate([prompt_vec.astype(np.float32), tag], axis=0)

        # === A. Contextual Memory (kNN key/value pairs) ===
        # base @ R^T = key
        if (self._R is None) or (self._cond_base_dim != base.size):
            rng = np.random.RandomState(12345)
            self._R = rng.normal(0, 1/np.sqrt(base.size), size=(KDIM, base.size)).astype(np.float32)
            self._cond_base_dim = base.size

        # R @ base
        key = self._R @ base
        return key.astype(np.float32)

    def _sample(self, logits, temperature: float = None, top_k: int = None, top_p: float = None):
        """M3 adaptive sampling wrapper (all params from M3 state)."""
        return self.m3_sampler.sample(
            logits, core=self.core,
            base_temp=temperature if temperature is not None else 0.8,
            base_top_k=top_k if top_k is not None else 50,
            base_top_p=top_p if top_p is not None else 0.9
        )

    def _build_hf_quality_gate_inputs(self, temperature, top_k, top_p, core, bridge_state, m3_sampler):
        tv_head = None
        internal_h = None
        beta_val = 0.0
        with guard_context(ctx='llm_adapter/llm_core.py:5888', catch_base=False) as __m3_guard_5879_8:
            enable_tv = (
                self._control_allows('token_value_bias')
                and os.getenv('LLM_ADAPTER_ENABLE_TOKEN_VALUE_BIAS', '0').lower() in ('1', 'true', 'yes', 'on')
            )
            if enable_tv and attr_has(self, 'model') and attr_has(self.model, 'token_value'):
                tv_head = self.model.token_value
                internal_h = attr_get_optional(self.model, 'hidden', 1024)
                beta_val = float(self._beta_schedule())

        if __m3_guard_5879_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

        return {
            'token_value_head': tv_head,
            'internal_hidden_dim': internal_h,
            'beta': beta_val,
            'prompt': None,
            'messages': None,
            'core': core,
            'm3_sampler': m3_sampler,
            'bridge_state': bridge_state,
            'decode_control': None,
            'temperature': temperature if temperature is not None else 0.8,
            'top_k': top_k if top_k is not None else 50,
            'top_p': top_p if top_p is not None else 0.9,
            'max_len': None,
        }

    def _generate_with_hf(self, hf, prompt, chat_messages, core, m3_sampler,
                         token_value_head=None, internal_hidden_dim=None, beta=0.0,
                         z_m3=None, max_new_tokens=0,
                         base_temperature=0.8, base_top_k=50, base_top_p=0.9,
                         decode_control: Optional[Dict[str, Any]] = None):
        return hf.generate_with_m3(
            prompt=prompt,
            messages=chat_messages,
            core=core,
            m3_sampler=m3_sampler,
            token_value_head=token_value_head,
            internal_hidden_dim=internal_hidden_dim,
            beta=beta,
            z_m3=z_m3,
            max_new_tokens=max_new_tokens,
            base_temperature=base_temperature,
            base_top_k=base_top_k,
            base_top_p=base_top_p,
            decode_control=decode_control,
        )

    def _apply_quality_gate_if_enabled(self, gate, gate_payload: dict, response: str, temperature, top_k, top_p):
        if not self._control_allows('quality_gate'):
            return response
        if gate is None:
            return response
        if not os.getenv('M3_ENABLE_QUALITY_GATE', '0').lower() in ('1', 'true', 'yes', 'on'):
            return response
        if gate is None or not response:
            return response

        with guard_context(ctx='llm_adapter/llm_core.py:5940', catch_base=False) as __m3_guard_5938_8:
            q0 = gate.evaluate(response)

        if __m3_guard_5938_8.error is not None:
            return response

        if not attr_get_optional(q0, 'reject', False):
            return response

        with guard_context(ctx='llm_adapter/llm_core.py:5968', catch_base=False) as __m3_guard_5946_8:
            retry = self._generate_with_hf(
                hf=gate_payload['hf'],
                prompt=gate_payload['prompt'],
                chat_messages=gate_payload['chat_messages'],
                core=gate_payload['core'],
                m3_sampler=gate_payload['m3_sampler'],
                token_value_head=gate_payload['token_value_head'],
                internal_hidden_dim=gate_payload['internal_hidden_dim'],
                beta=gate_payload['beta'],
                z_m3=gate_payload['z_m3'],
                max_new_tokens=gate_payload['max_new_tokens'],
                base_temperature=max(0.25, gate_payload['base_temperature'] * 0.8),
                base_top_k=max(20, int(gate_payload['base_top_k'] * 0.8)),
                base_top_p=min(0.98, gate_payload['base_top_p'] + 0.05),
                decode_control=gate_payload.get('decode_control'),
            )
            if not retry:
                return response
            q1 = gate.evaluate(retry)
            if (not q1.reject) or (q1.score > q0.score):
                return retry

        if __m3_guard_5946_8.error is not None:
            return response
        return response

    def _semantic_perspective_prefix(self, core=None) -> str:
        with guard_context(ctx='llm_adapter/llm_core.py:5975', catch_base=False) as __m3_guard_5973_8:
            enabled = os.getenv('M3_EMBED_PERSPECTIVE', '0').lower() in ('1', 'true', 'yes', 'on')

        if __m3_guard_5973_8.error is not None:
            enabled = False
        if not enabled:
            return ""
        with guard_context(ctx='llm_adapter/llm_core.py:5990', catch_base=False) as __m3_guard_5979_8:
            core_ref = core if core is not None else attr_get_optional(self, 'core', None)
            if core_ref is None:
                return ""
            subj = attr_get_optional(core_ref, 'unified_subject', None)
            if subj is None or not attr_has(subj, 'reflect_on_self'):
                return ""
            summary = str(subj.reflect_on_self()).strip()
            if not summary:
                return ""
            return f"Perspective: {summary}\n\n"

        if __m3_guard_5979_8.error is not None:
            return ""

    def embed_text(self, text: str, sys_identity: str = "", max_src_len: int = 512) -> np.ndarray:
        import numpy as _np
        import torch as _torch
        base_prompt = (str(sys_identity).strip() + "\n\n" + str(text)) if sys_identity else str(text)
        src_ids = self.tok.encode(base_prompt, add_special=False)
        if max_src_len and len(src_ids) > int(max_src_len):
            src_ids = src_ids[-int(max_src_len):]
        if not src_ids:
            src_ids = [int(self.tok.BOS)]
        src = _torch.tensor(src_ids, dtype=_torch.long, device=self.device).unsqueeze(0)
        with _torch.no_grad():
            e_src = self.model.emb(src)
            _, h = self.model.encoder(e_src)
        vec = h.squeeze(0).detach().cpu().numpy().astype(_np.float32)
        if not _np.all(_np.isfinite(vec)):
            vec = _np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(_np.float32)
        return vec

    def generate(
        self,
        prompt: str,
        max_len: int = 60,
        temperature: float = None,
        top_k: int = None,
        top_p: float = 0.9,
        mem: Optional[np.ndarray] = None,
        prefix_embed: Optional[np.ndarray] = None,
        knn_provider: Optional[Any] = None,
        knn_alpha: float = 0.0,
        affect_state: Optional[np.ndarray] = None,
        source: str = "generate",
        meaning_state: Optional[Dict[str, Any]] = None,
        response_plan: Optional[Dict[str, Any]] = None,
        generation_contract: Optional[Any] = None,
    ) -> str:
        t = self.tok
        core = attr_get_optional(self, 'core', None)
        raw_prompt = prompt
        control_decision = self._build_control_decision(prompt=str(raw_prompt), source=str(source or "generate"))
        control_mode = str(
            attr_get_optional(control_decision, "decision_mode", self._control_plane_mode()) or self._control_plane_mode()
        ).strip().lower()
        enforce_control_plane = control_mode == "enforce"
        with guard_context(ctx='llm_adapter/llm_core.py:6034', catch_base=False) as __m3_guard_6031_8:
            if attr_has(self.tok, "observe_unknown_rate"):
                self.tok.observe_unknown_rate(str(raw_prompt or ""))

        if __m3_guard_6031_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        if attr_get_optional(self, "_hf_circuit_open", False):
            self._note_control_health(False, "hf_circuit_open")
            return self._generate_safe_fallback(
                str(raw_prompt),
                chat_messages=None,
                max_len=max_len,
                meaning_state=meaning_state,
                response_plan=response_plan,
                generation_contract=generation_contract,
            )
        if self._hf_runtime_cooldown_active():
            self._note_control_health(False, "hf_runtime_cooldown")
            with guard_context(ctx='llm_adapter/llm_core.py:6049', catch_base=False) as __m3_guard_6044_16:
                self._log_jsonl(
                    os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                    {
                        "kind": "hf_runtime_failure",
                        "reason_code": "hf_runtime_cooldown",
                        "phase": "precheck",
                        "model_output_shape": "n/a",
                        "has_logits": False,
                        "failure_window_count": int(self._prune_hf_failure_events()),
                    },
                )

            if __m3_guard_6044_16.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            return self._generate_safe_fallback(
                str(raw_prompt),
                chat_messages=None,
                max_len=max_len,
                meaning_state=meaning_state,
                response_plan=response_plan,
                generation_contract=generation_contract,
            )
        # System identity / behavior prefix (configurable / optional)
        sys_identity = self._get_system_prompt()
        
        panels = None
        phi_trend = None
        if core is not None:
            panels = self._update_m3_cache(core)
            with guard_context(ctx='llm_adapter/llm_core.py:6055', catch_base=False) as __m3_guard_6053_12:
                phi_trend = self.m3_cache.get_phi_trend(core)

            if __m3_guard_6053_12.error is not None:
                phi_trend = None
        
        affect_state = self._get_affect_state(core, affect_state)
        main_prompt = raw_prompt
        # === Phase 4: Episodic Memory Retrieval (NO THRESHOLD) ===
        retrieval_allowed = bool(attr_get_optional(control_decision, "allow_memory_retrieval", True))
        retrieval_enabled = bool(
            core is not None
            and (
                retrieval_allowed
                if enforce_control_plane
                else self._control_allows('memory_retrieval')
            )
        )
        if core is not None and not retrieval_enabled:
            self._record_retrieval_outcome(
                outcome="skip",
                skip_reason="disabled_by_mode",
                selected_count=0,
                candidate_pool=0,
            )
        if retrieval_enabled:
            try:
                semantic_query = self._semantic_perspective_prefix(core) + str(raw_prompt)
                context_embedding = self.embed_text(semantic_query, sys_identity="")
                if context_embedding is None or int(np.asarray(context_embedding).size) <= 0:
                    self._record_retrieval_outcome(
                        outcome="skip",
                        skip_reason="no_query_vector",
                        selected_count=0,
                        candidate_pool=0,
                    )
                else:
                    relevant_episodes = self.m3_memory_retriever.retrieve_relevant_episodes(core, context_embedding)
                    episode_texts = []
                    with guard_context(ctx='llm_adapter/llm_core.py:6071', catch_base=False) as __m3_guard_6069_24:
                        max_chars = int(os.getenv('M3_EPISODIC_MAX_CHARS', '800'))

                    if __m3_guard_6069_24.error is not None:
                        max_chars = 800
                    with guard_context(ctx='llm_adapter/llm_core.py:6075', catch_base=False) as __m3_guard_6073_24:
                        item_chars = int(os.getenv('M3_EPISODIC_ITEM_CHARS', '200'))

                    if __m3_guard_6073_24.error is not None:
                        item_chars = 200
                    total_chars = 0
                    for ep in list(relevant_episodes or []):
                        if attr_get_optional(ep, 'kind', 'internal_state') not in {'dialog', 'research', 'knowledge'}:
                            continue
                        if attr_has(ep, 'content'):
                            txt = str(ep.content)
                        elif attr_has(ep, 'text'):
                            txt = str(ep.text)
                        elif attr_has(ep, 'description'):
                            txt = str(ep.description)
                        elif attr_has(ep, 'narrative'):
                            txt = str(ep.narrative)
                        elif attr_has(ep, 'context'):
                            txt = str(ep.context)
                        else:
                            txt = ""
                        if not txt:
                            continue
                        if self._is_disallowed_generation_output(txt):
                            continue
                        if item_chars > 0 and len(txt) > item_chars:
                            txt = txt[:item_chars] + "..."
                        if max_chars > 0 and (total_chars + len(txt)) > max_chars:
                            break
                        total_chars += len(txt) + 2
                        episode_texts.append(txt)
                    episode_pool = 0
                    with guard_context(ctx='llm_adapter/llm_core.py:6115', catch_base=False) as __m3_guard_6111_16:
                        em = attr_get_optional(core, "episodic_memory", None)
                        pool = attr_get_optional(em, "episodes", []) if em is not None else []
                        episode_pool = int(len(pool)) if pool is not None else 0

                    if __m3_guard_6111_16.error is not None:
                        episode_pool = 0
                    if episode_texts:
                        context_prefix = "Similar past context:\n" + "\n".join(f"- {txt}" for txt in episode_texts) + "\n\nCurrent context:\n"
                        main_prompt = context_prefix + raw_prompt
                    if relevant_episodes:
                        self._record_retrieval_outcome(
                            outcome="hit",
                            skip_reason="",
                            selected_count=len(list(relevant_episodes)),
                            candidate_pool=episode_pool,
                        )
                    else:
                        miss_reason = "empty_index" if episode_pool <= 0 else "below_threshold"
                        self._record_retrieval_outcome(
                            outcome="miss",
                            skip_reason=miss_reason,
                            selected_count=0,
                            candidate_pool=episode_pool,
                        )
            except Exception:
                self._record_retrieval_outcome(
                    outcome="skip",
                    skip_reason="no_query_vector",
                    selected_count=0,
                    candidate_pool=0,
                )
                logging.getLogger(__name__).exception("Swallowed exception")

        include_m3_state = bool(
            (
                bool(attr_get_optional(control_decision, "allow_state_context", True))
                if enforce_control_plane
                else self._should_include_m3_state(raw_prompt)
            )
            and self._should_include_m3_state(raw_prompt)
        )
        decode_control = None
        if not enforce_control_plane or bool(attr_get_optional(control_decision, "allow_decode_control", True)):
            decode_control = self._build_decode_control(raw_prompt)
        m3_context = (
            self._build_m3_context(core, affect_state=affect_state, phi_trend=phi_trend, panels=panels)
            if include_m3_state else ""
        )
        # Build chat messages for instruct models (HF) when possible.
        system_content_parts = []
        if sys_identity:
            system_content_parts.append(sys_identity)
        if m3_context:
            system_content_parts.append(m3_context)
        system_content = "\n\n".join(system_content_parts).strip()

        transcript_msgs = self._parse_user_m3_transcript(main_prompt)
        if transcript_msgs is not None:
            chat_messages = ([{'role': 'system', 'content': system_content}] if system_content else []) + transcript_msgs
        else:
            user_content = str(main_prompt).strip()
            if not user_content:
                user_content = self._autonomy_seed_prompt(core)
            chat_messages = ([{'role': 'system', 'content': system_content}] if system_content else []) + [
                {'role': 'user', 'content': user_content}
            ]

        # For logging/training-record compatibility, keep a readable prompt string.
        prompt_parts = []
        if system_content:
            prompt_parts.append(system_content)
        if transcript_msgs is not None:
            for m in transcript_msgs:
                role = 'User' if m.get('role') == 'user' else 'M3'
                prompt_parts.append(f"{role}: {m.get('content', '')}")
            prompt_parts.append('M3:')
        else:
            prompt_parts.append(chat_messages[-1].get('content', ''))
        prompt = "\n\n".join([p for p in prompt_parts if p is not None])

        m3_memory = mem
        if (
            m3_memory is None
            and core is not None
            and (
                bool(attr_get_optional(control_decision, "allow_state_context", True))
                if enforce_control_plane
                else self._control_allows('state_context')
            )
            and attr_get_optional(self.model, 'use_m3_integration', False)
        ):
            m3_memory = self._build_m3_memory(core=core, panels=panels, affect_state=affect_state)

        m3_memory_t = None
        if m3_memory is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:6167', catch_base=False) as __m3_guard_6158_12:
                if self.torch.is_tensor(m3_memory):
                    m3_memory_t = m3_memory.to(self.device)
                else:
                    m3_memory_t = torch.tensor(np.asarray(m3_memory), dtype=torch.float32, device=self.device)
                if m3_memory_t.ndim == 3 and m3_memory_t.size(0) == 1:
                    m3_memory_t = m3_memory_t.squeeze(0)
                if m3_memory_t.ndim == 1:
                    m3_memory_t = m3_memory_t.unsqueeze(0)

            if __m3_guard_6158_12.error is not None:
                m3_memory_t = None

        bridge_state = None
        bridge_allowed = bool(attr_get_optional(control_decision, "allow_bridge", True))
        if self._bridge_enabled_safe() and (bridge_allowed if enforce_control_plane else True):
            with guard_context(ctx='llm_adapter/llm_core.py:6178', catch_base=False) as __m3_guard_6172_12:
                bridge_state = self._build_full_state_vector(
                    core=core,
                    panels=panels,
                    affect_state=affect_state,
                )

            if __m3_guard_6172_12.error is not None:
                bridge_state = None
            if bridge_state is None and m3_memory_t is not None:
                with guard_context(ctx='llm_adapter/llm_core.py:6183', catch_base=False) as __m3_guard_6181_16:
                    bridge_state = m3_memory_t.detach().float().cpu().numpy().reshape(-1)

                if __m3_guard_6181_16.error is not None:
                    bridge_state = None
            # Optional ablation for evaluation: none|shuffle|zero
            with guard_context(ctx='llm_adapter/llm_core.py:6197', catch_base=False) as __m3_guard_6186_12:
                ablation = os.getenv('M3_BRIDGE_ABLATION', 'none').strip().lower()
                if bridge_state is not None and ablation in ('shuffle', 'permute', 'zero', 'zeros'):
                    v = np.asarray(bridge_state, dtype=np.float32).reshape(-1)
                    if ablation in ('shuffle', 'permute'):
                        if v.size > 1:
                            p = np.random.permutation(v.size)
                            v = v[p]
                    else:
                        v = np.zeros_like(v, dtype=np.float32)
                    bridge_state = v

            if __m3_guard_6186_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

        # === HuggingFace backend: unified generation + scoring loop ===
        used_hf = False
        if self.use_hf and HFBackend.is_available():
            used_hf = True
            try:
                hf = HFBackend.get_instance()
                resolved_temp, resolved_top_k, resolved_top_p = self._resolve_generation_sampling(
                    requested_temperature=temperature,
                    requested_top_k=top_k,
                    requested_top_p=top_p,
                    source=source,
                    core=core,
                )
                gate_payload = self._build_hf_quality_gate_inputs(
                    temperature=resolved_temp,
                    top_k=resolved_top_k,
                    top_p=resolved_top_p,
                    core=core,
                    bridge_state=bridge_state,
                    m3_sampler=self.m3_sampler,
                )
                gate_payload['decode_control'] = decode_control

                gate = attr_get_optional(self, '_quality_gate', None)
                if enforce_control_plane and not bool(attr_get_optional(control_decision, "quality_gate_on", True)):
                    gate = None
                use_semantic_scoring = self._should_use_semantic_scoring(
                    meaning_state=meaning_state,
                    response_plan=response_plan,
                    generation_contract=generation_contract,
                )
                legacy_retry = max(0, int(os.getenv("M3_CONTROL_RETRY", "1")))
                candidate_budget = max(1, self._candidate_count() if use_semantic_scoring else 1, legacy_retry + 1)
                if candidate_budget <= 0:
                    candidate_budget = 1

                # start from configured defaults / runtime inputs
                cur_temp = gate_payload['temperature']
                cur_top_k = gate_payload['top_k']
                cur_top_p = gate_payload['top_p']

                # Inject plan contract into system context when available.
                if response_plan is not None:
                    contract_text = self._plan_contract_text(response_plan, meaning_state, generation_contract)
                    if contract_text:
                        system_content = f"{system_content}\n\n{contract_text}".strip()
                        # keep training prompt in sync with injected plan contract context
                        prompt_parts = []
                        if system_content:
                            prompt_parts.append(system_content)
                        if transcript_msgs is not None:
                            for m in transcript_msgs:
                                role = 'User' if m.get('role') == 'user' else 'M3'
                                prompt_parts.append(f"{role}: {m.get('content', '')}")
                            prompt_parts.append('M3:')
                        else:
                            prompt_parts.append(chat_messages[-1].get('content', ''))
                        prompt = "\n\n".join([p for p in prompt_parts if p is not None])
                        if transcript_msgs is not None:
                            chat_messages = ([{'role': 'system', 'content': system_content}] if system_content else []) + transcript_msgs
                        else:
                            chat_messages = ([{'role': 'system', 'content': system_content}] if system_content else []) + [
                                {'role': 'user', 'content': chat_messages[-1].get('content', '')}
                            ]

                best_response: Optional[str] = None
                best_score: float = -1.0
                best_passed_quality = False
                best_passed_semantic = False
                for attempt in range(candidate_budget):
                    response = self._generate_with_hf(
                        hf=hf,
                        prompt=prompt,
                        chat_messages=chat_messages,
                        core=core,
                        m3_sampler=self.m3_sampler,
                        token_value_head=gate_payload['token_value_head'],
                        internal_hidden_dim=gate_payload['internal_hidden_dim'],
                        beta=gate_payload['beta'],
                        z_m3=bridge_state,
                        max_new_tokens=max_len,
                        base_temperature=cur_temp,
                        base_top_k=cur_top_k,
                        base_top_p=cur_top_p,
                        decode_control=decode_control,
                    )
                    if not response or not str(response).strip():
                        if attempt < candidate_budget - 1:
                            with guard_context(ctx='llm_adapter/llm_core.py:6292', catch_base=False) as __m3_guard_6286_28:
                                cur_temp, cur_top_k, cur_top_p = self._retry_generation_sampling(
                                    temperature=cur_temp,
                                    top_k=cur_top_k,
                                    top_p=cur_top_p,
                                )

                            if __m3_guard_6286_28.error is not None:
                                cur_temp = max(0.15, float(cur_temp) * 0.8)
                                cur_top_k = max(5, int(cur_top_k) if int(cur_top_k) < 24 else int(cur_top_k * 0.8))
                                cur_top_p = max(0.70, float(cur_top_p) * 0.95)
                        continue
                    repeat_blocked, repeat_meta = self._is_repeat_candidate_blocked(str(response), core=core)
                    if repeat_blocked:
                        self._log_jsonl(
                            os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                            {
                                "kind": "response_repeat_block",
                                "attempt": int(attempt + 1),
                                "candidate_budget": int(candidate_budget),
                                "similarity_last": float(repeat_meta.get("similarity_last", 0.0)),
                                "similarity_prev": float(repeat_meta.get("similarity_prev", 0.0)),
                                "similarity_min": float(repeat_meta.get("similarity_min", 0.0)),
                                "threshold": float(repeat_meta.get("threshold", 0.90)),
                            },
                        )
                        if attempt < candidate_budget - 1:
                            with guard_context(ctx='llm_adapter/llm_core.py:6314', catch_base=False) as __m3_guard_6310_28:
                                cur_temp, cur_top_k, cur_top_p = self._retry_generation_sampling(
                                    temperature=cur_temp,
                                    top_k=cur_top_k,
                                    top_p=cur_top_p,
                                )

                            if __m3_guard_6310_28.error is not None:
                                cur_temp = max(0.15, float(cur_temp) * 0.8)
                                cur_top_k = max(5, int(cur_top_k) if int(cur_top_k) < 24 else int(cur_top_k * 0.8))
                                cur_top_p = max(0.70, float(cur_top_p) * 0.95)
                        continue

                    if gate is not None:
                        with guard_context(ctx='llm_adapter/llm_core.py:6324', catch_base=False) as __m3_guard_6299_24:
                            gate_payload.update({
                                'hf': hf,
                                'prompt': prompt,
                                'chat_messages': chat_messages,
                                'core': core,
                                'm3_sampler': self.m3_sampler,
                                'token_value_head': gate_payload['token_value_head'],
                                'internal_hidden_dim': gate_payload['internal_hidden_dim'],
                                'beta': gate_payload['beta'],
                                'z_m3': bridge_state,
                                'max_new_tokens': max_len,
                                'base_temperature': cur_temp,
                                'base_top_k': cur_top_k,
                                'base_top_p': cur_top_p,
                                'decode_control': decode_control,
                            })
                            response = self._apply_quality_gate_if_enabled(
                                gate=gate,
                                gate_payload=gate_payload,
                                response=response,
                                temperature=cur_temp,
                                top_k=cur_top_k,
                                top_p=cur_top_p,
                            )

                        if __m3_guard_6299_24.error is not None:
                            logging.getLogger(__name__).exception("Swallowed exception")

                    if not response or not str(response).strip():
                        if attempt < candidate_budget - 1:
                            logger.warning(
                                f'[HFBackend] quality gate returned empty on attempt {attempt+1}/{candidate_budget}; re-sampling with stricter params'
                            )
                            with guard_context(ctx='llm_adapter/llm_core.py:6338', catch_base=False) as __m3_guard_6332_28:
                                cur_temp, cur_top_k, cur_top_p = self._retry_generation_sampling(
                                    temperature=cur_temp,
                                    top_k=cur_top_k,
                                    top_p=cur_top_p,
                                )

                            if __m3_guard_6332_28.error is not None:
                                cur_temp = max(0.15, float(cur_temp) * 0.8)
                                cur_top_k = max(5, int(cur_top_k) if int(cur_top_k) < 24 else int(cur_top_k * 0.8))
                                cur_top_p = max(0.70, float(cur_top_p) * 0.95)
                        continue

                    passed_quality, q_info = self._evaluate_generation_quality(prompt, response, source=source)
                    user_text = self._extract_last_user_text(prompt)
                    if use_semantic_scoring:
                        passed_semantic, sem_info, _ = self._evaluate_semantic_candidate(
                            response=str(response),
                            meaning_state=meaning_state,
                            response_plan=response_plan,
                            generation_contract=generation_contract,
                            user_text=user_text,
                        )
                    else:
                        passed_semantic = True
                        sem_info = {"overall": 0.0, "plan_adherence": 0.0, "identity_consistency": 1.0}

                    quality_score = float(q_info.get("score", 0.0))
                    semantic_overall = float(sem_info.get("overall", 0.0))
                    composite = float(0.35 * quality_score + 0.65 * semantic_overall) if use_semantic_scoring else float(quality_score)
                    if use_semantic_scoring and not passed_semantic:
                        composite *= 0.25
                    if not passed_quality:
                        composite *= 0.55

                    if self._is_disallowed_generation_output(response):
                        logger.warning('[HFBackend] filtered disallowed output candidate')
                    elif composite > best_score:
                        best_score = composite
                        best_response = response.strip()
                        best_passed_quality = bool(passed_quality)
                        best_passed_semantic = bool(passed_semantic)
                        if use_semantic_scoring and passed_semantic and passed_quality and quality_score >= 0.80:
                            break
                        if not use_semantic_scoring and passed_quality and composite > 0.30:
                            break

                    if attempt < candidate_budget - 1 and not passed_semantic:
                        logger.warning(
                            f'[HFBackend] semantic candidate rejected on attempt {attempt+1}/{candidate_budget}: '
                            f'{", ".join(sem_info.get("reasons", []))}'
                        )
                        with guard_context(ctx='llm_adapter/llm_core.py:6389', catch_base=False) as __m3_guard_6383_24:
                            cur_temp, cur_top_k, cur_top_p = self._retry_generation_sampling(
                                temperature=cur_temp,
                                top_k=cur_top_k,
                                top_p=cur_top_p,
                            )

                        if __m3_guard_6383_24.error is not None:
                            cur_temp = max(0.15, float(cur_temp) * 0.8)
                            cur_top_k = max(5, int(cur_top_k) if int(cur_top_k) < 24 else int(cur_top_k * 0.8))
                            cur_top_p = max(0.70, float(cur_top_p) * 0.95)
                    elif attempt < candidate_budget - 1 and not passed_quality:
                        logger.warning(
                            f'[HFBackend] quality rejected on attempt {attempt+1}/{candidate_budget}; re-sampling with stricter params'
                        )
                        with guard_context(ctx='llm_adapter/llm_core.py:6403', catch_base=False) as __m3_guard_6397_24:
                            cur_temp, cur_top_k, cur_top_p = self._retry_generation_sampling(
                                temperature=cur_temp,
                                top_k=cur_top_k,
                                top_p=cur_top_p,
                            )

                        if __m3_guard_6397_24.error is not None:
                            cur_temp = max(0.15, float(cur_temp) * 0.8)
                            cur_top_k = max(5, int(cur_top_k) if int(cur_top_k) < 24 else int(cur_top_k * 0.8))
                            cur_top_p = max(0.70, float(cur_top_p) * 0.95)

                if best_response is not None and not self._is_disallowed_generation_output(best_response):
                    if (not use_semantic_scoring and best_passed_quality) or (use_semantic_scoring and best_passed_semantic):
                        with guard_context(ctx='llm_adapter/llm_core.py:6412', catch_base=False) as __m3_guard_6410_24:
                            self._record_example(prompt, best_response, source='generate_hf')

                        if __m3_guard_6410_24.error is not None:
                            logging.getLogger(__name__).exception("Swallowed exception")
                        with guard_context(ctx='llm_adapter/llm_core.py:6421', catch_base=False) as __m3_guard_6414_24:
                            self._adapt_bridge_online(
                                hf=hf,
                                prompt=prompt,
                                response=best_response,
                                z_m3=bridge_state,
                            )

                        if __m3_guard_6414_24.error is not None:
                            logging.getLogger(__name__).exception("Swallowed exception")
                        with guard_context(ctx='llm_adapter/llm_core.py:6425', catch_base=False) as __m3_guard_6423_24:
                            self._maybe_auto_rebuild_tokenizer()

                        if __m3_guard_6423_24.error is not None:
                            logging.getLogger(__name__).exception("Swallowed exception")
                        self._note_control_health(True, "hf_generate_ok")
                        return best_response
                if use_semantic_scoring:
                    logger.warning('[HFBackend] filtered by semantic evaluation; switching to fallback')
                else:
                    logger.warning('[HFBackend] filtered by generation quality; switching to fallback')
            except Exception as e:
                details = self._extract_hf_failure_details(e, hf=hf)
                reason_code = str(details.get("reason_code", "hf_runtime_failure"))
                phase = str(details.get("phase", "unknown"))
                model_output_shape = str(details.get("model_output_shape", "unknown"))
                has_logits = bool(details.get("has_logits", False))
                if self._is_cuda_fatal_error(e):
                    self._trip_hf_circuit_breaker(e)
                    reason_code = "hf_cuda_fault"
                failure_state = self._register_hf_runtime_failure(
                    reason_code=reason_code,
                    phase=phase,
                    model_output_shape=model_output_shape,
                    has_logits=has_logits,
                )
                cooldown_remaining = float(
                    max(0.0, float(failure_state.get("cooldown_until", 0.0)) - float(time.time()))
                )
                self._log_jsonl(
                    os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                    {
                        "kind": "hf_runtime_failure",
                        "reason_code": str(reason_code),
                        "phase": str(phase),
                        "model_output_shape": str(model_output_shape),
                        "has_logits": bool(has_logits),
                        "failure_window_count": int(failure_state.get("failure_window_count", 0)),
                        "cooldown_remaining_sec": cooldown_remaining,
                    },
                )
                logger.warning(
                    f"[HFBackend] generation failed reason={reason_code} phase={phase} "
                    f"count={failure_state.get('failure_window_count', 0)} err=({e})"
                )
                self._note_control_health(False, reason_code)
                return self._generate_safe_fallback(
                    prompt,
                    chat_messages=chat_messages,
                    max_len=max_len,
                    meaning_state=meaning_state,
                    response_plan=response_plan,
                    generation_contract=generation_contract,
                )

        # HF backend disabled or unavailable: continue with safe fallback.
        if used_hf:
            self._note_control_health(False, "hf_filtered_or_fallback")
        else:
            self._note_control_health(False, "hf_unavailable_or_fallback")
        return self._generate_safe_fallback(
            prompt,
            chat_messages=chat_messages,
            max_len=max_len,
            meaning_state=meaning_state,
            response_plan=response_plan,
            generation_contract=generation_contract,
        )

    def score_value(self, prompt: str, candidate: str, mem: Optional[np.ndarray] = None) -> float:
        """Estimate value for a candidate response (scaffolding for Step 3)."""
        t = self.tok
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            src = t.encode(prompt, add_special=False)
            response = response.strip()
            if not src or not response:
                return 0.0
            src_ids = torch.tensor(src, dtype=torch.long, device=self.device).unsqueeze(0)
            e_src = self.model.emb(src_ids)
            _, h = self.model.encoder(e_src)
            # teacher forcing over candidate
            tgt = t.encode(response, add_special=True)
            if len(tgt) < 2:
                return 0.0
            tgt_in = tgt[:-1]
            tgt_in_ids = torch.tensor([tgt_in], dtype=torch.long, device=self.device)
            o, _ = self.model.decoder(self.model.emb(tgt_in_ids), h)
            dec_t = o[:, -1, :].contiguous()  # (1, H)
            
            # Optional memory context
            if mem is not None:
                m = np.array(mem)
                if m.ndim == 1:
                    m = m[None, :]
                m_t = torch.tensor(m, dtype=torch.float32, device=self.device)
                if m_t.ndim == 2:
                    m_t = m_t.unsqueeze(0)
                M, D = m_t.shape[1], m_t.shape[2]
                self.model._ensure_mem_layers(D)
                mem_h = self.model.mem_proj(m_t)
                mem_k = self.model.Wk(mem_h)
                mem_v = self.model.Wv(mem_h)
                q = self.model.Wq(dec_t).unsqueeze(1)
                att = torch.softmax((q @ mem_k.transpose(1, 2)) / np.sqrt(self.model.hidden + 1e-8), dim=-1)
                ctx = att @ mem_v
                dec_t = (dec_t + ctx.squeeze(1)) * 0.5
            
            val = self.model.value(dec_t)
            return float(val.squeeze().item())

    def collect_knn_from_teacher(self, prompt: str, response: str, core=None):
        """During supervised step, record (key, next-token logits) for each time step."""
        torch = self.torch
        t = self.tok
        encoded = t.encode(prompt)
        if not encoded:
            encoded = [t.BOS]
        src_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
        tgt = t.encode(response, add_special=True)
        if len(tgt) < 2: 
            return
        tgt_in_ids = torch.tensor([tgt[:-1]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            e_src = self.model.emb(src_ids)
            _, h = self.model.encoder(e_src)
            e_tgt_in = self.model.emb(tgt_in_ids)
            o, _ = self.model.decoder(e_tgt_in, h)
            logits = self.model.head(o).squeeze(0).cpu().numpy()   # (T, V)
            # conditional key per prompt (can also include rolling tags)
            key = self._build_cond_key(prompt, core=core)
            for tstep in range(logits.shape[0]):
                self._knn.add(key, logits[tstep])

    def learn_pair(self, prompt: str, response: str, max_len: int = 256) -> None:
        try:
            t = self.tok
            torch = self.torch
            self.model.train()
            encoded = t.encode(prompt)
            if not encoded:
                encoded = [t.BOS]
            src_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
            tgt = t.encode(response, add_special=True)
            if len(tgt) < 2:
                # print(f"Debug: Target too short for '{response}'")
                return
            tgt_in = tgt[:-1][:max_len]
            tgt_out = tgt[1:][:max_len]
            if not tgt_in:
                return
            tgt_in_ids = torch.tensor([tgt_in], dtype=torch.long, device=self.device)
            tgt_out_ids = torch.tensor(tgt_out, dtype=torch.long, device=self.device)
            # === Phase 5: M3 Supervised Training (NO MAGIC NUMBERS) ===
            with self._learn_lock:
                logits = self.model(src_ids, tgt_in_ids).squeeze(0).clone()
                base_loss = self.criterion(logits, tgt_out_ids)
                
                # M3-aware loss weighting: phi/entropy/energy 
                if attr_has(self, 'core') and self.core is not None:
                    with guard_context(ctx='llm_adapter/llm_core.py:6560', catch_base=False) as __m3_guard_6553_20:
                        phi = self.core.phi_calculator.phi_history[-1] if self.core.phi_calculator.phi_history else 0.5
                        entropy = attr_get_optional(self.core.qualia, 'entropy', 0.5)
                        energy_ratio = self.core.energy_ctrl.cognitive_energy / max(self.core.energy_ctrl.energy_capacity, 1.0) if attr_has(self.core, 'energy_ctrl') else 0.5
                        loss_weight = (1.0 - phi) * entropy * (1.0 - energy_ratio)
                        loss_weight = torch.tensor(max(0.1, min(2.0, loss_weight)), device=self.device)
                        loss = loss_weight * base_loss

                    if __m3_guard_6553_20.error is not None:
                        loss = base_loss
                else:
                    loss = base_loss
                
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self._guarded_optimizer_step(
                    optimizer=self.opt,
                    params=list(self.model.parameters()),
                    tag="learn_pair",
                )
            
            # === kNN: collect from teacher forcing ===
            with guard_context(ctx='llm_adapter/llm_core.py:6577', catch_base=False) as __m3_guard_6574_12:
                self.collect_knn_from_teacher(prompt, response, core=attr_get_optional(self, "core", None))

            if __m3_guard_6574_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            with guard_context(ctx='llm_adapter/llm_core.py:6581', catch_base=False) as __m3_guard_6579_12:
                self._record_example(prompt, response, source="learn_pair")

            if __m3_guard_6579_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            with guard_context(ctx='llm_adapter/llm_core.py:6585', catch_base=False) as __m3_guard_6583_12:
                self._maybe_auto_rebuild_tokenizer()

            if __m3_guard_6583_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        except Exception as e:
            print(f"Error in learn_pair: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def train_batch(self, examples: list[tuple[str, str]], max_len: int = 256, batch_size: int = 8) -> tuple[int, float]:
        """Train on a batch of examples to improve speed. Returns (num_trained, avg_loss)."""
        if not examples:
            return 0, 0.0
            
        try:
            t = self.tok
            torch = self.torch
            self.model.train()
            
            # Process in mini-batches
            num_trained = 0
            global_loss_sum = 0.0
            
            for i in range(0, len(examples), batch_size):
                batch = examples[i:i+batch_size]
                
                self.opt.zero_grad(set_to_none=True)
                valid_samples = 0
                
                # [Optimization] Mixed Precision Context
                # Use autocast for lower memory and higher speed
                with self.torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda'), dtype=self.amp_dtype):
                    for prompt, response in batch:
                        with guard_context(ctx='llm_adapter/llm_core.py:6644', catch_base=False) as __m3_guard_6617_24:
                            encoded = t.encode(prompt)
                            if not encoded:
                                encoded = [t.BOS]
                            src_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
                            tgt = t.encode(response, add_special=True)
                            if len(tgt) < 2:
                                continue
                            tgt_in = tgt[:-1][:max_len]
                            tgt_out = tgt[1:][:max_len]
                            if not tgt_in:
                                continue
                                
                            tgt_in_ids = torch.tensor([tgt_in], dtype=torch.long, device=self.device)
                            tgt_out_ids = torch.tensor(tgt_out, dtype=torch.long, device=self.device)
                            
                            logits = self.model(src_ids, tgt_in_ids).squeeze(0)
                            loss = self.criterion(logits, tgt_out_ids)
                            
                            # Accumulate gradients (Direct BF16 backward, no scaler)
                            loss.backward()
                            global_loss_sum += loss.item()
                            valid_samples += 1

                        if __m3_guard_6617_24.error is not None:
                            e = __m3_guard_6617_24.error
                            print(f"Error in batch item: {e}")
                            continue
                
                if valid_samples > 0:
                    # Optimizer Step (Direct)
                    if valid_samples > 1:
                        # Average gradients over batch
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad /= valid_samples
                    
                    # Optional: Clip grad norm
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self._guarded_optimizer_step(
                        optimizer=self.opt,
                        params=list(self.model.parameters()),
                        tag="train_batch",
                    )
                    num_trained += valid_samples
                    
                    # [Memory Safety] Clear CUDA cache to prevent fragmentation (Slows down but prevents OOM)
                    if self.device.type == 'cuda':
                        self.torch.cuda.empty_cache()

                    
        except Exception as e:
            print(f"Error in train_batch: {e}")
            import traceback
            traceback.print_exc()
            # Return whatever was trained before the error
            try:
                return int(num_trained), 0.0
            except Exception:
                return 0, 0.0

        avg_loss = global_loss_sum / num_trained if num_trained > 0 else 0.0
        # Inform how many examples were processed
        try:
            print(f"[LLM-ADAPTER] train_batch: processed {num_trained}/{len(examples)} examples, avg_loss={avg_loss:.4f}")
        except Exception:
            logging.getLogger(__name__).exception("Swallowed exception")
        with guard_context(ctx='llm_adapter/llm_core.py:6689', catch_base=False) as __m3_guard_6687_8:
            self._maybe_auto_rebuild_tokenizer()

        if __m3_guard_6687_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return num_trained, avg_loss

    def train_on_example(self, prompt: str, response: str, max_len: int = 120) -> None:
        """Alias for learn_pair to maintain API compatibility with GUI/data loaders."""
        return self.learn_pair(prompt, response, max_len)

    def _iter_dpo_records_from_dir(self, data_dir: str):
        import json
        import glob

        patterns = [
            os.path.join(data_dir, '**', '*preference*.json'),
            os.path.join(data_dir, '**', '*preference*.jsonl'),
            os.path.join(data_dir, '**', '*dpo*.json'),
            os.path.join(data_dir, '**', '*chosen*.json'),
            os.path.join(data_dir, '**', '*.preference.auto.jsonl'),
        ]
        paths = []
        for pat in patterns:
            paths.extend(glob.glob(pat, recursive=True))
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            try:
                if path.lower().endswith(".jsonl"):
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            with guard_context(ctx='llm_adapter/llm_core.py:6725', catch_base=False) as __m3_guard_6723_28:
                                obj = json.loads(line)

                            if __m3_guard_6723_28.error is not None:
                                continue
                            if not isinstance(obj, dict):
                                continue
                            prompt = obj.get('prompt') or obj.get('question') or obj.get('input')
                            chosen = obj.get('chosen') or obj.get('preferred') or obj.get('good')
                            rejected = obj.get('rejected') or obj.get('dispreferred') or obj.get('bad')
                            if prompt and chosen and rejected:
                                yield str(prompt), str(chosen), str(rejected)
                else:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        items = data
                    elif isinstance(data, dict):
                        items = data.get('samples') or data.get('examples') or data.get('items') or [data]
                    else:
                        items = []
                    for obj in items:
                        if not isinstance(obj, dict):
                            continue
                        prompt = obj.get('prompt') or obj.get('question') or obj.get('input')
                        chosen = obj.get('chosen') or obj.get('preferred') or obj.get('good')
                        rejected = obj.get('rejected') or obj.get('dispreferred') or obj.get('bad')
                        if prompt and chosen and rejected:
                            yield str(prompt), str(chosen), str(rejected)
            except Exception:
                continue

    def _stable_hash(self, text: str) -> int:
        s = str(text or "").encode("utf-8", errors="ignore")
        return int(hashlib.sha1(s).hexdigest(), 16)

    def _split_train_val(self, items: List[Any], key_fn, val_fraction: float) -> Tuple[List[Any], List[Any]]:
        vf = max(0.0, min(0.99, float(val_fraction)))
        train = []
        val = []
        for x in items:
            key = str(key_fn(x))
            h = self._stable_hash(key) % 1000000
            if (h / 1000000.0) < vf:
                val.append(x)
            else:
                train.append(x)
        if not train and val:
            train, val = list(val), []
        return train, val

    def collect_dpo_preferences_from_logs(
        self,
        logs_dir: Optional[str] = None,
        out_path: Optional[str] = None,
        max_pairs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Auto-build DPO preference triples from accepted/rejected/chat logs."""
        if not self.dpo_auto_cfg.enabled or str(os.getenv("M3_DPO_AUTO_COLLECT", "1")).lower() not in ("1", "true", "yes", "on"):
            return {"enabled": False, "num_pairs": 0}
        base = logs_dir or os.path.dirname(TRAINING_PATH)
        chosen_path = os.path.join(base, "llm_training_data.jsonl")
        if not os.path.exists(chosen_path):
            chosen_path = TRAINING_PATH
        rejected_path = os.path.join(base, "llm_training_data.rejected.jsonl")
        chat_path = os.path.join(base, "chat_history.jsonl")
        if out_path is None:
            name = str(self.dpo_auto_cfg.output_file or "llm_training_data.preference.auto.jsonl")
            if not name.endswith(".jsonl"):
                name += ".jsonl"
            if "preference.auto" not in name:
                name = "llm_training_data.preference.auto.jsonl"
            out_path = os.path.join(base, name)

        def load_jsonl(path: str) -> List[Dict[str, Any]]:
            out = []
            if not os.path.exists(path):
                return out
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        with guard_context(ctx='llm_adapter/llm_core.py:6810', catch_base=False) as __m3_guard_6806_24:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                out.append(obj)

                        if __m3_guard_6806_24.error is not None:
                            continue
            except Exception:
                return out
            return out

        chosen_rows = load_jsonl(chosen_path)
        rejected_rows = load_jsonl(rejected_path)
        chat_rows = load_jsonl(chat_path)

        time_window = int(max(1, self.dpo_auto_cfg.time_window_sec))
        min_chars = int(max(1, self.dpo_auto_cfg.min_response_chars))
        hard_repeat = float(self.dpo_auto_cfg.hard_negative_repeat_threshold)
        rejected_by_hash: Dict[str, List[Dict[str, Any]]] = {}
        chosen_pool: List[Dict[str, Any]] = []

        def row_prompt(row: Dict[str, Any]) -> str:
            return str(row.get("prompt") or row.get("prompt_raw") or row.get("input") or row.get("question") or "").strip()

        def row_resp(row: Dict[str, Any]) -> str:
            return str(row.get("response") or row.get("text") or row.get("output") or "").strip()

        def ts_of(row: Dict[str, Any]) -> float:
            t = row.get("ts") or row.get("time") or row.get("t") or 0.0
            try:
                tv = float(t)
            except Exception:
                tv = 0.0
            if tv > 1e12:
                tv /= 1000.0
            return float(tv)

        def prompt_hash(prompt: str) -> str:
            return hashlib.sha1(str(prompt).encode("utf-8", errors="ignore")).hexdigest()[:16]

        for row in chosen_rows:
            p = row_prompt(row)
            r = row_resp(row)
            if not p or len(r) < min_chars:
                continue
            chosen_pool.append({
                "prompt": p,
                "response": r,
                "ts": ts_of(row),
                "hash": prompt_hash(p),
            })

        for row in rejected_rows:
            p = row_prompt(row)
            r = row_resp(row)
            if not p or len(r) < min_chars:
                continue
            is_hard = False
            rep_pen = self._repetition_penalty(r)
            if rep_pen >= hard_repeat:
                is_hard = True
            if self._is_identity_drift_output(r) or self._is_refusal_disclaimer(r):
                is_hard = True
            if self._is_numeric_dump_response(r) or self._is_backend_status_text(r):
                is_hard = True
            if not is_hard:
                continue
            ph = prompt_hash(p)
            rejected_by_hash.setdefault(ph, []).append({
                "prompt": p,
                "response": r,
                "ts": ts_of(row),
                "hash": ph,
            })

        # chat_history fallback mining
        last_user = ""
        for row in chat_rows:
            role = str(row.get("role") or row.get("speaker") or "").strip().lower()
            text = str(row.get("content") or row.get("text") or row.get("message") or "").strip()
            if not text:
                continue
            if role in ("user", "human"):
                last_user = text
                continue
            if role not in ("assistant", "m3", "bot"):
                continue
            if not last_user:
                continue
            ph = prompt_hash(last_user)
            entry = {"prompt": last_user, "response": text, "ts": ts_of(row), "hash": ph}
            if self._is_backend_status_text(text) or self._is_refusal_disclaimer(text) or self._is_identity_drift_output(text) or self._is_numeric_dump_response(text):
                rejected_by_hash.setdefault(ph, []).append(entry)
            else:
                chosen_pool.append(entry)

        max_pairs_eff = int(max_pairs if max_pairs is not None else self.dpo_auto_cfg.max_pairs)
        out_rows = []
        dedupe = set()
        for ch in chosen_pool:
            ph = ch["hash"]
            cands = rejected_by_hash.get(ph, [])
            if not cands:
                continue
            cands.sort(key=lambda x: abs(float(x.get("ts", 0.0)) - float(ch.get("ts", 0.0))))
            picked = None
            for rj in cands:
                dt = abs(float(rj.get("ts", 0.0)) - float(ch.get("ts", 0.0)))
                if dt <= time_window:
                    picked = rj
                    break
            if picked is None:
                picked = cands[0]
            key = (
                hashlib.sha1((ch["prompt"] + "\n" + ch["response"]).encode("utf-8", errors="ignore")).hexdigest(),
                hashlib.sha1((picked["prompt"] + "\n" + picked["response"]).encode("utf-8", errors="ignore")).hexdigest(),
            )
            if key in dedupe:
                continue
            dedupe.add(key)
            out_rows.append({
                "prompt": ch["prompt"],
                "chosen": ch["response"],
                "rejected": picked["response"],
                "source": "auto_log_hard_negative",
                "prompt_hash": ph,
            })
            if len(out_rows) >= max_pairs_eff:
                break

        if out_rows:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for row in out_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._log_jsonl(
            os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
            {
                "kind": "dpo_auto_collect",
                "chosen_candidates": int(len(chosen_pool)),
                "rejected_candidates": int(sum(len(v) for v in rejected_by_hash.values())),
                "num_pairs": int(len(out_rows)),
                "output_path": out_path,
            },
        )
        return {"enabled": True, "num_pairs": len(out_rows), "output_path": out_path}

    def train_supervised_with_early_stopping(
        self,
        epochs: int = 1,
        data_dir: str = 'data_set',
        max_len: int = 120,
        limit: int | None = None,
    ) -> dict:
        pairs = list(self._iter_supervised_records_from_dir(data_dir))
        if limit is not None:
            pairs = pairs[:max(0, int(limit))]
        if not pairs:
            return {"num_pairs": 0, "epochs_run": 0, "stopped_early": False}

        cfg = self.early_stop_cfg
        train_pairs, val_pairs = self._split_train_val(
            pairs,
            key_fn=lambda x: f"{x[0]}|||{x[1]}",
            val_fraction=float(cfg.val_fraction),
        )
        max_epochs = int(max(1, min(int(cfg.max_epochs), int(max(1, epochs)))))
        best_metric = float("inf")
        best_state = None
        best_epoch = 0
        patience = int(max(0, cfg.patience))
        min_delta = float(cfg.min_delta)
        bad = 0
        epochs_run = 0
        for ep in range(1, max_epochs + 1):
            epochs_run = ep
            for p, r in train_pairs:
                with guard_context(ctx='llm_adapter/llm_core.py:6984', catch_base=False) as __m3_guard_6982_16:
                    self.learn_pair(p, r, max_len=max_len)

                if __m3_guard_6982_16.error is not None:
                    continue
            if not val_pairs:
                continue
            total_lp = 0.0
            total_tok = 0
            for p, r in val_pairs:
                with guard_context(ctx='llm_adapter/llm_core.py:6995', catch_base=False) as __m3_guard_6991_16:
                    lp = self._sequence_logprob(p, r, max_len=max_len)
                    total_lp += float(lp)
                    total_tok += max(1, len(self.tok.encode(r, add_special=True)) - 1)

                if __m3_guard_6991_16.error is not None:
                    continue
            val_nll = float(-total_lp / max(1, total_tok))
            improved = val_nll < (best_metric - min_delta)
            if improved:
                best_metric = val_nll
                best_epoch = ep
                bad = 0
                if bool(cfg.restore_best_weights):
                    with guard_context(ctx='llm_adapter/llm_core.py:7006', catch_base=False) as __m3_guard_7004_20:
                        best_state = copy.deepcopy(self.model.state_dict())

                    if __m3_guard_7004_20.error is not None:
                        best_state = None
            else:
                bad += 1
                if bad > patience:
                    break
        stopped_early = bool(epochs_run < max_epochs)
        if stopped_early and best_state is not None and bool(cfg.restore_best_weights):
            with guard_context(ctx='llm_adapter/llm_core.py:7016', catch_base=False) as __m3_guard_7014_12:
                self.model.load_state_dict(best_state, strict=False)

            if __m3_guard_7014_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        result = {
            "num_pairs": len(pairs),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "best_epoch": int(best_epoch),
            "epochs_run": int(epochs_run),
            "stopped_early": bool(stopped_early),
            "val_nll": float(best_metric if np.isfinite(best_metric) else 0.0),
        }
        self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"), {"kind": "early_stop", "phase": "supervised", **result})
        return result

    def train_dpo_with_early_stopping(
        self,
        epochs: int = 2,
        beta: float = 0.1,
        data_dir: str = 'data_set',
    ) -> dict:
        rows = list(self._iter_dpo_records_from_dir(data_dir))
        if not rows:
            return {"num_pairs": 0, "epochs_run": 0, "stopped_early": False}
        cfg = self.early_stop_cfg
        train_rows, val_rows = self._split_train_val(
            rows,
            key_fn=lambda x: f"{x[0]}|||{x[1]}|||{x[2]}",
            val_fraction=float(cfg.val_fraction),
        )
        max_epochs = int(max(1, min(int(cfg.max_epochs), int(max(1, epochs)))))
        best_score = -1e9
        best_epoch = 0
        best_state = None
        patience = int(max(0, cfg.patience))
        min_delta = float(cfg.min_delta)
        bad = 0
        epochs_run = 0
        best_rate = 0.0
        best_margin = -1e9
        for ep in range(1, max_epochs + 1):
            epochs_run = ep
            for p, ch, rj in train_rows:
                with guard_context(ctx='llm_adapter/llm_core.py:7060', catch_base=False) as __m3_guard_7058_16:
                    self.dpo_step(p, ch, rj, beta=beta)

                if __m3_guard_7058_16.error is not None:
                    continue
            if not val_rows:
                continue
            better = 0
            margins = []
            for p, ch, rj in val_rows:
                with guard_context(ctx='llm_adapter/llm_core.py:7073', catch_base=False) as __m3_guard_7067_16:
                    ch_lp = float(self._sequence_logprob(p, ch))
                    rj_lp = float(self._sequence_logprob(p, rj))
                    if ch_lp > rj_lp:
                        better += 1
                    margins.append(ch_lp - rj_lp)

                if __m3_guard_7067_16.error is not None:
                    continue
            rate = float(better / max(1, len(val_rows)))
            margin = float(np.mean(margins)) if margins else 0.0
            score = rate + 0.05 * margin
            if score > (best_score + min_delta):
                best_score = score
                best_rate = rate
                best_margin = margin
                best_epoch = ep
                bad = 0
                if bool(cfg.restore_best_weights):
                    with guard_context(ctx='llm_adapter/llm_core.py:7087', catch_base=False) as __m3_guard_7085_20:
                        best_state = copy.deepcopy(self.model.state_dict())

                    if __m3_guard_7085_20.error is not None:
                        best_state = None
            else:
                bad += 1
                if bad > patience:
                    break
        stopped_early = bool(epochs_run < max_epochs)
        if stopped_early and best_state is not None and bool(cfg.restore_best_weights):
            with guard_context(ctx='llm_adapter/llm_core.py:7097', catch_base=False) as __m3_guard_7095_12:
                self.model.load_state_dict(best_state, strict=False)

            if __m3_guard_7095_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        result = {
            "num_pairs": len(rows),
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
            "best_epoch": int(best_epoch),
            "epochs_run": int(epochs_run),
            "stopped_early": bool(stopped_early),
            "chosen_better_rate": float(best_rate),
            "margin": float(best_margin if np.isfinite(best_margin) else 0.0),
        }
        self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"), {"kind": "early_stop", "phase": "dpo", **result})
        return result

    def train_dpo_from_dir(self, epochs: int = 2, beta: float = 0.1, data_dir: str = 'data_set') -> dict:
        """Train using DPO from preference data in data_set directory."""
        import json
        import glob
        with guard_context(ctx='llm_adapter/llm_core.py:7124', catch_base=False) as __m3_guard_7116_8:
            if self.dpo_auto_cfg.enabled and str(os.getenv("M3_DPO_AUTO_COLLECT", "1")).lower() in ("1", "true", "yes", "on"):
                auto_out = os.path.join(data_dir, "llm_training_data.preference.auto.jsonl")
                self.collect_dpo_preferences_from_logs(
                    logs_dir=os.path.dirname(TRAINING_PATH),
                    out_path=auto_out,
                    max_pairs=int(self.dpo_auto_cfg.max_pairs),
                )

        if __m3_guard_7116_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        if self.early_stop_cfg.enabled and str(os.getenv("M3_TRAIN_EARLY_STOP", "1")).lower() in ("1", "true", "yes", "on"):
            return self.train_dpo_with_early_stopping(epochs=epochs, beta=beta, data_dir=data_dir)
        
        total_loss = 0.0
        num_samples = 0
        
        try:
            # Search for DPO-style preference data files
            patterns = [
                os.path.join(data_dir, '**', '*preference*.json'),
                os.path.join(data_dir, '**', '*preference*.jsonl'),
                os.path.join(data_dir, '**', '*dpo*.json'),
                os.path.join(data_dir, '**', '*chosen*.json'),
            ]
            
            files = []
            for pattern in patterns:
                files.extend(glob.glob(pattern, recursive=True))
            
            if not files:
                logger.warning(f'No DPO preference files found in {data_dir}')
                return {'avg_loss': 0.0, 'num_samples': 0}
            
            for epoch in range(epochs):
                epoch_samples = 0
                
                for filepath in files:
                    try:
                        if filepath.lower().endswith('.jsonl'):
                            data = []
                            with open(filepath, 'r', encoding='utf-8') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    with guard_context(ctx='llm_adapter/llm_core.py:7165', catch_base=False) as __m3_guard_7161_36:
                                        row = json.loads(line)
                                        if isinstance(row, dict):
                                            data.append(row)

                                    if __m3_guard_7161_36.error is not None:
                                        continue
                        else:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                        
                        # Handle different DPO data formats
                        if isinstance(data, list):
                            samples = data
                        elif isinstance(data, dict) and 'samples' in data:
                            samples = data['samples']
                        else:
                            samples = [data]
                        
                        for sample in samples:
                            if isinstance(sample, dict):
                                # Extract prompt, chosen, rejected
                                prompt = sample.get('prompt', sample.get('question', sample.get('input', '')))
                                chosen = sample.get('chosen', sample.get('preferred', sample.get('good', '')))
                                rejected = sample.get('rejected', sample.get('dispreferred', sample.get('bad', '')))
                                
                                if prompt and chosen and rejected:
                                    # Perform DPO step
                                    self.dpo_step(str(prompt), str(chosen), str(rejected), beta=beta)
                                    epoch_samples += 1
                    
                    except Exception as e:
                        logger.debug(f'Error processing DPO file {filepath}: {e}')
                        continue
                
                if epoch_samples > 0:
                    num_samples += epoch_samples
                    logger.info(f'DPO epoch {epoch+1}/{epochs}: {epoch_samples} samples processed')
            
            avg_loss = total_loss / max(num_samples, 1)
            return {'avg_loss': avg_loss, 'num_samples': num_samples}
        
        except Exception as e:
            logger.error(f'DPO training error: {e}')
            return {'avg_loss': 0.0, 'num_samples': 0}

    # === Supervised auto-train/test from data_set ===
    def _extract_supervised_pair(self, obj: dict) -> tuple | None:
        """Extract (prompt, response) from a generic object with common key variants.

        Supported key pairs:
        - ('prompt', 'response')
        - ('input', 'output') or ('inputs', 'targets')
        - ('question', 'answer')
        - ('instruction', 'output') / ('instruction', 'response')
        - ('context', 'completion')
        """
        key_pairs = [
            ('prompt', 'response'),
            ('input', 'output'), ('inputs', 'targets'),
            ('question', 'answer'),
            ('instruction', 'output'), ('instruction', 'response'),
            ('context', 'completion'),
        ]
        for a, b in key_pairs:
            if a in obj and b in obj:
                pa = str(obj.get(a, '')).strip()
                pb = str(obj.get(b, '')).strip()
                if pa and pb:
                    return pa, pb
        return None

    def _iter_supervised_records_from_dir(self, data_dir: str):
        import json, os
        for root, _, files in os.walk(data_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    if fn.lower().endswith('.jsonl'):
                        with open(fp, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                with guard_context(ctx='llm_adapter/llm_core.py:7246', catch_base=False) as __m3_guard_7244_32:
                                    obj = json.loads(line)

                                if __m3_guard_7244_32.error is not None:
                                    continue
                                pair = self._extract_supervised_pair(obj)
                                if pair:
                                    yield pair
                    elif fn.lower().endswith('.json'):
                        with open(fp, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            pair = self._extract_supervised_pair(data)
                            if pair:
                                yield pair
                            for k in ('data', 'examples', 'samples', 'items'):
                                arr = data.get(k)
                                if isinstance(arr, list):
                                    for obj in arr:
                                        if isinstance(obj, dict):
                                            pair = self._extract_supervised_pair(obj)
                                            if pair:
                                                yield pair
                        elif isinstance(data, list):
                            for obj in data:
                                if isinstance(obj, dict):
                                    pair = self._extract_supervised_pair(obj)
                                    if pair:
                                        yield pair
                except Exception:
                    continue

    def train_supervised_from_dir(self, epochs: int = 1, data_dir: str = 'data_set', max_len: int = 120, limit: int | None = None) -> dict:
        """Train with (prompt, response) pairs found under data_dir (recursive)."""
        if self.early_stop_cfg.enabled and str(os.getenv("M3_TRAIN_EARLY_STOP", "1")).lower() in ("1", "true", "yes", "on"):
            return self.train_supervised_with_early_stopping(
                epochs=epochs,
                data_dir=data_dir,
                max_len=max_len,
                limit=limit,
            )
        import random
        pairs = list(self._iter_supervised_records_from_dir(data_dir))
        if not pairs:
            logger.warning(f'No supervised (prompt,response) pairs found in {data_dir}')
            return {'num_pairs': 0, 'epochs': 0}

        if limit is not None:
            pairs = pairs[:max(0, int(limit))]

        random.shuffle(pairs)
        for e in range(max(1, int(epochs))):
            for prompt, response in pairs:
                with guard_context(ctx='llm_adapter/llm_core.py:7298', catch_base=False) as __m3_guard_7296_16:
                    self.learn_pair(prompt, response, max_len=max_len)

                if __m3_guard_7296_16.error is not None:
                    continue
        logger.info(f'Supervised training complete: pairs={len(pairs)}, epochs={epochs}')
        return {'num_pairs': len(pairs), 'epochs': int(epochs)}

    def evaluate_supervised_from_dir(self, data_dir: str = 'data_set', max_len: int = 120, limit: int | None = None) -> dict:
        """Evaluate average NLL and perplexity over supervised pairs under data_dir."""
        import math
        pairs = []
        for i, pair in enumerate(self._iter_supervised_records_from_dir(data_dir)):
            pairs.append(pair)
            if limit is not None and len(pairs) >= int(limit):
                break
        if not pairs:
            logger.warning(f'No supervised eval pairs found in {data_dir}')
            return {'num_pairs': 0}

        total_logp = 0.0
        total_tokens = 0
        for prompt, response in pairs:
            with guard_context(ctx='llm_adapter/llm_core.py:7322', catch_base=False) as __m3_guard_7318_12:
                lp = self._sequence_logprob(prompt, response, max_len=max_len)
                total_logp += float(lp)
                total_tokens += max(1, len(self.tok.encode(response, add_special=True)) - 1)

            if __m3_guard_7318_12.error is not None:
                continue

        avg_nll = float(-total_logp / max(1, total_tokens))
        ppl = float(math.exp(avg_nll)) if avg_nll < 30 else float('inf')
        metrics = {'num_pairs': len(pairs), 'avg_nll': avg_nll, 'ppl': ppl}
        with guard_context(ctx='llm_adapter/llm_core.py:7330', catch_base=False) as __m3_guard_7328_8:
            self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {"kind": "eval_supervised", **metrics})

        if __m3_guard_7328_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        logger.info(f'Supervised eval: pairs={metrics["num_pairs"]}, avg_nll={avg_nll:.4f}, ppl={ppl:.2f}')
        return metrics

    def evaluate_dpo_from_dir(self, data_dir: str = 'data_set', limit: int | None = None) -> dict:
        """Evaluate DPO preference consistency: rate(chosen_logp > rejected_logp)."""
        import json, glob
        samples = []
        try:
            patterns = [
                os.path.join(data_dir, '**', '*preference*.json'),
                os.path.join(data_dir, '**', '*preference*.jsonl'),
                os.path.join(data_dir, '**', '*dpo*.json'),
                os.path.join(data_dir, '**', '*chosen*.json'),
            ]
            paths = []
            for pat in patterns:
                paths.extend(glob.glob(pat, recursive=True))
            for filepath in paths:
                try:
                    if filepath.lower().endswith('.jsonl'):
                        data = []
                        with open(filepath, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                with guard_context(ctx='llm_adapter/llm_core.py:7362', catch_base=False) as __m3_guard_7358_32:
                                    row = json.loads(line)
                                    if isinstance(row, dict):
                                        data.append(row)

                                if __m3_guard_7358_32.error is not None:
                                    continue
                    else:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    if isinstance(data, dict):
                        items = data.get('data') or data.get('examples') or data.get('samples') or data.get('items') or []
                        if isinstance(items, list):
                            iterable = items
                        else:
                            iterable = [data]
                    elif isinstance(data, list):
                        iterable = data
                    else:
                        iterable = []
                    for obj in iterable:
                        prompt = obj.get('prompt') or obj.get('question') or obj.get('input')
                        chosen = obj.get('chosen') or obj.get('chosen_response') or obj.get('preferred')
                        rejected = obj.get('rejected') or obj.get('rejected_response') or obj.get('other')
                        if not prompt or not chosen or not rejected:
                            resp = obj.get('responses') or obj.get('candidates')
                            if isinstance(resp, list) and len(resp) >= 2:
                                ch = None; rj = None
                                for r in resp:
                                    txt = r.get('text') or r.get('response') or r.get('output')
                                    lab = (r.get('label') or r.get('preference') or '').lower()
                                    if lab in ('chosen', 'preferred', 'pos', 'positive', 'good'):
                                        ch = txt
                                    elif lab in ('rejected', 'neg', 'negative', 'bad'):
                                        rj = txt
                                if prompt and ch and rj:
                                    chosen, rejected = ch, rj
                        if prompt and chosen and rejected:
                            samples.append((str(prompt), str(chosen), str(rejected)))
                            if limit is not None and len(samples) >= int(limit):
                                break
                except Exception:
                    continue
        except Exception:
            logging.getLogger(__name__).exception("Swallowed exception")

        if not samples:
            logger.warning(f'No DPO preference samples found in {data_dir}')
            return {'num_pairs': 0}

        better = 0
        margins = []
        for prompt, chosen, rejected in samples:
            with guard_context(ctx='llm_adapter/llm_core.py:7416', catch_base=False) as __m3_guard_7410_12:
                ch_lp = float(self._sequence_logprob(prompt, chosen))
                rj_lp = float(self._sequence_logprob(prompt, rejected))
                if ch_lp > rj_lp:
                    better += 1
                margins.append(ch_lp - rj_lp)

            if __m3_guard_7410_12.error is not None:
                continue
        rate = better / max(1, len(samples))
        avg_margin = float(np.mean(margins)) if margins else 0.0
        metrics = {'num_pairs': len(samples), 'chosen_better_rate': rate, 'avg_logp_margin': avg_margin}
        with guard_context(ctx='llm_adapter/llm_core.py:7423', catch_base=False) as __m3_guard_7421_8:
            self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {"kind": "eval_dpo", **metrics})

        if __m3_guard_7421_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        logger.info(f'DPO eval: pairs={metrics["num_pairs"]}, chosen>rejected rate={rate:.3f}, avg_margin={avg_margin:.4f}')
        return metrics

    def auto_train_and_test_from_data_set(self, data_dir: str = 'data_set', epochs: int = 1,
                                           eval_fraction: float = 0.1, max_len: int = 120,
                                           enable_dpo: bool = True, limit: int | None = None) -> dict:
        """End-to-end: train on data_set and run simple eval.

        - Supervised: scan (prompt,response) pairs, train epochs, evaluate on held-out fraction.
        - DPO: optionally run train_dpo_from_dir() then evaluate preference consistency.
        """
        import random

        pairs = list(self._iter_supervised_records_from_dir(data_dir))
        if limit is not None:
            pairs = pairs[:max(0, int(limit))]

        sup_metrics = {}
        if pairs:
            random.shuffle(pairs)
            n_eval = int(len(pairs) * max(0.0, min(1.0, eval_fraction)))
            eval_pairs = pairs[:n_eval]
            train_pairs = pairs[n_eval:]

            for e in range(max(1, int(epochs))):
                for prompt, response in train_pairs:
                    with guard_context(ctx='llm_adapter/llm_core.py:7453', catch_base=False) as __m3_guard_7451_20:
                        self.learn_pair(prompt, response, max_len=max_len)

                    if __m3_guard_7451_20.error is not None:
                        continue

            total_logp = 0.0
            total_tokens = 0
            for prompt, response in eval_pairs:
                with guard_context(ctx='llm_adapter/llm_core.py:7463', catch_base=False) as __m3_guard_7459_16:
                    lp = self._sequence_logprob(prompt, response, max_len=max_len)
                    total_logp += float(lp)
                    total_tokens += max(1, len(self.tok.encode(response, add_special=True)) - 1)

                if __m3_guard_7459_16.error is not None:
                    continue
            avg_nll = float(-total_logp / max(1, total_tokens)) if total_tokens else float('inf')
            ppl = float(np.exp(avg_nll)) if np.isfinite(avg_nll) and avg_nll < 30 else float('inf')
            sup_metrics = {
                'supervised': True,
                'train_pairs': len(train_pairs),
                'eval_pairs': len(eval_pairs),
                'avg_nll': avg_nll,
                'ppl': ppl,
            }
            logger.info(f'Auto supervised: train={len(train_pairs)}, eval={len(eval_pairs)}, avg_nll={avg_nll:.4f}, ppl={ppl:.2f}')
            with guard_context(ctx='llm_adapter/llm_core.py:7477', catch_base=False) as __m3_guard_7475_12:
                self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {"kind": "auto_supervised_eval", **sup_metrics})

            if __m3_guard_7475_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
        else:
            logger.info('Auto supervised: no pairs found; skipping')

        dpo_metrics = {}
        if enable_dpo:
            with guard_context(ctx='llm_adapter/llm_core.py:7487', catch_base=False) as __m3_guard_7484_12:
                self.train_dpo_from_dir(epochs=max(1, int(epochs)), beta=0.1, data_dir=data_dir)
                dpo_metrics = self.evaluate_dpo_from_dir(data_dir=data_dir)

            if __m3_guard_7484_12.error is not None:
                e = __m3_guard_7484_12.error
                logger.warning(f'Auto DPO skipped due to error: {e}')

                return {**({'supervised_metrics': sup_metrics} if sup_metrics else {}),
                **({'dpo_metrics': dpo_metrics} if dpo_metrics else {})}

    def train_all_from_data_set(self,
                                data_dir: str = 'data_set',
                                epochs: int = 1,
                                max_len: int = 120,
                                limit: int | None = None,
                                enable_dpo: bool = True,
                                eval_fraction: float = 0.1) -> dict:
        """Unified training over data_set directory.
        - Detects datasets by files present and streams samples to training.
        - No fixed heuristics: evaluation sampling uses eval_fraction reservoir.
        - Returns summary metrics for supervised and DPO evaluations.
        """
        import json, os, csv, math, random, glob
        with guard_context(ctx='llm_adapter/llm_core.py:7513', catch_base=False) as __m3_guard_7506_8:
            if enable_dpo and self.dpo_auto_cfg.enabled and str(os.getenv("M3_DPO_AUTO_COLLECT", "1")).lower() in ("1", "true", "yes", "on"):
                self.collect_dpo_preferences_from_logs(
                    logs_dir=os.path.dirname(TRAINING_PATH),
                    out_path=os.path.join(data_dir, "llm_training_data.preference.auto.jsonl"),
                    max_pairs=int(self.dpo_auto_cfg.max_pairs),
                )

        if __m3_guard_7506_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        if self.early_stop_cfg.enabled and str(os.getenv("M3_TRAIN_EARLY_STOP", "1")).lower() in ("1", "true", "yes", "on"):
            sup = self.train_supervised_with_early_stopping(
                epochs=epochs,
                data_dir=data_dir,
                max_len=max_len,
                limit=limit,
            )
            dpo = {}
            if enable_dpo:
                dpo = self.train_dpo_with_early_stopping(
                    epochs=max(1, int(epochs)),
                    beta=0.1,
                    data_dir=data_dir,
                )
            return {
                **({"supervised_metrics": sup} if sup else {}),
                **({"dpo_metrics": dpo} if dpo else {}),
            }

        # Reservoir buffers for evaluation (fractional sampling)
        sup_eval = []  # list[(prompt, response)]
        dpo_eval = []  # list[(prompt, chosen, rejected)]

        def maybe_take_sup_eval(p, r):
            if random.random() < max(0.0, min(1.0, float(eval_fraction))):
                sup_eval.append((p, r))

        def maybe_take_dpo_eval(p, ch, rj):
            if random.random() < max(0.0, min(1.0, float(eval_fraction))):
                dpo_eval.append((p, ch, rj))

        # === A. SLURP ===
        def iter_slurp_pairs(base_dir: str):
            slurp_roots = []
            # Common layout: data_set/slurp-master/dataset/slurp/{train,devel}.jsonl
            for cand in glob.glob(os.path.join(base_dir, '**', 'dataset', 'slurp'), recursive=True):
                tr = os.path.join(cand, 'train.jsonl')
                dv = os.path.join(cand, 'devel.jsonl')
                if os.path.exists(tr):
                    slurp_roots.append((tr, dv if os.path.exists(dv) else None))
            for tr, dv in slurp_roots:
                for path in filter(None, [tr, dv]):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                obj = json.loads(line)
                                utt = str(obj.get('utt', '')).strip()
                                intent = str(obj.get('intent', '')).strip()
                                slots = obj.get('slots', {})
                                if isinstance(slots, list):
                                    # Some SLURP variants store list of {slot, value}
                                    with guard_context(ctx='llm_adapter/llm_core.py:7571', catch_base=False) as __m3_guard_7569_36:
                                        slots = {str(s.get('slot','')): str(s.get('value','')) for s in slots if isinstance(s, dict)}

                                    if __m3_guard_7569_36.error is not None:
                                        slots = {}
                                if utt and intent is not None:
                                    # Deterministic textualization of semantic target
                                    slot_str = ''
                                    if isinstance(slots, dict) and slots:
                                        parts = [f"{k}={v}" for k, v in slots.items()]
                                        slot_str = ' ' + ' '.join(parts)
                                    resp = f"intent={intent}{slot_str}"
                                    yield utt, resp
                    except Exception:
                        continue

        # === B. MIND ===
        def iter_mind_pairs(base_dir: str):
            # Locate any folder containing both news.tsv and behaviors.tsv
            mind_dirs = []
            for root, dirs, files in os.walk(base_dir):
                if {'news.tsv', 'behaviors.tsv'}.issubset(set(files)):
                    mind_dirs.append(root)
            for mdir in mind_dirs:
                news_path = os.path.join(mdir, 'news.tsv')
                beh_path = os.path.join(mdir, 'behaviors.tsv')
                with guard_context(ctx='llm_adapter/llm_core.py:7641', catch_base=False) as __m3_guard_7594_16:
                    id2title = {}
                    with open(news_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='\t')
                        for row in reader:
                            if not row:
                                continue
                            # Robust: NewsID at col 0, Title at col 3 as per MIND standard
                            nid = row[0] if len(row) > 0 else ''
                            ttl = row[3] if len(row) > 3 else ''
                            if nid:
                                id2title[nid] = ttl
                    # Iterate behaviors
                    with open(beh_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='\t')
                        for row in reader:
                            if len(row) < 5:
                                continue
                            hist = row[3] or ''
                            impr = row[4] or ''
                            hist_ids = [h for h in hist.split(' ') if h]
                            cand = [x for x in impr.split(' ') if x]
                            if not cand:
                                continue
                            # Titles
                            hist_titles = [id2title.get(h, h) for h in hist_ids]
                            cand_ids = [c.split('-')[0] for c in cand]
                            cand_labels = [c.split('-')[1] for c in cand if '-' in c]
                            cand_titles = [id2title.get(cid, cid) for cid in cand_ids]
                            # Build prompt (deterministic textualization, no magic numbers)
                            prompt = ''
                            if hist_titles:
                                prompt += 'History: ' + ' | '.join(hist_titles) + '\n'
                            prompt += 'Candidates: ' + ' | '.join(cand_titles)
                            # Train pairs for clicked items
                            for idx, lab in enumerate(cand_labels):
                                if lab == '1':
                                    resp = cand_titles[idx]
                                    yield prompt, resp
                                    # DPO triple for one negative
                                    if enable_dpo:
                                        # choose the first negative deterministically
                                        for j, lj in enumerate(cand_labels):
                                            if lj == '0':
                                                yield (prompt, ('__DPO__', cand_titles[idx], cand_titles[j]))
                                                break

                if __m3_guard_7594_16.error is not None:
                    continue

        # === C. Generic supervised pairs ===
        def iter_generic_pairs(base_dir: str):
            for p, r in self._iter_supervised_records_from_dir(base_dir):
                yield p, r

        # Unified streaming over all sources
        total_pairs = 0
        processed = 0
        # Pre-build sources list in stable order
        sources = [
            ('slurp', iter_slurp_pairs(data_dir)),
            ('mind', iter_mind_pairs(data_dir)),
            ('generic', iter_generic_pairs(data_dir)),
        ]

        # Training
        for e in range(max(1, int(epochs))):
            for name, it in sources:
                for item in it:
                    # MIND iterator may emit DPO triples as tagged tuples
                    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple) and item[1][0] == '__DPO__':
                        # DPO triple emitted; handle only in enable_dpo pass
                        if enable_dpo:
                            _, (tag, chosen, rejected) = item
                            prompt = item[0]
                            with guard_context(ctx='llm_adapter/llm_core.py:7672', catch_base=False) as __m3_guard_7669_28:
                                self.dpo_step(prompt, chosen, rejected)
                                maybe_take_dpo_eval(prompt, chosen, rejected)

                            if __m3_guard_7669_28.error is not None:
                                logging.getLogger(__name__).exception("Swallowed exception")
                        continue
                    with guard_context(ctx='llm_adapter/llm_core.py:7682', catch_base=False) as __m3_guard_7675_20:
                        prompt, response = item
                        self.learn_pair(prompt, response, max_len=max_len)
                        maybe_take_sup_eval(prompt, response)
                        processed += 1
                        if limit is not None and processed >= int(limit):
                            break

                    if __m3_guard_7675_20.error is not None:
                        continue
                if limit is not None and processed >= int(limit):
                    break
            if limit is not None and processed >= int(limit):
                break

        # Additionally, generic DPO files (if present)
        dpo_metrics = {}
        if enable_dpo:
            with guard_context(ctx='llm_adapter/llm_core.py:7695', catch_base=False) as __m3_guard_7692_12:
                self.train_dpo_from_dir(epochs=max(1, int(epochs)), beta=0.1, data_dir=data_dir)

            if __m3_guard_7692_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            # Eval: include proactively collected mind triples + generic files
            with guard_context(ctx='llm_adapter/llm_core.py:7701', catch_base=False) as __m3_guard_7698_12:
                gen_dpo = self.evaluate_dpo_from_dir(data_dir=data_dir)
                dpo_metrics.update(gen_dpo)

            if __m3_guard_7698_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            # Evaluate buffered triples
            better = 0
            margins = []
            for prompt, chosen, rejected in dpo_eval:
                with guard_context(ctx='llm_adapter/llm_core.py:7713', catch_base=False) as __m3_guard_7707_16:
                    ch_lp = float(self._sequence_logprob(prompt, chosen, max_len=max_len))
                    rj_lp = float(self._sequence_logprob(prompt, rejected, max_len=max_len))
                    if ch_lp > rj_lp:
                        better += 1
                    margins.append(ch_lp - rj_lp)

                if __m3_guard_7707_16.error is not None:
                    continue
            if margins:
                rate = better / max(1, len(margins))
                avg_margin = float(np.mean(margins))
                dpo_metrics.setdefault('chosen_better_rate_buffer', rate)
                dpo_metrics.setdefault('avg_logp_margin_buffer', avg_margin)

        # Supervised eval from reservoir sup_eval
        total_logp = 0.0
        total_tokens = 0
        for prompt, response in sup_eval:
            with guard_context(ctx='llm_adapter/llm_core.py:7729', catch_base=False) as __m3_guard_7725_12:
                lp = self._sequence_logprob(prompt, response, max_len=max_len)
                total_logp += float(lp)
                total_tokens += max(1, len(self.tok.encode(response, add_special=True)) - 1)

            if __m3_guard_7725_12.error is not None:
                continue
        sup_metrics = {}
        if total_tokens > 0:
            avg_nll = float(-total_logp / total_tokens)
            ppl = float(math.exp(avg_nll)) if avg_nll < 30 else float('inf')
            sup_metrics = {'num_eval_pairs': len(sup_eval), 'avg_nll': avg_nll, 'ppl': ppl}
            with guard_context(ctx='llm_adapter/llm_core.py:7738', catch_base=False) as __m3_guard_7736_12:
                self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {"kind": "train_all_eval", **sup_metrics, **({'dpo_metrics': dpo_metrics} if dpo_metrics else {})})

            if __m3_guard_7736_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

        return {**({'supervised_metrics': sup_metrics} if sup_metrics else {}),
                **({'dpo_metrics': dpo_metrics} if dpo_metrics else {})}

    def _sequence_logprob(self, prompt: str, target: str, max_len: int = 120) -> float:
        """Compute summed log-probability log p(target | prompt) under teacher forcing."""
        torch = self.torch
        t = self.tok
        with torch.no_grad():
            src = t.encode(prompt, add_special=False)
            tgt = t.encode(target, add_special=True)  # BOS ... EOS
            if len(src) == 0 or len(tgt) < 2:
                return 0.0
            tgt_in = tgt[:-1][:max_len]
            tgt_out = tgt[1:][:max_len]
            src_ids = torch.tensor([src], dtype=torch.long, device=self.device)
            tgt_in_ids = torch.tensor([tgt_in], dtype=torch.long, device=self.device)
            tgt_out_ids = torch.tensor([tgt_out], dtype=torch.long, device=self.device)
            logits = self.model(src_ids, tgt_in_ids)  # (1, T, V)
            logp = torch.log_softmax(logits, dim=-1)
            gathered = torch.gather(logp, dim=-1, index=tgt_out_ids.unsqueeze(-1)).squeeze(-1)
            # mask PAD targets
            mask = (tgt_out_ids != t.PAD)
            s = (gathered * mask).sum().item()
            return float(s)

    def dpo_step(self, prompt: str, chosen: str, rejected: str, beta: float = 0.1, max_len: int = 120) -> None:
        """One DPO update step to prefer chosen over rejected for a prompt."""
        torch = self.torch
        t = self.tok
        # Build tensors
        src = t.encode(prompt, add_special=False)
        ch = t.encode(chosen, add_special=True)
        rj = t.encode(rejected, add_special=True)
        if len(src) == 0 or len(ch) < 2 or len(rj) < 2:
            return
        ch_in = ch[:-1][:max_len]; ch_out = ch[1:][:max_len]
        rj_in = rj[:-1][:max_len]; rj_out = rj[1:][:max_len]
        src_ids = torch.tensor([src], dtype=torch.long, device=self.device)
        ch_in_ids = torch.tensor([ch_in], dtype=torch.long, device=self.device)
        ch_out_ids = torch.tensor([ch_out], dtype=torch.long, device=self.device)
        rj_in_ids = torch.tensor([rj_in], dtype=torch.long, device=self.device)
        rj_out_ids = torch.tensor([rj_out], dtype=torch.long, device=self.device)
        # Forward
        self.model.train()
        # Serialize to avoid concurrent modification during backward
        with self._learn_lock:
            e_src = self.model.emb(src_ids)
            _, h = self.model.encoder(e_src)
            # chosen
            e_ch = self.model.emb(ch_in_ids)
            o_ch, _ = self.model.decoder(e_ch, h)
            logits_ch = self.model.head(o_ch).contiguous().clone()
            logp_ch = torch.log_softmax(logits_ch, dim=-1)
            ch_lp = torch.gather(logp_ch, dim=-1, index=ch_out_ids.unsqueeze(-1)).squeeze(-1)
            ch_mask = (ch_out_ids != t.PAD)
            ch_sum = (ch_lp * ch_mask).sum()
            # rejected
            e_rj = self.model.emb(rj_in_ids)
            o_rj, _ = self.model.decoder(e_rj, h)
            logits_rj = self.model.head(o_rj).contiguous().clone()
            logp_rj = torch.log_softmax(logits_rj, dim=-1)
            rj_lp = torch.gather(logp_rj, dim=-1, index=rj_out_ids.unsqueeze(-1)).squeeze(-1)
            rj_mask = (rj_out_ids != t.PAD)
            rj_sum = (rj_lp * rj_mask).sum()
            
            # === Phase 5: M3-aware margin (NO MAGIC NUMBERS)
            beta = float(os.getenv("DPO_BETA", str(beta)))
            phi_margin = torch.tensor(0.0, device=self.device)
            with guard_context(ctx='llm_adapter/llm_core.py:7828', catch_base=False) as __m3_guard_7809_12:
                with torch.no_grad():
                    # Get phi predictions for chosen vs rejected
                    v_ch = self.model.v_phi(o_ch[:, -1, :]).squeeze(-1)
                    v_rj = self.model.v_phi(o_rj[:, -1, :]).squeeze(-1)
                    
                    # Dynamic phi_margin: core (NO MAGIC NUMBER 0.1)
                    if attr_has(self, 'core') and self.core is not None:
                        # phi/entropy/engagement  margin 
                        phi = self.core.phi_calculator.phi_history[-1] if self.core.phi_calculator.phi_history else 0.5
                        entropy = attr_get_optional(self.core.qualia, 'entropy', 0.5)
                        engagement = attr_get_optional(self.core.qualia, 'engagement', 0.5)
                        # margin_coef = phi * (1 - entropy) * engagement
                        # Compute margin coefficient
                        margin_coef = phi * (1.0 - entropy) * engagement
                        phi_margin = margin_coef * (v_ch - v_rj)
                    else:
                        # Fallback: phi difference (0.5)
                        phi_margin = (v_ch - v_rj) * 0.5

            if __m3_guard_7809_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")

            # DPO objective with phi-margin: -log(sigmoid(beta*((ch - rj) + phi_margin)))
            loss = -torch.log(torch.sigmoid(beta * ((ch_sum - rj_sum) + phi_margin)) + 1e-8)
            # Debugging: log intermediate scalars to help diagnose zero-loss/zero-grad issues
            with guard_context(ctx='llm_adapter/llm_core.py:7836', catch_base=False) as __m3_guard_7834_12:
                logger.debug(f'DPO step: ch_sum={float(ch_sum):.6f}, rj_sum={float(rj_sum):.6f}, phi_margin={float(phi_margin):.6f}, beta={beta}')

            if __m3_guard_7834_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            # compute grad norm for diagnostics
            try:
                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        with guard_context(ctx='llm_adapter/llm_core.py:7847', catch_base=False) as __m3_guard_7845_24:
                            total_grad_norm += float(p.grad.data.norm().item() or 0.0)

                        if __m3_guard_7845_24.error is not None:
                            continue
                logger.debug(f'DPO grads: total_grad_norm={total_grad_norm:.6f}, loss={float(loss):.6f}')
                if total_grad_norm == 0.0:
                    logger.warning('DPO step produced zero gradient norm check that chosen/rejected differ, model params require_grad, and optimizer has parameters')
            except Exception:
                logging.getLogger(__name__).exception("Swallowed exception")
            with guard_context(ctx='llm_adapter/llm_core.py:7856', catch_base=False) as __m3_guard_7854_12:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if __m3_guard_7854_12.error is not None:
                logging.getLogger(__name__).exception("Swallowed exception")
            self._guarded_optimizer_step(
                optimizer=self.opt,
                params=list(self.model.parameters()),
                tag="dpo_step",
            )
    
    def dpo_batch_step(
        self, 
        samples: List[Dict[str, Any]], 
        beta: float = 0.1, 
        max_len: int = 256,
        batch_size: int = 4
    ) -> None:
        """
        === F. DPO Batch Learning ===
         DPO with gradient clipping, sample mixing, hard negative filtering.
        
        Args:
            samples: [{'prompt': str, 'chosen': str, 'rejected': str, 'tool_failure': bool}, ...]
            beta: DPO temperature
            max_len: max token length for chosen/rejected
            batch_size: number of samples per batch
        """
        torch = self.torch
        t = self.tok
        
        if len(samples) == 0:
            return

        # === F1. Sampling Strategy: Recent 50% + Past 30% + Random 20% ===
        n = len(samples)
        n_recent = max(1, int(n * 0.5))
        n_past = max(1, int(n * 0.3))
        n_random = max(1, int(n * 0.2))

        recent = samples[-n_recent:]  # Recent samples
        past = samples[:n_past] if n_past < n else []  # Past samples
        random_pool = samples[n_past:-n_recent] if (n - n_recent - n_past) > 0 else []
        random_samples = []
        if len(random_pool) > 0:
            import random
            random_samples = random.sample(random_pool, min(n_random, len(random_pool)))
        
        mixed_samples = recent + past + random_samples
        
        # === F2. Hard Negative Filtering: tool_failure 2x rejection weight ===
        filtered_samples = []
        for s in mixed_samples:
            prompt = str(s.get('prompt', ''))
            chosen = str(s.get('chosen', ''))
            rejected = str(s.get('rejected', ''))
            tool_failure = bool(s.get('tool_failure', False))
            
            if self._is_empty_prompt_or_response(prompt, chosen) or not rejected:
                continue
            
            # Hard negative: tool_failure rejected2x
            if not tool_failure:
               filtered_samples.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
            if tool_failure:
               filtered_samples.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
        
        if len(filtered_samples) == 0:
            return
        
        # === F3.  DataLoader  ===
        self.model.train()
        beta = float(os.getenv("DPO_BETA", str(beta)))
        
        for i in range(0, len(filtered_samples), batch_size):
            batch = filtered_samples[i:i+batch_size]
            batch_loss = 0.0
            valid_count = 0
            
            with self._learn_lock:
                for sample in batch:
                    try:
                        prompt = sample['prompt']
                        chosen = sample['chosen']
                        rejected = sample['rejected']
                        
                        # Encode
                        src = t.encode(prompt, add_special=False)
                        ch = t.encode(chosen, add_special=True)
                        rj = t.encode(rejected, add_special=True)
                        if len(src) == 0 or len(ch) < 2 or len(rj) < 2:
                            continue
                        
                        ch_in = ch[:-1][:max_len]; ch_out = ch[1:][:max_len]
                        rj_in = rj[:-1][:max_len]; rj_out = rj[1:][:max_len]
                        src_ids = torch.tensor([src], dtype=torch.long, device=self.device)
                        ch_in_ids = torch.tensor([ch_in], dtype=torch.long, device=self.device)
                        ch_out_ids = torch.tensor([ch_out], dtype=torch.long, device=self.device)
                        rj_in_ids = torch.tensor([rj_in], dtype=torch.long, device=self.device)
                        rj_out_ids = torch.tensor([rj_out], dtype=torch.long, device=self.device)
                        
                        # Forward
                        e_src = self.model.emb(src_ids)
                        _, h = self.model.encoder(e_src)
                        
                        # chosen
                        e_ch = self.model.emb(ch_in_ids)
                        o_ch, _ = self.model.decoder(e_ch, h)
                        logits_ch = self.model.head(o_ch).contiguous().clone()
                        logp_ch = torch.log_softmax(logits_ch, dim=-1)
                        ch_lp = torch.gather(logp_ch, dim=-1, index=ch_out_ids.unsqueeze(-1)).squeeze(-1)
                        ch_mask = (ch_out_ids != t.PAD)
                        ch_sum = (ch_lp * ch_mask).sum()
                        
                        # rejected
                        e_rj = self.model.emb(rj_in_ids)
                        o_rj, _ = self.model.decoder(e_rj, h)
                        logits_rj = self.model.head(o_rj).contiguous().clone()
                        logp_rj = torch.log_softmax(logits_rj, dim=-1)
                        rj_lp = torch.gather(logp_rj, dim=-1, index=rj_out_ids.unsqueeze(-1)).squeeze(-1)
                        rj_mask = (rj_out_ids != t.PAD)
                        rj_sum = (rj_lp * rj_mask).sum()
                        
                        # Phase 5: M3-aware margin (NO MAGIC NUMBERS)
                        beta = float(os.getenv("DPO_BETA", str(beta)))
                        phi_margin = torch.tensor(0.0, device=self.device)
                        try:
                            with torch.no_grad():
                                v_ch = self.model.v_phi(o_ch[:, -1, :]).squeeze(-1)
                                v_rj = self.model.v_phi(o_rj[:, -1, :]).squeeze(-1)

                                # NO MAGIC NUMBERS: M3 margin coefficient
                                if self.core is not None:
                                    with guard_context(ctx='llm_adapter/llm_core.py:7994', catch_base=False) as __m3_guard_7986_36:
                                        phi = self.core.phi_calculator.calculate_phi()
                                        qualia = self.core.qualia_analyzer.analyze()
                                        entropy = qualia.get('entropy', 0.5)
                                        engagement = self.core.meta_cognitive_monitor.get_state().get('engagement', 0.5)
                                        # margin_coef = phi * (1 - entropy) * engagement
                                        # Compute margin coefficient
                                        margin_coef = phi * (1.0 - entropy) * engagement

                                    if __m3_guard_7986_36.error is not None:
                                        margin_coef = 0.1  # fallback
                                else:
                                    margin_coef = 0.1  # standalone mode
                                
                                phi_margin = margin_coef * (v_ch - v_rj)

                        except Exception:
                            logging.getLogger(__name__).exception("Swallowed exception")
                        
                        # DPO loss
                        loss = -torch.log(torch.sigmoid(beta * ((ch_sum - rj_sum) + phi_margin)) + 1e-8)
                        batch_loss += loss
                        # Debugging per-sample (aggregate will be logged per-batch)
                        with guard_context(ctx='llm_adapter/llm_core.py:8010', catch_base=False) as __m3_guard_8008_24:
                            logger.debug(f'DPO batch sample: ch_sum={float(ch_sum):.6f}, rj_sum={float(rj_sum):.6f}, phi_margin={float(phi_margin):.6f}')

                        if __m3_guard_8008_24.error is not None:
                            logging.getLogger(__name__).exception("Swallowed exception")
                        valid_count += 1
                    except Exception:
                        continue

                # === F4. Gradient Clipping (max_norm=1.0) ===
                if valid_count > 0:
                    avg_loss = batch_loss / valid_count
                    self.opt.zero_grad(set_to_none=True)
                    avg_loss.backward()
                    # diagnostics: compute total grad norm
                    try:
                        total_grad_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                with guard_context(ctx='llm_adapter/llm_core.py:8028', catch_base=False) as __m3_guard_8026_32:
                                    total_grad_norm += float(p.grad.data.norm().item() or 0.0)

                                if __m3_guard_8026_32.error is not None:
                                    continue
                        logger.debug(f'DPO batch: avg_loss={float(avg_loss):.6f}, total_grad_norm={total_grad_norm:.6f}, batch_size={valid_count}')
                        if total_grad_norm == 0.0:
                            logger.warning('DPO batch produced zero gradient norm check data and model configuration')
                    except Exception:
                        logging.getLogger(__name__).exception("Swallowed exception")
                    with guard_context(ctx='llm_adapter/llm_core.py:8037', catch_base=False) as __m3_guard_8035_20:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    if __m3_guard_8035_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                    self._guarded_optimizer_step(
                        optimizer=self.opt,
                        params=list(self.model.parameters()),
                        tag="dpo_batch_step",
                    )
    
    def _is_empty_prompt_or_response(self, prompt: str, response: str) -> bool:
        """Utility function to check if prompt or response is empty."""
        return not prompt or not response

    def train_value_head(self, records: List[Dict[str, Any]], epochs: int = 1, mem: Optional[np.ndarray] = None) -> None:
        """Train multi-head value predictors: phi_delta, stability_delta, tool_success."""
        torch = self.torch
        nn = torch.nn
        # Ensure we have parameters to train for the value heads
        params = []
        for name, p in self.model.named_parameters():
            if name.startswith(('value', 'v_phi', 'v_stab', 'v_tool')) or any(x in name for x in ['.value.', '.v_phi.', '.v_stab.', '.v_tool.']):
                params.append(p)
        if not params:
            return

        if self.value_opt is None:
            self.value_opt = self.torch.optim.AdamW(
                params,
                lr=1e-3,
                weight_decay=float(os.getenv("M3_STABILITY_WEIGHT_DECAY", str(self.stability_cfg.weight_decay))),
            )
        
        self.model.train()
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()
        t = self.tok
        batch_size = 32  # You can adjust the batch size as needed
        num_records = len(records)

        # Initialize accumulators if not already present
        if not attr_has(self, 'phi_targets'):
            self.phi_targets: List[float] = []
        if not attr_has(self, 'phi_preds'):
            self.phi_preds: List[float] = []

        import random
        for _ in range(max(1, int(epochs))):
            total_loss = 0.0
            n = 0
            random.shuffle(records)
            for batch_start in range(0, num_records, batch_size):
                batch = records[batch_start:batch_start + batch_size]
                batch_loss = 0.0
                valid_count = 0
                for rec in batch:
                    with guard_context(ctx='llm_adapter/llm_core.py:8138', catch_base=False) as __m3_guard_8091_20:
                        prompt = str(rec.get('prompt', ''))
                        response = str(rec.get('response', ''))
                        phi_d = float(rec.get('phi_delta', 0.0))
                        stab_d = float(rec.get('stability_delta', 0.0))
                        tool_s = float(1.0 if rec.get('tool_success', False) else 0.0)
                        
                        if not prompt or not response:
                            continue
                        
                        # Encode
                        src = t.encode(prompt, add_special=False)
                        tgt = t.encode(response, add_special=True)
                        if len(src) == 0 or len(tgt) < 2:
                            continue
                        
                        src_ids = torch.tensor([src], dtype=torch.long, device=self.device)
                        tgt_in = tgt[:-1]
                        tgt_in_ids = torch.tensor([tgt_in], dtype=torch.long, device=self.device)
                        
                        # Forward to last hidden
                        with torch.no_grad():
                            e_src = self.model.emb(src_ids)
                            _, h = self.model.encoder(e_src)
                            o, _ = self.model.decoder(self.model.emb(tgt_in_ids), h)
                            h_f = o[:, -1, :].contiguous()  # (1, H)
                            
                            # Optional memory context
                            if mem is not None:
                                m = np.array(mem)
                                if m.ndim == 1:
                                    m = m[None, :]
                                m_t = torch.tensor(m, dtype=torch.float32, device=self.device)
                                if m_t.ndim == 2:
                                    m_t = m_t.unsqueeze(0)
                                M, D = m_t.shape[1], m_t.shape[2]
                                self.model._ensure_mem_layers(D)
                                mem_h = self.model.mem_proj(m_t)
                                mem_k = self.model.Wk(mem_h)
                                mem_v = self.model.Wv(mem_h)
                                # Apply attention to each position
                                for t_pos in range(o.size(1)):
                                    dec_t = o[:, t_pos:t_pos+1, :]  # (1, 1, H)
                                    q = self.model.Wq(dec_t)
                                    att = torch.softmax((q @ mem_k.transpose(1, 2)) / np.sqrt(self.model.hidden + 1e-8), dim=-1)
                                    ctx = att @ mem_v
                                    o[:, t_pos:t_pos+1, :] = (dec_t + ctx) * 0.5

                    if __m3_guard_8091_20.error is not None:
                        continue

        # === G. Compute and log phi-delta correlation ===
        try:
            if len(self.phi_targets) > 1 and len(self.phi_preds) > 1:
                phi_t_arr = np.array(self.phi_targets)
                phi_p_arr = np.array(self.phi_preds)
                if phi_t_arr.size == phi_p_arr.size and phi_t_arr.size > 1:
                    corr_matrix = np.corrcoef(phi_t_arr, phi_p_arr)
                    phi_correlation = float(corr_matrix[0, 1])
                else:
                    phi_correlation = 0.0

                logger.info(f'Value head training complete: n={n}, avg_loss={(total_loss / max(1, n)):.4f}, phi_correlation={phi_correlation:.4f}')
                
                # Optional: log to file
                with guard_context(ctx='llm_adapter/llm_core.py:8166', catch_base=False) as __m3_guard_8156_16:
                    rec = {
                        "t": int(time.time() * 1000),
                        "kind": "value_train",
                        "n_samples": n,
                        "avg_loss": float(total_loss / max(1, n)),
                        "phi_correlation": phi_correlation,
                        "task_weights": dict(self._task_weights)
                    }
                    self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), rec)

                if __m3_guard_8156_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
        except Exception:
            logger.exception('Failed to compute phi correlation')
        
        return

    def train_token_value_head(self, records: List[Dict[str, Any]], epochs: int = 1, mem: Optional[np.ndarray] = None) -> None:
        """Train the token-value head to regress per-token credit from (prompt, response, token_credits).
        
        Args:
            records: List of dicts with 'prompt', 'response', and 'token_credits' (list of floats, one per token)
            epochs: Number of training epochs
            mem: Optional memory context
        """
        torch = self.torch
        nn = torch.nn
        if self.token_value_opt is None:
            # token_value head only
            params = [p for n, p in self.model.named_parameters() if 'token_value' in n]
            if not params:
                return
            self.token_value_opt = self.torch.optim.AdamW(
                params,
                lr=1e-3,
                weight_decay=float(os.getenv("M3_STABILITY_WEIGHT_DECAY", str(self.stability_cfg.weight_decay))),
            )
        self.model.train()
        mse = nn.MSELoss()
        for _ in range(max(1, epochs)):
            total = 0.0
            n = 0
            for rec in records:
                with guard_context(ctx='llm_adapter/llm_core.py:8267', catch_base=False) as __m3_guard_8199_16:
                    prompt = str(rec.get('prompt', ''))
                    response = str(rec.get('response', ''))
                    token_credits = rec.get('token_credits', [])  # list of per-token credits
                    if self._is_empty_prompt_or_response(prompt, response) or not token_credits:
                        continue
                    # Encode
                    src = self.tok.encode(prompt, add_special=False)
                    tgt = self.tok.encode(response, add_special=True)
                    if len(src) == 0 or len(tgt) < 2:
                        continue
                    src_ids = torch.tensor([src], dtype=torch.long, device=self.device)
                    tgt_in = tgt[:-1]
                    tgt_in_ids = torch.tensor([tgt_in], dtype=torch.long, device=self.device)
                    # Forward to get hidden states for each token position
                    e_src = self.model.emb(src_ids)
                    _, h = self.model.encoder(e_src)
                    e = self.model.emb(tgt_in_ids)
                    o, _ = self.model.decoder(e, h)  # (1, T, H)
                    # Optional memory context
                    if mem is not None:
                        m = np.array(mem)
                        if m.ndim == 1:
                            m = m[None, :]
                        m_t = torch.tensor(m, dtype=torch.float32, device=self.device)
                        if m_t.ndim == 2:
                            m_t = m_t.unsqueeze(0)
                        M, D = m_t.shape[1], m_t.shape[2]
                        self.model._ensure_mem_layers(D)
                        mem_h = self.model.mem_proj(m_t)
                        mem_k = self.model.Wk(mem_h)
                        mem_v = self.model.Wv(mem_h)
                        # Apply attention to each position
                        for t_pos in range(o.size(1)):
                            dec_t = o[:, t_pos:t_pos+1, :]  # (1, 1, H)
                            q = self.model.Wq(dec_t)
                            att = torch.softmax((q @ mem_k.transpose(1, 2)) / np.sqrt(self.model.hidden + 1e-8), dim=-1)
                            ctx = att @ mem_v
                            o[:, t_pos:t_pos+1, :] = (dec_t + ctx) * 0.5
                    # Train token_value head with per-token targets
                    with self._learn_lock:
                        # o: (1, T, H), predict token_value for each position
                        token_q_pred = self.model.token_value(o)  # (1, T, V)
                        # Build targets: for each position, we want to predict the Q-value of the *next* token
                        # But we have token_credits which are per-token in response
                        # Align: tgt_in corresponds to positions, tgt[1:] are the actual tokens
                        tgt_out = tgt[1:][:len(tgt_in)]
                        # Truncate credits to match
                        credits = token_credits[:len(tgt_out)]
                        if len(credits) < len(tgt_out):
                            credits = credits + [0.0] * (len(tgt_out) - len(credits))
                        tgt_out_ids = torch.tensor([tgt_out], dtype=torch.long, device=self.device)  # (1, T)
                        # Gather predicted Q for the actual next tokens
                        pred_q = torch.gather(token_q_pred, dim=-1, index=tgt_out_ids.unsqueeze(-1)).squeeze(-1)  # (1, T)
                        # Target Q values from credits
                        target_q = torch.tensor([credits], dtype=torch.float32, device=self.device)  # (1, T)
                        # MSE loss
                        loss = mse(pred_q, target_q)
                        self.token_value_opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.token_value.parameters(), max_norm=1.0)
                        self._guarded_optimizer_step(
                            optimizer=self.token_value_opt,
                            params=list(self.model.token_value.parameters()),
                            tag="token_value_head",
                        )
                    total += float(loss.item())
                    n += 1

                if __m3_guard_8199_16.error is not None:
                    continue
        return


from .m3_plastic_policy import M3PlasticPolicy

class UnifiedM3Policy(TorchConversationalPolicy):
    """
    Unified policy that manages both conversational (LLM) and motor (RL) policies.
    This allows a single point of management for the agent's behavior.
    """
    def __init__(self, config: Optional[TorchPolicyConfig] = None, device: Optional[str] = None, motor_policy=None):
        super().__init__(config, device)
        self.motor_policy = motor_policy
        
    def set_motor_policy(self, policy):
        """Attach an external motor policy (e.g. TorchPolicy from m3/torch_policy.py)."""

        self.motor_policy = policy
        
    def sample_action(self, obs, affect=None):
        """Sample motor action from the internal motor policy."""
        if self.motor_policy:
            return self.motor_policy.sample(obs, affect)
        return None
    
    def update_motor(self, batch):
        """Update the motor policy."""
        if self.motor_policy:
            return self.motor_policy.update(batch)
        return {}


class PlasticBrainPolicy(UnifiedM3Policy):
    """
    M3-Binary Brain Adapter.
    Replaces standard GRU with 1-bit Plasticity Engine.
    """
    def __init__(self, config: Optional[TorchPolicyConfig] = None, device: Optional[str] = None, motor_policy=None):
        super().__init__(config, device, motor_policy)
        # Replace model with Plastic Brain
        logger.info("Initializing PlasticBrainPolicy: Rewiring synapses to 1-bit PlasticBitLinear...")
        self.model = M3PlasticPolicy(device=str(self.device)).to(self.device)
        # CPU?먯꽌??bfloat16 鍮꾩?????float32濡?媛뺤젣
        if self.device.type == 'cpu':
            self.model = self.model.float()
        self.model.train() # Default to plastic mode (Online Learning)

    def sample(self, obs, affect_state=None, **kwargs):
        """
        Sample from Plastic Brain.
        Supports direct affect-gating.
        """
        # M3PlasticPolicy.sample signature: (obs, affect_state=None, temperature=1.0, top_k=50, ...)
        return self.model.sample(obs, affect_state=affect_state, **kwargs)

    def _state_vector(self, core=None):
        """Expose plastic hidden state."""
        h = self.model._hidden_state
        # CPU?먯꽌??float32濡?媛뺤젣 蹂??
        if h is not None and self.device.type == 'cpu' and h.dtype == self.torch.bfloat16:
            h = h.float()
        return h
        
    def learn(self, text: str, arousal: float = 1.0, sleep_after: bool = False):
        """Unified interface for offline learning."""
        return self.model.learn_from_text(text, arousal=arousal, sleep_after=sleep_after)

    def generate(
        self,
        prompt: str,
        mem: Optional[np.ndarray] = None,
        affect_state: Optional[np.ndarray] = None,
        max_len: int = 50,
        **kwargs,
    ) -> str:
        """Use shared generation pipeline from the base policy."""
        return super().generate(
            prompt=prompt,
            mem=mem,
            affect_state=affect_state,
            max_len=max_len,
            **kwargs,
        )


# ============================================================================
# Public API: attach_llm_to_core
# ============================================================================

def _attach_control_compat(adapter):
    """Attach minimal control hooks to legacy adapters that predate this API."""

    def _selection_mode(self) -> str:
        raw = str(os.getenv("M3_CONTROL_SELECTION_MODE", "state") or "state").strip().lower()
        if raw in {"0", "off", "none", "disable", "disabled", "no", "false"}:
            return "off"
        if raw in {"1", "state", "state_only", "context", "context_only", "low"}:
            return "state"
        if raw in {"2", "memory", "mid", "mixed", "medium"}:
            return "memory"
        if raw in {"3", "full", "high", "all", "strict"}:
            return "full"
        if raw in {"auto", "adaptive", "self", "self_adjust"}:
            return "state"
        if raw in {"on", "true", "yes"}:
            return "full"
        return "state"

    def _allows(self, feature: str) -> bool:
        mode = _selection_mode(self)
        allowed = {
            "off": set(),
            "state": {"state_context"},
            "memory": {"state_context", "memory_retrieval"},
            "full": {"state_context", "memory_retrieval", "bridge", "decode_control", "adaptive_sampler", "token_value_bias", "quality_gate"},
        }.get(mode, {"state_context"})
        try:
            return feature in allowed
        except Exception:
            return False

    def _bridge(self) -> bool:
        try:
            allows = bool(_allows(self, "bridge"))
        except Exception:
            allows = False
        return allows and os.getenv("M3_ENABLE_CONTROL_BRIDGE", "0").lower() in ("1", "true", "yes", "on")

    def _bridge_safe(self) -> bool:
        with guard_context(ctx='llm_adapter/llm_core.py:8402', catch_base=False) as __m3_guard_8398_8:
            fn = attr_get_optional(self, "_bridge_enabled", None)
            if callable(fn):
                return bool(fn())

        if __m3_guard_8398_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        return False

    def _note(self, success: bool, reason: str = "") -> None:
        reason_code = _normalize_control_reason(reason)
        with guard_context(ctx='llm_adapter/llm_core.py:8412', catch_base=False) as __m3_guard_8407_8:
            if attr_get_optional(self, '_control_health_window', None) is None:
                window = int(os.getenv("M3_CONTROL_HEALTH_WINDOW", "24"))
                self._control_health_window = deque(maxlen=max(1, window))
            self._control_health_window.append((time.time(), bool(success), str(reason_code)))

        if __m3_guard_8407_8.error is not None:
            return
        if success:
            self._auto_mode_fail_streak = 0
        else:
            self._auto_mode_fail_streak = attr_get_optional(self, '_auto_mode_fail_streak', 0) + 1

    if not callable(attr_get_optional(adapter, '_control_selection_mode', None)):
        adapter._control_selection_mode = types.MethodType(_selection_mode, adapter)
    if not callable(attr_get_optional(adapter, '_control_allows', None)):
        adapter._control_allows = types.MethodType(_allows, adapter)
    if not callable(attr_get_optional(adapter, '_note_control_health', None)):
        adapter._note_control_health = types.MethodType(_note, adapter)
    if not callable(attr_get_optional(adapter, '_bridge_enabled', None)):
        adapter._bridge_enabled = types.MethodType(_bridge, adapter)
    if not callable(attr_get_optional(adapter, '_bridge_enabled_safe', None)):
        adapter._bridge_enabled_safe = types.MethodType(_bridge_safe, adapter)


def attach_llm_to_core(core, adapter=None, record: bool = True):
    """
    Attach LLM adapter to core for conversational learning.
    
    Args:
        core: M3Core instance
        adapter: Optional TorchConversationalPolicy instance (auto-created if None)
        record: Whether to record training data to JSONL
    """
    try:
        required_methods = (
            "_control_allows",
            "_note_control_health",
            "_control_selection_mode",
            "_bridge_enabled",
            "_bridge_enabled_safe",
        )
        if adapter is None:
            # Auto-detect device with env override
            import torch
            device = _resolve_torch_device(torch)
            
            # Use PlasticBrain based on env var or config
            use_plastic = os.getenv("M3_PLASTIC_BRAIN", "1") == "1"
            
            if use_plastic:
                adapter = PlasticBrainPolicy(device=device)
                logger.info(f'Created PlasticBrainPolicy (M3-BB) adapter with device={device}')
            else:
                adapter = UnifiedM3Policy(device=device)
                logger.info(f'Created UnifiedM3Policy adapter with device={device}')
            
            # If core has a policy, attach it as motor policy
            if attr_has(core, 'policy') and core.policy is not None:
                adapter.set_motor_policy(core.policy)
                logger.info('Attached existing core policy as motor policy')
        else:
            missing_methods = [m for m in required_methods if not callable(attr_get_optional(adapter, m, None))]
            if missing_methods:
                # First, try compatibility binding in-place for legacy adapters.
                _attach_control_compat(adapter)
                still_missing = [m for m in required_methods if not callable(attr_get_optional(adapter, m, None))]
                if still_missing:
                    logger.warning(
                        f'Attached adapter missing control hooks {still_missing}; '
                        f'attempting to rebuild with TorchConversationalPolicy'
                    )
                    try:
                        import torch
                        device = _resolve_torch_device(torch)
                        adapter = TorchConversationalPolicy(device=device)
                    except Exception as e:
                        logger.warning(
                            f'Failed to rebuild adapter for missing control hooks {still_missing}: {e}'
                        )
                        _attach_control_compat(adapter)
        # Ensure attached adapter always has control API, even if newly-created.
        missing_final = [m for m in required_methods if not callable(attr_get_optional(adapter, m, None))]
        if missing_final:
            _attach_control_compat(adapter)
        
        # Attach adapter to core
        core.llm_adapter = adapter
        adapter.core = core

        # Enable M3 integration by default unless disabled
        with guard_context(ctx='llm_adapter/llm_core.py:8503', catch_base=False) as __m3_guard_8497_8:
            m3_flag = os.getenv('M3_INTEGRATION', '1').lower()
            if m3_flag in ('1', 'true', 'yes', 'on'):
                if attr_has(adapter, 'enable_m3_integration'):
                    adapter.enable_m3_integration()
                    logger.info('M3 integration enabled for LLM adapter')

        if __m3_guard_8497_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")
        
        # Connect MessageBus inbox for credit assignment
        if attr_has(core, 'message_bus') and core.message_bus is not None:
            with guard_context(ctx='llm_adapter/llm_core.py:8513', catch_base=False) as __m3_guard_8508_12:
                if attr_has(core.message_bus, 'inboxes') and 'llm_adapter' in core.message_bus.inboxes:
                    adapter._bus_inbox = core.message_bus.inboxes['llm_adapter']
                    adapter.start_credit_consumer()
                    logger.info('LLM adapter connected to MessageBus for credit assignment')

            if __m3_guard_8508_12.error is not None:
                e = __m3_guard_8508_12.error
                logger.warning(f'Failed to connect LLM adapter to MessageBus: {e}')
        
        # Enable training data recording if requested
        if record:
            adapter._record_training = True
            logger.info('LLM adapter attached to core with training data recording enabled')
        else:
            adapter._record_training = False
            logger.info('LLM adapter attached to core (recording disabled)')

        # Optionally start autonomy loop if enabled via environment
        with guard_context(ctx='llm_adapter/llm_core.py:8530', catch_base=False) as __m3_guard_8525_8:
            auto_flag = os.getenv('LLM_AUTONOMY', '0').lower()
            if auto_flag in ('1', 'true', 'yes', 'on'):
                adapter.start_autonomy_loop()
                logger.info('Autonomy loop started (LLM_AUTONOMY enabled)')

        if __m3_guard_8525_8.error is not None:
            logging.getLogger(__name__).exception("Swallowed exception")

        return adapter
        
    except Exception as e:
        logger.exception('Failed to attach LLM adapter to core')
        raise
