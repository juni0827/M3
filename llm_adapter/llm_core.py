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
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import threading
import time
from collections import deque
import types
import torch
from m3.device import resolve_torch_device_string

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

try:
    from transformers.utils import logging as _tlog
    _tlog.set_verbosity_error()
    try:
        _tlog.disable_progress_bar()
    except Exception:
        pass
except Exception:
    pass

try:
    from huggingface_hub.utils import logging as _hlog
    try:
        _hlog.set_verbosity_error()
    except Exception:
        pass
    try:
        _hlog.disable_progress_bars()
    except Exception:
        pass
except Exception:
    pass

try:
    from .config import (
        KNNIndexConfig,
        M3AdaptiveSamplerConfig,
        M3AwareDecoderLayerConfig,
        M3EpisodicMemoryConfig,
        M3LLMConfig,
        M3StateCacheConfig,
        M3StateEncoderConfig,
        TokenizerConfig,
        TorchPolicyConfig,
        get_global_config,
    )
    from .memory import ConditionalKNNIndex, KNNItem, M3EpisodicMemoryRetriever
    from .tokenization import M3Tokenizer, AutoTokenizer
except Exception:
    # Support running file directly (script) where package-relative imports fail
    from llm_adapter.config import (
        KNNIndexConfig,
        M3AdaptiveSamplerConfig,
        M3AwareDecoderLayerConfig,
        M3EpisodicMemoryConfig,
        M3LLMConfig,
        M3StateCacheConfig,
        M3StateEncoderConfig,
        TokenizerConfig,
        TorchPolicyConfig,
        get_global_config,
    )
    from llm_adapter.memory import ConditionalKNNIndex, KNNItem, M3EpisodicMemoryRetriever
    from llm_adapter.tokenization import M3Tokenizer, AutoTokenizer

try:
    from .m3_control_bridge import (
        M3ControlBridge,
        LayerGateRuntime,
        GenerationQualityGate,
        find_decoder_layers,
    )
except Exception:
    try:
        from llm_adapter.m3_control_bridge import (  # type: ignore
            M3ControlBridge,
            LayerGateRuntime,
            GenerationQualityGate,
            find_decoder_layers,
        )
    except Exception:
        M3ControlBridge = None  # type: ignore
        LayerGateRuntime = None  # type: ignore
        GenerationQualityGate = None  # type: ignore
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
        try:
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
        except Exception as e:
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
        if hasattr(core, 'phi_calculator') and hasattr(core.phi_calculator, 'phi_history'):
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
        try:
            panels = core.feature_bank.panels(core)
            self._panels_cache.append(panels)
        except Exception as e:
            # Graceful degradation: core might not have feature_bank
            self._panels_cache.append(None)
        
        # Store qualia
        try:
            self._qualia_cache.append({
                'arousal': core.qualia.arousal,
                'valence': core.qualia.valence,
                'entropy': core.qualia.entropy,
                'engagement': core.qualia.engagement,
                'frustration': core.qualia.frustration
            })
        except Exception:
            self._qualia_cache.append(None)
        
        # Store phi
        try:
            phi = core.phi_calculator.phi_history[-1] if core.phi_calculator.phi_history else 0.0
            self._phi_cache.append(phi)
        except Exception:
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
        
        try:
            # 1. Affect Kernel adjustment
            if hasattr(core, 'affect_kernel'):
                affect_state = core.affect_kernel.get_state() # [valence, arousal, dominance, novelty, clarity]
                # Map to temp_predictor input (5D)
                qualia_vec = self.torch.tensor(affect_state, dtype=self.torch.float32).to(self.device)
            elif hasattr(core, 'qualia'):
                # Fallback to old qualia
                qualia_vec = self.torch.tensor([
                    getattr(core.qualia, 'arousal', 0.5),
                    getattr(core.qualia, 'valence', 0.5),
                    getattr(core.qualia, 'entropy', 0.5),
                    getattr(core.qualia, 'engagement', 0.5),
                    getattr(core.qualia, 'frustration', 0.0)
                ], dtype=self.torch.float32).to(self.device)
            else:
                return base_temp
            
            with self.torch.no_grad():
                temp_factor = self.temp_predictor(qualia_vec).item()
            
            temp = self.config.temp_min + temp_factor * (self.config.temp_max - self.config.temp_min)

            # 2. Phi adjustment (phi * temp)
            if hasattr(core, 'phi_calculator') and core.phi_calculator.phi_history:
                phi = core.phi_calculator.phi_history[-1]
                # phi [0, 1],  temp
                temp = temp * (1.0 - self.config.phi_influence * phi)

            # 3. Energy adjustment (energy * temp)
            if hasattr(core, 'energy_ctrl'):
                energy_ratio = core.energy_ctrl.cognitive_energy / max(core.energy_ctrl.energy_capacity, 1.0)
                temp = temp * (0.8 + 0.4 * (1.0 - energy_ratio) * self.config.energy_influence)

            # 4. Meta-awareness adjustment (meta * temp)
            if hasattr(core, 'self_model') and hasattr(core.self_model, 'meta_awareness'):
                meta = core.self_model.meta_awareness
                temp = temp * (1.0 - self.config.meta_influence * meta)
            
            # Clamp to bounds
            return max(self.config.temp_min, min(self.config.temp_max, temp))
            
        except Exception:
            # Graceful degradation
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
        
        try:
            # Exploration = entropy * engagement (qualia)
            if hasattr(core, 'affect_kernel'):
                # Use Novelty (index 3) and Arousal (index 1) as proxy for exploration
                affect = core.affect_kernel.get_state()
                novelty = affect[3]
                arousal = affect[1]
                exploration = novelty * arousal
            elif hasattr(core, 'qualia'):
                entropy = getattr(core.qualia, 'entropy', 0.5)
                engagement = getattr(core.qualia, 'engagement', 0.5)
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
                
        except Exception:
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
    - Raw logits  → token-critic Q-value injection
    - Hidden states → cross-attention / projection
    - Sampling params → M3AdaptiveSampler (phi/qualia/energy → temperature/top_k)

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

        # transformers 최신 버전에서는 `torch_dtype`가 deprecated이고 `dtype`를 권장함.
        # 다만 구버전 호환을 위해 from_pretrained 호출에서 필요 시 `torch_dtype`로 폴백한다.
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

        # CPU는 bfloat16 비지원 → float32로 강제
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
        try:
            fn = getattr(self, "_bridge_enabled", None)
            if callable(fn):
                return bool(fn())
        except Exception:
            pass
        return False

    def _note_control_health(self, success: bool, reason: str = "") -> None:
        try:
            self._control_health_window.append((time.time(), bool(success), str(reason or "")))
        except Exception:
            return
        if success:
            self._auto_mode_fail_streak = 0
        else:
            self._auto_mode_fail_streak += 1

    def _compute_recent_control_stats(self) -> Dict[str, float]:
        now = time.time()
        window_sec = float(os.getenv("M3_CONTROL_HEALTH_WINDOW_SEC", "180"))
        events = [
            (ts, ok, reason)
            for ts, ok, reason in list(self._control_health_window)
            if now - float(ts) <= window_sec
        ]
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
        if getattr(self, "_hf_circuit_open", False):
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

        core = getattr(self, 'core', None)
        if core is not None:
            em = getattr(core, 'episodic_memory', None)
            try:
                retrieved = int(getattr(em, 'total_retrieved', 0))
            except Exception:
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

    def _prepare_bridge_state(self, z_m3, state_dim: int, device):
        if z_m3 is None:
            return None
        try:
            z = np.asarray(z_m3, dtype=np.float32).ravel()
        except Exception:
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
        try:
            if hasattr(core, 'energy_ctrl'):
                ec = core.energy_ctrl
                before = float(getattr(ec, 'cognitive_energy', 0.0))
                ec.cognitive_energy = max(0.0, before - 0.05 * interval)
                if ec.cognitive_energy != before:
                    updated = True
            if hasattr(core, 'qualia'):
                q = core.qualia
                before = float(getattr(q, 'entropy', 0.0))
                window = generated_ids[-interval:] if generated_ids else []
                n_unique = len(set(window)) if window else 0
                token_diversity = n_unique / max(interval, 1)
                q.entropy = 0.8 * before + 0.2 * token_diversity
                if q.entropy != before:
                    updated = True
        except Exception:
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
            try:
                temperature = float(m3_sampler._compute_temperature(core, temperature))
                top_k = int(m3_sampler._compute_top_k(core, top_k))
            except Exception:
                pass
        return temperature, top_k, top_p

    @staticmethod
    def _apply_bridge_logit_bias(logits, bridge_controls):
        if bridge_controls is None:
            return logits
        try:
            lb = bridge_controls.logit_bias
            if lb is not None and lb.shape[-1] == logits.shape[-1]:
                return logits + lb.to(device=logits.device, dtype=logits.dtype)
        except Exception:
            pass
        return logits

    def _apply_token_value_injection(self, logits, hidden, token_value_head, beta, internal_hidden_dim):
        if not self._token_critic_enabled(token_value_head, internal_hidden_dim):
            return logits
        try:
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
        except Exception:
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
                try:
                    token_ids = self._tokenizer.encode(term, add_special_tokens=False)
                except Exception:
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
        try:
            temperature = min(float(temperature), float(decode_control.get("max_temperature", temperature)))
        except Exception:
            pass
        try:
            ctrl_k = int(decode_control.get("max_top_k", top_k))
            if ctrl_k > 0 and int(top_k) > 0:
                top_k = min(int(top_k), ctrl_k)
        except Exception:
            pass
        try:
            top_p = min(float(top_p), float(decode_control.get("max_top_p", top_p)))
        except Exception:
            pass
        return temperature, top_k, top_p

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
                try:
                    system_max_tokens = int(os.getenv('M3_SYSTEM_MAX_TOKENS', '320'))
                except Exception:
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
                            try:
                                ids = self._tokenizer(system_content, add_special_tokens=False).get('input_ids', [])
                                if len(ids) > system_max_tokens:
                                    system_content = self._tokenizer.decode(
                                        ids[:system_max_tokens],
                                        skip_special_tokens=True,
                                    ).strip()
                            except Exception:
                                pass
                        safe_messages[mi]['content'] = system_content
                        break
                if hasattr(self._tokenizer, 'apply_chat_template'):
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
        try:
            if max_input > 0:
                cur_len = int(inputs['input_ids'].shape[1])
                if cur_len > max_input:
                    start = cur_len - max_input
                    for k, v in list(inputs.items()):
                        if hasattr(v, 'shape') and len(v.shape) == 2 and int(v.shape[1]) == cur_len:
                            inputs[k] = v[:, start:]
        except Exception:
            pass
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        inputs_embeds = None
        bridge_controls = None
        bridge_runtime = None

        # Lazy-init projection: HF hidden → internal model hidden
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
            try:
                state_dim = int(os.getenv('M3_BRIDGE_STATE_DIM', '256'))
            except Exception:
                state_dim = 256
            try:
                bridge = self._ensure_control_bridge(state_dim=state_dim)
                z_t = self._prepare_bridge_state(z_m3, state_dim=state_dim, device=self.device)
                if bridge is not None and z_t is not None:
                    strength = float(os.getenv('M3_BRIDGE_STRENGTH', '1.0'))
                    with _torch.no_grad():
                        bridge_controls = bridge(z_t, strength=strength)
                    if bridge_controls.prefix_embeddings is not None:
                        try:
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
                        except Exception:
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

        generated_ids: list = []
        past_key_values = None
        forbidden_token_ids = self._resolve_forbidden_token_ids(decode_control)
        try:
            forbidden_penalty = float(decode_control.get("forbidden_penalty", 0.0)) if decode_control else 0.0
        except Exception:
            forbidden_penalty = 0.0
        # How often to micro-update core state during decoding
        try:
            _core_update_interval = int(os.getenv('M3_HF_CORE_UPDATE_INTERVAL', '8'))
        except Exception:
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

            logits = out.logits[:, -1, :].float()          # (1, V)
            hidden = out.hidden_states[-1][:, -1, :]       # (1, H)
            past_key_values = out.past_key_values
            inputs_embeds = None

            # M3ControlBridge logit bias
            logits = self._apply_bridge_logit_bias(logits, bridge_controls)

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
                    pass

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

            # KV-cache: feed only the new token
            input_ids = next_token
            attention_mask = _torch.cat(
                [attention_mask,
                 _torch.ones(1, 1, device=self.device, dtype=attention_mask.dtype)],
                dim=1,
            )

        if bridge_runtime is not None:
            try:
                bridge_runtime.close()
            except Exception:
                pass

        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# Redirecting all outputs to the unified logs folder (with safe fallback)
DEFAULT_OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docs&tests&data_sets/tests/logs'))
OUT_DIR = os.getenv('LLM_ADAPTER_LOG_DIR', DEFAULT_OUT_DIR)
try:
    os.makedirs(OUT_DIR, exist_ok=True)
except Exception:
    try:
        import tempfile
        OUT_DIR = tempfile.gettempdir()
        os.makedirs(OUT_DIR, exist_ok=True)
    except Exception:
        OUT_DIR = os.path.abspath(os.path.dirname(__file__))
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
        except Exception:
            pass
TRAINING_PATH = os.path.join(OUT_DIR, 'llm_training_data.jsonl')

LOG_PATH = os.getenv('LLM_ADAPTER_LOG_PATH', os.path.join(OUT_DIR, 'llm_adapter.log'))

logger = logging.getLogger('llm_adapter')

DEFAULT_SYSTEM_PROMPT = (
    "You are M3. Respond as M3. "
    "Use internal control state for response policy, not as quoted prompt context. "
    "Do not claim to be an AI assistant or language model. "
    "Do not say you cannot feel. "
    "Do not report phi/qualia/state values unless the user explicitly asks for them. "
    "Be concise and factual. Reply in the user's language."
)
if not logger.handlers:
    level = logging.DEBUG if os.environ.get('LLM_ADAPTER_DEBUG', '0') in ('1', 'true', 'TRUE') else logging.WARNING
    handler = None
    log_paths = [LOG_PATH]
    try:
        alt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
        os.makedirs(alt_dir, exist_ok=True)
        log_paths.append(os.path.join(alt_dir, 'llm_adapter.log'))
    except Exception:
        pass
    try:
        import tempfile
        log_paths.append(os.path.join(tempfile.gettempdir(), f'llm_adapter_{os.getpid()}.log'))
    except Exception:
        pass
    for path in log_paths:
        try:
            handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
            LOG_PATH = path
            break
        except Exception:
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
            try:
                logger.warning(f"Failed to open log file at {LOG_PATH}; falling back to stderr logging")
            except Exception:
                pass

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
        embed_dim = self.config.embed_dim
        hidden = self.config.hidden_dim
        lr = self.config.learning_rate
        
        # Initialize tokenizer
        try:
            self.tok = AutoTokenizer.from_config(get_global_config().tokenizer)
            logger.info(f"Using {self.tok.__class__.__name__}")
        except Exception as e:
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
                try:
                    if hasattr(self.q_head, 'bias') and self.q_head.bias is not None:
                        # bias[0]=Q_wait, bias[1]=Q_speak
                        with torch_module.no_grad():
                            self.q_head.bias[0].fill_(0.0)
                            self.q_head.bias[1].fill_(0.5)
                except Exception:
                    pass
                
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
                    try:
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
                    except Exception:
                        pass
                
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
                        try:
                            stability = float(core_state.get('stability', 0.5))
                            drift = float(core_state.get('drift', 0.0))
                            phi_delta = float(core_state.get('phi_delta', 0.0))
                            bias = (
                                self.config.stability_weight * stability
                                + self.config.drift_weight * abs(drift)
                                + self.config.phi_delta_weight * abs(phi_delta)
                            )
                            gate = self.torch.clamp(gate + bias, 0.0, 1.0)
                        except Exception:
                            pass
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

        num_layers = getattr(self.config, 'num_layers', 6)
        self.model = Model(self.vocab_size, embed_dim, hidden, num_layers, self.torch, self.device, self.config.init_gate_value, self.tok.PAD, self.config).to(self.device)

        # Load checkpoint if available (safe)
        checkpoint_path = os.getenv(
            "LLM_CHECKPOINT_PATH",
            os.path.join('docs&tests&data_sets', 'tests', 'logs', 'llm_checkpoint.pt')
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
                try:
                    bad_path = checkpoint_path + ".bad"
                    os.replace(checkpoint_path, bad_path)
                    logger.warning(f"Moved bad checkpoint to {bad_path}")
                except Exception:
                    pass

        # M3 State Cache (M3 integration)
        self.m3_cache = M3StateCache(get_global_config().state_cache)

        # M3 Adaptive Sampler (M3 integration)
        self.m3_sampler = M3AdaptiveSampler(self.torch, self.device, get_global_config().adaptive_sampler)

        # M3 Episodic Memory Retriever (M3 integration)
        self.m3_memory_retriever = M3EpisodicMemoryRetriever(get_global_config().episodic_memory)
        
        # Core reference (attach_llm_to_core)
        self.core = None
        
        self.criterion = self.torch.nn.CrossEntropyLoss(ignore_index=self.tok.PAD)
        self.opt = self.torch.optim.Adam(self.model.parameters(), lr=lr)
        self.value_opt = None  # lazy init for value-head training
        self.token_value_opt = None  # lazy init for token-value-head training
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
        try:
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
        except Exception:
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
        # CPU는 bfloat16을 지원 안 함 → float32로 강제
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
        self._record_scope = str(os.getenv("M3_TRAIN_RECORD_SCOPE", "user_only") or "user_only").strip().lower()
        self._last_record_reject_reason = ""
        self._phi_zero_streak = 0
        self._phi_zero_warn_every = max(10, int(os.getenv("M3_PHI_ZERO_WARN_EVERY", "50")))
        control_window = max(8, int(os.getenv("M3_CONTROL_HEALTH_WINDOW", "24")))
        self._control_health_window = deque(maxlen=control_window)
        self._auto_mode_fail_streak = 0

    def _is_numeric_dump_response(self, text: str) -> bool:
        """Detect low-quality CSV-like numeric dumps."""
        try:
            s = str(text).strip()
        except Exception:
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
                pass
        return numeric >= max(6, int(0.85 * len(parts)))

    def _is_backend_status_text(self, text: str) -> bool:
        try:
            s = str(text or "").strip()
        except Exception:
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
            "현재 생성 경로가 일시적으로 안전 모드입니다.",
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
        try:
            s = str(text or "").strip().lower()
        except Exception:
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
            "인공지능",
            "언어 모델",
            "감정을 느끼지 못",
            "개인적인 감정이나 경험이 없",
            "훈련 버튼",
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
        try:
            s = str(text or "").strip()
        except Exception:
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
            "인공지능",
            "언어 모델",
            "알리바바",
            "퀸웬",
        )
        return any(m in sl for m in drift_markers)

    def _is_disallowed_generation_output(self, text: str) -> bool:
        try:
            s = str(text or "").strip()
        except Exception:
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

    def _evaluate_generation_quality(self, prompt: str, response: str, source: str = "generate") -> Tuple[bool, Dict[str, float]]:
        info: Dict[str, float] = {
            "score": 0.0,
            "overlap": 0.0,
            "len_ratio": 0.0,
            "disallowed": 1.0,
        }
        if self._is_disallowed_generation_output(response):
            return False, info
        info["disallowed"] = 0.0

        response_text = str(response or "").strip()
        if not response_text:
            return False, info

        try:
            min_chars = max(2, int(os.getenv("M3_CONTROL_MIN_RESPONSE_CHARS", "8")))
        except Exception:
            min_chars = 24
        if len(response_text) < min_chars:
            return False, info

        prompt_text = self._extract_last_user_text(prompt)

        # Primary verifier: delegate to the core's existing accuracy metric when available.
        core = getattr(self, "core", None)
        if core is not None and hasattr(core, "_evaluate_dialog_accuracy"):
            try:
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
            except Exception:
                pass

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
        info["score"] = score

        min_score = float(os.getenv("M3_CONTROL_MIN_RESPONSE_SCORE", "0.16"))
        if source == "autonomy":
            min_score = float(os.getenv("M3_CONTROL_MIN_AUTONOMY_SCORE", "0.08"))
        return score >= min_score, info

    def _read_float(self, name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    def _read_int(self, name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return int(default)

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
            try:
                phi = 0.0
                if hasattr(core, 'phi_calculator') and getattr(core.phi_calculator, 'phi_history', None):
                    hist = core.phi_calculator.phi_history
                    if hist:
                        phi = float(hist[-1])
                q = getattr(core, 'qualia', None)
                arousal = float(getattr(q, 'arousal', 0.0)) if q is not None else 0.0
                valence = float(getattr(q, 'valence', 0.0)) if q is not None else 0.0
                world_state = getattr(core, 'world_state', None)
                stability = float(getattr(world_state, 'get', lambda k, d: d)('stability', 1.0)) if isinstance(world_state, dict) else float(getattr(world_state, 'stability', 1.0))
                if abs(phi) < 1e-8:
                    state_strict_bonus += 0.06
                if arousal < 0.2 or arousal > 0.8:
                    state_strict_bonus += 0.08
                if abs(valence) < 0.2:
                    state_strict_bonus += 0.04
                if stability < 0.4:
                    state_strict_bonus += 0.08
            except Exception:
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
        self._gpu_fault_count = int(getattr(self, "_gpu_fault_count", 0)) + 1
        failover = str(os.getenv("M3_HF_CUDA_FAILOVER", "1")).lower() in ("1", "true", "yes", "on")
        if not failover:
            return
        if not getattr(self, "_hf_circuit_open", False):
            self._hf_circuit_open = True
            self._hf_circuit_reason = str(err)
            self.use_hf = False
            try:
                os.environ["M3_USE_HF"] = "0"
            except Exception:
                pass
            try:
                if hasattr(self, "torch") and hasattr(self.torch, "cuda") and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
            except Exception:
                pass
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

        try:
            penalty = float(os.getenv("M3_CONTROL_TERM_PENALTY", "8.0"))
        except Exception:
            penalty = 8.0
        try:
            identity_penalty = float(os.getenv("M3_CONTROL_IDENTITY_PENALTY", "14.0"))
        except Exception:
            identity_penalty = 14.0
        try:
            max_temp = float(os.getenv("M3_CONTROL_MAX_TEMP", "0.65"))
        except Exception:
            max_temp = 0.65
        try:
            max_top_k = int(os.getenv("M3_CONTROL_MAX_TOP_K", "40"))
        except Exception:
            max_top_k = 40
        try:
            max_top_p = float(os.getenv("M3_CONTROL_MAX_TOP_P", "0.92"))
        except Exception:
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
            core = getattr(self, "core", None)
            if core is not None:
                ph = []
                try:
                    ph = list(core.phi_calculator.phi_history) if hasattr(core, "phi_calculator") else []
                except Exception:
                    ph = []
                phi = float(ph[-1]) if ph else 0.0
                phi_mean10 = float(np.mean(ph[-10:])) if len(ph) >= 1 else 0.0
                phi_delta = float(ph[-1] - ph[-2]) if len(ph) >= 2 else 0.0
                phi_nonzero_recent = int(any(abs(float(v)) > 1e-8 for v in ph[-10:])) if ph else 0

                if abs(phi) <= 1e-12:
                    self._phi_zero_streak = int(getattr(self, "_phi_zero_streak", 0)) + 1
                    if self._phi_zero_streak % int(getattr(self, "_phi_zero_warn_every", 50)) == 0:
                        logger.warning(f"[TrainingRecord] phi remains zero for {self._phi_zero_streak} samples")
                else:
                    self._phi_zero_streak = 0

                rec["phi"] = float(phi)
                rec["phi_mean10"] = float(phi_mean10)
                rec["phi_delta"] = float(phi_delta)
                rec["phi_nonzero_recent"] = int(phi_nonzero_recent)

                try:
                    q = core.qualia
                    rec["qualia"] = {
                        "arousal": float(getattr(q, "arousal", 0.0)),
                        "valence": float(getattr(q, "valence", 0.0)),
                        "entropy": float(getattr(q, "entropy", 0.0)),
                        "engagement": float(getattr(q, "engagement", 0.0)),
                        "frustration": float(getattr(q, "frustration", 0.0)),
                    }
                except Exception:
                    rec["qualia"] = {}

                try:
                    ec = getattr(core, "energy_ctrl", None)
                    if ec is not None:
                        energy = float(getattr(ec, "cognitive_energy", 0.0))
                        cap = float(max(getattr(ec, "energy_capacity", 1.0), 1e-6))
                        rec["energy"] = {
                            "value": energy,
                            "ratio": float(energy / cap),
                        }
                except Exception:
                    pass
        except Exception:
            pass
        return rec

    def _generate_safe_fallback(self, prompt: str, chat_messages: Optional[List[Dict[str, str]]] = None, max_len: int = 60) -> str:
        user_text = self._extract_last_user_text(prompt)
        if not user_text and chat_messages:
            try:
                user_msgs = [str(m.get("content", "")) for m in chat_messages if str(m.get("role", "")) == "user"]
                if user_msgs:
                    user_text = user_msgs[-1].strip()
            except Exception:
                pass
        if not user_text:
            user_text = str(prompt or "").strip()
        safe_temp = None
        safe_top_k = None
        safe_top_p = None
        try:
            safe_temp, safe_top_k, safe_top_p = self._resolve_generation_sampling(
                requested_temperature=None,
                requested_top_k=None,
                requested_top_p=None,
                source='fallback',
                core=getattr(self, 'core', None),
            )
        except Exception:
            safe_temp = None
            safe_top_k = None
            safe_top_p = None
        use_ollama_fallback = str(os.getenv("M3_DISABLE_OLLAMA", "1")).lower() not in ("1", "true", "yes", "on")
        if use_ollama_fallback:
            try:
                try:
                    fb_timeout = float(os.getenv("M3_SAFE_FALLBACK_TIMEOUT", "12"))
                except Exception:
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
                pass

        lang = self._detect_language(user_text)
        q = str(user_text or "").strip().lower()

        if lang == "ko":
            if any(k in q for k in ("의식", "자아", "감각", "기억", "정체성", "너", "m3")):
                return "내 관점에서 의식은 감각, 기억, 목표를 통합해 다음 행동을 선택하는 실시간 과정이다."
            return "출력 안정화 중입니다. 질문을 한 문장으로 다시 보내 주면 M3 관점으로 바로 답하겠습니다."
        if lang == "zh":
            return "当前在输出稳定模式。请用一句话重发问题，我会以 M3 视角直接回答。"
        if lang == "ru":
            return "Сейчас включен режим стабилизации вывода. Повторите вопрос одной фразой, и я отвечу от лица M3."
        return "Output is in stabilization mode. Send a single-sentence question and I will answer from the M3 perspective."


    def save_model(self, path: str):
        """Saves the model state dictionary."""
        if getattr(self, "_hf_circuit_open", False) and self.device.type == "cuda":
            logger.warning(f"Skipping model save while HF circuit is open: {path}")
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}")
            if self._is_cuda_fatal_error(e):
                self._trip_hf_circuit_breaker(e)

    def _record_example(self, prompt: str, response: str, source: str = "generate") -> None:
        if not getattr(self, "_record_training", False):
            return
        try:
            prompt_raw = str(prompt or "")
            response_text = str(response or "")
            rec = self._sanitize_training_record(prompt_raw, response_text, source=source)
            reject_reason = str(getattr(self, "_last_record_reject_reason", "") or "")
            try:
                os.makedirs(os.path.dirname(TRAINING_PATH), exist_ok=True)
            except Exception:
                pass
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
                    try:
                        os.makedirs(os.path.dirname(reject_path), exist_ok=True)
                    except Exception:
                        pass
                    with open(reject_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(reject_rec, ensure_ascii=False) + "\n")
        except Exception as e:
            try:
                logger.warning(f"Failed to record training example: {e}")
            except Exception:
                pass

    def enable_m3_integration(self) -> bool:
        if hasattr(self, 'model') and hasattr(self.model, 'enable_m3_integration'):
            try:
                self.model.enable_m3_integration()
                return True
            except Exception:
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
            try:
                return np.array([float(affect_state[k]) for k in sorted(affect_state.keys())], dtype=np.float32)
            except Exception:
                return None
        return None

    def _get_affect_state(self, core, affect_state=None):
        if affect_state is not None:
            return self._normalize_affect_state(affect_state)
        if core is None:
            return None
        try:
            if hasattr(core, 'rewards') and getattr(core.rewards, 'last_affect', None) is not None:
                return np.asarray(core.rewards.last_affect, dtype=np.float32)
        except Exception:
            pass
        if hasattr(core, 'affect_kernel'):
            for name in ('get_state', 'get_state_vector'):
                try:
                    fn = getattr(core.affect_kernel, name, None)
                    if callable(fn):
                        return self._normalize_affect_state(fn())
                except Exception:
                    pass
        if hasattr(core, 'qualia'):
            try:
                return np.array([
                    float(getattr(core.qualia, 'valence', 0.0)),
                    float(getattr(core.qualia, 'arousal', 0.0)),
                    float(getattr(core.qualia, 'entropy', 0.0)),
                    float(getattr(core.qualia, 'engagement', 0.0)),
                    float(getattr(core.qualia, 'frustration', 0.0)),
                ], dtype=np.float32)
            except Exception:
                pass
        return None

    def _update_m3_cache(self, core):
        panels = None
        if core is None:
            return panels
        try:
            self.m3_cache.update(core)
            panels = self.m3_cache.get_current_panels()
        except Exception:
            panels = None
        return panels

    def _build_m3_memory(self, core=None, panels=None, affect_state=None, drive_state=None):
        if panels is None and core is not None:
            try:
                if hasattr(core, 'feature_bank') and hasattr(core.feature_bank, 'panels'):
                    panels = core.feature_bank.panels(core)
            except Exception:
                panels = None
        if panels is None:
            return None
        if affect_state is not None:
            affect_state = self._normalize_affect_state(affect_state)
        if not hasattr(self, 'model'):
            return None
        if not hasattr(self.model, 'm3_encoder') or self.model.m3_encoder is None:
            try:
                self.model.enable_m3_integration()
            except Exception:
                return None
        try:
            return self.model.m3_encoder(panels, affect_state=affect_state, drive_state=drive_state)
        except Exception:
            return None

    def build_m3_memory(self, core=None, panels=None, affect_state=None, drive_state=None):
        return self._build_m3_memory(core=core, panels=panels, affect_state=affect_state, drive_state=drive_state)

    def _vectorize_panels(self, panels):
        if panels is None:
            return None
        try:
            arr = np.asarray(panels, dtype=np.float32)
        except Exception:
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
        try:
            max_len = int(os.getenv('M3_VEC_MAX_LEN', '256'))
        except Exception:
            max_len = 256
        if max_len > 0:
            vec = vec[:max_len]
        return vec

    def _snapshot_vector(self, core, affect_state=None):
        if core is None:
            return None
        vecs = []
        try:
            snap = core.snapshot() if hasattr(core, 'snapshot') and callable(core.snapshot) else None
        except Exception:
            snap = None
        if isinstance(snap, dict):
            for key in ('phi', 'energy', 'activation', 'unity', 'policy_lr', 'steps'):
                try:
                    val = snap.get(key, None)
                    if isinstance(val, (int, float, np.floating, np.integer)):
                        vecs.append(np.array([float(val)], dtype=np.float32))
                except Exception:
                    pass
            try:
                emb = snap.get('embeddings', None)
                if emb is None:
                    emb = snap.get('embedding', None)
                if emb is not None:
                    vecs.append(np.asarray(emb, dtype=np.float32).ravel())
            except Exception:
                pass
        try:
            q = getattr(core, 'qualia', None)
            if q is not None:
                qv = np.array(
                    [
                        float(getattr(q, 'arousal', 0.0)),
                        float(getattr(q, 'valence', 0.0)),
                        float(getattr(q, 'entropy', 0.0)),
                        float(getattr(q, 'engagement', 0.0)),
                        float(getattr(q, 'frustration', 0.0)),
                    ],
                    dtype=np.float32,
                )
                vecs.append(qv)
        except Exception:
            pass
        try:
            if hasattr(core, 'energy_ctrl'):
                energy_ratio = core.energy_ctrl.cognitive_energy / max(core.energy_ctrl.energy_capacity, 1.0)
                vecs.append(np.array([float(energy_ratio)], dtype=np.float32))
        except Exception:
            pass
        try:
            if hasattr(core, 'episodic_memory'):
                stats = core.episodic_memory.get_statistics()
                mem_vals = [
                    float(stats.get('total_memories', 0.0)),
                    float(stats.get('total_encoded', 0.0)),
                    float(stats.get('avg_consolidation', 0.0)),
                    float(stats.get('avg_retrieval_count', 0.0)),
                ]
                vecs.append(np.asarray(mem_vals, dtype=np.float32))
        except Exception:
            pass
        if affect_state is not None:
            try:
                vecs.append(np.asarray(affect_state, dtype=np.float32).ravel())
            except Exception:
                pass
        if not vecs:
            return None
        try:
            vec = np.concatenate(vecs).astype(np.float32)
        except Exception:
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
        try:
            vec = np.concatenate(vecs).astype(np.float32)
        except Exception:
            return None
        try:
            max_len = int(os.getenv('M3_VEC_MAX_LEN', '256'))
        except Exception:
            max_len = 256
        try:
            hard_max = int(os.getenv('M3_VEC_HARD_MAX', '2048'))
        except Exception:
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
        try:
            phi_hist = core.phi_calculator.phi_history if hasattr(core, 'phi_calculator') else []
            phi = phi_hist[-1] if phi_hist else 0.0
        except Exception:
            phi = 0.0
        lines.append(f"phi={float(phi):.4f}")
        if phi_trend:
            lines.append(f"phi_trend={phi_trend}")
        try:
            qualia = getattr(core, 'qualia', None)
            if qualia is not None:
                lines.append(
                    "qualia="
                    f"arousal:{float(getattr(qualia, 'arousal', 0.0)):.3f},"
                    f"valence:{float(getattr(qualia, 'valence', 0.0)):.3f},"
                    f"entropy:{float(getattr(qualia, 'entropy', 0.0)):.3f},"
                    f"engagement:{float(getattr(qualia, 'engagement', 0.0)):.3f},"
                    f"frustration:{float(getattr(qualia, 'frustration', 0.0)):.3f}"
                )
        except Exception:
            pass
        if affect_state is not None:
            try:
                a = np.asarray(affect_state, dtype=np.float32).ravel()
                a_vals = ",".join(f"{v:.3f}" for v in a[:8])
                lines.append(f"affect=[{a_vals}]")
            except Exception:
                pass
        try:
            if hasattr(core, 'energy_ctrl'):
                energy_ratio = core.energy_ctrl.cognitive_energy / max(core.energy_ctrl.energy_capacity, 1.0)
                lines.append(f"energy_ratio={float(energy_ratio):.3f}")
        except Exception:
            pass
        try:
            stability = float(getattr(core, 'world_state', {}).get('stability', 0.0))
            drift = float(getattr(core, 'world_state', {}).get('delta_hat', 0.0))
            lines.append(f"stability={stability:.3f},drift={drift:.3f}")
        except Exception:
            pass
        try:
            if hasattr(core, 'episodic_memory'):
                stats = core.episodic_memory.get_statistics()
                lines.append(f"memories={int(stats.get('total_memories', 0))}")
                lines.append(f"memory_consolidation={float(stats.get('avg_consolidation', 0.0)):.3f}")
        except Exception:
            pass
        vec = None
        try:
            if hasattr(core, 'export_state_vector'):
                vec = core.export_state_vector()
        except Exception:
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
                    try:
                        head_elems = int(os.getenv('M3_STATE_VECTOR_MAX_ELEMS', '32'))
                    except Exception:
                        head_elems = 32
                    head = vec_arr[:head_elems] if head_elems > 0 else vec_arr
                    vec_str = ",".join(f"{v:.4f}" for v in head.tolist())
                    suffix = ",..." if head_elems > 0 and vec_arr.size > head_elems else ""
                    lines.append(f"vector_head=[{vec_str}{suffix}]")
            except Exception:
                pass
        try:
            max_chars = int(os.getenv('M3_STATE_MAX_CHARS', '4000'))
        except Exception:
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
    def _state_vector(self, core=None):
        """Build state vector s from encoder hidden; uses BOS-only source when prompt absent."""
        torch = self.torch
        t = self.tok
        with torch.no_grad():
            src_ids = torch.tensor([[t.BOS]], dtype=torch.long, device=self.device)
            e_src = self.model.emb(src_ids)
            _, h = self.model.encoder(e_src)
            s = h[-1, :, :]  # (1, H)
            return s

    def _build_autonomy_prefix(self, s):
        """From state vector s, build continuous prefix embeddings with learned gating."""
        torch = self.torch
        try:
            if hasattr(s, 'dtype') and s.dtype == torch.bfloat16 and self.device.type == 'cpu':
                s = s.float()
        except Exception:
            pass
        with torch.no_grad():
            H = s.shape[-1]
            P = int(getattr(self.model, 'prefix_len', 1))
            E = int(self.model.emb.embedding_dim)
            raw = self.model.state2prefix(s)  # (1, P*E)
            pref = raw.view(1, P, E)
            gate = torch.sigmoid(self.model.prefix_gate(s)).view(1, P, 1)
            # numpy는 bfloat16을 지원하지 않음 → float32로 변환
            return (gate * pref).float().detach().cpu().numpy()

    def _run_core_steps(self, core, count: int = 0):
        if core is None:
            return
        for _ in range(max(0, int(count))):
            try:
                if hasattr(core, '_single_consciousness_step'):
                    core._single_consciousness_step()
            except Exception:
                pass

    def _run_checkpoint_if_enabled(self, core):
        try:
            if os.getenv('M3_SAVE_EVERY_TURN', '0') in ('1', 'true', 'yes', 'on'):
                if core is not None and hasattr(core, '_save_checkpoint'):
                    core._save_checkpoint()
        except Exception:
            pass

    def _drain_user_message(self):
        import queue
        if not hasattr(self, '_user_queue') or self._user_queue is None:
            return None
        try:
            return self._user_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception:
            return None

    def _handle_user_turn(self, user_msg: str, steps_per_cycle: int) -> str:
        core = getattr(self, 'core', None)
        response = ''
        if core is not None:
            self._run_core_steps(core, steps_per_cycle)
        try:
            if core is not None and hasattr(core, 'handle_user_message'):
                response = core.handle_user_message(user_msg)
            else:
                try:
                    user_max_len = int(os.getenv('M3_USER_MAX_LEN', '320'))
                except Exception:
                    user_max_len = 320
                response = self.generate(user_msg, max_len=max(32, user_max_len))
        except Exception as e:
            response = f'[Error: {e}]'

        try:
            cb = getattr(self, '_on_response', None)
            if cb is not None:
                cb(response)
        except Exception:
            pass

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

    def _run_autonomy_turn(self, cycle_count: int, autonomy_check_every: int):
        if autonomy_check_every <= 0 or cycle_count % autonomy_check_every != 0:
            return
        if getattr(self, "_hf_circuit_open", False):
            # CUDA fault detected: skip GPU-dependent autonomy policy path.
            return
        core = getattr(self, 'core', None)
        try:
            s = self._state_vector(core)
            torch = self.torch
            try:
                if hasattr(s, 'dtype') and s.dtype == torch.bfloat16 and self.device.type == 'cpu':
                    s = s.float()
            except Exception:
                pass
            with torch.no_grad():
                q = self.model.q_head(s.float())  # (1, 2): [Q_wait, Q_speak]
                try:
                    min_prob = float(os.getenv('M3_MIN_SPEAK_PROB', '0.2'))
                except Exception:
                    min_prob = 0.2
                diff = float(q[0, 1].item() - q[0, 0].item())
                prob = 1.0 / (1.0 + np.exp(-diff))
                prob = max(min_prob, prob)
                speak = bool(np.random.rand() < prob)
                lam = torch.nn.functional.softplus(self.model.intensity_head(s.float())).item() + 1e-8

            if not speak:
                try:
                    dt = float(np.random.exponential(1.0 / max(lam, 1e-12)))
                    dt = min(dt, 5.0)
                    self._wait_for_user_interrupt(dt)
                except Exception:
                    pass
                return

            prefix = self._build_autonomy_prefix(s)
            seed = self._autonomy_seed_prompt(core)
            try:
                auto_max_len = int(os.getenv('M3_AUTONOMY_MAX_LEN', '160'))
            except Exception:
                auto_max_len = 160
            text = self.generate(seed, max_len=max(32, auto_max_len), prefix_embed=prefix, source="autonomy")
            if not text or not text.strip():
                return
            text = text.strip()
            if self._is_disallowed_generation_output(text):
                logger.warning('Autonomy generation filtered by safety policy')
                return
            numeric_dump = self._is_numeric_dump_response(text)
            if (
                os.getenv('M3_AUTONOMY_LEARN_FROM_SELF', '0').lower() in ('1', 'true', 'yes', 'on')
                and not numeric_dump
            ):
                try:
                    self.learn_pair(seed, text, max_len=max(32, auto_max_len))
                except Exception:
                    pass

            if core is not None and hasattr(core, 'bus') and core.bus is not None:
                try:
                    vec = np.zeros((32,), dtype=np.float32)
                    if hasattr(core, 'feature_bank') and hasattr(core.feature_bank, '_hash_embed'):
                        vec = core.feature_bank._hash_embed(
                            text, core.feature_bank.embed_dim
                        ).astype(np.float32)
                    core.bus.push('system', 'utter.self', vec, salience=0.8, confidence=1.0, ttl=10)
                except Exception:
                    pass

            scb = getattr(self, '_on_spontaneous', None)
            if scb is not None:
                try:
                    scb(text, float(q[0, 1].item()), lam)
                except Exception:
                    pass

            try:
                self._log_jsonl(
                    os.getenv("LLM_ADAPTER_LOG", "llm_adapter.log"),
                    {
                        "kind": "autonomy_event",
                        "lambda": lam,
                        "q_wait": float(q[0, 0].item()),
                        "q_speak": float(q[0, 1].item()),
                        "text": text,
                    },
                )
            except Exception:
                pass
        except Exception as e:
            try:
                logger.error(f'Autonomy decision error: {e}')
            except Exception:
                pass
            self._note_control_health(False, f"autonomy_error:{e}")

    def start_autonomy_loop(self):
        if getattr(self, '_auto_running', False):
            return
        self._auto_running = True
        self._user_queue = getattr(self, '_user_queue', None)
        if self._user_queue is None:
            import queue
            self._user_queue = queue.Queue()
        self._auto_thread = threading.Thread(target=self._unified_loop, daemon=True)
        self._auto_thread.start()

    def stop_autonomy_loop(self):
        self._auto_running = False
        th = getattr(self, '_auto_thread', None)
        if th is not None:
            th.join(timeout=2.0)

    def submit_user_message(self, text: str):
        """사용자 메시지를 자율사고 루프에 비동기 전달 (논블로킹)"""
        import queue
        if not hasattr(self, '_user_queue') or self._user_queue is None:
            self._user_queue = queue.Queue()
        self._user_queue.put(text)

    def _autonomy_seed_prompt(self, core=None) -> str:
        """Build a minimal non-empty prompt for spontaneous generation.

        Empty prompts often cause instruct models to drift into low-quality text.
        This seed stays short and uses current M3 state when available.
        """
        try:
            lang = os.getenv('M3_AUTONOMY_LANGUAGE', 'ko').lower()
        except Exception:
            lang = 'ko'

        # Light state summary (best-effort)
        bits = []
        try:
            if core is not None and hasattr(core, 'energy_ctrl'):
                ec = core.energy_ctrl
                ratio = float(ec.cognitive_energy / max(ec.energy_capacity, 1.0))
                bits.append(f"energy={ratio:.2f}")
        except Exception:
            pass
        try:
            if core is not None and hasattr(core, 'qualia'):
                q = core.qualia
                bits.append(f"arousal={float(getattr(q, 'arousal', 0.0)):.2f}")
                bits.append(f"valence={float(getattr(q, 'valence', 0.0)):.2f}")
        except Exception:
            pass
        try:
            if core is not None and hasattr(core, 'phi_calculator') and core.phi_calculator.phi_history:
                bits.append(f"phi={float(core.phi_calculator.phi_history[-1]):.3f}")
        except Exception:
            pass
        state_line = (" (" + ", ".join(bits) + ")") if bits else ""

        if lang.startswith('ko'):
            return (
                "[자율 모드] 현재 M3_STATE를 바탕으로, 지금 해야 할 한 가지를 짧게 말해줘." + state_line
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
        core = getattr(self, 'core', None)
        consciousness_interval = float(os.getenv('M3_CONSCIOUSNESS_INTERVAL', '0.1'))
        steps_per_cycle = int(os.getenv('M3_STEPS_PER_CYCLE', '5'))
        autonomy_check_every = int(os.getenv('M3_AUTONOMY_CHECK_EVERY', '10'))
        cycle_count = 0

        while getattr(self, '_auto_running', False):
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
                try:
                    logger.error(f'Unified loop error: {e}')
                except Exception:
                    pass
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
                try:
                    msg = self._bus_inbox.get_nowait()
                except:
                    time.sleep(0.05)
                    continue
                
                # Process credit message
                if hasattr(msg, "type") and msg.type == "credit":
                    self._process_credit_message(msg)
            except Exception:
                time.sleep(0.1)

    def _process_credit_message(self, msg):
        """Process a single credit message from MessageBus."""
        try:
            payload = getattr(msg, "payload", {})
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
            
            # Optional: micro-update token value head with credit signal
            # (requires span_id token mapping, skipped for now)
        except Exception:
            pass

    def load_model(self, path: str):
        """Loads the model state dictionary."""
        try:
            if not os.path.exists(path):
                logger.warning(f"Model file not found at {path}, skipping load.")
                return
            self.model.load_state_dict(self.torch.load(path, map_location=self.device))
            self.model.eval()  # Set to evaluation mode after loading
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")

    def _beta_schedule(self) -> float:
        """
        D. Temporal Contextualization (Dynamic Memory)
        
        Returns:
             [beta_min, beta_max]
        """
        try:
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
        except Exception:
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

        try:
            # r_i(t) = L_i(t) / L_i(0)
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
        
        except Exception:
            pass  # Log error if needed

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
        try:
            with torch.no_grad():
                logits = self.model.head(h_final)  # (1, V)
                p = torch.softmax(logits, dim=-1)
                H = -torch.sum(p * torch.log(p + 1e-8), dim=-1).item()
                # Normalize
                H_norm = float(H / (np.log(self.vocab_size) + 1e-8))
                uncertainty = float(np.clip(H_norm, 0.0, 1.0))
        except Exception:
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
        try:
            with open(path, 'a', encoding='utf-8') as f:
                import json as _json
                f.write(_json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception:
            pass

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
        try:
            if core is not None and hasattr(core, "sdm") and hasattr(core.sdm, "quick_eval"):
                return float(core.sdm.quick_eval(candidate))
        except Exception:
            pass
        return 0.0

    def _predict_value_scalar(self, prompt: str, candidate: str) -> float:
        """Get last hidden and run value head for quick value estimate."""
        torch = self.torch
        t = self.tok
        try:
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
        except Exception:
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
        try:
            if core is not None:
                last_tool = getattr(core, "last_tool", "none")
                tool_ok   = float(getattr(core, "last_tool_ok", 0.0))
                phi_hist  = np.asarray(getattr(core, "phi_calculator", None).phi_history[-8:], dtype=np.float32) \
                            if hasattr(getattr(core,"phi_calculator", None), "phi_history") else np.zeros(0, np.float32)
                tag[0] = float(hash(str(last_tool)) % 997) / 997.0
                tag[1] = tool_ok
                if phi_hist.size:
                    tag[2] = float(phi_hist[-1])
                    tag[3] = float(np.mean(phi_hist))
                # bus depth/latency if present
                bus = getattr(core, "bus", None)
                if bus is not None and hasattr(bus, "depth"):
                    tag[4] = float(getattr(bus, "depth", 0))
                if bus is not None and hasattr(bus, "latency_ms"):
                    tag[5] = float(getattr(bus, "latency_ms", 0.0))
        except Exception:
            pass

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
        try:
            enable_tv = (
                self._control_allows('token_value_bias')
                and os.getenv('LLM_ADAPTER_ENABLE_TOKEN_VALUE_BIAS', '0').lower() in ('1', 'true', 'yes', 'on')
            )
            if enable_tv and hasattr(self, 'model') and hasattr(self.model, 'token_value'):
                tv_head = self.model.token_value
                internal_h = getattr(self.model, 'hidden', 1024)
                beta_val = float(self._beta_schedule())
        except Exception:
            pass

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

        try:
            q0 = gate.evaluate(response)
        except Exception:
            return response

        if not getattr(q0, 'reject', False):
            return response

        try:
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
        except Exception:
            return response
        return response

    def _semantic_perspective_prefix(self, core=None) -> str:
        try:
            enabled = os.getenv('M3_EMBED_PERSPECTIVE', '0').lower() in ('1', 'true', 'yes', 'on')
        except Exception:
            enabled = False
        if not enabled:
            return ""
        try:
            core_ref = core if core is not None else getattr(self, 'core', None)
            if core_ref is None:
                return ""
            subj = getattr(core_ref, 'unified_subject', None)
            if subj is None or not hasattr(subj, 'reflect_on_self'):
                return ""
            summary = str(subj.reflect_on_self()).strip()
            if not summary:
                return ""
            return f"Perspective: {summary}\n\n"
        except Exception:
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

    def generate(self, prompt: str, max_len: int = 60, temperature: float = None,
                 top_k: int = None, top_p: float = None,
                 mem: Optional[np.ndarray] = None,
                 prefix_embed: Optional[np.ndarray] = None,
                 knn_provider: Optional[Any] = None,
                 knn_alpha: float = 0.0,
                 affect_state: Optional[np.ndarray] = None,
                 source: str = "generate") -> str:
        t = self.tok
        core = getattr(self, 'core', None)
        raw_prompt = prompt
        if getattr(self, "_hf_circuit_open", False):
            self._note_control_health(False, "hf_circuit_open")
            return self._generate_safe_fallback(str(raw_prompt), chat_messages=None, max_len=max_len)
        # System identity / behavior prefix (configurable / optional)
        sys_identity = self._get_system_prompt()
        
        panels = None
        phi_trend = None
        if core is not None:
            panels = self._update_m3_cache(core)
            try:
                phi_trend = self.m3_cache.get_phi_trend(core)
            except Exception:
                phi_trend = None
        
        affect_state = self._get_affect_state(core, affect_state)
        main_prompt = raw_prompt
        # === Phase 4: Episodic Memory Retrieval (NO THRESHOLD) ===
        if core is not None and self._control_allows('memory_retrieval'):
            try:
                semantic_query = self._semantic_perspective_prefix(core) + str(raw_prompt)
                context_embedding = self.embed_text(semantic_query, sys_identity="")
                if context_embedding is not None:
                    relevant_episodes = self.m3_memory_retriever.retrieve_relevant_episodes(core, context_embedding)
                    if relevant_episodes:
                        episode_texts = []
                        try:
                            max_chars = int(os.getenv('M3_EPISODIC_MAX_CHARS', '800'))
                        except Exception:
                            max_chars = 800
                        try:
                            item_chars = int(os.getenv('M3_EPISODIC_ITEM_CHARS', '200'))
                        except Exception:
                            item_chars = 200
                        total_chars = 0
                        for ep in relevant_episodes:
                            if getattr(ep, 'kind', 'internal_state') not in {'dialog', 'research', 'knowledge'}:
                                continue
                            if hasattr(ep, 'content'):
                                txt = str(ep.content)
                            elif hasattr(ep, 'text'):
                                txt = str(ep.text)
                            elif hasattr(ep, 'description'):
                                txt = str(ep.description)
                            elif hasattr(ep, 'narrative'):
                                txt = str(ep.narrative)
                            elif hasattr(ep, 'context'):
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
                        if episode_texts:
                            context_prefix = "Similar past context:\n" + "\n".join(f"- {txt}" for txt in episode_texts) + "\n\nCurrent context:\n"
                            main_prompt = context_prefix + raw_prompt
            except Exception:
                pass

        include_m3_state = self._should_include_m3_state(raw_prompt)
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
            and self._control_allows('state_context')
            and getattr(self.model, 'use_m3_integration', False)
        ):
            m3_memory = self._build_m3_memory(core=core, panels=panels, affect_state=affect_state)

        m3_memory_t = None
        if m3_memory is not None:
            try:
                if self.torch.is_tensor(m3_memory):
                    m3_memory_t = m3_memory.to(self.device)
                else:
                    m3_memory_t = torch.tensor(np.asarray(m3_memory), dtype=torch.float32, device=self.device)
                if m3_memory_t.ndim == 3 and m3_memory_t.size(0) == 1:
                    m3_memory_t = m3_memory_t.squeeze(0)
                if m3_memory_t.ndim == 1:
                    m3_memory_t = m3_memory_t.unsqueeze(0)
            except Exception:
                m3_memory_t = None

        bridge_state = None
        if self._bridge_enabled_safe():
            try:
                bridge_state = self._build_full_state_vector(
                    core=core,
                    panels=panels,
                    affect_state=affect_state,
                )
            except Exception:
                bridge_state = None
            if bridge_state is None and m3_memory_t is not None:
                try:
                    bridge_state = m3_memory_t.detach().float().cpu().numpy().reshape(-1)
                except Exception:
                    bridge_state = None
            # Optional ablation for evaluation: none|shuffle|zero
            try:
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
            except Exception:
                pass

        # === HuggingFace backend: full M3 parameter control ===
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
                attempts = max(0, int(os.getenv("M3_CONTROL_RETRY", "1")))
                gate = getattr(self, '_quality_gate', None)
                # start from configured defaults / runtime inputs
                cur_temp = gate_payload['temperature']
                cur_top_k = gate_payload['top_k']
                cur_top_p = gate_payload['top_p']
                for attempt in range(attempts + 1):
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
                    if response and response.strip():
                        if gate is not None:
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
                        passed, _ = self._evaluate_generation_quality(prompt, response, source=source)
                        if passed:
                            if not self._is_disallowed_generation_output(response):
                                try:
                                    self._record_example(prompt, response, source='generate_hf')
                                except Exception:
                                    pass
                                self._note_control_health(True, "hf_generate_ok")
                                return response
                            logger.warning('[HFBackend] filtered disallowed output; switching to fallback')
                        elif attempt < attempts:
                            logger.warning(
                                f'[HFBackend] quality gate rejected output on attempt {attempt+1}/{attempts+1}; re-sampling with stricter params'
                            )
                            try:
                                cur_temp, cur_top_k, cur_top_p = self._retry_generation_sampling(
                                    temperature=cur_temp,
                                    top_k=cur_top_k,
                                    top_p=cur_top_p,
                                )
                            except Exception:
                                cur_temp = max(0.15, float(cur_temp) * 0.8)
                                cur_top_k = max(5, int(cur_top_k) if int(cur_top_k) < 24 else int(cur_top_k * 0.8))
                                cur_top_p = max(0.70, float(cur_top_p) * 0.95)
                            continue
                        logger.warning('[HFBackend] filtered by generation quality; switching to fallback')
            except Exception as e:
                logger.warning(f'[HFBackend] generation failed ({e})')
                if self._is_cuda_fatal_error(e):
                    self._trip_hf_circuit_breaker(e)
                self._note_control_health(False, f"hf_exception:{e}")
                return self._generate_safe_fallback(prompt, chat_messages=chat_messages, max_len=max_len)

        # HF backend disabled or unavailable: continue with safe fallback.
        if used_hf:
            self._note_control_health(False, "hf_filtered_or_fallback")
        else:
            self._note_control_health(False, "hf_unavailable_or_fallback")
        return self._generate_safe_fallback(prompt, chat_messages=chat_messages, max_len=max_len)

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
                if hasattr(self, 'core') and self.core is not None:
                    try:
                        phi = self.core.phi_calculator.phi_history[-1] if self.core.phi_calculator.phi_history else 0.5
                        entropy = getattr(self.core.qualia, 'entropy', 0.5)
                        energy_ratio = self.core.energy_ctrl.cognitive_energy / max(self.core.energy_ctrl.energy_capacity, 1.0) if hasattr(self.core, 'energy_ctrl') else 0.5
                        loss_weight = (1.0 - phi) * entropy * (1.0 - energy_ratio)
                        loss_weight = torch.tensor(max(0.1, min(2.0, loss_weight)), device=self.device)
                        loss = loss_weight * base_loss
                    except Exception:
                        loss = base_loss
                else:
                    loss = base_loss
                
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
            
            # === kNN: collect from teacher forcing ===
            try:
                # optional: provide core via self.core if attached
                self.collect_knn_from_teacher(prompt, response, core=getattr(self, "core", None))
            except Exception:
                pass
            try:
                self._record_example(prompt, response, source="learn_pair")
            except Exception:
                pass
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
                        try:
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
                            
                            # kNN collection (optional, can be slow so maybe skip or sample)
                            # self.collect_knn_from_teacher(prompt, response, core=getattr(self, "core", None))
                            
                        except Exception as e:
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
                    
                    self.opt.step()
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
            pass
        return num_trained, avg_loss

    def train_on_example(self, prompt: str, response: str, max_len: int = 120) -> None:
        """Alias for learn_pair to maintain API compatibility with GUI/data loaders."""
        return self.learn_pair(prompt, response, max_len)

    def train_dpo_from_dir(self, epochs: int = 2, beta: float = 0.1, data_dir: str = 'data_set') -> dict:
        """Train using DPO from preference data in data_set directory."""
        import json
        import glob
        
        total_loss = 0.0
        num_samples = 0
        
        try:
            # Search for DPO-style preference data files
            patterns = [
                os.path.join(data_dir, '**', '*preference*.json'),
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
                                try:
                                    obj = json.loads(line)
                                except Exception:
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
                try:
                    self.learn_pair(prompt, response, max_len=max_len)
                except Exception:
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
            try:
                lp = self._sequence_logprob(prompt, response, max_len=max_len)
                total_logp += float(lp)
                total_tokens += max(1, len(self.tok.encode(response, add_special=True)) - 1)
            except Exception:
                continue

        avg_nll = float(-total_logp / max(1, total_tokens))
        ppl = float(math.exp(avg_nll)) if avg_nll < 30 else float('inf')
        metrics = {'num_pairs': len(pairs), 'avg_nll': avg_nll, 'ppl': ppl}
        try:
            self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {"kind": "eval_supervised", **metrics})
        except Exception:
            pass
        logger.info(f'Supervised eval: pairs={metrics["num_pairs"]}, avg_nll={avg_nll:.4f}, ppl={ppl:.2f}')
        return metrics

    def evaluate_dpo_from_dir(self, data_dir: str = 'data_set', limit: int | None = None) -> dict:
        """Evaluate DPO preference consistency: rate(chosen_logp > rejected_logp)."""
        import json, glob
        samples = []
        try:
            patterns = [
                os.path.join(data_dir, '**', '*preference*.json'),
                os.path.join(data_dir, '**', '*dpo*.json'),
                os.path.join(data_dir, '**', '*chosen*.json'),
            ]
            paths = []
            for pat in patterns:
                paths.extend(glob.glob(pat, recursive=True))
            for filepath in paths:
                try:
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
            pass

        if not samples:
            logger.warning(f'No DPO preference samples found in {data_dir}')
            return {'num_pairs': 0}

        better = 0
        margins = []
        for prompt, chosen, rejected in samples:
            try:
                ch_lp = float(self._sequence_logprob(prompt, chosen))
                rj_lp = float(self._sequence_logprob(prompt, rejected))
                if ch_lp > rj_lp:
                    better += 1
                margins.append(ch_lp - rj_lp)
            except Exception:
                continue
        rate = better / max(1, len(samples))
        avg_margin = float(np.mean(margins)) if margins else 0.0
        metrics = {'num_pairs': len(samples), 'chosen_better_rate': rate, 'avg_logp_margin': avg_margin}
        try:
            self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {"kind": "eval_dpo", **metrics})
        except Exception:
            pass
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
                    try:
                        self.learn_pair(prompt, response, max_len=max_len)
                    except Exception:
                        continue

            total_logp = 0.0
            total_tokens = 0
            for prompt, response in eval_pairs:
                try:
                    lp = self._sequence_logprob(prompt, response, max_len=max_len)
                    total_logp += float(lp)
                    total_tokens += max(1, len(self.tok.encode(response, add_special=True)) - 1)
                except Exception:
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
            try:
                self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {"kind": "auto_supervised_eval", **sup_metrics})
            except Exception:
                pass
        else:
            logger.info('Auto supervised: no pairs found; skipping')

        dpo_metrics = {}
        if enable_dpo:
            try:
                self.train_dpo_from_dir(epochs=max(1, int(epochs)), beta=0.1, data_dir=data_dir)
                dpo_metrics = self.evaluate_dpo_from_dir(data_dir=data_dir)
            except Exception as e:
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
                                    try:
                                        slots = {str(s.get('slot','')): str(s.get('value','')) for s in slots if isinstance(s, dict)}
                                    except Exception:
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
                try:
                    # Build id -> title mapping
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
                except Exception:
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
                            try:
                                self.dpo_step(prompt, chosen, rejected)
                                maybe_take_dpo_eval(prompt, chosen, rejected)
                            except Exception:
                                pass
                        continue
                    try:
                        prompt, response = item
                        self.learn_pair(prompt, response, max_len=max_len)
                        maybe_take_sup_eval(prompt, response)
                        processed += 1
                        if limit is not None and processed >= int(limit):
                            break
                    except Exception:
                        continue
                if limit is not None and processed >= int(limit):
                    break
            if limit is not None and processed >= int(limit):
                break

        # Additionally, generic DPO files (if present)
        dpo_metrics = {}
        if enable_dpo:
            try:
                # Train (uses existing parser that supports multiple schemas)
                self.train_dpo_from_dir(epochs=max(1, int(epochs)), beta=0.1, data_dir=data_dir)
            except Exception:
                pass
            # Eval: include proactively collected mind triples + generic files
            try:
                gen_dpo = self.evaluate_dpo_from_dir(data_dir=data_dir)
                dpo_metrics.update(gen_dpo)
            except Exception:
                pass
            # Evaluate buffered triples
            better = 0
            margins = []
            for prompt, chosen, rejected in dpo_eval:
                try:
                    ch_lp = float(self._sequence_logprob(prompt, chosen, max_len=max_len))
                    rj_lp = float(self._sequence_logprob(prompt, rejected, max_len=max_len))
                    if ch_lp > rj_lp:
                        better += 1
                    margins.append(ch_lp - rj_lp)
                except Exception:
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
            try:
                lp = self._sequence_logprob(prompt, response, max_len=max_len)
                total_logp += float(lp)
                total_tokens += max(1, len(self.tok.encode(response, add_special=True)) - 1)
            except Exception:
                continue
        sup_metrics = {}
        if total_tokens > 0:
            avg_nll = float(-total_logp / total_tokens)
            ppl = float(math.exp(avg_nll)) if avg_nll < 30 else float('inf')
            sup_metrics = {'num_eval_pairs': len(sup_eval), 'avg_nll': avg_nll, 'ppl': ppl}
            try:
                self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {"kind": "train_all_eval", **sup_metrics, **({'dpo_metrics': dpo_metrics} if dpo_metrics else {})})
            except Exception:
                pass

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
            try:
                with torch.no_grad():
                    # Get phi predictions for chosen vs rejected
                    v_ch = self.model.v_phi(o_ch[:, -1, :]).squeeze(-1)
                    v_rj = self.model.v_phi(o_rj[:, -1, :]).squeeze(-1)
                    
                    # Dynamic phi_margin: core (NO MAGIC NUMBER 0.1)
                    if hasattr(self, 'core') and self.core is not None:
                        # phi/entropy/engagement  margin 
                        phi = self.core.phi_calculator.phi_history[-1] if self.core.phi_calculator.phi_history else 0.5
                        entropy = getattr(self.core.qualia, 'entropy', 0.5)
                        engagement = getattr(self.core.qualia, 'engagement', 0.5)
                        # margin_coef = phi * (1 - entropy) * engagement
                        # Compute margin coefficient
                        margin_coef = phi * (1.0 - entropy) * engagement
                        phi_margin = margin_coef * (v_ch - v_rj)
                    else:
                        # Fallback: phi difference (0.5)
                        phi_margin = (v_ch - v_rj) * 0.5  # fallback margin scaling
            except Exception:
                pass

            # DPO objective with phi-margin: -log(sigmoid(beta*((ch - rj) + phi_margin)))
            loss = -torch.log(torch.sigmoid(beta * ((ch_sum - rj_sum) + phi_margin)) + 1e-8)
            # Debugging: log intermediate scalars to help diagnose zero-loss/zero-grad issues
            try:
                logger.debug(f'DPO step: ch_sum={float(ch_sum):.6f}, rj_sum={float(rj_sum):.6f}, phi_margin={float(phi_margin):.6f}, beta={beta}')
            except Exception:
                pass
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            # compute grad norm for diagnostics
            try:
                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        try:
                            total_grad_norm += float(p.grad.data.norm().item() or 0.0)
                        except Exception:
                            continue
                logger.debug(f'DPO grads: total_grad_norm={total_grad_norm:.6f}, loss={float(loss):.6f}')
                if total_grad_norm == 0.0:
                    logger.warning('DPO step produced zero gradient norm check that chosen/rejected differ, model params require_grad, and optimizer has parameters')
            except Exception:
                pass
            try:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            except Exception:
                pass
            self.opt.step()
    
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
                                    try:
                                        phi = self.core.phi_calculator.calculate_phi()
                                        qualia = self.core.qualia_analyzer.analyze()
                                        entropy = qualia.get('entropy', 0.5)
                                        engagement = self.core.meta_cognitive_monitor.get_state().get('engagement', 0.5)
                                        # margin_coef = phi * (1 - entropy) * engagement
                                        # Compute margin coefficient
                                        margin_coef = phi * (1.0 - entropy) * engagement
                                    except Exception:
                                        margin_coef = 0.1  # fallback
                                else:
                                    margin_coef = 0.1  # standalone mode
                                
                                phi_margin = margin_coef * (v_ch - v_rj)

                        except Exception:
                            pass
                        
                        # DPO loss
                        loss = -torch.log(torch.sigmoid(beta * ((ch_sum - rj_sum) + phi_margin)) + 1e-8)
                        batch_loss += loss
                        # Debugging per-sample (aggregate will be logged per-batch)
                        try:
                            logger.debug(f'DPO batch sample: ch_sum={float(ch_sum):.6f}, rj_sum={float(rj_sum):.6f}, phi_margin={float(phi_margin):.6f}')
                        except Exception:
                            pass
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
                                try:
                                    total_grad_norm += float(p.grad.data.norm().item() or 0.0)
                                except Exception:
                                    continue
                        logger.debug(f'DPO batch: avg_loss={float(avg_loss):.6f}, total_grad_norm={total_grad_norm:.6f}, batch_size={valid_count}')
                        if total_grad_norm == 0.0:
                            logger.warning('DPO batch produced zero gradient norm check data and model configuration')
                    except Exception:
                        pass
                    try:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    except Exception:
                        pass
                    self.opt.step()
    
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
            self.value_opt = self.torch.optim.Adam(params, lr=1e-3)
        
        self.model.train()
        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()
        t = self.tok
        batch_size = 32  # You can adjust the batch size as needed
        num_records = len(records)

        # Initialize accumulators if not already present
        if not hasattr(self, 'phi_targets'):
            self.phi_targets: List[float] = []
        if not hasattr(self, 'phi_preds'):
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
                    try:
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
                    except Exception:
                        # Skip problematic sample quietly
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
                try:
                    rec = {
                        "t": int(time.time() * 1000),
                        "kind": "value_train",
                        "n_samples": n,
                        "avg_loss": float(total_loss / max(1, n)),
                        "phi_correlation": phi_correlation,
                        "task_weights": dict(self._task_weights)
                    }
                    self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), rec)
                except Exception:
                    pass
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
            self.token_value_opt = self.torch.optim.Adam(params, lr=1e-3)
        self.model.train()
        mse = nn.MSELoss()
        for _ in range(max(1, epochs)):
            total = 0.0
            n = 0
            for rec in records:
                try:
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
                        self.token_value_opt.step()
                    total += float(loss.item())
                    n += 1
                except Exception:
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
        # CPU에서는 bfloat16 비지원 → float32로 강제
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
        # CPU에서는 float32로 강제 변환
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
        try:
            fn = getattr(self, "_bridge_enabled", None)
            if callable(fn):
                return bool(fn())
        except Exception:
            pass
        return False

    def _note(self, success: bool, reason: str = "") -> None:
        try:
            if getattr(self, '_control_health_window', None) is None:
                window = int(os.getenv("M3_CONTROL_HEALTH_WINDOW", "24"))
                self._control_health_window = deque(maxlen=max(1, window))
            self._control_health_window.append((time.time(), bool(success), str(reason or "")))
        except Exception:
            return
        if success:
            self._auto_mode_fail_streak = 0
        else:
            self._auto_mode_fail_streak = getattr(self, '_auto_mode_fail_streak', 0) + 1

    if not callable(getattr(adapter, '_control_selection_mode', None)):
        adapter._control_selection_mode = types.MethodType(_selection_mode, adapter)
    if not callable(getattr(adapter, '_control_allows', None)):
        adapter._control_allows = types.MethodType(_allows, adapter)
    if not callable(getattr(adapter, '_note_control_health', None)):
        adapter._note_control_health = types.MethodType(_note, adapter)
    if not callable(getattr(adapter, '_bridge_enabled', None)):
        adapter._bridge_enabled = types.MethodType(_bridge, adapter)
    if not callable(getattr(adapter, '_bridge_enabled_safe', None)):
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
            if hasattr(core, 'policy') and core.policy is not None:
                adapter.set_motor_policy(core.policy)
                logger.info('Attached existing core policy as motor policy')
        else:
            missing_methods = [m for m in required_methods if not callable(getattr(adapter, m, None))]
            if missing_methods:
                # First, try compatibility binding in-place for legacy adapters.
                _attach_control_compat(adapter)
                still_missing = [m for m in required_methods if not callable(getattr(adapter, m, None))]
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
        missing_final = [m for m in required_methods if not callable(getattr(adapter, m, None))]
        if missing_final:
            _attach_control_compat(adapter)
        
        # Attach adapter to core
        core.llm_adapter = adapter
        adapter.core = core

        # Enable M3 integration by default unless disabled
        try:
            m3_flag = os.getenv('M3_INTEGRATION', '1').lower()
            if m3_flag in ('1', 'true', 'yes', 'on'):
                if hasattr(adapter, 'enable_m3_integration'):
                    adapter.enable_m3_integration()
                    logger.info('M3 integration enabled for LLM adapter')
        except Exception:
            pass
        
        # Connect MessageBus inbox for credit assignment
        if hasattr(core, 'message_bus') and core.message_bus is not None:
            try:
                if hasattr(core.message_bus, 'inboxes') and 'llm_adapter' in core.message_bus.inboxes:
                    adapter._bus_inbox = core.message_bus.inboxes['llm_adapter']
                    adapter.start_credit_consumer()
                    logger.info('LLM adapter connected to MessageBus for credit assignment')
            except Exception as e:
                logger.warning(f'Failed to connect LLM adapter to MessageBus: {e}')
        
        # Enable training data recording if requested
        if record:
            adapter._record_training = True
            logger.info('LLM adapter attached to core with training data recording enabled')
        else:
            adapter._record_training = False
            logger.info('LLM adapter attached to core (recording disabled)')

        # Optionally start autonomy loop if enabled via environment
        try:
            auto_flag = os.getenv('LLM_AUTONOMY', '0').lower()
            if auto_flag in ('1', 'true', 'yes', 'on'):
                adapter.start_autonomy_loop()
                logger.info('Autonomy loop started (LLM_AUTONOMY enabled)')
        except Exception:
            pass

        return adapter
        
    except Exception as e:
        logger.exception('Failed to attach LLM adapter to core')
        raise


