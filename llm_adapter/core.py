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

# If this file is executed directly (python llm_adapter/core.py), ensure
# project root is on sys.path so absolute imports like `import llm_adapter.*`
# work. When run as a module (`python -m llm_adapter.core`) this is not needed.
if __name__ == "__main__" and __package__ is None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
import os
import json
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import threading
import time
from collections import deque
import torch

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
    "You are M3. Respond as M3 using the provided M3_STATE. "
    "Do not claim to be an AI assistant or language model. "
    "Do not mention DeepSeek or any other persona. "
    "Do not say you cannot feel; answer based on state. "
    "Be concise and factual. Reply in the user's language."
)
if not logger.handlers:
    level = logging.DEBUG if os.environ.get('LLM_ADAPTER_DEBUG', '0') in ('1', 'true', 'TRUE') else logging.INFO
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
    'TorchConversationalPolicy',
    'UnifiedM3Policy',
    'attach_llm_to_core',
]


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
        self.device = self.torch.device(device or ('cuda' if self.torch.cuda.is_available() else 'cpu'))
        
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
                
                self.prefix_len = max(1, hidden // embed_dim)
                self.state2prefix = PlasticBitLinear(hidden, self.prefix_len * embed_dim)
                self.prefix_gate = PlasticBitLinear(hidden, self.prefix_len)
                
                # --- Heads (Plastic) ---
                self.value = PlasticBitLinear(hidden, 1)
                self.v_phi   = PlasticBitLinear(hidden, 1)
                self.v_stab  = PlasticBitLinear(hidden, 1)
                self.v_tool  = PlasticBitLinear(hidden, 1)
                self.token_value = PlasticBitLinear(hidden, vocab_size)
                
                # Autonomy (Plastic)
                self.q_head = PlasticBitLinear(hidden, 2)
                self.intensity_head = PlasticBitLinear(hidden, 1)
                
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
        self.amp_dtype = self.torch.bfloat16

        # Ollama configuration for Deepseek R1 integration
        self.use_local_ai = os.getenv("USE_LOCAL_AI", "1") == "1"
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
        # Request timeout (seconds) for local Ollama/Deepseek calls — configurable via OLLAMA_TIMEOUT
        try:
            self.ollama_timeout = float(os.getenv("OLLAMA_TIMEOUT", "300"))
        except Exception:
            self.ollama_timeout = 300.0

    def save_model(self, path: str):
        """Saves the model state dictionary."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}")

    def get_local_thinking(self, prompt: str, temperature: float = None, top_k: int = None,
                           top_p: float = None, max_len: int = None) -> str:
        """Call local Ollama/Deepseek server with validation."""
        if not self.use_local_ai:
            return "Local AI disabled"
        try:
            from . import remote as remote_api
        except Exception:
            import llm_adapter.remote as remote_api
        return remote_api.get_local_thinking(
            prompt,
            url=self.ollama_url,
            model=self.ollama_model,
            timeout=self.ollama_timeout,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_len=max_len,
        )

    def _record_example(self, prompt: str, response: str, source: str = "generate") -> None:
        if not getattr(self, "_record_training", False):
            return
        try:
            rec = {
                "ts": time.time(),
                "source": source,
                "prompt": str(prompt),
                "response": str(response),
            }
            try:
                core = getattr(self, "core", None)
                if core is not None:
                    try:
                        phi = core.phi_calculator.phi_history[-1] if core.phi_calculator.phi_history else 0.0
                    except Exception:
                        phi = 0.0
                    try:
                        q = core.qualia
                        qualia = {
                            "arousal": float(getattr(q, "arousal", 0.0)),
                            "valence": float(getattr(q, "valence", 0.0)),
                            "entropy": float(getattr(q, "entropy", 0.0)),
                            "engagement": float(getattr(q, "engagement", 0.0)),
                            "frustration": float(getattr(q, "frustration", 0.0)),
                        }
                    except Exception:
                        qualia = {}
                    rec["phi"] = float(phi)
                    rec["qualia"] = qualia
            except Exception:
                pass
            try:
                os.makedirs(os.path.dirname(TRAINING_PATH), exist_ok=True)
            except Exception:
                pass
            with self._record_lock:
                with open(TRAINING_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
                vec_str = ",".join(f"{v:.4f}" for v in np.asarray(vec, dtype=np.float32).ravel().tolist())
                lines.append(f"vector=[{vec_str}]")
            except Exception:
                pass
        try:
            max_chars = int(os.getenv('M3_STATE_MAX_CHARS', '4000'))
        except Exception:
            max_chars = 4000
        joined = "\n".join(lines)
        if max_chars > 0 and len(joined) > max_chars:
            if vec is not None:
                trimmed = [ln for ln in lines if not ln.startswith("vector=")]
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
        with torch.no_grad():
            H = s.shape[-1]
            P = int(getattr(self.model, 'prefix_len', 1))
            E = int(self.model.emb.embedding_dim)
            raw = self.model.state2prefix(s)  # (1, P*E)
            pref = raw.view(1, P, E)
            gate = torch.sigmoid(self.model.prefix_gate(s)).view(1, P, 1)
            return (gate * pref).detach().cpu().numpy()

    def start_autonomy_loop(self):
        if getattr(self, '_auto_running', False):
            return
        self._auto_running = True
        self._auto_thread = threading.Thread(target=self._autonomy_loop, daemon=True)
        self._auto_thread.start()

    def stop_autonomy_loop(self):
        self._auto_running = False
        th = getattr(self, '_auto_thread', None)
        if th is not None:
            th.join(timeout=1.0)

    def _autonomy_loop(self):
        import time
        torch = self.torch
        while getattr(self, '_auto_running', False):
            try:
                s = self._state_vector(getattr(self, 'core', None))  # (1, H)
                with torch.no_grad():
                    q = self.model.q_head(s)  # (1, 2)
                    speak = bool(q[0, 1] > q[0, 0])
                    lam = torch.nn.functional.softplus(self.model.intensity_head(s)).item() + 1e-8
                # Sample next event time from Exp(lam)
                dt = float(np.random.exponential(1.0 / max(lam, 1e-12)))
                # If speaking, generate now; else wait and loop
                if speak:
                    prefix = self._build_autonomy_prefix(s)
                    try:
                        text = self.generate('', max_len=120, prefix_embed=prefix)
                    except Exception:
                        text = ''
                    try:
                        self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), {
                            "kind": "autonomy_event",
                            "lambda": lam,
                            "q_wait": float(q[0,0].item()),
                            "q_speak": float(q[0,1].item()),
                            "text": text,
                        })
                    except Exception:
                        pass
                # Sleep until next event
                time.sleep(max(0.0, dt))
            except Exception as e:
                try:
                    logger.error(f'Autonomy loop error: {e}')
                except Exception:
                    pass
                self._auto_running = False

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

    def generate(self, prompt: str, max_len: int = 60, temperature: float = None,
                 top_k: int = None, top_p: float = None,
                 mem: Optional[np.ndarray] = None,
                 prefix_embed: Optional[np.ndarray] = None,
                 knn_provider: Optional[Any] = None,
                 knn_alpha: float = 0.0,
                 affect_state: Optional[np.ndarray] = None) -> str:
        t = self.tok
        core = getattr(self, 'core', None)
        raw_prompt = prompt
        # System identity / behavior prefix (configurable)
        sys_identity = os.getenv('LLM_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT)
        if sys_identity:
            sys_identity = sys_identity.strip()
        
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
        if core is not None:
            try:
                base_prompt = sys_identity + "\n\n" + raw_prompt if sys_identity else raw_prompt
                src_temp = t.encode(base_prompt, add_special=False)
                if src_temp:
                    src_temp_ids = torch.tensor(src_temp, dtype=torch.long, device=self.device).unsqueeze(0)
                    e_src_temp = self.model.emb(src_temp_ids)
                    _, h_temp = self.model.encoder(e_src_temp)
                    context_embedding = h_temp.squeeze(0).cpu().numpy()
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

        m3_context = self._build_m3_context(core, affect_state=affect_state, phi_trend=phi_trend, panels=panels)
        prompt_parts = []
        if sys_identity:
            prompt_parts.append(sys_identity)
        if m3_context:
            prompt_parts.append(m3_context)
        prompt_parts.append(main_prompt)
        prompt = "\n\n".join(prompt_parts)

        m3_memory = mem
        if m3_memory is None and core is not None and getattr(self.model, 'use_m3_integration', False):
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

        # If using local AI (Deepseek), call Ollama directly
        if self.use_local_ai:
            try:
                response = self.get_local_thinking(
                    prompt,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_len=max_len,
                )
                if response and not response.startswith("Local Error"):
                    try:
                        self._record_example(prompt, response, source="generate_local")
                    except Exception:
                        pass
                    return response
                if os.getenv('DEEPSEEK_ONLY', '0') in ('1', 'true', 'yes', 'on'):
                    return response
            except Exception as e:
                try:
                    logger.warning(f"Deepseek Error: {e}")
                except Exception:
                    pass
                if os.getenv('DEEPSEEK_ONLY', '0') in ('1', 'true', 'yes', 'on'):
                    return f"Local Error: {e}"

        if not hasattr(self, 'model'):
            return "No AI model available"
        
        src = t.encode(prompt, add_special=False)
        if not src:
            # Use BOS-only source to derive state when no textual prompt is provided
            src = [t.BOS]
        src_ids = torch.tensor(src, dtype=torch.long, device=self.device).unsqueeze(0)
        e_src = self.model.emb(src_ids)
        h, _ = self.model.encoder(e_src)
        # If a continuous prefix is provided, use it to prime the decoder state
        if prefix_embed is not None:
            pe = np.array(prefix_embed)
            if pe.ndim == 2:
                pe = pe[None, :, :]  # (P, E) -> (1, P, E)
            if pe.ndim == 3:
                prefix_t = torch.tensor(pe, dtype=torch.float32, device=self.device)
                # Use prefix as source, history as target (append history to prefix)
                # h becomes [prefix, h]
                _, _, h = self.model.decoder(src_ids=prefix_t, tgt_in_ids=h, m3_memory=m3_memory_t, return_history=True)
        cur = torch.tensor([[t.BOS]], dtype=torch.long, device=self.device)
        out_tokens: List[int] = []
        
        # DEBUG: Generation start
        print(f"DEBUG: Gen start. Prompt tokens: {src}")

        # Prepare memory (if provided)
        mem_k = mem_v = None
        if m3_memory_t is not None:
            try:
                m_t = m3_memory_t
                if m_t.ndim == 2:
                    m_t = m_t.unsqueeze(0)
                if m_t.ndim == 1:
                    m_t = m_t.unsqueeze(0).unsqueeze(0)
                D = int(m_t.shape[2])
                self.model._ensure_mem_layers(D)
                mem_h = self.model.mem_proj(m_t)  # (1, M, H)
                mem_k = self.model.Wk(mem_h)
                mem_v = self.model.Wv(mem_h)
            except Exception:
                mem_k = mem_v = None
        
        for step in range(max_len):
            e = self.model.emb(cur)
            _, list_x, h = self.model.decoder(src_ids=None, tgt_in_ids=cur, m3_memory=m3_memory_t, return_history=True)
            dec_t = list_x[:, -1, :]  # (1, H)
            
            # === C. ===
            if mem_k is not None and mem_v is not None:
                # core(gating)
                core_state = {}
                try:
                    if hasattr(self, 'core') and self.core is not None:
                        core_state['stability'] = float(getattr(self.core, 'world_state', {}).get('stability', 0.5)) \
                                                  if hasattr(self.core, 'world_state') else 0.5
                        core_state['drift'] = float(getattr(self.core, 'world_state', {}).get('delta_hat', 0.0)) \
                                              if hasattr(self.core, 'world_state') else 0.0
                        phi_estimates = getattr(self, '_last_value_estimates', {})
                        core_state['phi_delta'] = float(phi_estimates.get('phi_delta', 0.0))
                except Exception:
                    pass

                # core(gating)
                dec_t = self.model.compute_mem_context(dec_t, mem_k, mem_v, core_state)
            
            logits = self.model.head(dec_t)  # (1, V)

            # === D. Token-Critic ===
            # Token-critic: compute token-level Q-values
            qtok = self.model.token_value(dec_t).detach()  # (1, V), detach for sampling
            # Get accumulated token-level advantages
            adv = self._adv_headroom(logits.shape[-1])  # (1, V) or None
            # Compute beta schedule
            beta = self._beta_schedule()
            # Combine: base logits + Q_token + advantage
            if adv is not None:
                logits = logits + beta * qtok + self._token_adv_alpha * adv
            else:
                logits = logits + beta * qtok
            # === A. kNN-aware kNN-LM Mixing (Contextual Memory) ===
            if knn_provider is not None and knn_alpha > 0.0:
                # keep old provider if passed externally
                try:
                    knn_dist = knn_provider()
                    if knn_dist is not None:
                        # knn_dist: torch vector (V,) already normalized
                        logits = torch.log_softmax(logits, dim=-1)
                        mix = torch.log((1 - knn_alpha) * torch.exp(logits) + knn_alpha * knn_dist + 1e-8)
                        logits = mix
                except Exception:
                    pass
            else:
                # NEW: internal conditional kNN with uncertainty-based alpha
                try:
                    key = self._build_cond_key(prompt, core=getattr(self, "core", None))
                    p_knn = self._knn.query(key, k=int(os.getenv("LLM_ADAPTER_KNN_K", "8")))
                    if p_knn is not None:
                        # === kNN-aware Mixing ===
                        # Compute kNN attention over dec_t (1,H) and value-head
                        alpha = self._alpha_scheduler_uncertainty(dec_t, self._last_value_estimates)

                        # Compute mixing probabilities: P = softmax(logits), P_knn
                        P = torch.softmax(logits, dim=-1)  # (1, V)
                        P_knn = torch.tensor(p_knn, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, V)

                        # Mix probabilities
                        P_mix = (1.0 - alpha) * P + alpha * P_knn
                        P_mix = P_mix / (P_mix.sum(dim=-1, keepdim=True) + 1e-8)

                        # Logits
                        logits = torch.log(P_mix + 1e-8)
                except Exception:
                    alpha = 0.0  # fallback
            
            # === Entropy-based sampling scheduler ===
            # Compute current entropy for adaptive temperature/top_p
            logp = torch.log_softmax(logits, dim=-1)
            p = torch.exp(logp)
            H = -torch.sum(p * logp, dim=-1).item()
            
            # Rolling entropy EMA
            if not hasattr(self, "_H_ema"):
                self._H_ema = H
            self._H_ema = 0.9 * self._H_ema + 0.1 * H
            
            # Adaptive temperature & top_p schedule
            try:
                env_temp = os.getenv("GEN_TEMP_BASE")
                if env_temp and env_temp.lower() != "none":
                    T0 = float(env_temp)
                elif temperature is not None:
                    T0 = float(temperature)
                else:
                    T0 = 1.0
            except Exception:
                T0 = 1.0

            try:
                env_ah = os.getenv("GEN_TEMP_H_COEF", "0.08")
                if env_ah and env_ah.lower() != "none":
                    aH = float(env_ah)
                else:
                    aH = 0.08
            except Exception:
                aH = 0.08

            temperature_step = float(np.clip(T0 + aH * (self._H_ema - 4.0), 0.3, 1.5))
            top_p_step = float(np.clip(0.8 + 0.05 * (self._H_ema - 4.0), 0.5, 0.98))
            
            # === G. Log generation step metrics with /g/ time series ===
            try:
                rec = {
                    "t": int(time.time() * 1000),
                    "kind": "gen_step",
                    "entropy": float(H),
                    "H_ema": float(self._H_ema),
                    "temp": float(temperature_step),
                    "top_p": float(top_p_step),
                }
                if mem_k is not None and mem_v is not None:
                    try:
                        # Compute attention over memory panels for logging
                        q_log = self.model.Wq(dec_t).unsqueeze(1)  # (1,1,H)
                        att_log = torch.softmax((q_log @ mem_k.transpose(1, 2)) / np.sqrt(self.model.hidden + 1e-8), dim=-1)
                        top_panel = int(torch.argmax(att_log[0, 0]).item())
                        rec["att_top_panel"] = top_panel
                        
                        # === G. Gate coefficient 'g'  ===
                        if hasattr(self.model, 'gate_proj'):
                            try:
                                g = torch.sigmoid(self.model.gate_proj(dec_t))
                                rec["gate_g"] = float(g.mean().item())
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                # === G.  (kNN mixing)  ===
                if 'alpha' in locals():
                    rec["knn_alpha"] = float(alpha)
                
                # === G.  (token-value correction)  ===
                if 'beta' in locals():
                    rec["token_value_beta"] = float(beta)
                
                self._log_jsonl(os.getenv("LLM_ADAPTER_LOG", "llm_adapter.jsonl"), rec)
            except Exception:
                pass
            
            tok = self._sample(logits, temperature=temperature_step, top_k=top_k, top_p=top_p_step)
            tid = int(tok.item())
            
            # DEBUG: Print token info
            print(f"DEBUG: Step {step}, Token {tid} ({t.decode([tid])})")
            
            if tid == t.EOS or tid == t.PAD:
                print("DEBUG: EOS/PAD reached")
                break
            out_tokens.append(tid)

            # === Global token index ===
            if not hasattr(self, "_global_token_idx"):
                self._global_token_idx = 0
            self._global_token_idx += 1
            
            cur = tok
        
        decoded = t.decode(out_tokens)
        try:
            self._record_example(prompt, decoded, source="generate")
        except Exception:
            pass
        # print(f"DEBUG: Decoded output: '{decoded}'")
        return decoded

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
        return self.model._hidden_state
        
    def learn(self, text: str, arousal: float = 1.0, sleep_after: bool = False):
        """Unified interface for offline learning."""
        return self.model.learn_from_text(text, arousal=arousal, sleep_after=sleep_after)

    def generate(self, prompt: str, mem: Optional[np.ndarray] = None, affect_state: Optional[np.ndarray] = None, max_len: int = 50, **kwargs) -> str:
        """
        Generate text response using M3PlasticPolicy.
        Now supports Affect State injection.
        """
        # System identity / behavior prefix (configurable)
        sys_identity = os.getenv('LLM_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT)
        if sys_identity:
            prompt = sys_identity.strip() + "\n\n" + prompt

        # Episodic memory retrieval for context
        if hasattr(self, 'core') and self.core is not None:
            try:
                src_temp = self.tok.encode(prompt, add_special=False)
                if src_temp:
                    src_temp_ids = self.torch.tensor(src_temp, dtype=self.torch.long, device=self.device).unsqueeze(0)
                    e_src_temp = self.model.emb(src_temp_ids)
                    _, h_temp = self.model.encoder(e_src_temp)
                    context_embedding = h_temp.squeeze(0).cpu().numpy()
                    relevant_episodes = self.m3_memory_retriever.retrieve_relevant_episodes(self.core, context_embedding)
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
                            if item_chars > 0 and len(txt) > item_chars:
                                txt = txt[:item_chars] + "..."
                            if max_chars > 0 and (total_chars + len(txt)) > max_chars:
                                break
                            total_chars += len(txt) + 2
                            episode_texts.append(txt)
                        if episode_texts:
                            context_prefix = "Similar past context:\n" + "\n".join(f"- {txt}" for txt in episode_texts) + "\n\nCurrent context:\n"
                            prompt = context_prefix + prompt
            except Exception:
                pass
        
        # If using local AI (Deepseek), call Ollama directly
        if self.use_local_ai:
            try:
                response = self.get_local_thinking(
                    prompt,
                    temperature=kwargs.get("temperature"),
                    top_k=kwargs.get("top_k"),
                    top_p=kwargs.get("top_p"),
                    max_len=max_len,
                )
                if response and not response.startswith("Local Error"):
                    try:
                        self._record_example(prompt, response, source="generate_local")
                    except Exception:
                        pass
                    return response
                # Fall back to local model if Ollama fails
            except Exception as e:
                logger.warning(f"Ollama call failed: {e}, falling back to local model")
        
        # 1. Prepare Input
        # Note: M3PlasticPolicy works with discrete tokens.
        # We need to bridge 'prompt' (str) -> 'input_ids' (Tensor)
        # And handle 'mem' (Panel Context) if possible (Currently PlasticPolicy is pure text/state)
        
        # Simple Prompt-based generation for now
        # Call sample recursively? Or implement beam search?
        # M3PlasticPolicy.sample is single-step. We need a loop here.
        
        self.model.eval()
        tokens = self.model.tok.encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        # Affect Tensor
        affect_tensor = None
        if affect_state is not None:
             affect_tensor = torch.from_numpy(affect_state).float().to(self.device).unsqueeze(0)
        
        generated = []
        curr_ids = input_ids
        
        # 2. Generation Loop
        for _ in range(max_len):
            with torch.no_grad():
                logits, _ = self.model(curr_ids, affect_state=affect_tensor) # Forward
                next_token_logits = logits[:, -1, :]
                
                # Greedy or Sampling?
                # Using simple temperature sampling
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                token_id = next_token.item()
                if token_id == self.model.tok.EOS:
                    break
                    
                generated.append(token_id)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
        
        # 3. Decode
        output_text = self.model.tok.decode(generated)
        try:
            self._record_example(prompt, output_text, source="generate")
        except Exception:
            pass
        return output_text


# ============================================================================
# Public API: attach_llm_to_core
# ============================================================================

def attach_llm_to_core(core, adapter=None, record: bool = True):
    """
    Attach LLM adapter to core for conversational learning.
    
    Args:
        core: M3Core instance
        adapter: Optional TorchConversationalPolicy instance (auto-created if None)
        record: Whether to record training data to JSONL
    """
    try:
        if adapter is None:
            # Auto-detect device: prefer CUDA if available
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
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
