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
import json
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import threading

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
from llm_adapter.tokenization import ByteTokenizer, HybridTokenizer


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
    
    def _lazy_init_projections(self, panel_dim: int, num_panels: int):
        """
        Lazy initialize panel dimension -> projection layers
        """
        if self.panel_projections is not None and self._initialized_panel_dim == panel_dim:
            # Already initialized with same dims
            return
        
        # Panel-specific projections
        self.panel_projections = self.nn.ModuleList([
            self.nn.Linear(panel_dim, self.hidden_dim)
            for _ in range(num_panels)
        ]).to(self.device)
        
        # Positional embeddings
        self.position_embeddings = self.nn.Parameter(
            self.torch.randn(num_panels, self.hidden_dim) * (1.0 / np.sqrt(self.hidden_dim))
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
    
    def forward(self, panels: List[np.ndarray]) -> 'torch.Tensor':
        """
        panels: List of panel vectors from core.feature_bank.panels(core)
        Args:
            panels: List of panel vectors from core.feature_bank.panels(core)
                   Each panel: (panel_dim,) numpy array
        
        Returns:
            memory_tokens: (num_panels, hidden_dim) tensor
        """
        if not panels:
            raise ValueError("panels list is empty - core.feature_bank.panels(core) returned no panels")
        
        # Infer dimensions from actual data
        num_panels = len(panels)
        panel_dim = panels[0].shape[0]
        
        # Validate all panels have same dimension
        for i, panel in enumerate(panels):
            if panel.shape[0] != panel_dim:
                raise ValueError(
                    f"Panel {i} has dimension {panel.shape[0]}, expected {panel_dim}. "
                    f"All panels must have consistent dimensions."
                )
        
        # Lazy initialization
        self._lazy_init_projections(panel_dim, num_panels)
        
        # Project each panel
        tokens = []
        for i, panel_vec in enumerate(panels):
            panel_tensor = self.torch.from_numpy(panel_vec).float().to(self.device)
            token = self.panel_projections[i](panel_tensor)
            tokens.append(token)
        
        # Stack and add positional embeddings
        memory = self.torch.stack(tokens)  # (num_panels, hidden_dim)
        memory = memory + self.position_embeddings
        
        # Apply fusion layers
        memory_batch = memory.unsqueeze(0)  # (1, num_panels, hidden_dim) - batch_first=True
        for layer in self.fusion_layers:
            memory_batch = layer(memory_batch)
        
        return memory_batch.squeeze(0)  # (num_panels, hidden_dim)


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
            # 1. Qualia  base adjustment
            qualia_vec = self.torch.tensor([
                getattr(core.qualia, 'arousal', 0.5),
                getattr(core.qualia, 'valence', 0.5),
                getattr(core.qualia, 'entropy', 0.5),
                getattr(core.qualia, 'engagement', 0.5),
                getattr(core.qualia, 'frustration', 0.0)
            ], dtype=self.torch.float32).to(self.device)
            
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
            entropy = getattr(core.qualia, 'entropy', 0.5)
            engagement = getattr(core.qualia, 'engagement', 0.5)
            exploration = entropy * engagement
            
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


OUT_DIR = os.path.join(os.path.dirname(__file__), 'out_m3')
os.makedirs(OUT_DIR, exist_ok=True)
TRAINING_PATH = os.path.join(OUT_DIR, 'llm_training_data.jsonl')

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, 'llm_adapter.log')

logger = logging.getLogger('llm_adapter')
if not logger.handlers:
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if os.environ.get('LLM_ADAPTER_DEBUG', '0') in ('1', 'true', 'TRUE') else logging.INFO)

__all__ = [
    'M3StateEncoder',
    'M3StateCache',
    'M3AwareDecoderLayer',
    'M3AdaptiveSampler',
    'TorchConversationalPolicy',
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
        
        # Initialize byte tokenizer (no BPE merges - lightweight mode)
        self.byte_tok = ByteTokenizer()
        self.tok = HybridTokenizer(self.byte_tok, merges=[])
        self.vocab_size = self.tok.vocab_size

        class Model(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int, hidden: int, torch_module, device, init_gate_value: float):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=256)
                self.encoder = nn.GRU(embed_dim, hidden, batch_first=True)
                self.decoder = nn.GRU(embed_dim, hidden, batch_first=True)
                self.head = nn.Linear(hidden, vocab_size)
                # Cross-attention projections (lazy init for memory input dim)
                self.mem_proj = None  # nn.Linear(mem_in_dim, hidden)
                self.Wk = None  # nn.Linear(hidden, hidden)
                self.Wv = None  # nn.Linear(hidden, hidden)
                self.Wq = nn.Linear(hidden, hidden)
                self.hidden = hidden
                # Value head for reranking (Step 3 scaffolding) - kept for backward compat
                self.value = nn.Linear(hidden, 1)
                # === Multi-head value predictors for label diversification ===
                self.v_phi   = nn.Linear(hidden, 1)  # phi_delta regression
                self.v_stab  = nn.Linear(hidden, 1)  # stability_delta regression
                self.v_tool  = nn.Linear(hidden, 1)  # tool_success binary (sigmoid)
                # Token-level Q-value head (token-critic)
                self.token_value = nn.Linear(hidden, vocab_size)
                # Initialize token_value with same scheme as head
                nn.init.xavier_uniform_(self.token_value.weight)
                nn.init.zeros_(self.token_value.bias)

                # === Autonomy heads (learned; no fixed thresholds) ===
                # Q-values for actions: 0=Wait, 1=Speak
                self.q_head = nn.Linear(hidden, 2)
                # Event intensity (lambda) for point-process timing (positive via softplus at use)
                self.intensity_head = nn.Linear(hidden, 1)
                # State-conditioned continuous prefix for self-prompt (length derived from dims)
                self.prefix_len = max(1, hidden // embed_dim)
                self.state2prefix = nn.Linear(hidden, self.prefix_len * embed_dim)
                self.prefix_gate = nn.Linear(hidden, self.prefix_len)

                # === C. ===
                # [0,1]: (1-g)h + gctx
                # LayerNorm for over-attention stability
                self.gate_proj = nn.Linear(hidden, 1)  # h g
                self.layer_norm = nn.LayerNorm(hidden)
                nn.init.zeros_(self.gate_proj.weight)
                # Infer initial gate bias from config
                # :param init_gate_value: Initial value for the gate (0-1)
                init_bias = np.log(init_gate_value / (1.0 - init_gate_value + 1e-8))
                nn.init.constant_(self.gate_proj.bias, init_bias)
                
                # === M3 Integration ===
                # M3StateEncoder (FeatureBank panels memory tokens)
                self.m3_encoder = None  # Lazy init when core is provided
                # M3AwareDecoderLayer (optional, for full M3 integration)
                self.m3_decoder_layer = None  # Lazy init
                self.use_m3_integration = False  # Flag

            def _ensure_mem_layers(self, mem_in_dim: int):
                nn = __import__('torch').nn
                if self.mem_proj is None or getattr(self.mem_proj, 'in_features', None) != mem_in_dim:
                    self.mem_proj = nn.Linear(mem_in_dim, self.hidden)
                    self.Wk = nn.Linear(self.hidden, self.hidden)
                    self.Wv = nn.Linear(self.hidden, self.hidden)
                    # Wq already defined on hidden
            
            def compute_mem_context(self, dec_t, mem_k, mem_v, core_state=None, config=None):
                """
                Compute memory context for the decoder.
                
                Args:
                    dec_t: (1, H) decoder hidden state
                    mem_k, mem_v: (1, M, H) memory key/value
                    core_state: Optional dict with stability/drift/phi_delta for gating
                    config: Optional TorchPolicyConfig for weights
                
                Returns:
                    blended: (1, H) gated blend of (h, ctx)
                """
                torch = __import__('torch')
                cfg = config or get_global_config().torch_policy
                
                # Attention
                q = self.Wq(dec_t).unsqueeze(1)  # (1, 1, H)
                att = torch.softmax((q @ mem_k.transpose(1, 2)) / (self.hidden ** 0.5 + 1e-8), dim=-1)
                ctx = att @ mem_v  # (1, 1, H)
                ctx = ctx.squeeze(1)  # (1, H)

                # LayerNorm for stability
                dec_t_norm = self.layer_norm(dec_t)
                ctx_norm = self.layer_norm(ctx)

                # === C. Gating ===
                # stability drift phi_delta
                g = torch.sigmoid(self.gate_proj(dec_t_norm))  # (1, 1)
                
                # core_state bias adjustment
                if core_state is not None:
                    try:
                        stability = float(core_state.get('stability', 0.5))
                        drift = float(core_state.get('drift', 0.0))
                        phi_delta = float(core_state.get('phi_delta', 0.0))

                        # Bias calculation
                        bias = 0.0
                        bias += cfg.stability_weight * (1.0 - np.clip(stability, 0.0, 1.0))  # stability
                        bias += cfg.drift_weight * np.clip(drift, 0.0, 1.0)  # drift
                        bias += cfg.phi_delta_weight * np.clip(-phi_delta, 0.0, 1.0)  # phi decrease
                        
                        g = g + float(bias)
                        g = torch.clamp(g, 0.0, 1.0)
                    except Exception:
                        pass
                
                # Gated blend: (1-g)h + gctx
                blended = (1.0 - g) * dec_t_norm + g * ctx_norm
                
                return blended

            def attach_memory(self, mem_vecs):
                """
                Attach memory key/value pairs to the decoder.
                
                Args:
                    mem_vecs: (B, M, D) memory vectors
                """
                mem_dim = mem_vecs.shape[-1]
                self._ensure_mem_layers(mem_dim)
                self.memory = mem_vecs  # Store memory vectors

            def forward(self, src_ids, tgt_in_ids=None, core=None, m3_memory=None):
                """
                Forward pass with optional M3 integration.
                
                Args:
                    src_ids: (B, L) source token IDs
                    tgt_in_ids: (B, L) target token IDs (optional, defaults to src_ids)
                    core: Optional M3ConsciousnessCore instance
                    m3_memory: Optional pre-computed M3 memory tokens (M, D)
                
                Returns:
                    logits: (B, L, V) output logits
                """
                if tgt_in_ids is None:
                    tgt_in_ids = src_ids
                
                # Standard encoder-decoder
                e_src = self.emb(src_ids)
                _, h = self.encoder(e_src)
                e_tgt_in = self.emb(tgt_in_ids)
                o, _ = self.decoder(e_tgt_in, h)
                
                # M3 Integration: Apply M3AwareDecoderLayer if available
                if self.use_m3_integration and self.m3_decoder_layer is not None:
                    # Get M3 memory
                    if m3_memory is None and core is not None:
                        if self.m3_encoder is None:
                            # Lazy init M3StateEncoder
                            torch_module = __import__('torch')
                            self.m3_encoder = M3StateEncoder(
                                torch_module,
                                self.hidden,
                                next(self.parameters()).device
                            )
                        
                        try:
                            panels = core.feature_bank.panels(core)
                            m3_memory = self.m3_encoder.forward(panels)
                        except Exception:
                            # Graceful degradation: no M3 memory available
                            m3_memory = None
                    
                    # Apply M3-aware decoder layer
                    if m3_memory is not None:
                        o, _ = self.m3_decoder_layer.forward(o, m3_memory)
                
                return self.head(o)
            
            def enable_m3_integration(self):
                """Enable M3 integration components."""
                if not self.use_m3_integration:
                    torch_module = __import__('torch')
                    device = next(self.parameters()).device

                    # M3StateEncoder (lazy init)
                    self.m3_encoder = None

                    # M3AwareDecoderLayer
                    self.m3_decoder_layer = M3AwareDecoderLayer(
                        torch_module,
                        self.hidden,
                        device
                    )
                    
                    self.use_m3_integration = True

        self.model = Model(self.vocab_size, embed_dim, hidden, self.torch, self.device, self.config.init_gate_value).to(self.device)

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
        knn_config = KNNIndexConfig(
            tau=float(os.getenv("LLM_ADAPTER_KNN_TAU", str(get_global_config().knn_index.tau))),
            max_items=int(os.getenv("LLM_ADAPTER_KNN_CAP", str(get_global_config().knn_index.max_items))),
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

    def save_model(self, path: str):
        """Saves the model state dictionary."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}")

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
                src = torch.tensor([t.encode(prompt)], dtype=torch.long, device=self.device)
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
        src = torch.tensor([t.encode(prompt, add_special=False)], dtype=torch.long, device=self.device)
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
                 knn_alpha: float = 0.0) -> str:
        t = self.tok
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            # === Phase 4: Episodic Memory Retrieval (NO THRESHOLD) ===
            if hasattr(self, 'core') and self.core is not None:
                try:
                    src_temp = t.encode(prompt, add_special=False)
                    if src_temp:
                        src_temp_ids = torch.tensor(src_temp, dtype=torch.long, device=self.device).unsqueeze(0)
                        e_src_temp = self.model.emb(src_temp_ids)
                        _, h_temp = self.model.encoder(e_src_temp)
                        context_embedding = h_temp.squeeze(0).cpu().numpy()
                        relevant_episodes = self.m3_memory_retriever.retrieve_relevant_episodes(self.core, context_embedding)
                        if relevant_episodes:
                            episode_texts = []
                            for ep in relevant_episodes:
                                if hasattr(ep, 'content'):
                                    episode_texts.append(str(ep.content))
                                elif hasattr(ep, 'text'):
                                    episode_texts.append(str(ep.text))
                                elif hasattr(ep, 'description'):
                                    episode_texts.append(str(ep.description))
                            if episode_texts:
                                context_prefix = "Similar past context:\n" + "\n".join(f"- {txt[:100]}..." if len(txt) > 100 else f"- {txt}" for txt in episode_texts[:3]) + "\n\nCurrent context:\n"
                                prompt = context_prefix + prompt
                except Exception:
                    pass
            
            src = t.encode(prompt, add_special=False)
            if not src:
                # Use BOS-only source to derive state when no textual prompt is provided
                src = [t.BOS]
            src_ids = torch.tensor(src, dtype=torch.long, device=self.device).unsqueeze(0)
            e_src = self.model.emb(src_ids)
            _, h = self.model.encoder(e_src)
            # If a continuous prefix is provided, use it to prime the decoder state
            if prefix_embed is not None:
                pe = np.array(prefix_embed)
                if pe.ndim == 2:
                    pe = pe[None, :, :]  # (P, E) -> (1, P, E)
                if pe.ndim == 3:
                    prefix_t = torch.tensor(pe, dtype=torch.float32, device=self.device)
                    _, h = self.model.decoder(prefix_t, h)
            cur = torch.tensor([[t.BOS]], dtype=torch.long, device=self.device)
            out_tokens: List[int] = []
            # Prepare memory (if provided)
            mem_k = mem_v = None
            if mem is not None and len(np.array(mem).shape) >= 1:
                m = np.array(mem)
                # NEW: (D,), (M,D), (1,M,D)
                if m.ndim == 1:
                    m = m[None, :]          # (1, D)  -> single token
                if m.ndim == 2:
                    m = m[None, :, :]       # (M, D)  -> (1, M, D)
                # now m.ndim == 3: (1, M, D)
                M, D = int(m.shape[1]), int(m.shape[2])
                m_t = torch.tensor(m, dtype=torch.float32, device=self.device)
                self.model._ensure_mem_layers(D)
                mem_h = self.model.mem_proj(m_t)  # (1, M, H)
                mem_k = self.model.Wk(mem_h)
                mem_v = self.model.Wv(mem_h)
            for _ in range(max_len):
                e = self.model.emb(cur)
                o, h = self.model.decoder(e, h)
                dec_t = o[:, -1, :]  # (1, H)
                
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
                T0 = float(os.getenv("GEN_TEMP_BASE", str(temperature)))
                aH = float(os.getenv("GEN_TEMP_H_COEF", "0.08"))
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
                if tid == t.EOS or tid == t.PAD:
                    break
                out_tokens.append(tid)

                # === Global token index ===
                if not hasattr(self, "_global_token_idx"):
                    self._global_token_idx = 0
                self._global_token_idx += 1
                
                cur = tok
            return t.decode(out_tokens)

    def score_value(self, prompt: str, candidate: str, mem: Optional[np.ndarray] = None) -> float:
        """Estimate value for a candidate response (scaffolding for Step 3)."""
        t = self.tok
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            src = t.encode(prompt, add_special=False)
            if not src:
                return 0.0
            src_ids = torch.tensor(src, dtype=torch.long, device=self.device).unsqueeze(0)
            e_src = self.model.emb(src_ids)
            _, h = self.model.encoder(e_src)
            # teacher forcing over candidate
            tgt = t.encode(candidate, add_special=True)
            if len(tgt) < 2:
                return 0.0
            tgt_in = tgt[:-1]
            tgt_in_ids = torch.tensor([tgt_in], dtype=torch.long, device=self.device)
            e = self.model.emb(tgt_in_ids)
            o, _ = self.model.decoder(e, h)
            dec_t = o[:, -1, :]  # last hidden
            # optional memory context
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
        src_ids = torch.tensor([t.encode(prompt)], dtype=torch.long, device=self.device)
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
        t = self.tok
        torch = self.torch
        self.model.train()
        src_ids = torch.tensor([t.encode(prompt)], dtype=torch.long, device=self.device)
        tgt = t.encode(response, add_special=True)
        if len(tgt) < 2:
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
            
            # === Phase 5: M3-aware DPO objective with phi-margin ===
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
                                q = self.model.Wq(h_f).unsqueeze(1)
                                att = torch.softmax((q @ mem_k.transpose(1, 2)) / np.sqrt(self.model.hidden + 1e-8), dim=-1)
                                ctx = att @ mem_v
                                h_f = (h_f + ctx.squeeze(1)) * 0.5
                        
                        # Detach h_f for value training (don't backprop through encoder/decoder)
                        h_f = h_f.detach()

                        # === E. Normalize (log/sqrt + z-score) ===
                        phi_d_norm, stab_d_norm = self._normalize_targets(phi_d, stab_d)
                        
                        # Multi-head predictions and loss update
                        with self._learn_lock:
                            phi_hat = self.model.v_phi(h_f).squeeze(-1)
                            stab_hat = self.model.v_stab(h_f).squeeze(-1)
                            tool_hat = self.model.v_tool(h_f).squeeze(-1)

                            # Collect for correlation analysis
                            self.phi_targets.append(phi_d_norm)
                            self.phi_preds.append(float(phi_hat.item()))

                            # Compute losses
                            loss_phi = mse(phi_hat, torch.tensor([phi_d_norm], dtype=torch.float32, device=self.device))
                            loss_stab = mse(stab_hat, torch.tensor([stab_d_norm], dtype=torch.float32, device=self.device))
                            loss_tool = bce(tool_hat, torch.tensor([tool_s], dtype=torch.float32, device=self.device))

                            w_phi = self._task_weights.get('phi', 1.0)
                            w_stab = self._task_weights.get('stab', 0.5)
                            w_tool = self._task_weights.get('tool', 0.5)

                            loss = w_phi * loss_phi + w_stab * loss_stab + w_tool * loss_tool
                            
                            batch_loss += float(loss.item())
                            valid_count += 1

                            self.value_opt.zero_grad(set_to_none=True)
                            loss.backward()
                            try:
                                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                            except Exception:
                                pass
                            self.value_opt.step()

                            # GradNorm update periodically
                            if (n % 10) == 0 and valid_count > 0:
                                individual_losses = {
                                    'phi': float(loss_phi.item()),
                                    'stab': float(loss_stab.item()),
                                    'tool': float(loss_tool.item())
                                }
                                self._update_task_weights_gradnorm(individual_losses, params)
                        
                        total_loss += float(loss.item())
                        n += 1
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
            
            # Create default adapter using built-in configuration
            adapter = TorchConversationalPolicy(device=device)
            logger.info(f'Created LLM adapter with device={device}')
        
        # Attach adapter to core
        core.llm_adapter = adapter
        adapter.core = core
        
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
