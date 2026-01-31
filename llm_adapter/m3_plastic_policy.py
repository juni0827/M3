import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import logging

from llm_adapter.config import get_global_config, M3PlasticPolicyConfig
from llm_adapter.layers import PlasticBitLinear
from llm_adapter.tokenization import AutoTokenizer

logger = logging.getLogger('m3_plastic_policy')

class M3PlasticPolicy(nn.Module):
    """
    M3-Binary Brain Policy.
    - Uses PlasticBitLinear for 1.58-bit weights + Hebbian Plasticity.
    - Neuromodulated by AffectKernel states.
    - Replaces TorchConversationalPolicy for 'learning from scratch'.
    """
    def __init__(self, config: Optional[M3PlasticPolicyConfig] = None, device: Optional[str] = None):
        super().__init__()
        
        # Load config
        self.config = config or get_global_config().plastic_policy
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Tokenizer
        try:
            self.tok = AutoTokenizer.from_config(get_global_config().tokenizer)
        except Exception as e:
            logger.warning(f"Tokenizer init failed: {e}, using default")
            self.tok = AutoTokenizer()
            
        self.vocab_size = self.tok.vocab_size
        embed_dim = self.config.embed_dim
        hidden_dim = self.config.hidden_dim
        
        # Architecture
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.tok.PAD)
        
        # Input projection (Plastic)
        self.layer_in = PlasticBitLinear(embed_dim, hidden_dim, 
                                         trace_decay=self.config.linear_config.trace_decay)
        
        # Recurrent Plastic Logic (The "Brain")
        # We model this as a stack of plastic layers that feed into themselves or each other
        self.layers = nn.ModuleList([
            PlasticBitLinear(hidden_dim, hidden_dim, 
                             trace_decay=self.config.linear_config.trace_decay)
            for _ in range(self.config.num_layers)
        ])
        
        # Output Projection (High Precision for probable token selection)
        self.layer_out = nn.Linear(hidden_dim, self.vocab_size)
        
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.to(self.device)
        
        # State for recurrence (persistent across steps)
        self.reset_state()

    def reset_state(self):
        """Reset the short-term hidden state (not the Hebbian traces)."""
        self._hidden_state = torch.zeros(1, self.config.hidden_dim, device=self.device)

    def forward(self, input_ids: torch.Tensor, 
                hidden_state: Optional[torch.Tensor] = None, 
                affect_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass. 
        Note: Hebbian updates happen HERE if self.training is True.
        """
        # Calculate affect gating
        gating = 1.0
        if affect_state is not None:
            # Simple heuristic: Higher arousal/magnitude = higher plasticity
            # affect_state is typically a vector (e.g. 5 dims). Norm gives intensity.
            arousal = affect_state.norm(dim=-1).mean().item()
            # Map typical arousal (0~1) to plastic scale (0.5 ~ 3.0)
            base_scale = self.config.linear_config.neuromodulation_scale
            # If arousal is high, we learn faster.
            gating = max(0.5, min(5.0, 1.0 + (arousal * base_scale * 10.0)))
        
        x = self.embedding(input_ids) # (B, S, E)
        batch_size, seq_len, _ = x.shape
        
        if hidden_state is None:
            hidden_state = self._hidden_state.repeat(batch_size, 1) if self._hidden_state.shape[0] == 1 else self._hidden_state
            if hidden_state.shape[0] != batch_size:
                hidden_state = torch.zeros(batch_size, self.config.hidden_dim, device=self.device)

        curr_h = hidden_state
        outputs = []

        # Manual recurrence logic to allow step-by-step plasticity
        for t in range(seq_len):
            xt = x[:, t, :] # (B, E)
            
            # Input projection
            h_in = self.layer_in(xt, affect_gating=gating) # (B, H)
            
            # Recurrent dynamics
            h_combined = h_in
            
            # Recurrent/Deep processing relies on previous state `curr_h`
            for layer in self.layers:
                # Recurrent connection: Previous H -> Current H'
                h_ff = layer(curr_h, affect_gating=gating)
                h_combined = h_combined + h_ff
            
            curr_h = self.norm(self.act(curr_h + h_combined)) # Residual update
            outputs.append(curr_h)
        
        self._hidden_state = curr_h.detach() # Update persistent state
        
        outputs_stack = torch.stack(outputs, dim=1) # (B, S, H)
        logits = self.layer_out(outputs_stack)
        
        return logits, curr_h

    def sample(self, obs: np.ndarray, 
               affect_state: Optional[np.ndarray] = None,
               temperature: float = 1.0, 
               top_k: int = 50, 
               **kwargs) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """
        Interface for M3ConsciousnessCore integration.
        Returns: (action_array, logp, value_array, entropy)
        """
        self.eval() # Ensure standard eval mode, BUT we manually control plasticity via internal logic if needed
        # Actually, for 'Online Learning', we want self.train() to be True usually, or pass update_trace=True.
        # But `PlasticBitLinear` uses `self.training` check.
        # We should set `self.train()` if we want plasticity ON during this "sample" phase (which is 'Inference' in user terms but 'Learning' in M3 terms).
        self.train() # Enable Plasticity!
        
        with torch.no_grad(): # But no gradients, this is Hebbian
            obs_tensor = torch.from_numpy(obs).float().to(self.device).unsqueeze(0) # (1, D)
            
            affect_tensor = None
            if affect_state is not None:
                affect_tensor = torch.from_numpy(affect_state).float().to(self.device).unsqueeze(0)
            
            # Feed "BOS" token to prompt the brain
            input_ids = torch.tensor([[self.tok.BOS]], device=self.device)
            
            # Forward pass (triggers Hebbian updates in layers)
            logits, _ = self.forward(input_ids, hidden_state=self._hidden_state, affect_state=affect_tensor)
            
            # Sampling logic
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-4)
            
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            action_idx = next_token.item()
            logp = torch.log(probs[0, action_idx] + 1e-8).item()
            
            # Dummy value/entropy
            value = np.array([0.0]) 
            entropy = 0.0
            
            return np.array([action_idx]), logp, value, entropy

    def sleep(self):
        """
        Trigger Deep Sleep cycle for Memory Consolidation.
        Moves all Traces to Weights in all plastic layers.
        """
        logger.info("M3 Entering Deep Sleep: Consolidating Memories...")
        
        lr = self.config.linear_config.learning_rate
        
        # 1. Consolidate Input Layer
        self.layer_in.consolidate(learning_rate=lr)
        
        # 2. Consolidate Recurrent Layers
        for layer in self.layers:
            layer.consolidate(learning_rate=lr)
            
        logger.info("Memory Consolidation Complete. Traces reset.")

    def learn_from_text(self, text: str, arousal: float = 1.0, sleep_after: bool = False):
        """
        Offline Learning (Manual Study).
        Feeds text into the brain, causing Hebbian updates in traces.
        Optinally sleeps (consolidates) immediately.
        
        Args:
            text: Content to learn
            arousal: Importance of this text (higher = learned faster)
            sleep_after: Whether to trigger consolidation immediately
        """
        self.train() # Enable plasticity
        
        # 1. Tokenize
        tokens = self.tok.encode(text) # List[int]
        # Chunking if too long? For now, we assume reasonable length or truncate
        max_len = 1024
        if len(tokens) > max_len:
            logger.info(f"Breaking long text ({len(tokens)} tokens) into chunks...")
            
        # Process in chunks of max_len
        for i in range(0, len(tokens), max_len):
            chunk = tokens[i : i + max_len]
            input_ids = torch.tensor([chunk], device=self.device) # (1, Seq)
            
            # Create synthetic affect state relative to arousal
            # Assuming affect dimension 5, we fill it with 'arousal' intensity
            affect_tensor = torch.ones(1, self.config.affect_dim, device=self.device) * arousal
            
            # 2. Forward Pass (Learn)
            # This triggers internal Hebbian trace updates
            with torch.no_grad():
                self.forward(input_ids, affect_state=affect_tensor)
                
            logger.debug(f"Learned chunk {i//max_len + 1} with arousal {arousal}")

        # 3. Consolidate if requested
        if sleep_after:
            self.sleep()
