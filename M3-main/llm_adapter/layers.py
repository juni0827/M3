import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger('plastic_bit_linear')

class PlasticBitLinear(nn.Module):
    """
    1.58-bit Plastic Linear Layer.
    - Ternary weights: -1, 0, 1 (1.58 bits: 2^1.58 ≈ 3 values)
    - Hebbian Plasticity: Maintains floating-point traces for learning
    - Neuromodulation: Affect-gating scales plasticity
    """
    
    def __init__(self, in_features: int, out_features: int, trace_decay: float = 0.95, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.trace_decay = trace_decay
        
        # [Optimization] Latent Weights in BFloat16/FP16 instead of FP32
        # RTX 4060 supports BFloat16 nicely. This halves the parameter memory.
        factory_kwargs = {'dtype': torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32}
        
        # Ternary weights: -1, 0, 1
        # Initialize with small random values, will be quantized to ternary
        self.weight = nn.Parameter(torch.randn(out_features, in_features, **factory_kwargs) * 0.1)
        
        # Bias (optional, but typically True for hidden layers)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # Hebbian trace: Optimize memory (FP16/BF16)
        # Use bfloat16 of float16 to save 50% memory on traces
        dtype = factory_kwargs['dtype']
            
        self.register_buffer('trace', torch.zeros(out_features, in_features, dtype=dtype))
        
        # Scale factor for ternary weights (learned)
        self.register_buffer('weight_scale', torch.ones(1))
        
    def forward(self, x: torch.Tensor, affect_gating: float = 1.0) -> torch.Tensor:
        """
        Forward pass with optional Hebbian update.
        
        Args:
            x: Input tensor (..., in_features)
            affect_gating: Neuromodulation factor (1.0 = normal, >1 = enhanced plasticity)
        
        Returns:
            Output tensor (..., out_features)
        """
        # Ensure operations happen in the weight's dtype (e.g. BFloat16)
        dtype = self.weight.dtype
        x_inner = x.to(dtype)
        scale = self.weight_scale.to(dtype)

        # Quantize weights to ternary: -1, 0, 1
        ternary_weight = self._quantize_weight(self.weight)
        
        # Linear operation
        y = F.linear(x_inner, ternary_weight * scale, self.bias)
        
        # Hebbian plasticity update (only during training)
        if self.training:
            self._hebbian_update(x_inner, y, affect_gating)
        
        return y
    
    def _quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize floating-point weights to ternary {-1, 0, 1}."""
        # Use sign with threshold for stability
        abs_weight = torch.mean(torch.abs(weight)) * 0.1  # Adaptive threshold
        ternary = torch.where(torch.abs(weight) > abs_weight, torch.sign(weight), torch.zeros_like(weight))
        return ternary
    
    def _hebbian_update(self, x: torch.Tensor, y: torch.Tensor, gating: float):
        """
        Hebbian learning: "Neurons that fire together wire together"
        Update trace based on input-output correlation.
        """
        # Detach inputs to prevent graph memory leak during trace update
        # We don't want to backprop through the trace update itself
        x_detached = x.detach()
        y_detached = y.detach()
        
        # Flatten batch dimensions for outer product
        x_flat = x_detached.view(-1, self.in_features)  # (batch_size, in_features)
        y_flat = y_detached.view(-1, self.out_features)  # (batch_size, out_features)
        
        # Hebbian rule: trace += outer(y, x) * gating
        # This correlates pre-synaptic (x) with post-synaptic (y) activity
        delta_trace = torch.einsum('bi,bo->oi', x_flat, y_flat) / x_flat.shape[0]
        
        # Apply gating and decay
        # Cast trace to calculations dtype (float32) then back to storage dtype (float16)
        current_trace = self.trace.to(delta_trace.dtype)
        updated_trace = current_trace * self.trace_decay + delta_trace * gating
        self.trace.copy_(updated_trace) # Save back as low precision
    
    def consolidate(self, learning_rate: float = 1e-4):
        """
        Memory consolidation: Move traces to weights.
        Called during "sleep" cycles.
        """
        # Update weights based on accumulated traces
        weight_update = self.trace * learning_rate
        
        # Add to current weights
        self.weight.data = self.weight.data + weight_update
        
        # Update scale factor based on weight magnitude
        # We want the effective weight (ternary * scale) to approximate the latent weight magnitude.
        # So scale should be the mean absolute value of the weights.
        weight_magnitude = torch.mean(torch.abs(self.weight))
        if weight_magnitude > 1e-6:
            self.weight_scale.data.fill_(weight_magnitude)
        else:
             self.weight_scale.data.fill_(1.0)
        
        # Reset trace after consolidation
        self.trace.zero_()
        
        logger.debug(f"Consolidated PlasticBitLinear {self.in_features}->{self.out_features}, "
                    f"weight_scale={self.weight_scale.item():.4f}")
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'trace_decay={self.trace_decay}, bias={self.bias is not None}'