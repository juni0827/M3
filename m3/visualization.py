from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from PIL import Image, ImageOps
from functools import lru_cache

# Optional scipy for efficient Gaussian blur (DoG)
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

# =============================================================================
# 1. Utility: Hilbert Curve Mapping (Space-Preserving)
# =============================================================================

@lru_cache(maxsize=32)
def get_hilbert_map(side: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-calculates Hilbert curve coordinates for a given grid side length.
    Returns: (xs, ys) integer arrays
    """
    size = side * side
    xs = np.zeros(size, dtype=np.int32)
    ys = np.zeros(size, dtype=np.int32)
    
    for i in range(size):
        x = y = 0
        t = i
        s = 1
        while s < side:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        xs[i] = x
        ys[i] = y
    return xs, ys

def hilbert_index_to_xy(n: int, d: int):
    """Legacy single-point conversion (kept for compatibility)"""
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def vector_to_grid(vec: np.ndarray, side: int) -> np.ndarray:
    """
    Maps a 1D vector to a 2D grid using Hilbert curve to preserve locality.
    Optimized with caching.
    """
    grid = np.zeros((side, side), dtype=np.float32)
    L = min(vec.size, side * side)
    
    if L > 0:
        xs, ys = get_hilbert_map(side)
        # Use cached coordinates for fast mapping
        grid[ys[:L], xs[:L]] = vec[:L]
        
        # Fill remaining space with mean value if vector is shorter than grid
        if L < side * side:
            remaining = vec[L:]
            fill_val = float(remaining.mean()) if remaining.size > 0 else 0.0
            grid[ys[L:], xs[L:]] = fill_val
            
    return grid

# =============================================================================
# 2. BioRetina: Biological Retina Simulation
# =============================================================================

@dataclass
class Retinizer:
    """
    [BioRetina]
    Simulates the biological retina's Ganglion Cells.
    
    Key Features:
    1. Temporal Integration: Retains 'afterimages' to detect motion/change.
    2. Center-Surround (DoG): Enhances edges and local contrast (ON/OFF cells).
    3. Luminance Adaptation: Normalizes input to handle varying light levels.
    """
    target_size: tuple[int, int] = (128, 128)
    temporal_decay: float = 0.6  # 0.0=Instant, 1.0=Frozen
    
    # Internal state for temporal integration
    _prev_retina: np.ndarray | None = field(default=None, init=False)

    def __call__(self, arr: np.ndarray | None) -> np.ndarray:
        if arr is None:
            return np.zeros(self.target_size, dtype=np.float32)

        # --- Stage 1: Photoreceptor Transduction (Input -> Gray -> Resize) ---
        try:
            a = np.asarray(arr)
            # Handle RGB
            if a.ndim == 3 and a.shape[2] >= 3:
                # Human eye sensitivity weights (Rec. 601)
                a = np.dot(a[..., :3], [0.299, 0.587, 0.114])
            
            # Normalize
            if a.max() > 1.0:
                a = a / 255.0
            
            # Resize using PIL (High quality Lanczos)
            im = Image.fromarray((a * 255).astype(np.uint8))
            im = ImageOps.fit(im, self.target_size, method=Image.Resampling.LANCZOS)
            current_frame = np.asarray(im).astype(np.float32) / 255.0
            
        except Exception:
            return np.zeros(self.target_size, dtype=np.float32)

        # --- Stage 2: Temporal Integration (Motion Sensitivity) ---
        if self._prev_retina is None:
            self._prev_retina = current_frame
            
        # I_t = (1-alpha)*I_new + alpha*I_old
        integrated_frame = (1.0 - self.temporal_decay) * current_frame + \
                           self.temporal_decay * self._prev_retina
        self._prev_retina = integrated_frame

        # --- Stage 3: Ganglion Cell Processing (DoG Filter) ---
        # Difference of Gaussians approximates the Center-Surround receptive field
        if gaussian_filter is not None:
            # Excitatory center
            center = gaussian_filter(integrated_frame, sigma=0.5)
            # Inhibitory surround
            surround = gaussian_filter(integrated_frame, sigma=2.0)
            # Response
            dog_response = center - surround
        else:
            # Fallback if scipy is missing: simple unsharp mask
            mean_val = integrated_frame.mean()
            dog_response = integrated_frame - mean_val

        # Normalize response to 0.0-1.0 range (0.5 = neutral)
        # Ganglion cells fire above baseline for ON, below for OFF
        retina_out = 0.5 + 2.0 * dog_response
        retina_out = np.clip(retina_out, 0.0, 1.0)

        return retina_out

# =============================================================================
# 3. ActiveVisualCortex: Self-Organizing Visual Cortex
# =============================================================================

@dataclass
class FeatureSummarizer:
    """
    [Affective Visual Cortex]
    Simulates V1 Simple Cells with Affective Modulation.
    
    Key Concept:
    - Visual memories are tagged with emotional valence (Affect).
    - Learning is modulated by the current Drive state (e.g., Curiosity).
    - "Neurons that fire together, wire together" (Hebbian), but emotion acts as the glue.
    """
    patch: int = 8
    n_filters: int = 16  # Number of cortical columns/filters
    base_learning_rate: float = 0.005
    
    # Learned synaptic weights (filters)
    _filters: np.ndarray | None = field(default=None, init=False)
    
    # Affective Tags: (n_filters, affect_dim) - Each filter remembers an emotion
    # Assuming 5D affect space from RewardSystem
    _affect_tags: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        # Initialize filters with small random noise
        # Shape: (n_filters, patch, patch)
        self._filters = np.random.normal(0, 0.1, (self.n_filters, self.patch, self.patch))
        self._affect_tags = np.zeros((self.n_filters, 5), dtype=np.float32)

    def __call__(self, retina: np.ndarray, affect_vector: np.ndarray | None = None, drive_state: dict | None = None) -> np.ndarray:
        """
        Args:
            retina: Visual input
            affect_vector: Current emotional state (5D vector) from AffectKernel.
            drive_state: Current drive activations (e.g. {'curiosity': 0.8}).
        """
        H, W = retina.shape
        ph, pw = self.patch, self.patch
        
        # --- Step 1: Receptive Field Extraction ---
        # Extract all non-overlapping patches
        h_steps = H // ph
        w_steps = W // pw
        
        # Crop to multiple of patch size
        crop = retina[:h_steps*ph, :w_steps*pw]
        
        if crop.size == 0:
            return np.zeros(self.n_filters, dtype=np.float32)
        
        # Reshape to (N_patches, patch_h, patch_w)
        # (h_steps, ph, w_steps, pw) -> (h_steps, w_steps, ph, pw) -> (N, ph, pw)
        patches = crop.reshape(h_steps, ph, w_steps, pw).transpose(0, 2, 1, 3).reshape(-1, ph, pw)
        
        # Local Contrast Normalization (LCN) per patch
        # V1 neurons respond to contrast structure, not absolute brightness
        p_mean = patches.mean(axis=(1, 2), keepdims=True)
        p_std = patches.std(axis=(1, 2), keepdims=True) + 1e-6
        patches_norm = (patches - p_mean) / p_std

        # Initialize filters if needed (lazy init)
        if self._filters is None:
             self._filters = np.random.normal(0, 0.1, (self.n_filters, self.patch, self.patch))
             self._affect_tags = np.zeros((self.n_filters, 5), dtype=np.float32)

        # --- Step 2: Attention Modulation (Drive-based) ---
        # If 'curiosity' drive is high, boost response to novel/complex patterns
        attention_gain = 1.0
        if drive_state and 'curiosity' in drive_state:
            # Curiosity amplifies visual processing
            attention_gain += float(drive_state['curiosity']) * 0.5

        # --- Step 3: Cortical Response (Feedforward) ---
        # Compute dot product (similarity) between every patch and every filter
        # response[n, k] = patch[n] . filter[k]
        response = np.einsum('nij,kij->nk', patches_norm, self._filters)
        
        # --- Step 4: Lateral Inhibition & Competition (WTA) ---
        # For each patch, only the best matching filter "fires"
        winners = np.argmax(np.abs(response), axis=1) # (N_patches,)
        
        # --- Step 5: Affective Learning (Emotional Tagging) ---
        # If strong emotion is present, associate it with the active visual pattern
        current_affect = np.asarray(affect_vector, dtype=np.float32) if affect_vector is not None else np.zeros(5, dtype=np.float32)
        affect_intensity = float(np.linalg.norm(current_affect))
        
        # Learning rate boosted by emotional intensity (Flashbulb Memory effect)
        # Effective LR = Base * (1 + Emotion) * Attention
        effective_lr = self.base_learning_rate * (1.0 + affect_intensity) * attention_gain
        
        # Calculate activation statistics for output
        activation_profile = np.zeros(self.n_filters, dtype=np.float32)
        
        for k in range(self.n_filters):
            # Find all patches that activated filter k
            mask = (winners == k)
            if np.any(mask):
                # Strength of activation
                count = np.sum(mask)
                avg_strength = np.mean(np.abs(response[mask, k]))
                activation_profile[k] = count * avg_strength
                
                # Hebbian Update (Pattern Learning)
                # "The filter moves towards the pattern it recognized"
                target_pattern = np.mean(patches_norm[mask], axis=0)
                self._filters[k] += effective_lr * (target_pattern - self._filters[k])
                
                # Homeostasis: Normalize filter energy
                f_norm = np.sqrt(np.sum(self._filters[k]**2)) + 1e-6
                self._filters[k] /= f_norm
                
                # Affective Update (Emotion Learning)
                # The filter 'learns' the emotion associated with this visual experience
                # Tag_new = Tag_old + lr * (Current_Affect - Tag_old)
                if affect_intensity > 0.1:
                    self._affect_tags[k] += effective_lr * (current_affect - self._affect_tags[k])

        return activation_profile

# =============================================================================
# 4. GlitchEncoder: Visual Feedback & Hallucination
# =============================================================================

@dataclass
class GlitchEncoder:
    """
    Visualizes the internal state by projecting it back onto the visual field.
    Creates 'hallucinatory' overlays based on system arousal and context.
    """
    out_size: tuple[int, int] = (768, 1024)
    one_bit: bool = True

    def __call__(self, retina: np.ndarray, context_vec: np.ndarray, arousal: float = 0.5) -> np.ndarray:
        H, W = self.out_size
        
        # Resize retina to output size (Nearest Neighbor for digital/glitch look)
        im = Image.fromarray((retina * 255).astype(np.uint8))
        im = im.resize((W, H), resample=Image.Resampling.NEAREST)
        arr = np.asarray(im).astype(np.float32) / 255.0
        
        # Process context vector
        ctx = np.asarray(context_vec, dtype=np.float32)
        if ctx.size == 0: ctx = np.zeros(1)
        # Normalize context
        ctx = (ctx - ctx.mean()) / (ctx.std() + 1e-6)
        
        # Create context map (vertical tiling)
        reps = int(np.ceil(H / max(1, ctx.size)))
        ctx_map = np.tile(ctx, reps)[:H]
        
        # --- Effect 1: Data-moshing (Pixel Displacement) ---
        A = np.clip(arousal, 0.0, 1.0)
        # Shift amount depends on arousal and context
        shifts = (ctx_map * 50 * A).astype(np.int32)
        
        # Apply row-wise shifts
        # (Vectorized approach isn't easy for variable shifts, list comp is fast enough here)
        shifted = np.array([np.roll(row, s) for row, s in zip(arr, shifts)])
        
        # --- Effect 2: Interference Patterns (Sine Waves) ---
        y_idx = np.arange(H).reshape(-1, 1)
        # Frequency modulated by context
        wave = np.sin(y_idx * 0.05 + ctx_map.reshape(-1, 1))
        
        # Combine
        out = shifted + (wave * 0.1 * A)
        
        # --- Effect 3: Quantization (1-bit or Grayscale) ---
        if self.one_bit:
            # Dithering-like threshold
            out = (out > 0.5).astype(np.uint8) * 255
        else:
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
            
        return out

__all__ = [
    'hilbert_index_to_xy',
    'vector_to_grid',
    'Retinizer',
    'FeatureSummarizer',
    'GlitchEncoder',
]
