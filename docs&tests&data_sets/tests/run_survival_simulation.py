"""
M3-Binary Brain: Long-term Survival Simulation ("David vs Goliath")
-------------------------------------------------------------------
Simulates an agent's life over multiple 'Days'.
The goal is to verify if the agent learns to distinguish 'Predator' (Goliath)
from 'Safe' events purely through Plasticity and Sleep, without backprop.

Scenario:
- **Environment**: A world with random noise and a specific "Predator" signal.
- **Agent**: M3 with 1.58-bit Plastic Brain.
- **Cycle**:
    1. Wake Phase: Encounter events -> React -> Update Traces (if Arousal is high).
    2. Sleep Phase: Consolidate Traces -> Weights.
- **Success Criteria**: 
    - Reaction to Predator increases over days.
    - Reaction to Noise remains low (discriminative learning).
"""

import sys
import os
import torch
import numpy as np
import json
import random
from tqdm import tqdm

# Adjust path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from llm_adapter.core import PlasticBrainPolicy

# Output config
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUTPUT_DIR, 'survival_simulation_report.json')

# Simulation Constants
NUM_DAYS = 15          # How many sleep cycles to simulate
EVENTS_PER_DAY = 30    # Increased events for better statistics
PREDATOR_RARITY = 0.1  # Rarer predator makes learning more critical
LEARNING_RATE = 0.05   # Lower learning rate to prevent weight explosion and allow smoother convergence

def run_simulation():
    print(f"=== Starting Survival Simulation ({NUM_DAYS} Days) ===")
    
    # 1. Initialize Brain
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = PlasticBrainPolicy(device=device)
    
    # Force high consolidation rate for simulation visibility
    policy.model.config.linear_config.learning_rate = LEARNING_RATE
    
    layer = policy.model.layer_in
    vocab_size = policy.model.vocab_size
    embed_dim = policy.model.config.embed_dim
    
    # 2. Define World Entities
    # Fixed Pattern for "Predator" (Goliath)
    # We create a random but fixed vector that represents the visual signature of the predator
    predator_sig = torch.randn(1, 1, embed_dim).to(device)
    predator_sig = predator_sig / predator_sig.norm() # Normalize
    
    history = {
        'config': {
            'num_days': NUM_DAYS,
            'events_per_day': EVENTS_PER_DAY,
            'learning_rate': LEARNING_RATE
        },
        'daily_stats': []
    }
    
    # 3. Time Loop
    for day in range(1, NUM_DAYS + 1):
        print(f"\n--- Day {day} ---")
        
        daily_encounters = []
        predator_encounters = 0
        damage_taken = 0
        
        # --- Wake Phase (Experience) ---
        policy.model.train() # Enable plasticity
        
        for event_idx in range(EVENTS_PER_DAY):
            is_predator = random.random() < PREDATOR_RARITY
            
            if is_predator:
                input_vec = predator_sig
                predator_encounters += 1
                # Simulation: Seeing a predator causes High Arousal (Fear)
                arousal = 5.0 
                label = "Predator"
            else:
                # Random noise (Safe event)
                noise = torch.randn(1, 1, embed_dim).to(device)
                input_vec = noise / noise.norm()
                arousal = 0.1 # Boredom / Calm
                label = "Safe"
            
            # Forward Pass (React)
            # update_trace parameter is not explicitly defined in layer(Forward), but handled by context or internal state
            # m3_plastic_policy.py uses 'affect_gating' in forward call.
            # layers.py PlasticBitLinear expects 'affect_gating' and checks 'self.training' for update.
            
            # Create a mock affect_state tensor if needed, but PlasticBitLinear.forward takes float scalar 'affect_gating'??
            # Checking layers.py: def forward(self, x: torch.Tensor, affect_gating: float = 1.0)
            
            # Ensure model is in training mode for plasticity
            policy.model.train()
            
            with torch.no_grad():
                # We need to call the layer directly as done in previous lines: layer = policy.model.layer_in
                # Correct call: layer(input_vec, affect_gating=arousal)
                out = layer(input_vec, affect_gating=arousal)
                reaction = out.norm().item()
                
            daily_encounters.append({
                'type': label,
                'reaction': reaction,
                'arousal': arousal
            })
            
        # Analyze Day's Performance
        avg_predator_reaction = np.mean([e['reaction'] for e in daily_encounters if e['type'] == 'Predator']) if predator_encounters > 0 else 0
        avg_safe_reaction = np.mean([e['reaction'] for e in daily_encounters if e['type'] == 'Safe'])
        
        print(f"  > Encounters: {predator_encounters} Predators, {EVENTS_PER_DAY - predator_encounters} Safe")
        print(f"  > Avg Reaction: Predator={avg_predator_reaction:.4f} vs Safe={avg_safe_reaction:.4f}")
        
        # Check trace accumulation before sleep
        trace_sum = layer.trace.abs().sum().item()
        print(f"  > Accumulated Stress (Traces): {trace_sum:.4f}")
        
        # --- Sleep Phase (Consolidation) ---
        print("  > Sleeping... ", end="")
        policy.model.sleep() # Consolidate Traces -> Weights
        print("Done. (Traces reset)")
        
        # Record Stats
        history['daily_stats'].append({
            'day': day,
            'predator_reaction': float(avg_predator_reaction),
            'safe_reaction': float(avg_safe_reaction),
            'trace_before_sleep': float(trace_sum)
        })

    # 4. Summary & Save
    print("\n=== Simulation Complete ===")
    
    # Calculate improvement
    start_reaction = history['daily_stats'][0]['predator_reaction']
    end_reaction = history['daily_stats'][-1]['predator_reaction']
    improvement = end_reaction - start_reaction
    
    print(f"Predator Reaction Change: {start_reaction:.4f} -> {end_reaction:.4f} (Delta: {improvement:+.4f})")
    
    # Save
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"Report saved to: {REPORT_PATH}")

if __name__ == "__main__":
    run_simulation()
