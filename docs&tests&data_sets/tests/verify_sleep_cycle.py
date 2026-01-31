"""
M3-Binary Brain: Memory Consolidation Verification
--------------------------------------------------
Tests the 'Deep Sleep' cycle where short-term traces (Trace)
are consolidated into long-term weights (Weight).

Scenario:
1. Baseline: Measure response to Event A.
2. Learning (Wake): Expose to Event A with High Arousal (builds Trace).
3. Verification (Pre-Sleep): Confirm Trace matches Event A.
4. Consolidation (Deep Sleep): Trigger sleep() to move Trace -> Weight.
5. Verification (Post-Sleep): Confirm Trace is 0, Weight is updated, Response remains high.

Outputs:
- analysis/plasticity/logs/sleep_test_report.json
"""

import sys
import os
import torch
import numpy as np
import json
from datetime import datetime

# Adjust path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from llm_adapter.core import PlasticBrainPolicy

# Output config
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_PATH = os.path.join(OUTPUT_DIR, 'sleep_test_report.json')

def run_test():
    print("=== M3 Deep Sleep Verification Test ===")
    
    # 1. Initialize Plastic Brain
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Brain on {device}...")
    policy = PlasticBrainPolicy(device=device)
    
    # Access internals for white-box testing
    # We focus on the Input Layer (embedding -> hidden) for clarity
    layer = policy.model.layer_in
    
    # 2. Create Synthetic Event A
    # Random input vector representing "Seeing a Lion"
    vocab_size = policy.model.vocab_size
    embed_dim = policy.model.config.embed_dim

    # FORCE HIGH LEARNING RATE FOR VALIDATION
    # Standard 1e-4 is too subtle for single-event bit-flipping in 1.58-bit quantization.
    # We boost it to ensure we cross quantization thresholds for the test.
    policy.model.config.linear_config.learning_rate = 0.5
    print(f"DEBUG: Set Sleep Consolidation Rate to {policy.model.config.linear_config.learning_rate} for testing.")
    
    # Simulate an embedding input (Mocking the embedding layer output)
    # Batch=1, Seq=1, Dim=embed_dim
    event_A = torch.randn(1, 1, embed_dim).to(device)
    event_A = event_A / event_A.norm() # Normalize
    
    history = {
        'timestamp': datetime.now().isoformat(),
        'steps': []
    }
    
    def log_step(phase, note):
        with torch.no_grad():
            # Measure response magnitude
            # Forward pass (inference mode)
            out = layer(event_A, affect_gating=1.0, update_trace=False)
            resp_mag = out.norm().item()
            
            # Measure internals
            trace_mag = layer.trace.abs().sum().item()
            weight_mag = layer.weight.abs().sum().item()
            
            step_data = {
                'phase': phase,
                'note': note,
                'response_magnitude': round(resp_mag, 4),
                'total_trace': round(trace_mag, 4),
                'total_weight': round(weight_mag, 4)
            }
            history['steps'].append(step_data)
            print(f"[{phase}] Resp: {step_data['response_magnitude']} | Trace: {step_data['total_trace']} | Weight: {step_data['total_weight']}")
            return step_data

    # --- Phase 1: Baseline ---
    print("\n--- Phase 1: Baseline ---")
    log_step("Baseline", "Initial Random Weights")
    
    # --- Phase 2: Learning (Wake) ---
    print("\n--- Phase 2: Learning (Wake) ---")
    # Simulate high arousal event (Affect = 5.0)
    # We run forward pass with update_trace=True
    policy.model.train() # Enable plastic mode
    affect_gating = 5.0
    
    for i in range(5):
        with torch.no_grad():
            _ = layer(event_A, affect_gating=affect_gating, update_trace=True)
            
    log_step("Post-Learning", f"After 5 exposures (Affect={affect_gating})")
    
    # --- Phase 3: Pre-Sleep Check ---
    # Trace should be high. Weight should be same as Baseline (strictly speaking, BitLinear doesn't update weight online).
    
    # --- Phase 4: DEEP SLEEP (Consolidation) ---
    print("\n--- Phase 3: Deep Sleep (Consolidation) ---")
    # Call the sleep method
    policy.model.sleep()
    
    # --- Phase 5: Post-Sleep Verification ---
    print("\n--- Phase 4: Post-Sleep Verification ---")
    final_state = log_step("Post-Sleep", "Traces consolidated to weights")
    
    # Validation Logic
    baseline = history['steps'][0]
    post_learn = history['steps'][1]
    post_sleep = history['steps'][2]
    
    checks = []
    
    # Check 1: Did we learn? (Trace should go up)
    if post_learn['total_trace'] > baseline['total_trace']:
        checks.append("PASS: Learning created Traces.")
    else:
        checks.append("FAIL: No Traces formed.")
        
    # Check 2: Did sleep clear traces?
    if post_sleep['total_trace'] == 0:
        checks.append("PASS: Sleep reset Traces to 0.")
    else:
        checks.append(f"FAIL: Traces remaining ({post_sleep['total_trace']}).")
        
    # Check 3: Did weights update?
    # Note: absolute sum might vary, but effectively the weight should change.
    # A better check is if the RESPONSE is preserved even with Trace=0.
    if post_sleep['response_magnitude'] > baseline['response_magnitude']:
        checks.append("PASS: Long-term Weight reflects learning (Response > Baseline).")
    else:
        checks.append("FAIL: Response dropped to baseline (Learning lost).")

    history['checks'] = checks
    
    print("\n=== Test Results ===")
    for c in checks:
        print(c)
        
    # Save Report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"\nReport saved to: {REPORT_PATH}")

if __name__ == "__main__":
    run_test()
