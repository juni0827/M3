import sys
import os
import torch
import numpy as np

# Adjust path to find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def test_integration():
    print("=== M3 Core & LLM Adapter Integration Test ===")
    
    try:
        from m3.core import M3ConsciousnessCore
        from llm_adapter.core import PlasticBrainPolicy
    except ImportError as e:
        print(f"FAILED to import modules: {e}")
        return

    print("Initializing M3ConsciousnessCore...")
    # Initialize Core (disable GUI for test)
    # We might need to mock some things if M3Core starts threads or heavy processes
    # Based on core.py, it seems safe to init with basic params
    core = M3ConsciousnessCore(n=100, K=5, outdir='tests/out_test')
    
    # Check if adapter is attached
    adapter = getattr(core, 'llm_adapter', None)
    
    if adapter is None:
        print("FAIL: LLM Adapter not attached to Core.")
    else:
        print("SUCCESS: LLM Adapter attached.")
        print(f"Adapter Type: {type(adapter)}")
        
        if isinstance(adapter, PlasticBrainPolicy):
            print("Verified: Adapter is PlasticBrainPolicy (1.58-bit M3 Brain).")
            
            # Simple functionality test
            print("Testing minimal forward update...")
            try:
                # Basic mock observation
                obs = np.random.rand(8).astype(np.float32) # Assuming obs matches some dim or handled
                
                # The sample method expects obs to be manageable. 
                # PlasticBrainPolicy -> UnifiedM3Policy -> internal logic
                # Let's just check if we can call sample without crashing
                # Note: UnifiedM3Policy usually expects encoded state or handles raw input. 
                # M3PlasticPolicy.sample takes (obs: np.ndarray, ...)
                
                # Affect state mock
                affect = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                
                # Run sample
                action, logp, val, ent = adapter.sample(obs, affect_state=affect)
                
                print(f"Sample output: Action={action}, LogP={logp:.4f}")
                print("SUCCESS: Adapter sample execution.")
            except Exception as e:
                print(f"FAIL during adapter sampling: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"WARNING: Adapter is not PlasticBrainPolicy. Is M3_PLASTIC_BRAIN env set? Type: {type(adapter)}")

    print("=== Test Complete ===")

if __name__ == "__main__":
    test_integration()
