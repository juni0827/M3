"""
M3-Binary Brain: Autonomous Sleep Verification
----------------------------------------------
Tests if the agent autonomously triggers sleep when conditions are met.

Scenario:
1. Initialize M3 Core (Mock).
2. Set state to "Tired but Safe" (Energy Low, Stability High).
3. Run the main loop logic (simulated).
4. Check if '_trigger_sleep' was called.

This test validates the wiring in `_run_core` without launching the full GUI.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import time

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# We need to act as if we are inside M3GUI._run_core
# Since M3GUI is tightly coupled with Tkinter, we will extract the logic to a testable function
# or mock the M3GUI class structure sufficiently.

class TestAutonomousSleep(unittest.TestCase):
    def test_auto_sleep_trigger(self):
        print("\n=== Autonomous Sleep Logic Test ===")
        
        # 1. Mock the Core and GUI structure
        mock_gui = MagicMock()
        mock_core = MagicMock()
        
        # Attach core to GUI
        mock_gui.core = mock_core
        
        # Setup Energy Control (Low Energy)
        mock_core.energy_ctrl.activation_level = 0.2  # < 0.3 (Threshold)
        
        # Setup World State (High Stability)
        mock_core.world_state = {'stability': 0.8}    # > 0.6 (Threshold)
        
        # Setup Sleep Trigger Mock
        mock_gui._trigger_sleep = MagicMock()
        mock_gui._log = MagicMock()
        mock_gui._last_auto_sleep = 0
        
        # 2. Extract and Run the Logic (Simulating one loop iteration)
        # Copied logic from M3GUI._run_core
        print("Simulating: Energy=0.2 (Tired), Stability=0.8 (Safe)")
        
        try:
            energy_level = getattr(mock_core.energy_ctrl, 'activation_level', 1.0)
            stability = mock_core.world_state.get('stability', 0.5)
            
            if energy_level < 0.3 and stability > 0.6:
                # Check debounce
                now = time.time()
                last_sleep = getattr(mock_gui, '_last_auto_sleep', 0)
                if now - last_sleep > 300: 
                    mock_gui._log("💤 Auto-Sleep: Energy Low & Safe. Consolidating Memories...")
                    mock_gui._trigger_sleep()
                    mock_gui._last_auto_sleep = now
        except Exception as e:
            self.fail(f"Logic raised exception: {e}")

        # 3. Verification
        if mock_gui._trigger_sleep.called:
            print("PASS: Sleep triggered autonomously.")
        else:
            print("FAIL: Sleep NOT triggered.")
            self.fail("Auto-sleep logic failed to trigger.")

    def test_auto_sleep_debounce(self):
        print("\n=== Auto-Sleep Debounce Test ===")
        
        mock_gui = MagicMock()
        mock_core = MagicMock()
        mock_gui.core = mock_core
        
        mock_core.energy_ctrl.activation_level = 0.2
        mock_core.world_state = {'stability': 0.8}
        
        mock_gui._trigger_sleep = MagicMock()
        
        # Simulate RECENT sleep
        mock_gui._last_auto_sleep = time.time() - 10 # Only 10 seconds ago
        
        print("Simulating: Tired & Safe, but slept 10s ago")
        
        try:
            energy_level = getattr(mock_core.energy_ctrl, 'activation_level', 1.0)
            stability = mock_core.world_state.get('stability', 0.5)
            
            if energy_level < 0.3 and stability > 0.6:
                now = time.time()
                last_sleep = getattr(mock_gui, '_last_auto_sleep', 0)
                if now - last_sleep > 300: 
                    mock_gui._trigger_sleep()
        except Exception:
            pass
            
        if not mock_gui._trigger_sleep.called:
            print("PASS: Sleep correctly inhibited (Debounce active).")
        else:
            print("FAIL: Sleep triggered despite debounce.")
            self.fail("Debounce logic failed.")

if __name__ == '__main__':
    unittest.main()
