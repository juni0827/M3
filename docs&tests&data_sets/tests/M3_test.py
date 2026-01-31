import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import json
import time
from llm_adapter.remote import get_local_thinking

# --- Configuration (4060 GPU 로컬 모드) ---
USE_LOCAL_AI = True 
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:8b" # deepseek r1 8b 모델로 변경 (설치된 모델 이름에 맞게 조정)


# --- Mathematical Constants (Based on Friston's Free Energy Principle) ---
DT = 0.1             # Time step
TAU = 20.0           # Time constant
SIGMA_NOISE = 0.05   # Sensory noise
THRESHOLD_ACTION = 0.85 # Surprisal threshold for action

@dataclass
class AgentState:
    phi: float = 0.0     # Integrated Information (Proxy)
    surprise: float = 0.0 # Free Energy / Prediction Error
    action_log: list = None

class FreeEnergyAgent:
    def __init__(self):
        # [Internal_State, Internal_External_Model]
        self.state = np.array([0.5, 0.5]) 
        self.target = np.array([0.5, 0.5]) # Homeostatic Set-point (Target density)
        self.history = []

    def sensory_input(self, t):
        # Environmental noise acting as sensory perturbation
        return np.random.normal(0, SIGMA_NOISE, 2)

    def update(self, t):
        """
        Active Inference Loop (Variational Free Energy Minimization):
        State changes to minimize prediction error (Surprise).
        F = (Sensory - Predicted)^2 + Complexity
        """
        sensory = self.sensory_input(t)
        
        # 1. Perception Step (Gradient Descent on Free Energy)
        # Prediction Error: The discrepancy between sensory input and the agent's target state
        prediction_error = sensory - (self.state - self.target)
        d_state = - (1/TAU) * prediction_error * DT + np.random.normal(0, 0.01, 2)
        
        # Entropic Drift: Without action, systems dissipate and error accumulates
        self.state += d_state + 0.005 # Constant entropic drift (Second Law)
        
        # Calculate Variational Free Energy (Simplified as Scalar Prediction Error)
        free_energy = np.sum(prediction_error**2)
        
        self.history.append(free_energy)
        return free_energy

    def act(self):
        """
        Action (Active Inference):
        The agent intervenes in its own state/environment to restore homeostasis.
        """
        self.state = self.target + np.random.normal(0, 0.05, 2)
        return True

# Use `get_local_thinking` from llm_adapter.remote to avoid embedding request logic in tests

def run_rigorous_trial(duration_steps=1000):
    agent = FreeEnergyAgent()
    data = []
    action_indices = [] # 액션이 발생한 시점들을 기록
    
    print(f"Running Physics Simulation (Offline Mode)...")
    
    # 1. 물리 시뮬레이션 먼저 실행 (API 호출 없이 순수 수학 모델링)
    for t in range(duration_steps):
        F = agent.update(t)
        action_triggered = 0
        thought_content = ""
        
        # 물리적 엔트로피가 임계치를 넘으면 즉시 '로컬 지능'이 개입
        if F > THRESHOLD_ACTION:
            prompt = (
                f"As an FEP agent, my entropy is critical ({F:.2f}). "
                f"Give me one single abstract word to restore my internal order. (No explanation)"
            )
            thought_content = get_local_thinking(prompt)
            agent.act() 
            action_triggered = 1
            print(f"[{t}] 4060 ACTION: {thought_content}")

        data.append({
            "step": t,
            "free_energy": F,
            "action": action_triggered,
            "content": thought_content
        })

    df = pd.DataFrame(data)
    action_count = len(action_indices)

    return df
if __name__ == "__main__":
    df = run_rigorous_trial()
    if len(df) > 0:
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/fep_experiment_data.csv", index=False)
        print(f"\nData saved to results/fep_experiment_data.csv")
