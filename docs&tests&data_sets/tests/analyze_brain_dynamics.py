import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from typing import List, Dict
import sys

# Add project root to path
# Adjusted for nested structure: tests -> docs&tests&data_sets -> root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from llm_adapter.layers import PlasticBitLinear
from llm_adapter.m3_plastic_policy import M3PlasticPolicy
from llm_adapter.config import M3PlasticPolicyConfig, PlasticBitLinearConfig

class BrainAnalyzer:
    def __init__(self, log_dir):
        # Ensure log_dir is absolute
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.report_path = os.path.join(self.log_dir, "summary_report.txt")
        # Clear or init report file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write("=== M3 Binary Brain Analysis Report ===\n\n")
            
        self.log(f"Analyzer initialized on {self.device}")
        self.log(f"Saving logs to {self.log_dir}")

    def log(self, message: str):
        """Print to console and append to report file."""
        print(message)
        try:
            with open(self.report_path, 'a', encoding='utf-8') as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}")

    def setup_model(self, hidden_dim=64):
        config = M3PlasticPolicyConfig(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            linear_config=PlasticBitLinearConfig(
                trace_decay=0.9,     # Fast decay for testing short-term dynamics
                learning_rate=0.01,
                neuromodulation_scale=0.5
            )
        )
        model = M3PlasticPolicy(config=config, device=self.device)
        model.train() # Enable plasticity
        return model

    def run_learning_simulation(self):
        """
        Simulate a learning session:
        1. Baseline (Random noise)
        2. High Arousal Event (Pattern A) - Flashbulb memory
        3. Low Arousal Event (Pattern B) - Routine
        4. Decay/Interference period
        5. Recall test
        """
        print("Running Learning Simulation...")
        model = self.setup_model()
        
        # Patterns
        dim = model.config.embed_dim
        pattern_A = torch.randn(1, 1, dim).to(self.device) # The "Trauma" or "Joy"
        pattern_B = torch.randn(1, 1, dim).to(self.device) # The "Lunch"
        noise_input = torch.randn(1, 1, dim).to(self.device)

        # Token IDs usually used, but we'll bypass embedding for direct layer analysis 
        # or just use embedding outputs. Let's use forward() normally.
        # Construct inputs as "token-like" embeddings roughly. 
        # Actually M3PlasticPolicy takes input_ids. Let's mock the internal layers directly 
        # for precise control, or assume input_ids map to these patterns.
        # For simplicity in this analysis, I'll inspect the FIRST plastic layer directly.
        
        layer = model.layer_in
        
        history = {
            "step": [],
            "event": [],
            "arousal": [],
            "trace_mag": [],
            "response_A": [],
            "response_B": []
        }

        def measure(step_name, arousal_val):
            with torch.no_grad():
                # Measure traces
                trace_mag = layer.trace.abs().sum().item()
                
                # Measure response to A and B (Frozen weights + Current Trace)
                resp_A = layer(pattern_A, update_trace=False).norm().item()
                resp_B = layer(pattern_B, update_trace=False).norm().item()
                
                history["step"].append(len(history["step"]))
                history["event"].append(step_name)
                history["arousal"].append(arousal_val)
                history["trace_mag"].append(trace_mag)
                history["response_A"].append(resp_A)
                history["response_B"].append(resp_B)

        # 1. Baseline
        measure("Baseline", 0)

        # 2. Event A (High Arousal) - 3 exposures
        print("Experiencing Event A (High Arousal)...")
        for _ in range(3):
            layer(pattern_A, affect_gating=5.0, update_trace=True)
            measure("Event A (High)", 5.0)

        # 3. Distraction (Noise) - 5 steps
        print("Distraction (Time passing)...")
        for _ in range(5):
            layer(torch.randn(1, 1, dim).to(self.device), affect_gating=0.5, update_trace=True)
            measure("Distraction", 0.5)

        # 4. Event B (Low Arousal) - 3 exposures
        print("Experiencing Event B (Low Arousal)...")
        for _ in range(3):
            layer(pattern_B, affect_gating=0.5, update_trace=True)
            measure("Event B (Low)", 0.5)
            
        # 5. Long Delay (Decay) - 10 steps
        self.log("Long Delay...")
        for _ in range(10):
            layer(torch.randn(1, 1, dim).to(self.device), affect_gating=0.1, update_trace=True)
            measure("Decay", 0.1)

        self.save_results(history, "simulation_metrics.json")
        self.plot_results(history)
        
    def analyze_quantization(self):
        self.log("Analyzing Quantization Stats...")
        model = self.setup_model()
        layer = model.layer_in
        
        # Force some updates
        x = torch.randn(10, 1, model.config.embed_dim).to(self.device)
        layer(x)
        
        w = layer.weight.detach().cpu().numpy().flatten()
        w_scale = np.abs(w).mean()
        w_quant = np.round(w / (w_scale + 1e-6)) * w_scale
        
        stats = {
            "mean": float(w.mean()),
            "std": float(w.std()),
            "scale_factor": float(w_scale),
            "distribution": {
                "-1": int(np.sum(w_quant < -0.1)),
                "0": int(np.sum(np.abs(w_quant) < 0.1)),
                "1": int(np.sum(w_quant > 0.1))
            }
        }
        
        self.save_results(stats, "quantization_stats.json")
        self.log(f"Quantization Stas: {json.dumps(stats, indent=2)}")

    def save_results(self, data, filename):
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.log(f"Saved data to {path}")

    def plot_results(self, history):
        try:
            steps = history["step"]
            plt.figure(figsize=(12, 6))
            
            # Plot Trace Magnitude
            plt.subplot(1, 2, 1)
            plt.plot(steps, history["trace_mag"], label="Total Plasticity (Trace)", color='purple')
            plt.xlabel("Step")
            plt.ylabel("Magnitude")
            plt.title("Hebbian Trace Dynamics")
            plt.grid(True, alpha=0.3)
            
            # Plot Response Strength
            plt.subplot(1, 2, 2)
            plt.plot(steps, history["response_A"], label="Response via A (High Arousal)", color='red')
            plt.plot(steps, history["response_B"], label="Response via B (Low Arousal)", color='blue')
            plt.xlabel("Step")
            plt.ylabel("Activation Norm")
            plt.title("Memory Retention (A vs B)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(self.log_dir, "dynamics_plot.png")
            plt.savefig(plot_path)
            self.log(f"Saved plot to {plot_path}")
            plt.close()
        except Exception as e:
            self.log(f"Plotting failed (matplotlib issues?): {e}")

if __name__ == "__main__":
    # Unified log directory: tests/logs
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    analyzer = BrainAnalyzer(log_dir)
    analyzer.analyze_quantization()
    analyzer.run_learning_simulation()
