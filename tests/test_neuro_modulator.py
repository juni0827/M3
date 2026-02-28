"""Functional tests for NeuroModulator."""
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_adapter.m3_control_bridge import (
    NeuroModulator,
    NeuroModulatorRuntime,
    NeuroModControls,
)


def _set_test_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


def test_creation_and_shapes():
    _set_test_seed(1001)
    nm = NeuroModulator(
        state_dim=256, num_layers=4, model_hidden_dim=512,
        vocab_size=1000, trunk_dim=128, hidden_rank=8, logit_rank=16,
    )
    n_params = sum(p.numel() for p in nm.parameters())
    print(f"  params={n_params}")
    z = torch.randn(1, 256)
    c = nm(z, strength=1.0)
    assert c.layer_gain.shape == (1, 4), f"gain shape: {c.layer_gain.shape}"
    assert c.hidden_bias.shape == (1, 4, 512), f"bias shape: {c.hidden_bias.shape}"
    assert c.logit_bias.shape == (1, 1000), f"logit shape: {c.logit_bias.shape}"
    assert 0.0 <= c.phi_gate <= 1.0
    print("  PASS")

def test_identity_at_warmup_zero():
    _set_test_seed(1002)
    nm = NeuroModulator(state_dim=64, num_layers=2, model_hidden_dim=128, vocab_size=100)
    z = torch.randn(1, 64)
    c = nm(z, strength=1.0)
    # At step 0, warmup_factor=0 -> gains = 1.0, biases = 0.0
    for g in c.layer_gain[0].tolist():
        assert abs(g - 1.0) < 1e-5, f"gain not identity: {g}"
    assert c.hidden_bias.abs().max().item() < 1e-6, "bias not zero"
    assert c.logit_bias.abs().max().item() < 1e-6, "logit bias not zero"
    print("  PASS")

def test_warmup_progression():
    _set_test_seed(1003)
    nm = NeuroModulator(state_dim=64, num_layers=2, model_hidden_dim=128, vocab_size=100)
    z = torch.randn(1, 64)
    # Run 100 steps to fully warm up
    for _ in range(100):
        nm(z, strength=1.0)
    c = nm(z, strength=1.0)
    deviated = any(abs(g - 1.0) > 0.0005 for g in c.layer_gain[0].tolist())
    assert deviated, "gains should deviate after warmup"
    assert c.hidden_bias.abs().max().item() > 1e-5, "bias should be non-zero after warmup"
    print(f"  gains: {c.layer_gain[0].tolist()}")
    print("  PASS")

def test_gradient_flow_positive_reward():
    _set_test_seed(1004)
    nm = NeuroModulator(state_dim=64, num_layers=2, model_hidden_dim=128, vocab_size=100)
    z = torch.randn(1, 64)
    nm.train()
    loss = nm.online_loss(z, reward=0.5, strength=1.0)
    assert loss.item() > 0, f"loss should be > 0: {loss.item()}"
    loss.backward()
    grad_found = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in nm.parameters()
    )
    assert grad_found, "gradients should flow"
    print(f"  loss={loss.item():.6f}")
    print("  PASS")

def test_gradient_flow_negative_reward():
    _set_test_seed(1005)
    nm = NeuroModulator(state_dim=64, num_layers=2, model_hidden_dim=128, vocab_size=100)
    z = torch.randn(1, 64)
    nm.train()
    loss = nm.online_loss(z, reward=-0.8, strength=1.0)
    assert loss.item() > 0, f"loss should be > 0: {loss.item()}"
    loss.backward()
    grad_found = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in nm.parameters()
    )
    assert grad_found, "gradients should flow with negative reward"
    print(f"  loss={loss.item():.6f}")
    print("  PASS")

def test_runtime_hooks():
    _set_test_seed(1006)
    class DummyLayer(torch.nn.Module):
        def forward(self, x):
            return (x, None)

    nm = NeuroModulator(state_dim=64, num_layers=3, model_hidden_dim=128, vocab_size=100)
    # Warm up to get non-trivial controls
    z = torch.randn(1, 64)
    for _ in range(50):
        nm(z, strength=1.0)

    layers = [DummyLayer() for _ in range(3)]
    runtime = NeuroModulatorRuntime(layers)

    with torch.no_grad():
        c = nm(z, strength=1.0)
    runtime.apply(c)

    x = torch.randn(1, 5, 128)
    x_original = x.clone()
    out, _ = layers[0](x)
    # Output should be modulated (not identical to input)
    assert not torch.allclose(out, x_original), "hook should modulate layer output"
    runtime.close()
    assert len(runtime._hooks) == 0, "hooks should be removed"
    print("  PASS")

def test_numpy_input():
    _set_test_seed(1007)
    nm = NeuroModulator(state_dim=64, num_layers=2, model_hidden_dim=128, vocab_size=100)
    z_np = np.random.randn(64).astype(np.float32)
    c = nm(z_np, strength=1.0)
    assert c.layer_gain.shape == (1, 2)
    print("  PASS")

def test_state_padding_truncation():
    _set_test_seed(1008)
    nm = NeuroModulator(state_dim=64, num_layers=2, model_hidden_dim=128, vocab_size=100)
    # Too short
    z_short = torch.randn(32)
    c = nm(z_short, strength=1.0)
    assert c.layer_gain.shape == (1, 2)
    # Too long
    z_long = torch.randn(128)
    c = nm(z_long, strength=1.0)
    assert c.layer_gain.shape == (1, 2)
    print("  PASS")

def test_online_learning_step():
    """Simulate a full online learning cycle."""
    _set_test_seed(1009)
    nm = NeuroModulator(state_dim=64, num_layers=2, model_hidden_dim=128, vocab_size=100)
    opt = torch.optim.Adam(nm.parameters(), lr=1e-3)
    z = torch.randn(1, 64)

    # Record initial params
    init_params = {n: p.data.clone() for n, p in nm.named_parameters()}

    # Run a training step with positive reward
    nm.train()
    opt.zero_grad()
    loss = nm.online_loss(z, reward=0.8, strength=1.0)
    loss.backward()
    opt.step()

    # Check that parameters changed
    changed = False
    for n, p in nm.named_parameters():
        if not torch.allclose(init_params[n], p.data):
            changed = True
            break
    assert changed, "parameters should change after optimizer step"
    print("  PASS")

if __name__ == "__main__":
    tests = [
        ("test_creation_and_shapes", test_creation_and_shapes),
        ("test_identity_at_warmup_zero", test_identity_at_warmup_zero),
        ("test_warmup_progression", test_warmup_progression),
        ("test_gradient_flow_positive_reward", test_gradient_flow_positive_reward),
        ("test_gradient_flow_negative_reward", test_gradient_flow_negative_reward),
        ("test_runtime_hooks", test_runtime_hooks),
        ("test_numpy_input", test_numpy_input),
        ("test_state_padding_truncation", test_state_padding_truncation),
        ("test_online_learning_step", test_online_learning_step),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{passed+failed} passed")
    if failed:
        print(f"FAILED: {failed} tests")
        sys.exit(1)
    else:
        print("All tests PASSED")
