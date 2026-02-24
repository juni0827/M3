import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

# Allow direct execution: `python tests/test_m3_plan_features.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_adapter.config import BridgeAdaptConfig, EarlyStopConfig, TorchPolicyConfig
from llm_adapter.llm_core import TorchConversationalPolicy
from llm_adapter.m3_control_bridge import M3ControlBridge
from llm_adapter.memory import M3EpisodicMemoryRetriever
from m3.m3_core import CauseEffectStructure, ConsciousnessBus, EvolutionVisualizer, GrowingSOM
from m3.torch_policy import MoEFFN


def _make_policy() -> TorchConversationalPolicy:
    os.environ["M3_USE_HF"] = "0"
    os.environ["M3_AUTONOMY_RL_ENABLE"] = "1"
    os.environ["M3_TRAIN_EARLY_STOP"] = "1"
    os.environ["M3_DPO_AUTO_COLLECT"] = "1"
    os.environ["M3_TOKENIZER_AUTO_VOCAB"] = "1"
    os.environ["M3_BRIDGE_ONLINE_ADAPT"] = "1"
    os.environ["M3_STABILITY_WEIGHT_DECAY"] = "0.01"
    os.environ["M3_STABILITY_SPECTRAL_NORM"] = "0"
    os.environ["LLM_CHECKPOINT_PATH"] = os.path.join(tempfile.gettempdir(), "missing_llm_checkpoint.pt")
    cfg = TorchPolicyConfig(embed_dim=16, hidden_dim=32, num_layers=1, learning_rate=3e-3)
    p = TorchConversationalPolicy(config=cfg, device="cpu")
    p.autonomy_rl_cfg.batch_size = 1
    p.autonomy_rl_cfg.replay_size = 64
    p.autonomy_rl_cfg.learning_rate = 1e-3
    p.early_stop_cfg = EarlyStopConfig(
        enabled=True,
        val_fraction=0.5,
        patience=1,
        min_delta=0.0,
        max_epochs=5,
        restore_best_weights=True,
    )
    return p


class _DummyEpisode:
    def __init__(self, idx: int, emb: np.ndarray):
        self.idx = idx
        self.embedding = emb.astype(np.float32)
        self.affect_state = {}
        self.drive_reduction = {}
        self.retrieval_count = 0


class _DummyBus:
    def push(self, *args, **kwargs):
        return None


class _DummyFeatureBank:
    embed_dim = 32

    @staticmethod
    def _hash_embed(text: str, dim: int) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)


class _DummyCore:
    def __init__(self):
        self.bus = _DummyBus()
        self.feature_bank = _DummyFeatureBank()
        self.affect_kernel = types.SimpleNamespace(get_state=lambda: [0.0, 0.1, 0.2, 0.7, 0.0])
        self.drives = types.SimpleNamespace(get_drive_state=lambda: {})
        self.qualia = types.SimpleNamespace(entropy=0.2, engagement=0.7, arousal=0.0, valence=0.1, frustration=0.0)

    @staticmethod
    def _evaluate_dialog_accuracy(prompt: str, response: str):
        score = 0.7 if response else 0.0
        return {"score": score, "overlap": 0.5}


class _FakeTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True):
        ids = [max(1, (ord(c) % max(2, self.vocab_size - 1))) for c in (text or "x")]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _FakeModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden: int = 12):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, output_hidden_states=False):
        h = self.emb(input_ids)
        logits = self.head(h)
        return types.SimpleNamespace(logits=logits)


class AdaptiveSamplerInputShapeTests(unittest.TestCase):
    def test_adaptive_sampler_accepts_affect_dict_without_crash(self):
        from llm_adapter.llm_core import M3AdaptiveSampler

        sampler = M3AdaptiveSampler(torch, device="cpu")
        core = types.SimpleNamespace(
            affect_kernel=types.SimpleNamespace(get_state=lambda: {"arousal": 0.3, "novelty": 0.8}),
            qualia=types.SimpleNamespace(entropy=0.2, engagement=0.7, arousal=0.1, valence=0.1, frustration=0.0),
        )
        t = sampler._compute_temperature(core, 0.9)
        self.assertTrue(np.isfinite(t))

    def test_adaptive_sampler_topk_affect_vector_indexing_is_stable(self):
        from llm_adapter.llm_core import M3AdaptiveSampler

        sampler = M3AdaptiveSampler(torch, device="cpu")
        core = types.SimpleNamespace(
            affect_kernel=types.SimpleNamespace(get_state=lambda: [0.2]),
            qualia=types.SimpleNamespace(entropy=0.4, engagement=0.6, arousal=0.2, valence=0.1, frustration=0.0),
        )
        k = sampler._compute_top_k(core, 50)
        self.assertIsInstance(k, int)


    def test_micro_update_updates_decode_entropy_not_qualia_entropy(self):
        from llm_adapter.llm_core import HFBackend

        core = types.SimpleNamespace(
            qualia=types.SimpleNamespace(entropy=0.25),
            decode_entropy=0.10,
        )
        updated = HFBackend._micro_update_step_state(core, _step=2, generated_ids=[1, 2, 1, 3], interval=2)
        self.assertTrue(updated)
        self.assertNotEqual(float(core.decode_entropy), 0.10)
        self.assertEqual(float(core.qualia.entropy), 0.25)

    def test_sampler_prefers_decode_entropy_when_present(self):
        from llm_adapter.llm_core import M3AdaptiveSampler

        sampler = M3AdaptiveSampler(torch, device="cpu")
        core = types.SimpleNamespace(
            qualia=types.SimpleNamespace(entropy=0.05, engagement=1.0, arousal=0.2, valence=0.1, frustration=0.0),
            decode_entropy=0.95,
        )
        k = sampler._compute_top_k(core, 50)
        self.assertEqual(k, sampler.config.top_k_high_exploration)


    def test_resolve_decode_entropy_prefers_decode_field(self):
        from llm_adapter.llm_core import M3AdaptiveSampler

        sampler = M3AdaptiveSampler(torch, device="cpu")
        core = types.SimpleNamespace(
            decode_entropy=0.77,
            token_entropy=0.22,
            qualia=types.SimpleNamespace(entropy=0.11),
        )
        self.assertAlmostEqual(sampler._resolve_decode_entropy(core, 0.5), 0.77)

    def test_micro_update_writes_decode_entropy_without_qualia(self):
        from llm_adapter.llm_core import HFBackend

        core = types.SimpleNamespace(token_entropy=0.20)
        updated = HFBackend._micro_update_step_state(core, _step=3, generated_ids=[1, 2, 3], interval=3)
        self.assertTrue(updated)
        self.assertTrue(hasattr(core, "decode_entropy"))

    def test_dummy_core_affect_kernel_returns_5d_vector(self):
        core = _DummyCore()
        vec = core.affect_kernel.get_state()
        self.assertEqual(len(vec), 5)

    def test_phi_influence_bounded_after_normalization(self):
        from llm_adapter.llm_core import M3AdaptiveSampler

        sampler = M3AdaptiveSampler(torch, device="cpu")
        sampler.config.temp_min = 0.3
        sampler.config.temp_max = 2.0
        sampler.config.phi_influence = 0.8
        sampler.config.phi_norm_mode = "dynamic"
        sampler.config.phi_norm_quantile = 0.9

        core = types.SimpleNamespace(
            qualia=types.SimpleNamespace(entropy=0.2, engagement=0.7, arousal=0.4, valence=0.3, frustration=0.1),
            phi_calculator=types.SimpleNamespace(phi_history=[0.2, 0.4, 0.6, 0.8, 10.0]),
        )

        t = sampler._compute_temperature(core, 0.9)
        self.assertGreaterEqual(t, sampler.config.temp_min)
        self.assertLessEqual(t, sampler.config.temp_max)


class PhiPathConsistencyTests(unittest.TestCase):
    def test_single_step_uses_compute_phi_api(self):
        from m3.m3_core import M3ConsciousnessCore

        core = M3ConsciousnessCore.__new__(M3ConsciousnessCore)
        called = {}

        core.energy_ctrl = types.SimpleNamespace(
            internal_clock=0,
            update_activation=lambda *a, **k: None,
            should_continue=lambda: (True, 0.5),
        )
        core._get_current_world_state = lambda: {'delta_hat': 0.1, 'stability': 0.9, 'energy_level': 0.8}
        core.goal_gen = types.SimpleNamespace(generate_goal=lambda *a, **k: {'goal': 'x'})
        core.self_model = types.SimpleNamespace(state_history=[], update_meta_awareness=lambda *_: None)
        core.qualia = types.SimpleNamespace()
        core._decide_action = lambda *a, **k: {'act': 'noop'}
        core._execute_action = lambda *_: (0.0, None)
        core._experience_qualia = lambda *a, **k: None
        core.conceptual_space = types.SimpleNamespace(ground_experience=lambda q: {'q': 1})
        core._submit_to_workspace = lambda *a, **k: None
        core.global_workspace = types.SimpleNamespace(compete_for_consciousness=lambda: [{'priority': 0.7}])
        core.t = 0

        def _fake_compute_phi(*args, **kwargs):
            called["args"] = args
            called["kwargs"] = kwargs
            return 0.42

        core.phi_calculator = types.SimpleNamespace()
        core.phi_calculator = types.SimpleNamespace()
        core.phi_calculator.compute_phi = _fake_compute_phi
        core._single_consciousness_step()

        self.assertIn("kwargs", called)
        self.assertEqual(called["kwargs"].get("method"), "simple")
        self.assertIn("state", called["kwargs"])
        self.assertIsInstance(called["kwargs"]["state"], np.ndarray)

    def test_phi_feedback_reaches_qualia_message_bus(self):
        from m3.m3_core import IITPhiCalculator, MessageBus

        bus = MessageBus(capacity=32)
        bus.register_module('qualia_monitor')
        phi_calc = IITPhiCalculator(n_elements=8, message_bus=bus)
        phi_calc.compute_phi(state=np.array([0.1, 0.8, 0.3, 0.9], dtype=np.float32), method='simple')
        msg = bus.receive('qualia_monitor', timeout=0.05)
        self.assertIsNotNone(msg)
        self.assertEqual(msg.type, 'phi_update')



class M3PlanFeatureTests(unittest.TestCase):
    def test_01_q_head_rl_updates_for_speak_and_wait(self):
        policy = _make_policy()
        policy.autonomy_rl_cfg.gamma = 0.0
        policy.model.q_head = torch.nn.Linear(policy.model.hidden, 2, bias=True).to(policy.device)
        policy.model.intensity_head = torch.nn.Linear(policy.model.hidden, 1, bias=True).to(policy.device)
        s = torch.ones((1, policy.model.hidden), dtype=torch.float32, device=policy.device)

        def reset_model():
            with torch.no_grad():
                policy.model.q_head.weight.zero_()
                policy.model.q_head.bias.zero_()
                policy.model.intensity_head.weight.zero_()
                policy.model.intensity_head.bias.zero_()
            policy._autonomy_replay.clear()
            wd = float(os.environ.get("M3_STABILITY_WEIGHT_DECAY", "0.01"))
            policy._autonomy_opt = torch.optim.AdamW(
                list(policy.model.q_head.parameters()) + list(policy.model.intensity_head.parameters()),
                lr=0.2,
                weight_decay=wd,
            )

        reset_model()
        for _ in range(30):
            policy._update_autonomy_q(s, action=1, reward=1.0, next_state_t=s, done=False)
        q_speak_pos = float(policy.model.q_head(s).detach().cpu().numpy()[0, 1])

        reset_model()
        for _ in range(30):
            policy._update_autonomy_q(s, action=1, reward=-1.0, next_state_t=s, done=False)
        q_speak_neg = float(policy.model.q_head(s).detach().cpu().numpy()[0, 1])

        reset_model()
        for _ in range(30):
            policy._update_autonomy_q(s, action=0, reward=1.0, next_state_t=s, done=False)
        q_wait_pos = float(policy.model.q_head(s).detach().cpu().numpy()[0, 0])

        reset_model()
        for _ in range(30):
            policy._update_autonomy_q(s, action=0, reward=-1.0, next_state_t=s, done=False)
        q_wait_neg = float(policy.model.q_head(s).detach().cpu().numpy()[0, 0])

        self.assertGreater(q_speak_pos, q_speak_neg)
        self.assertGreater(q_wait_pos, q_wait_neg)

    def test_02_ann_backend_auto_selection(self):
        retriever = M3EpisodicMemoryRetriever()
        retriever._supports_module = lambda m: m == "faiss"
        retriever.set_ann_backend("auto")
        self.assertEqual(retriever._ann_backend_name, "faiss")
        retriever._supports_module = lambda m: m == "annoy"
        retriever.set_ann_backend("auto")
        self.assertEqual(retriever._ann_backend_name, "annoy")
        retriever._supports_module = lambda m: False
        retriever.set_ann_backend("auto")
        self.assertEqual(retriever._ann_backend_name, "numpy")

    def test_03_ann_query_matches_bruteforce_topk(self):
        retriever = M3EpisodicMemoryRetriever()
        retriever.set_ann_backend("numpy")
        rng = np.random.default_rng(7)
        episodes = [_DummyEpisode(i, rng.normal(size=(16,)).astype(np.float32)) for i in range(50)]
        core = types.SimpleNamespace(
            episodic_memory=types.SimpleNamespace(episodes=episodes),
            affect_kernel=types.SimpleNamespace(get_state=lambda: {}),
            drives=types.SimpleNamespace(get_drive_state=lambda: {}),
            qualia=types.SimpleNamespace(entropy=0.2, engagement=0.8, arousal=0.0, valence=0.0, frustration=0.0),
        )
        retriever.refresh_index(core)
        q = rng.normal(size=(16,)).astype(np.float32)
        ann = retriever.query_candidates(q, k=8)
        qn = q / (np.linalg.norm(q) + 1e-8)
        sims = []
        for ep in episodes:
            en = ep.embedding / (np.linalg.norm(ep.embedding) + 1e-8)
            sims.append((ep.idx, float(np.dot(qn, en))))
        sims.sort(key=lambda x: x[1], reverse=True)
        brute_ids = [i for i, _ in sims[:8]]
        ann_ids = [ep.idx for ep in ann]
        self.assertEqual(ann_ids, brute_ids)

    def test_04_moe_aux_loss_penalizes_collapse(self):
        moe = MoEFFN(d=8, d_ff=16, n_experts=4, top_k=1)
        moe.eval()
        x = torch.randn(128, 8)
        with torch.no_grad():
            moe.gate.weight.zero_()
            moe.gate.bias.zero_()
            moe.gate.bias[0] = 10.0
            _, aux_c, stats_c = moe(x)
            moe.gate.weight.normal_(0.0, 0.2)
            moe.gate.bias.zero_()
            _, aux_u, stats_u = moe(x)
        self.assertGreater(float(aux_c.item()), float(aux_u.item()))
        self.assertLess(float(stats_c["expert_usage_entropy"].item()), float(stats_u["expert_usage_entropy"].item()))

    def test_05_dpo_auto_collect_from_logs(self):
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as td:
            chosen_path = os.path.join(td, "llm_training_data.jsonl")
            rejected_path = os.path.join(td, "llm_training_data.rejected.jsonl")
            chat_path = os.path.join(td, "chat_history.jsonl")
            out_path = os.path.join(td, "llm_training_data.preference.auto.jsonl")
            chosen = {"ts": 1000.0, "prompt_raw": "p", "response": "good answer"}
            rejected = {"ts": 1002.0, "prompt_raw": "p", "response": "1,2,3,4,5,6,7,8,9"}
            with open(chosen_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(chosen, ensure_ascii=False) + "\n")
                f.write(json.dumps(chosen, ensure_ascii=False) + "\n")
            with open(rejected_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(rejected, ensure_ascii=False) + "\n")
            with open(chat_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"role": "user", "content": "p", "ts": 1003.0}, ensure_ascii=False) + "\n")
                f.write(json.dumps({"role": "assistant", "content": "as an ai i cannot", "ts": 1004.0}, ensure_ascii=False) + "\n")
            res = policy.collect_dpo_preferences_from_logs(logs_dir=td, out_path=out_path, max_pairs=8)
            self.assertGreaterEqual(int(res["num_pairs"]), 1)
            with open(out_path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            self.assertGreaterEqual(len(lines), 1)
            self.assertEqual(len(lines), len(set(lines)))

    def test_06_supervised_early_stopping_stops_on_plateau(self):
        policy = _make_policy()
        pairs = [(f"p{i}", f"r{i}") for i in range(20)]
        policy._iter_supervised_records_from_dir = lambda _d: iter(pairs)
        policy.learn_pair = lambda _p, _r, max_len=120: None
        policy._sequence_logprob = lambda _p, _r, max_len=120: -1.0
        policy.early_stop_cfg = EarlyStopConfig(
            enabled=True,
            val_fraction=0.5,
            patience=1,
            min_delta=0.0,
            max_epochs=5,
            restore_best_weights=False,
        )
        res = policy.train_supervised_with_early_stopping(epochs=5, data_dir="dummy")
        self.assertTrue(bool(res["stopped_early"]))
        self.assertLess(int(res["epochs_run"]), 5)

    def test_07_bridge_adaptation_on_off_parameter_change(self):
        policy = _make_policy()
        policy.bridge_adapt_cfg = BridgeAdaptConfig(
            enabled=True,
            learning_rate=5e-2,
            reward_scale=1.0,
            gate_reg=1e-3,
            bias_reg=1e-4,
            min_quality_score=0.0,
            cooldown_steps=3,
        )
        policy._evaluate_generation_quality = lambda prompt, response, source="generate": (True, {"score": 0.8})
        hf = types.SimpleNamespace()
        hf.device = torch.device("cpu")
        hf._tokenizer = _FakeTokenizer(vocab_size=24)
        hf._model = _FakeModel(vocab_size=24, hidden=12)
        hf._control_bridge = M3ControlBridge(
            state_dim=4,
            model_hidden_dim=12,
            vocab_size=24,
            num_layers=1,
            prefix_len=2,
            logit_rank=4,
        )
        hf._prepare_bridge_state = lambda z_m3, state_dim, device: torch.as_tensor(np.asarray(z_m3, dtype=np.float32)).view(1, -1).to(device)

        before = [p.detach().clone() for p in hf._control_bridge.parameters()]
        policy._bridge_adapt_enabled = True
        policy._adapt_bridge_online(hf, prompt="abc", response="def", z_m3=np.ones((4,), dtype=np.float32))
        after = [p.detach().clone() for p in hf._control_bridge.parameters()]
        delta_on = float(sum((a - b).abs().sum().item() for a, b in zip(after, before)))
        self.assertGreater(delta_on, 0.0)

        before2 = [p.detach().clone() for p in hf._control_bridge.parameters()]
        policy._bridge_adapt_enabled = False
        policy._adapt_bridge_online(hf, prompt="abc", response="def", z_m3=np.ones((4,), dtype=np.float32))
        after2 = [p.detach().clone() for p in hf._control_bridge.parameters()]
        delta_off = float(sum((a - b).abs().sum().item() for a, b in zip(after2, before2)))
        self.assertEqual(delta_off, 0.0)

    def test_08_tokenizer_reload_and_vocab_resize_forward(self):
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as td:
            out_tok = os.path.join(td, "tok.json")
            try:
                from tokenizers import Tokenizer, decoders, models, pre_tokenizers
                from tokenizers.trainers import BpeTrainer
            except Exception as e:
                self.skipTest(f"tokenizers unavailable: {e}")
            tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
            tok.pre_tokenizer = pre_tokenizers.ByteLevel()
            tok.decoder = decoders.ByteLevel()
            trainer = BpeTrainer(vocab_size=200, special_tokens=policy.tok.special_tokens, show_progress=False)
            tok.train_from_iterator(["hello world", "autonomy learning", "m3 bridge"], trainer=trainer)
            tok.save(out_tok)
            res = policy.reload_tokenizer_and_resize(out_tok)
            self.assertTrue(bool(res.get("ok", False)))
            src = torch.tensor([[policy.tok.BOS]], dtype=torch.long)
            tgt = torch.tensor([[policy.tok.BOS]], dtype=torch.long)
            out = policy.model(src, tgt)
            self.assertEqual(int(out.shape[-1]), int(policy.vocab_size))

    def test_09_stability_guard_skips_non_finite_and_renorms(self):
        policy = _make_policy()
        params = list(policy.model.parameters())
        p0 = params[0]
        p0.grad = torch.full_like(p0.data, float("nan"))
        ok = policy._guarded_optimizer_step(policy.opt, params, tag="unit_non_finite")
        self.assertFalse(ok)
        self.assertGreater(policy._stability_skip_steps, 0)

        with torch.no_grad():
            p0.data.fill_(1e5)
        for p in params:
            p.grad = torch.zeros_like(p.data)
        ok2 = policy._guarded_optimizer_step(policy.opt, params, tag="unit_finite")
        self.assertTrue(ok2)
        with torch.no_grad():
            n = torch.norm(p0.data)
            self.assertLessEqual(float(n.item()), float(policy.stability_cfg.max_weight_norm) + 0.2)

    def test_10_e2e_smoke_autonomy_cycle_logs(self):
        policy = _make_policy()
        core = _DummyCore()
        policy.core = core
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "llm_adapter.log")
            os.environ["LLM_ADAPTER_LOG"] = log_path
            with torch.no_grad():
                if getattr(policy.model.q_head, "bias", None) is not None:
                    policy.model.q_head.bias[0].fill_(-2.0)
                    policy.model.q_head.bias[1].fill_(2.0)
            with mock.patch.object(policy, "generate", return_value="smoke response"), mock.patch("numpy.random.rand", return_value=0.0):
                policy._run_autonomy_turn(cycle_count=1, autonomy_check_every=1)
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("autonomy_rl", content)

    def test_11_ann_backend_selection_dedup_keeps_state(self):
        retriever = M3EpisodicMemoryRetriever()
        retriever.ann_config.log_select_once = True
        retriever._supports_module = lambda _m: False
        retriever.set_ann_backend("auto")
        retriever._ann_refresh_count = 7
        retriever.set_ann_backend("auto")
        self.assertEqual(retriever._ann_backend_name, "numpy")
        self.assertEqual(retriever._ann_refresh_count, 7)

    def test_12_consciousness_bus_priority_filter_async(self):
        bus = ConsciousnessBus(
            top_k=4,
            outdir=None,
            async_dispatch=True,
            max_queue=32,
            drop_policy="drop_low_priority",
        )
        seen = []
        sid = bus.subscribe(lambda payload: seen.append(payload.get("id")), topic="token", min_priority=0.3)
        bus.publish("token", {"id": "low"}, priority=0.1, async_dispatch=True)
        bus.publish("other", {"id": "wrong_topic"}, priority=1.0, async_dispatch=True)
        bus.publish("token", {"id": "mid"}, priority=0.5, async_dispatch=True)
        bus.publish("token", {"id": "high"}, priority=0.9, async_dispatch=True)
        drained = bus.drain()
        bus.unsubscribe(sid)
        bus.stop_async_worker()
        self.assertGreaterEqual(int(drained), 2)
        self.assertEqual(seen, ["high", "mid"])

    def test_13_phi_threshold_policy_shared_consumers(self):
        system_state = {
            "u_matrix": np.zeros((4, 4), dtype=np.float32),
            "qualia": {},
            "current_experience": "x",
            "memories": 0,
            "unity": 0.5,
            "neuron_count": 4,
            "connection_count": 4,
            "growth_events": 0,
            "strange_loop": False,
            "meta_awareness": 0.1,
            "phi": 0.18,
            "energy": 0.5,
            "phi_policy": {
                "floor": 0.01,
                "low": 0.05,
                "mid": 0.15,
                "high": 0.25,
                "very_high": 0.40,
                "announce_high": 0.25,
            },
        }
        ev = EvolutionVisualizer()
        ev.update(dict(system_state))
        gs = GrowingSOM(input_dim=5, initial_size=2)
        gs.update(dict(system_state))
        self.assertEqual(int(ev.consciousness_level), 3)
        self.assertEqual(int(gs.consciousness_level), 3)

        ces = CauseEffectStructure(n_elements=4)
        for v in np.linspace(0.02, 0.5, num=32):
            ces.phi_history.append(float(v))
        lvl = ces.get_consciousness_level(0.2)
        self.assertIn(lvl, {"(low)", "(moderate)", "(high)", "(very high)"})

    def test_14_neuro_modulator_config_defaults_and_json_roundtrip(self):
        from llm_adapter.config import M3LLMConfig, NeuroModulatorConfig, validate_config

        cfg = M3LLMConfig()
        nm = cfg.neuro_modulator
        self.assertIsInstance(nm, NeuroModulatorConfig)
        self.assertTrue(nm.enabled)
        self.assertEqual(nm.state_dim, 256)
        self.assertEqual(nm.hidden_rank, 16)
        self.assertEqual(nm.logit_rank, 32)
        self.assertGreater(nm.learning_rate, 0.0)
        self.assertGreater(nm.grad_clip_norm, 0.0)
        self.assertTrue(validate_config(cfg))

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test_cfg.json")
            cfg.to_json(path)
            loaded = M3LLMConfig.from_json(path)
            self.assertEqual(loaded.neuro_modulator.state_dim, nm.state_dim)
            self.assertEqual(loaded.neuro_modulator.hidden_rank, nm.hidden_rank)
            self.assertEqual(loaded.neuro_modulator.strength, nm.strength)
            self.assertEqual(loaded.neuro_modulator.checkpoint_file, nm.checkpoint_file)

    def test_15_neuro_modulator_config_validation_rejects_invalid(self):
        from llm_adapter.config import M3LLMConfig, NeuroModulatorConfig, validate_config
        import logging

        cfg = M3LLMConfig()
        cfg.neuro_modulator.state_dim = 0
        logging.disable(logging.CRITICAL)
        try:
            ok = validate_config(cfg)
        finally:
            logging.disable(logging.NOTSET)
        self.assertFalse(ok)

    def test_16_neuro_modulator_save_and_load_checkpoint(self):
        from llm_adapter.config import M3LLMConfig, NeuroModulatorConfig, set_global_config, get_global_config
        from llm_adapter.m3_control_bridge import NeuroModulator

        nm = NeuroModulator(state_dim=32, num_layers=2, model_hidden_dim=64, vocab_size=100)
        z = torch.randn(1, 32)
        # Advance a few steps so _step > 0
        for _ in range(5):
            nm(z, strength=1.0)
        opt = torch.optim.Adam(nm.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as td:
            ckpt_path = os.path.join(td, "nm_test.pt")
            cfg = get_global_config()
            old_path = cfg.neuro_modulator.checkpoint_file
            cfg.neuro_modulator.checkpoint_file = ckpt_path
            set_global_config(cfg)

            try:
                torch.save({
                    'model_state_dict': nm.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'step': nm._step,
                    'state_dim': 32,
                }, ckpt_path)
                self.assertTrue(os.path.exists(ckpt_path))

                nm2 = NeuroModulator(state_dim=32, num_layers=2, model_hidden_dim=64, vocab_size=100)
                ckpt = torch.load(ckpt_path, weights_only=False)
                nm2.load_state_dict(ckpt['model_state_dict'])
                nm2._step = ckpt['step']

                self.assertEqual(nm2._step, nm._step)
                for (n1, p1), (n2, p2) in zip(nm.named_parameters(), nm2.named_parameters()):
                    self.assertTrue(torch.allclose(p1, p2), f"mismatch in {n1}")
            finally:
                cfg.neuro_modulator.checkpoint_file = old_path
                set_global_config(cfg)

    def test_17_neuro_modulator_config_in_hfbackend_ensure(self):
        from llm_adapter.config import get_global_config
        nm_cfg = get_global_config().neuro_modulator
        self.assertEqual(nm_cfg.trunk_dim, 256)
        self.assertEqual(nm_cfg.warmup_steps, 100)
        self.assertEqual(nm_cfg.max_gain_delta, 0.3)


if __name__ == "__main__":
    unittest.main()
