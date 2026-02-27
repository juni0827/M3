#!/usr/bin/env python3
"""M3 command-line runner with integrated control profile and runtime setting APIs."""

import os
import sys
import threading
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# -------------------------
# 기본 런타임 환경
# -------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are M3. Respond as m3. "
    "Use internal control state for response policy, not as quoted prompt context. "
    "Do not claim to be an AI assistant or language model. "
    "Do not say you cannot feel. "
    "Do not report phi/qualia/state values unless the user explicitly asks for them. "
    "Be concise and factual. Reply in the user's language."
)

os.environ.setdefault("LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("M3_CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("M3_CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("M3_TORCH_DEVICE", "cuda")

os.environ.setdefault("M3_USE_HF", "1")
os.environ.setdefault("M3_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
os.environ.setdefault("M3_HF_QUANTIZE", "4bit")
os.environ.setdefault("M3_SYSTEM_PROMPT_MODE", "param")
os.environ.setdefault("M3_HF_MAX_TOKENS", "320")
os.environ.setdefault("M3_HF_MAX_INPUT", "2048")
os.environ.setdefault("M3_USER_MAX_LEN", "320")
os.environ.setdefault("M3_AUTONOMY_MAX_LEN", "160")
os.environ.setdefault("M3_AUTO_ATTACH_LLM", "0")
os.environ.setdefault("M3_HF_CUDA_FAILOVER", "1")
os.environ.setdefault("M3_DISABLE_OLLAMA", "1")
os.environ.setdefault("USE_LOCAL_AI", "0")
os.environ.setdefault("M3_TRAIN_RECORD_SCOPE", "user_only")
os.environ.setdefault("M3_TRAIN_EXCLUDE_SIMILAR_CONTEXT", "1")
os.environ.setdefault("M3_TRAIN_LANG_MATCH", "1")
os.environ.setdefault("M3_STATE_CONTEXT_POLICY", "off")
os.environ.setdefault("M3_ENABLE_DECODE_CONTROL", "1")
os.environ.setdefault("M3_CONTROL_TERM_PENALTY", "8.0")
os.environ.setdefault("M3_CONTROL_IDENTITY_LOCK", "1")
os.environ.setdefault("M3_CONTROL_IDENTITY_PENALTY", "14.0")
os.environ.setdefault("M3_CONTROL_RETRY", "2")
os.environ.setdefault("M3_CONTROL_MIN_RESPONSE_CHARS", "8")
os.environ.setdefault("M3_CONTROL_MAX_TEMP", "0.65")
os.environ.setdefault("M3_CONTROL_MAX_TOP_K", "40")
os.environ.setdefault("M3_CONTROL_SELECTION_MODE", "state")
os.environ.setdefault("M3_CONTROL_AUTO_FULL", "1")
os.environ.setdefault("M3_CONTROL_HEALTH_WINDOW", "24")
os.environ.setdefault("M3_CONTROL_HEALTH_WINDOW_SEC", "180")
os.environ.setdefault("M3_CONTROL_QUALITY_TARGET", "0.42")

# NeuroModulator: weight-level M3 consciousness control
os.environ.setdefault("M3_ENABLE_CONTROL_BRIDGE", "1")
os.environ.setdefault("M3_ENABLE_NEURO_MODULATOR", "1")
os.environ.setdefault("M3_NEURO_STRENGTH", "1.0")
os.environ.setdefault("M3_NEUROMOD_LR", "1e-4")
os.environ.setdefault("M3_NEURO_STATE_DIM", "256")
os.environ.setdefault("M3_NEURO_HIDDEN_RANK", "16")
os.environ.setdefault("M3_NEURO_LOGIT_RANK", "32")
os.environ.setdefault("M3_NEURO_TRUNK_DIM", "256")

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

for _name in (
    "httpx",
    "httpcore",
    "hpack",
    "h2",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "accelerate",
    "bitsandbytes",
):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger("llm_adapter.tokenization").setLevel(logging.WARNING)

try:
    from transformers.utils import logging as _tlog
    _tlog.set_verbosity_error()
    try:
        _tlog.disable_progress_bar()
    except Exception:
        pass
except Exception:
    pass

try:
    from huggingface_hub.utils import logging as _hlog
    try:
        _hlog.set_verbosity_error()
    except Exception:
        pass
    try:
        _hlog.disable_progress_bars()
    except Exception:
        pass
except Exception:
    pass

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Centralize adapter logs under docs_tests_data by default.
default_log_dir = (project_root / "docs_tests_data").resolve()
try:
    default_log_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
default_log_path = str((default_log_dir / "llm_adapter.log").resolve())
os.environ.setdefault("LLM_ADAPTER_LOG_DIR", str(default_log_dir))
os.environ.setdefault("LLM_ADAPTER_LOG_PATH", default_log_path)
os.environ.setdefault("LLM_ADAPTER_LOG", default_log_path)

print(f"Project root: {project_root}")

from llm_adapter import attach_llm_to_core, TorchConversationalPolicy
import m3.m3_core as m3core

core_cls = None
for name in ("M3Core", "M3ConsciousnessCore", "M3CoreEngine"):
    if hasattr(m3core, name):
        core_cls = getattr(m3core, name)
        break
if core_cls is None:
    for attr in dir(m3core):
        if "Core" in attr and attr[0].isupper():
            core_cls = getattr(m3core, attr)
            break
if core_cls is None:
    raise ImportError("No suitable Core class found in m3.m3_core")

print(f"Found core class: {core_cls.__name__}. Initializing...")
core = core_cls()

force_attach = os.getenv("FORCE_ATTACH_LLM", "0").lower() in ("1", "true", "yes", "on")
if getattr(core, "llm_adapter", None) is not None and not force_attach:
    adapter = core.llm_adapter
    required_control_methods = (
        "_control_allows",
        "_note_control_health",
        "_control_selection_mode",
        "_bridge_enabled",
        "_bridge_enabled_safe",
    )
    has_required_control = all(callable(getattr(adapter, m, None)) for m in required_control_methods)
    if not has_required_control:
        print("Existing adapter is missing new control hooks; rebuilding adapter.")
        force_attach = True
    else:
        print("Reusing existing adapter:", type(adapter))

if getattr(core, "llm_adapter", None) is None or force_attach:
    adapter = TorchConversationalPolicy()
    try:
        adapter.core = core
    except Exception:
        pass
    adapter = attach_llm_to_core(core, adapter=adapter)
    print("Attached adapter:", type(adapter))
else:
    pass

_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)
        try:
            sys.stdout.flush()
        except Exception:
            pass


def _bool_env(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _coerce_env_value(text: str):
    value = text.strip()
    low = value.lower()
    if low in ("1", "true", "yes", "on"):
        return "1"
    if low in ("0", "false", "no", "off"):
        return "0"
    if value == "":
        return ""
    try:
        if "." in value:
            return str(float(value))
        return str(int(value))
    except ValueError:
        return value


def _set_env_vars(kv: Dict[str, str]) -> List[Tuple[str, str, str]]:
    changed = []
    for k, v in kv.items():
        before = os.environ.get(k)
        os.environ[k] = v
        changed.append((k, str(before), v))
    return changed


def _on_response(text):
    _safe_print("\n" + "=" * 50)
    _safe_print("  M3 RESPONSE")
    _safe_print("=" * 50)
    _safe_print(text)
    _safe_print("=" * 50)
    _safe_print("prompt> ", end="", flush=True)


def _on_spontaneous(text, q_speak, lam):
    if not _bool_env("M3_SHOW_AUTONOMY", False):
        return
    _safe_print(f"\n{'-' * 50}")
    _safe_print(f"[AUTONOMY] Q_speak={q_speak:.3f} lam={lam:.3f}")
    _safe_print("-" * 50)
    _safe_print(text)
    _safe_print("-" * 50)
    _safe_print("prompt> ", end="", flush=True)


adapter._on_response = _on_response
adapter._on_spontaneous = _on_spontaneous


def _start_research(max_cycles: int = 20, topic: str = None, force: bool = True):
    try:
        os.environ["M3_RESEARCH_FORCE"] = "1" if force else "0"
    except Exception:
        pass

    def _run():
        try:
            if hasattr(core, "run_autonomous_research"):
                try:
                    core.run_autonomous_research(max_cycles=int(max_cycles), topic=topic)
                except TypeError:
                    core.run_autonomous_research(max_cycles=int(max_cycles))
            else:
                _safe_print("  M3.autonomous_research is not available.")
        except Exception as e:
            _safe_print(f"  research failed: {e}")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


def _start_creation(cycles: int = 20, topic: str = None):
    def _run():
        try:
            if hasattr(core, "run_autonomous_creation"):
                try:
                    core.run_autonomous_creation(cycles=int(cycles), topic=topic)
                except TypeError:
                    core.run_autonomous_creation(cycles=int(cycles))
            else:
                _safe_print("  M3.autonomous_creation is not available.")
        except Exception as e:
            _safe_print(f"  creation failed: {e}")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


def _start_learning(dataset_root: str, max_iterations: int = -1):
    if not hasattr(core, "run_autonomous_learning"):
        _safe_print("  run_autonomous_learning is not available.")
        return None

    def _run():
        try:
            core.run_autonomous_learning(
                dataset_root=dataset_root,
                max_iterations=int(max_iterations),
            )
        except TypeError:
            core.run_autonomous_learning(dataset_root, int(max_iterations))
        except Exception as e:
            _safe_print(f"  learning failed: {e}")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


PROFILE_DEFS = {
    "obs-only": {
        "description": "Observation only. keep sampling/bridge/quality/token-q off.",
        "vars": {
            "M3_HF_ENABLE_M3_SAMPLER": "0",
            "M3_ENABLE_CONTROL_BRIDGE": "0",
            "LLM_ADAPTER_ENABLE_TOKEN_VALUE_BIAS": "0",
            "M3_ENABLE_QUALITY_GATE": "0",
            "M3_HF_CORE_UPDATE_INTERVAL": "0",
        },
    },
    "sampler-only": {
        "description": "Adaptive sampling only.",
        "vars": {
            "M3_HF_ENABLE_M3_SAMPLER": "1",
            "M3_ENABLE_CONTROL_BRIDGE": "0",
            "LLM_ADAPTER_ENABLE_TOKEN_VALUE_BIAS": "0",
            "M3_ENABLE_QUALITY_GATE": "0",
            "M3_HF_CORE_UPDATE_INTERVAL": "48",
        },
    },
    "bridge-only": {
        "description": "ControlBridge only.",
        "vars": {
            "M3_HF_ENABLE_M3_SAMPLER": "0",
            "M3_ENABLE_CONTROL_BRIDGE": "1",
            "LLM_ADAPTER_ENABLE_TOKEN_VALUE_BIAS": "0",
            "M3_ENABLE_QUALITY_GATE": "0",
            "M3_BRIDGE_STATE_DIM": "256",
            "M3_BRIDGE_STRENGTH": "1.0",
            "M3_HF_CORE_UPDATE_INTERVAL": "8",
        },
    },
    "tokenq-only": {
        "description": "Token-Q (logit bias) only.",
        "vars": {
            "M3_HF_ENABLE_M3_SAMPLER": "0",
            "M3_ENABLE_CONTROL_BRIDGE": "0",
            "LLM_ADAPTER_ENABLE_TOKEN_VALUE_BIAS": "1",
            "M3_ENABLE_QUALITY_GATE": "0",
            "M3_HF_CORE_UPDATE_INTERVAL": "64",
        },
    },
    "quality-only": {
        "description": "Quality gate only.",
        "vars": {
            "M3_HF_ENABLE_M3_SAMPLER": "0",
            "M3_ENABLE_CONTROL_BRIDGE": "0",
            "LLM_ADAPTER_ENABLE_TOKEN_VALUE_BIAS": "0",
            "M3_ENABLE_QUALITY_GATE": "1",
            "M3_HF_CORE_UPDATE_INTERVAL": "8",
        },
    },
}


def _parse_profile_name(raw: str) -> str:
    return str(raw or "").strip().lower()


def _apply_profile(name: str):
    name = _parse_profile_name(name)
    cfg = PROFILE_DEFS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown profile: {name}. Available: {', '.join(sorted(PROFILE_DEFS))}")
    set_log = _set_env_vars(cfg["vars"])
    _safe_print(f"[profile:{name}] {cfg['description']}")
    for k, before, after in set_log:
        _safe_print(f"  {k}: {before!r} -> {after!r}")
    return name


def _snapshot_state():
    gs = []
    for k, default in [
        ("M3_CONTROL_SELECTION_MODE", "state"),
        ("M3_CONTROL_AUTO_FULL", "1"),
        ("M3_CONTROL_HEALTH_WINDOW", "24"),
        ("M3_CONTROL_HEALTH_WINDOW_SEC", "180"),
        ("M3_HF_ENABLE_M3_SAMPLER", "1"),
        ("M3_ENABLE_CONTROL_BRIDGE", "0"),
        ("LLM_ADAPTER_ENABLE_TOKEN_VALUE_BIAS", "0"),
        ("M3_ENABLE_QUALITY_GATE", "0"),
        ("M3_HF_CORE_UPDATE_INTERVAL", "8"),
        ("M3_BRIDGE_STRENGTH", "1.0"),
        ("M3_BRIDGE_STATE_DIM", "256"),
        ("M3_BRIDGE_ABLATION", "none"),
    ]:
        gs.append(f"{k}={os.getenv(k, default)}")
    return gs


def _normalize_control_mode(value: str) -> str:
    raw = str(value or "").strip().lower()
    if raw in ("", "off", "0", "false", "no", "disable", "disabled"):
        return "off"
    if raw in ("1", "state", "state_only", "context", "context_only", "low"):
        return "state"
    if raw in ("2", "memory", "mid", "medium", "mixed"):
        return "memory"
    if raw in ("3", "full", "high", "strict", "all", "on", "true", "yes"):
        return "full"
    if raw in ("auto", "adaptive", "self", "self_adjust"):
        return "auto"
    return "state"


def _print_status():
    try:
        energy = core.energy_ctrl.cognitive_energy
        arousal = core.qualia.arousal
        valence = core.qualia.valence
    except Exception:
        energy = arousal = valence = None

    _safe_print("[status]")
    _safe_print(f"  auto loop: {getattr(adapter, '_auto_running', False)}")
    if energy is not None:
        _safe_print(f"  energy={energy:.2f} arousal={arousal:.3f} valence={valence:.3f}")
    else:
        _safe_print("  core state unavailable")

    for line in _snapshot_state():
        _safe_print(f"  {line}")

    mode_desc_map = {
        "off": "off: M3 제어 비활성화, 기본 LLM 응답만 사용",
        "state": "state: M3 상태 컨텍스트만 포함하여 제어",
        "memory": "memory: 상태 + 대화/연구/지식 메모리 회상 주입",
        "full": "full: 상태/메모리 + bridge/sampler/token-critic/quality gate 모두 활성",
        "auto": "auto: 상황에 따라 제어 강도를 자동으로 조절(adaptive self-adjust)",
    }
    mode = _normalize_control_mode(os.environ.get("M3_CONTROL_SELECTION_MODE", "state"))
    resolved_mode = mode
    if mode == "auto" and hasattr(adapter, "_control_selection_mode"):
        try:
            resolved_mode = str(adapter._control_selection_mode())
        except Exception:
            resolved_mode = "auto"
    if mode == "auto" and mode != resolved_mode:
        _safe_print(f"  M3_CONTROL_SELECTION_MODE={mode} -> resolved: {resolved_mode}")
        mode_for_desc = resolved_mode
    else:
        _safe_print(f"  M3_CONTROL_SELECTION_MODE={mode}")
        mode_for_desc = mode
    if mode_for_desc not in mode_desc_map:
        mode_for_desc = mode
    _safe_print(f"  control level: {mode_desc_map.get(mode_for_desc, mode_for_desc)}")

    if hasattr(core, "growing_som") and hasattr(core.growing_som, "get_topology_health"):
        try:
            health = core.growing_som.get_topology_health()
            _safe_print("  topology: " + ", ".join(f"{k}={v:.4f}" for k, v in health.items()))
        except Exception:
            _safe_print("  topology: (not available)")
    if hasattr(core, "global_workspace"):
        gw = core.global_workspace
        try:
            _safe_print(f"  gw allowlist size: {len(getattr(gw, '_policy_param_allowlist', []))}")
        except Exception:
            pass

    # NeuroModulator status
    try:
        from llm_adapter.llm_core import HFBackend
        if HFBackend.is_available():
            hf = HFBackend.get_instance()
            ns = hf._neuro_status()
            if ns.get('active'):
                _safe_print(
                    f"  neuro_modulator: active step={ns['step']} "
                    f"layers={ns['num_layers']} params={ns['params']}"
                )
            else:
                _safe_print("  neuro_modulator: inactive")
    except Exception:
        pass


def _print_help():
    _safe_print("Commands:")
    _safe_print("  :help                          show this help")
    _safe_print("  :status                        show run/core status")
    _safe_print("  :pause / :resume               autonomy loop control")
    _safe_print("  :research [cycles] [topic...]   start autonomous research")
    _safe_print("  :create [cycles] [topic...]     start autonomous creation")
    _safe_print("  :learn <dataset_root> [iters]   run autonomous learning")
    _safe_print("  :topology                      print topology health")
    _safe_print("  :control <off|state|memory|full|auto> set M3 제어 레벨")
    _safe_print("  :control list                   list control presets")
    _safe_print("  :profile <name>                 set profile (obs-only|sampler-only|bridge-only|tokenq-only|quality-only)")
    _safe_print("  :profile list                   list profiles")
    _safe_print("  :env [KEY VALUE ...]           print or set runtime env values")
    _safe_print("  :set KEY VALUE                 set one env value")
    _safe_print("  :shadow [N]                    print latest N allowlist shadow writes")
    _safe_print("  :q / :quit / exit             quit")


def _print_topology():
    _safe_print("[topology]")
    if not hasattr(core, "growing_som"):
        _safe_print("  core.growing_som unavailable")
        return
    gs = core.growing_som
    for name, fn in [
        ("stats", "get_statistics"),
        ("topology", "get_topology_health"),
        ("state", "get_network_state"),
    ]:
        try:
            fn_obj = getattr(gs, fn)
        except Exception:
            continue
        try:
            val = fn_obj()
            _safe_print(f"  {name}: {val}")
        except Exception as exc:
            _safe_print(f"  {name}: error={exc}")


def _print_shadow_writes(limit: int = 20):
    if not hasattr(core, "global_workspace"):
        _safe_print("  global workspace unavailable")
        return
    ws = core.global_workspace
    logs = []
    try:
        logs = ws.get_policy_param_shadow_writes(limit)
    except Exception:
        logs = []
    if not logs:
        _safe_print("  shadow log empty")
        return
    for item in logs:
        _safe_print(
            f"  {item.get('timestamp')} source={item.get('source')} "
            f"{item.get('param')} {item.get('current_value')} -> {item.get('requested_value')}"
        )


def _handle_command(raw: str) -> bool:
    cmdline = raw.strip()
    if not cmdline.startswith(":"):
        return False
    parts = cmdline.split()
    if not parts:
        return True

    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in (":q", ":quit", ":exit"):
        raise SystemExit
    if cmd == ":help":
        _print_help()
        return True
    if cmd == ":status":
        _print_status()
        return True
    if cmd == ":pause":
        adapter.stop_autonomy_loop()
        _safe_print("  autonomy paused")
        return True
    if cmd == ":resume":
        adapter.start_autonomy_loop()
        _safe_print("  autonomy resumed")
        return True
    if cmd == ":topology":
        _print_topology()
        return True
    if cmd == ":research":
        cycles = 20
        topic = None
        if args:
            try:
                cycles = int(args[0])
                if len(args) > 1:
                    topic = " ".join(args[1:])
            except ValueError:
                topic = " ".join(args)
        _safe_print(f"  start research (cycles={cycles}, topic={topic})")
        _start_research(max_cycles=cycles, topic=topic, force=True)
        return True
    if cmd == ":create":
        cycles = 20
        topic = None
        if args:
            try:
                cycles = int(args[0])
                if len(args) > 1:
                    topic = " ".join(args[1:])
            except ValueError:
                topic = " ".join(args)
        _safe_print(f"  start creation (cycles={cycles}, topic={topic})")
        _start_creation(cycles=cycles, topic=topic)
        return True
    if cmd == ":learn":
        if not args:
            _safe_print("  usage: :learn <dataset_root> [iters]")
            return True
        dataset_root = args[0]
        iters = -1
        if len(args) > 1:
            try:
                iters = int(args[1])
            except ValueError:
                pass
        _safe_print(f"  start autonomous learning: {dataset_root}, iters={iters}")
        _start_learning(dataset_root=dataset_root, max_iterations=iters)
        return True
    if cmd == ":profile":
        if not args:
            _safe_print("  usage: :profile list | <name>")
            return True
        if args[0].lower() == "list":
            for name, cfg in sorted(PROFILE_DEFS.items()):
                _safe_print(f"  {name}: {cfg['description']}")
            return True
        try:
            name = _apply_profile(args[0])
            _safe_print(f"  active profile: {name}")
        except Exception as e:
            _safe_print(f"  error: {e}")
        return True
    if cmd == ":control":
        if not args:
            _safe_print("  usage: :control off|state|memory|full|auto")
            return True
        if args[0].lower() == "list":
            _safe_print("  control presets:")
            _safe_print("    off      : disable M3 제어, 기본 LLM 반응만 사용")
            _safe_print("    state    : M3 상태 컨텍스트만 포함")
            _safe_print("    memory   : 상태 + dialog/research/knowledge 회상 주입")
            _safe_print("    full     : state/memory + bridge/sampler/token-critic/quality gate")
            _safe_print("    auto     : adaptive self-adjust control")
            return True
        mode = _normalize_control_mode(args[0])
        before = _normalize_control_mode(os.environ.get("M3_CONTROL_SELECTION_MODE", "state"))
        os.environ["M3_CONTROL_SELECTION_MODE"] = mode
        _safe_print(f"  control mode: {before} -> {mode}")
        return True
    if cmd == ":env":
        if not args:
            for k in sorted(_snapshot_state()):
                _safe_print(f"  {k}")
            return True
        if len(args) == 1:
            key = args[0]
            _safe_print(f"  {key}={os.environ.get(key)}")
            return True
        kv = {args[i]: _coerce_env_value(args[i + 1]) for i in range(0, len(args) - 1, 2)}
        setlog = _set_env_vars(kv)
        for k, before, after in setlog:
            _safe_print(f"  {k}: {before!r} -> {after!r}")
        return True
    if cmd == ":set":
        if len(args) < 2:
            _safe_print("  usage: :set KEY VALUE")
            return True
        key = args[0]
        value = _coerce_env_value(" ".join(args[1:]))
        before, after = os.environ.get(key), value
        os.environ[key] = value
        _safe_print(f"  set {key}: {before!r} -> {after!r}")
        return True
    if cmd == ":shadow":
        n = 20
        if args:
            try:
                n = max(1, int(args[0]))
            except ValueError:
                n = 20
        _print_shadow_writes(n)
        return True

    _safe_print(f"  unknown command: {cmd}. :help for command list.")
    return True


print("\nM3 adapter ready.")
print("  GPU/Device:", os.environ.get("CUDA_VISIBLE_DEVICES"), "/", os.environ.get("M3_TORCH_DEVICE"))
print("  initial profile vars:")
for line in _snapshot_state():
    _safe_print(f"  {line}")
print("  command tips: :help, :control list, :profile list, :topology, :status, :shadow 20")
print("  exit: :q, :quit, :exit")

adapter.start_autonomy_loop()

try:
    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:
            break

        if prompt is None:
            break
        prompt = str(prompt).strip()
        if not prompt:
            continue

        if prompt.startswith(":"):
            handled = _handle_command(prompt)
            if handled:
                continue

        adapter.submit_user_message(prompt)
except KeyboardInterrupt:
    _safe_print("\n[quit] Ctrl-C")
except SystemExit:
    pass
except Exception as e:
    _safe_print(f"\nFatal error: {e}")
finally:
    _safe_print("[shutdown] stopping autonomy loop")
    adapter.stop_autonomy_loop()
    try:
        if hasattr(core, "_save_checkpoint"):
            core._save_checkpoint()
            _safe_print("  checkpoint saved")
    except Exception:
        pass
    _safe_print("goodbye.")
