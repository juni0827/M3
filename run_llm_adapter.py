 #!/usr/bin/env python3
"""
Simple entrypoint for starting / testing the LLM adapter.
- If `m3.core.M3Core` is available, creates a core and attaches the LLM adapter via `attach_llm_to_core`.
- Otherwise, creates a standalone `TorchConversationalPolicy` (no generate call).

Run from project root:
    python run_llm_adapter.py
or as a module:
    python -m run_llm_adapter
"""
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "You are M3. Respond as M3 using the provided M3_STATE. "
    "Do not claim to be an AI assistant or language model. "
    "Do not mention DeepSeek or any other persona. "
    "Do not say you cannot feel; answer based on state. "
    "Be concise and factual. Reply in the user's language."
)

os.environ.setdefault('LLM_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT)
os.environ.setdefault('USE_LOCAL_AI', '1')
os.environ.setdefault('DEEPSEEK_ONLY', '1')
os.environ.setdefault('M3_AUTO_ATTACH_LLM', '0')

# Reasoning models often need a larger generation budget: `thinking` may consume tokens before `response` appears.
os.environ.setdefault('OLLAMA_NUM_PREDICT_MIN', '4096')
os.environ.setdefault('OLLAMA_NUM_PREDICT_MAX', '16384')
os.environ.setdefault('OLLAMA_NUM_PREDICT_ESCALATIONS', '4')
# Local inference can be slow; 60s is often too short.
os.environ.setdefault('OLLAMA_TIMEOUT', '300')

# Try to set the max ctx for the current Ollama model when not explicitly provided.
def _set_default_ollama_ctx():
    if os.getenv('OLLAMA_NUM_CTX'):
        return
    model = os.getenv('OLLAMA_MODEL', 'deepseek-r1:8b')
    detected = None
    try:
        out = subprocess.check_output(
            ['ollama', 'show', model],
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='ignore',
        )
        for pattern in (
            r'context length\\s+(\\d+)',
            r'context_length\\s*[:=]\\s*(\\d+)',
            r'n_ctx\\s*[:=]\\s*(\\d+)',
        ):
            m = re.search(pattern, out, flags=re.IGNORECASE)
            if m:
                detected = m.group(1)
                break
    except Exception:
        detected = None
    if not detected:
        detected = os.getenv('M3_OLLAMA_NUM_CTX', '32768')
    os.environ.setdefault('OLLAMA_NUM_CTX', str(detected))

_set_default_ollama_ctx()

# Ensure project root is on sys.path when run from elsewhere
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

try:
    from llm_adapter import attach_llm_to_core, TorchConversationalPolicy
except Exception as e:
    print("Failed to import llm_adapter package:", e)
    raise

# Try to attach to real M3 core if available
try:
    # Prefer canonical M3 core class; try common names
    from m3 import core as m3core
    core_cls = None
    for name in ('M3Core', 'M3ConsciousnessCore', 'M3CoreEngine'):
        if hasattr(m3core, name):
            core_cls = getattr(m3core, name)
            break
    if core_cls is None:
        # Fallback: pick first class with 'Core' in its name
        for attr in dir(m3core):
            if 'Core' in attr and attr[0].isupper():
                core_cls = getattr(m3core, attr)
                break
    if core_cls is None:
        raise ImportError('No suitable Core class found in m3.core')
    print(f"Found core class: {core_cls.__name__}. Initializing core and attaching adapter...")
    core = core_cls()
    # If core already attached an adapter, reuse it unless forced to reattach
    force_attach = os.getenv('FORCE_ATTACH_LLM', '0').lower() in ('1', 'true', 'yes', 'on')
    if getattr(core, 'llm_adapter', None) is not None and not force_attach:
        adapter = core.llm_adapter
        print("Reusing existing adapter:", type(adapter))
    else:
        # Instantiate the real adapter immediately (using local fine-tuned model)
        adapter = TorchConversationalPolicy()
        # bind core reference on the adapter and attach
        try:
            adapter.core = core
        except Exception:
            pass
        adapter = attach_llm_to_core(core, adapter=adapter)
        print("Attached adapter:", type(adapter))
    print("LLM adapter ready. Use `core.llm_adapter.generate(prompt)` to generate.")
except Exception as e:
    print("M3Core import/init failed:", e)
    print("")
    print("This script requires the real M3 core to attach the LLM adapter. No DummyCore will be used.")
    print("Please ensure the `m3` package in the workspace is importable and that `M3Core` can be initialized.")
    print("")
    print("Quick checks:")
    print("  - Run: python -c \"from m3.core import M3Core; print('M3Core OK')\"")
    print("  - Ensure your current working directory is the project root (contains folders 'm3' and 'llm_adapter')")
    print("")
    raise RuntimeError("M3Core not available — cannot attach LLM adapter without real core.")
 
print("Entrypoint finished.")

# Lightweight state stepping to keep M3 internals moving in REPL mode.
def _parse_steps(env_key: str, default_val: str) -> int:
    try:
        return int(os.getenv(env_key, default_val))
    except Exception:
        try:
            return int(default_val)
        except Exception:
            return 0


def _advance_core_state(core_obj, steps=None):
    if steps is None:
        steps = _parse_steps('M3_STEPS_PER_TURN', '20')
    if steps <= 0:
        return
    for _ in range(steps):
        try:
            if hasattr(core_obj, '_single_consciousness_step'):
                core_obj._single_consciousness_step()
            elif hasattr(core_obj, 'generate_internal_events'):
                core_obj.generate_internal_events()
            elif hasattr(core_obj, 'message_bus') and core_obj.message_bus is not None:
                core_obj.message_bus.step()
        except Exception:
            pass

# Keep the process alive and provide a minimal interactive REPL for testing
try:
    print("Interactive LLM REPL started. Type a prompt and press Enter. Ctrl-C to exit.")
    while True:
        prompt = input('prompt> ')
        if prompt is None:
            break
        prompt = prompt.strip()
        if prompt == '':
            continue
        try:
            steps_before = _parse_steps('M3_STEPS_BEFORE_TURN', os.getenv('M3_STEPS_PER_TURN', '20'))
            _advance_core_state(core, steps_before)
            if hasattr(core, 'handle_user_message'):
                resp = core.handle_user_message(prompt)
            else:
                resp = core.llm_adapter.generate(prompt)
            print('\n=== RESPONSE ===')
            print(resp)
            print('================\n')
            try:
                if os.getenv('M3_SAVE_EVERY_TURN', '0') in ('1','true','TRUE','yes','on'):
                    if hasattr(core, '_save_checkpoint'):
                        core._save_checkpoint()
            except Exception:
                pass
            steps_after = _parse_steps('M3_STEPS_AFTER_TURN', '5')
            _advance_core_state(core, steps_after)
        except Exception as e:
            print('generate() error:', e)
except KeyboardInterrupt:
    print('\nREPL interrupted by user. Exiting.')
    try:
        if 'core' in globals() and hasattr(core, '_save_checkpoint'):
            core._save_checkpoint()
    except Exception:
        pass
except Exception as e:
    print('REPL fatal error:', e)
