import os
import time
import re
from typing import Tuple
import requests
import logging

logger = logging.getLogger('llm_adapter.remote')

DEFAULT_SYSTEM_PROMPT = (
    "You are M3. Respond as M3 using the provided M3_STATE. "
    "Do not claim to be an AI assistant or language model. "
    "Do not mention DeepSeek or any other persona. "
    "Do not say you cannot feel; answer based on state. "
    "Do not refuse by saying you cannot. "
    "Be concise and factual. Reply in the user's language."
)


def _get_system_prompt() -> str:
    try:
        return os.getenv('LLM_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT).strip()
    except Exception:
        return DEFAULT_SYSTEM_PROMPT


def _build_prompt(prompt: str, sys_identity: str) -> str:
    if sys_identity:
        stripped = prompt.lstrip()
        if stripped.startswith(sys_identity):
            return prompt
        return f"{sys_identity}\n\n{prompt}"


    return prompt


def _dedupe_response(text: str) -> str:
    if not text:
        return text
    # Normalize newlines
    norm = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not norm:
        return norm
    lines = [ln.rstrip() for ln in norm.split("\n")]
    # Remove consecutive duplicate lines
    deduped = []
    prev = None
    for ln in lines:
        if prev is not None and ln == prev:
            continue
        deduped.append(ln)
        prev = ln
    # Remove repeated blocks (exact duplicate paragraphs)
    paras = []
    seen = set()
    for block in "\n".join(deduped).split("\n\n"):
        blk = block.strip()
        if not blk:
            continue
        if blk in seen:
            continue
        seen.add(blk)
        paras.append(blk)
    return "\n\n".join(paras).strip()


def get_local_thinking(
    prompt: str,
    *,
    url: str = None,
    model: str = None,
    timeout: float = None,
    retries: int = None,
    backoff: float = None,
    num_ctx: int = None,
    temperature: float = None,
    top_k: int = None,
    top_p: float = None,
    max_len: int = None,
) -> str:
    """Call local Ollama/Deepseek server with retries/backoff.

    Uses environment variables:
      OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_RETRIES, OLLAMA_BACKOFF,
      OLLAMA_NUM_CTX,
      OLLAMA_NUM_PREDICT_MIN, OLLAMA_NUM_PREDICT_MAX, OLLAMA_NUM_PREDICT_ESCALATIONS
    """
    use_local = os.getenv('USE_LOCAL_AI', '1') == '1'
    if not use_local:
        return 'Local AI disabled'

    url = url or os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
    model = model or os.getenv('OLLAMA_MODEL', 'deepseek-r1:8b')
    if timeout is None:
        try:
            # Reasoning models (e.g. deepseek-r1) can take much longer; 60s is often too short.
            timeout = float(os.getenv('OLLAMA_TIMEOUT', '300'))
        except Exception:
            timeout = 300.0
    if retries is None:
        try:
            retries = int(os.getenv('OLLAMA_RETRIES', '10'))
        except Exception:
            retries = 10
    if backoff is None:
        try:
            backoff = float(os.getenv('OLLAMA_BACKOFF', '2.0'))
        except Exception:
            backoff = 2.0
    if num_ctx is None:
        try:
            num_ctx = int(os.getenv('OLLAMA_NUM_CTX', '0'))
        except Exception:
            num_ctx = 0

    # Output token budgeting: some "thinking" models may consume all tokens in `thinking`
    # and leave `response` empty unless `num_predict` is sufficiently large.
    try:
        num_predict_min = int(os.getenv('OLLAMA_NUM_PREDICT_MIN', '4096'))
    except Exception:
        num_predict_min = 4096
    try:
        num_predict_max = int(os.getenv('OLLAMA_NUM_PREDICT_MAX', '16384'))
    except Exception:
        num_predict_max = 16384
    try:
        num_predict_escalations = int(os.getenv('OLLAMA_NUM_PREDICT_ESCALATIONS', '4'))
    except Exception:
        num_predict_escalations = 4

    # Unicode ranges (explicit) to avoid mojibake from source encoding
    RE_HANGUL = r"[\uAC00-\uD7A3]"
    RE_CJK = r"[\u4E00-\u9FFF]"
    RE_HIRAGANA = r"[\u3040-\u309F]"
    RE_KATAKANA = r"[\u30A0-\u30FF]"
    prompt_has_korean = bool(re.search(RE_HANGUL, prompt))
    allow_english_when_korean = os.getenv('LLM_ALLOW_ENGLISH_WHEN_KOREAN', '1').lower() in ('1', 'true', 'yes', 'on')

    def _sanitize_and_validate(text: str) -> Tuple[bool, str]:
        # Remove common emojis
        try:
            emoji_pattern = re.compile(
                r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]",
                flags=re.UNICODE,
            )
            cleaned = emoji_pattern.sub('', text)
        except Exception:
            cleaned = text

        # Strip common emoticon glyphs
        cleaned = re.sub(r"[:;=8][\-~]?[)DPOp]", '', cleaned)

        # Validate: allow Korean (Hangul) or English (Latin letters); block Chinese/Japanese
        has_korean = bool(re.search(RE_HANGUL, cleaned))
        has_english = bool(re.search(r"[A-Za-z]", cleaned))
        # Detect Chinese (CJK Unified Ideographs) and Japanese (Hiragana/Katakana)
        has_chinese = bool(re.search(RE_CJK, cleaned))
        has_japanese = bool(re.search(RE_HIRAGANA, cleaned) or re.search(RE_KATAKANA, cleaned))
        persona_terms = ['miya', 'Miya', 'DeepSeek', 'Deepseek', 'DeepSeek R1', 'DeepSeekR1', 'DeepSeek R1']
        has_other_persona = any(term in cleaned for term in persona_terms)
        lower = cleaned.lower()
        forbidden_phrases = [
            'ai assistant',
            'language model',
            'as an ai',
            "i'm an ai",
            'i am an ai',
            'as a language model',
            'i am a language model',
            'i do not have feelings',
            "i don't have feelings",
            'cannot feel',
            'do not have feelings',
            'i cannot',
            "i can't",
        ]
        has_forbidden = any(phrase in lower for phrase in forbidden_phrases)

        # Block responses that contain Chinese or Japanese characters
        if has_chinese or has_japanese:
            valid = False
        else:
            if prompt_has_korean and not has_korean:
                if allow_english_when_korean and has_english:
                    valid = (not has_other_persona) and (not has_forbidden)
                else:
                    valid = False
            else:
                # Allow if Korean or English present, and no forbidden persona mention
                valid = (has_korean or has_english) and (not has_other_persona) and (not has_forbidden)
        return valid, cleaned.strip()

    for attempt in range(1, retries + 1):
        try:
            # Prepend system instruction to strongly enforce identity/format
            sys_identity = _get_system_prompt()
            final_prompt = _build_prompt(prompt, sys_identity)

            options = {}
            if num_ctx and num_ctx > 0:
                options['num_ctx'] = int(num_ctx)
            if temperature is not None:
                options['temperature'] = float(temperature)
            if top_k is not None:
                try:
                    options['top_k'] = int(top_k)
                except Exception:
                    pass
            if top_p is not None:
                try:
                    options['top_p'] = float(top_p)
                except Exception:
                    pass

            # Resolve output length budget (Ollama `num_predict`)
            requested_predict = None
            if max_len is not None:
                try:
                    requested_predict = int(max_len)
                except Exception:
                    requested_predict = None
            num_predict = requested_predict if requested_predict is not None else num_predict_min
            if num_predict_min and num_predict < num_predict_min:
                num_predict = num_predict_min
            if num_predict_max and num_predict > num_predict_max:
                num_predict = num_predict_max
            if num_predict:
                options['num_predict'] = int(num_predict)

            def _call_ollama(prompt_text: str, options_dict: dict) -> dict:
                payload = {'model': model, 'prompt': prompt_text, 'stream': False}
                if options_dict:
                    payload['options'] = options_dict
                resp = requests.post(url, json=payload, timeout=timeout)
                if 500 <= getattr(resp, 'status_code', 0) < 600:
                    txt = ''
                    try:
                        txt = resp.text[:1000]
                    except Exception:
                        pass
                    raise requests.exceptions.RequestException(f'Server error {resp.status_code}: {txt}')
                resp.raise_for_status()
                return resp.json()

            data = _call_ollama(final_prompt, options)

            # Adaptive retry: if the model produced only `thinking` and hit length, bump `num_predict`.
            # This is common for deepseek-r1 style models.
            try:
                dr = str(data.get('done_reason', '') or '')
            except Exception:
                dr = ''
            try:
                resp_preview = str(data.get('response', '') or '').strip()
            except Exception:
                resp_preview = ''
            try:
                thinking_preview = str(data.get('thinking', '') or '')
            except Exception:
                thinking_preview = ''

            if (not resp_preview) and thinking_preview and dr.lower() == 'length' and num_predict_escalations > 0:
                cur_predict = int(options.get('num_predict') or 0)
                for _ in range(num_predict_escalations):
                    if not cur_predict:
                        cur_predict = max(num_predict_min, 1)
                    next_predict = cur_predict * 2
                    if num_predict_max and next_predict > num_predict_max:
                        next_predict = int(num_predict_max)
                    if next_predict <= cur_predict:
                        break
                    try:
                        logger.warning(
                            f"Empty response with thinking (done_reason=length). "
                            f"Escalating num_predict {cur_predict} -> {next_predict}"
                        )
                    except Exception:
                        pass
                    options['num_predict'] = int(next_predict)
                    cur_predict = int(next_predict)
                    data = _call_ollama(final_prompt, options)
                    try:
                        dr = str(data.get('done_reason', '') or '')
                    except Exception:
                        dr = ''
                    try:
                        resp_preview = str(data.get('response', '') or '').strip()
                    except Exception:
                        resp_preview = ''
                    try:
                        thinking_preview = str(data.get('thinking', '') or '')
                    except Exception:
                        thinking_preview = ''
                    if resp_preview:
                        break

            if 'response' in data:
                resp_text = data['response']
                if resp_text is None:
                    resp_text = ''
                resp_text = str(resp_text).strip()
                orig_text = resp_text
                # If model echoed the system prompt or the full prompt, strip that prefix
                try:
                    if sys_identity and resp_text.startswith(sys_identity):
                        resp_text = resp_text[len(sys_identity):].strip()
                    if resp_text.startswith(final_prompt):
                        resp_text = resp_text[len(final_prompt):].strip()
                except Exception:
                    pass
                if not resp_text and orig_text:
                    # Avoid stripping away entire response
                    resp_text = orig_text

                valid, cleaned = _sanitize_and_validate(resp_text)
                if valid:
                    return _dedupe_response(cleaned)
                if attempt < retries:
                    try:
                        logger.warning(
                            f"Response failed validation (attempt {attempt}), retrying. Raw: {resp_text[:400]!r}"
                        )
                    except Exception:
                        pass
                    strong_sys = _get_system_prompt() + " STRICT: Reply as M3 only. No AI disclaimers."
                    final_prompt = _build_prompt(prompt, strong_sys)
                    try:
                        data2 = _call_ollama(final_prompt, options)
                        if 'response' in data2:
                            resp2_text = data2['response'].strip()
                            valid2, cleaned2 = _sanitize_and_validate(resp2_text)
                            if valid2:
                                return _dedupe_response(cleaned2)
                    except Exception:
                        pass
                if not resp_text:
                    try:
                        logger.warning(
                            f"Empty response from Ollama. "
                            f"done_reason={data.get('done_reason')!r} "
                            f"eval_count={data.get('eval_count')!r} "
                            f"num_predict={options.get('num_predict')!r} "
                            f"thinking_len={len(str(data.get('thinking') or ''))} "
                            f"keys={list(data.keys())}"
                        )
                    except Exception:
                        pass
                return f'Local Error: Unexpected or invalid response: {resp_text}'
            return f'Local Error: Unexpected response format: {data}'
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                wait = backoff ** (attempt - 1)
                try:
                    logger.warning(f'Ollama request failed (attempt {attempt}/{retries}): {e}. Retrying in {wait}s')
                except Exception:
                    pass
                time.sleep(wait)
                continue
            else:
                try:
                    logger.error(f'Ollama request failed after {retries} attempts: {e}')
                except Exception:
                    pass
                return f'Local Error: Request failed after {retries} attempts: {e}'
        except Exception as e:
            if attempt < retries:
                time.sleep(backoff ** (attempt - 1))
                continue
            return f'Local Error: {e}'
