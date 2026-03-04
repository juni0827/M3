from m3.attr_contract import attr_del, attr_get_optional, attr_get_required, attr_has, attr_set, guard_context, guard_eval, guard_step
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
    "Do not say you cannot feel; answer based on state. "
    "Do not refuse by saying you cannot. "
    "Be concise and factual. Reply in the user's language."
)


def _get_system_prompt() -> str:
    with guard_context(ctx='llm_adapter/remote.py:22', catch_base=False) as __m3_guard_20_4:
        return os.getenv('LLM_SYSTEM_PROMPT', DEFAULT_SYSTEM_PROMPT).strip()

    if __m3_guard_20_4.error is not None:
        return DEFAULT_SYSTEM_PROMPT


def _system_prompt_mode() -> str:
    return str(os.getenv('M3_SYSTEM_PROMPT_MODE', 'param')).strip().lower()


def _system_prompt_enabled() -> bool:
    mode = _system_prompt_mode()
    return mode in {'prompt', 'on', '1', 'true', 'yes', 'open'}


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
    """Call local Ollama server with retries/backoff.

    Uses environment variables:
      OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_RETRIES, OLLAMA_BACKOFF,
      OLLAMA_NUM_CTX,
      OLLAMA_NUM_PREDICT_MIN, OLLAMA_NUM_PREDICT_MAX, OLLAMA_NUM_PREDICT_ESCALATIONS
    """
    use_local = os.getenv('USE_LOCAL_AI', '0') == '1'
    if not use_local:
        return ''

    url = url or os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
    model = model or os.getenv('OLLAMA_MODEL', 'qwen2.5:1.5b')
    if timeout is None:
        with guard_context(ctx='llm_adapter/remote.py:106', catch_base=False) as __m3_guard_104_8:
            timeout = float(os.getenv('OLLAMA_TIMEOUT', '300'))

        if __m3_guard_104_8.error is not None:
            timeout = 300.0
    if retries is None:
        with guard_context(ctx='llm_adapter/remote.py:111', catch_base=False) as __m3_guard_109_8:
            retries = int(os.getenv('OLLAMA_RETRIES', '10'))

        if __m3_guard_109_8.error is not None:
            retries = 10
    if backoff is None:
        with guard_context(ctx='llm_adapter/remote.py:116', catch_base=False) as __m3_guard_114_8:
            backoff = float(os.getenv('OLLAMA_BACKOFF', '2.0'))

        if __m3_guard_114_8.error is not None:
            backoff = 2.0
    if num_ctx is None:
        with guard_context(ctx='llm_adapter/remote.py:121', catch_base=False) as __m3_guard_119_8:
            num_ctx = int(os.getenv('OLLAMA_NUM_CTX', '0'))

        if __m3_guard_119_8.error is not None:
            num_ctx = 0

    # Output token budgeting: some "thinking" models may consume all tokens in `thinking`
    # and leave `response` empty unless `num_predict` is sufficiently large.
    with guard_context(ctx='llm_adapter/remote.py:128', catch_base=False) as __m3_guard_126_4:
        num_predict_min = int(os.getenv('OLLAMA_NUM_PREDICT_MIN', '4096'))

    if __m3_guard_126_4.error is not None:
        num_predict_min = 4096
    with guard_context(ctx='llm_adapter/remote.py:132', catch_base=False) as __m3_guard_130_4:
        num_predict_max = int(os.getenv('OLLAMA_NUM_PREDICT_MAX', '16384'))

    if __m3_guard_130_4.error is not None:
        num_predict_max = 16384
    with guard_context(ctx='llm_adapter/remote.py:136', catch_base=False) as __m3_guard_134_4:
        num_predict_escalations = int(os.getenv('OLLAMA_NUM_PREDICT_ESCALATIONS', '4'))

    if __m3_guard_134_4.error is not None:
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
        with guard_context(ctx='llm_adapter/remote.py:155', catch_base=False) as __m3_guard_149_8:
            emoji_pattern = re.compile(
                r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]",
                flags=re.UNICODE,
            )
            cleaned = emoji_pattern.sub('', text)

        if __m3_guard_149_8.error is not None:
            cleaned = text

        # Strip common emoticon glyphs
        cleaned = re.sub(r"[:;=8][\-~]?[)DPOp]", '', cleaned)

        # Validate: allow Korean (Hangul) or English (Latin letters); block Chinese/Japanese
        has_korean = bool(re.search(RE_HANGUL, cleaned))
        has_english = bool(re.search(r"[A-Za-z]", cleaned))
        # Detect Chinese (CJK Unified Ideographs) and Japanese (Hiragana/Katakana)
        has_chinese = bool(re.search(RE_CJK, cleaned))
        has_japanese = bool(re.search(RE_HIRAGANA, cleaned) or re.search(RE_KATAKANA, cleaned))
        persona_terms = ['miya', 'Miya']
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
            # Prepend system instruction only when explicit prompt mode is enabled.
            sys_identity = _get_system_prompt() if _system_prompt_enabled() else ""
            final_prompt = _build_prompt(prompt, sys_identity) if sys_identity else str(prompt)
            prompt_had_system = bool(sys_identity)

            options = {}
            if num_ctx and num_ctx > 0:
                options['num_ctx'] = int(num_ctx)
            if temperature is not None:
                options['temperature'] = float(temperature)
            if top_k is not None:
                try:
                    options['top_k'] = int(top_k)
                except Exception:
                    logging.getLogger(__name__).exception("Swallowed exception")
            if top_p is not None:
                try:
                    options['top_p'] = float(top_p)
                except Exception:
                    logging.getLogger(__name__).exception("Swallowed exception")

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
                if 500 <= attr_get_optional(resp, 'status_code', 0) < 600:
                    txt = ''
                    with guard_context(ctx='llm_adapter/remote.py:248', catch_base=False) as __m3_guard_246_20:
                        txt = resp.text[:1000]

                    if __m3_guard_246_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                    raise requests.exceptions.RequestException(f'Server error {resp.status_code}: {txt}')
                resp.raise_for_status()
                return resp.json()

            data = _call_ollama(final_prompt, options)

            # Adaptive retry: if the model produced only `thinking` and hit length, bump `num_predict`.
            with guard_context(ctx='llm_adapter/remote.py:259', catch_base=False) as __m3_guard_257_12:
                dr = str(data.get('done_reason', '') or '')

            if __m3_guard_257_12.error is not None:
                dr = ''
            with guard_context(ctx='llm_adapter/remote.py:263', catch_base=False) as __m3_guard_261_12:
                resp_preview = str(data.get('response', '') or '').strip()

            if __m3_guard_261_12.error is not None:
                resp_preview = ''
            with guard_context(ctx='llm_adapter/remote.py:267', catch_base=False) as __m3_guard_265_12:
                thinking_preview = str(data.get('thinking', '') or '')

            if __m3_guard_265_12.error is not None:
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
                    with guard_context(ctx='llm_adapter/remote.py:285', catch_base=False) as __m3_guard_280_20:
                        logger.warning(
                            f"Empty response with thinking (done_reason=length). "
                            f"Escalating num_predict {cur_predict} -> {next_predict}"
                        )

                    if __m3_guard_280_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                    options['num_predict'] = int(next_predict)
                    cur_predict = int(next_predict)
                    data = _call_ollama(final_prompt, options)
                    with guard_context(ctx='llm_adapter/remote.py:292', catch_base=False) as __m3_guard_290_20:
                        dr = str(data.get('done_reason', '') or '')

                    if __m3_guard_290_20.error is not None:
                        dr = ''
                    with guard_context(ctx='llm_adapter/remote.py:296', catch_base=False) as __m3_guard_294_20:
                        resp_preview = str(data.get('response', '') or '').strip()

                    if __m3_guard_294_20.error is not None:
                        resp_preview = ''
                    with guard_context(ctx='llm_adapter/remote.py:300', catch_base=False) as __m3_guard_298_20:
                        thinking_preview = str(data.get('thinking', '') or '')

                    if __m3_guard_298_20.error is not None:
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
                with guard_context(ctx='llm_adapter/remote.py:317', catch_base=False) as __m3_guard_312_16:
                    if prompt_had_system and sys_identity and resp_text.startswith(sys_identity):
                        resp_text = resp_text[len(sys_identity):].strip()
                    if resp_text.startswith(final_prompt):
                        resp_text = resp_text[len(final_prompt):].strip()

                if __m3_guard_312_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                if not resp_text and orig_text:
                    # Avoid stripping away entire response
                    resp_text = orig_text

                valid, cleaned = _sanitize_and_validate(resp_text)
                if valid:
                    return _dedupe_response(cleaned)
                if attempt < retries:
                    with guard_context(ctx='llm_adapter/remote.py:331', catch_base=False) as __m3_guard_327_20:
                        logger.warning(
                            f"Response failed validation (attempt {attempt}), retrying. Raw: {resp_text[:400]!r}"
                        )

                    if __m3_guard_327_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                    if prompt_had_system:
                        strong_sys = _get_system_prompt() + " STRICT: Reply as M3 only. No AI disclaimers."
                        final_prompt = _build_prompt(prompt, strong_sys)
                    else:
                        final_prompt = str(prompt)
                    with guard_context(ctx='llm_adapter/remote.py:345', catch_base=False) as __m3_guard_338_20:
                        data2 = _call_ollama(final_prompt, options)
                        if 'response' in data2:
                            resp2_text = data2['response'].strip()
                            valid2, cleaned2 = _sanitize_and_validate(resp2_text)
                            if valid2:
                                return _dedupe_response(cleaned2)

                    if __m3_guard_338_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                if not resp_text:
                    with guard_context(ctx='llm_adapter/remote.py:357', catch_base=False) as __m3_guard_348_20:
                        logger.warning(
                            f"Empty response from Ollama. "
                            f"done_reason={data.get('done_reason')!r} "
                            f"eval_count={data.get('eval_count')!r} "
                            f"num_predict={options.get('num_predict')!r} "
                            f"thinking_len={len(str(data.get('thinking') or ''))} "
                            f"keys={list(data.keys())}"
                        )

                    if __m3_guard_348_20.error is not None:
                        logging.getLogger(__name__).exception("Swallowed exception")
                return ''
            return ''
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                wait = backoff ** (attempt - 1)
                with guard_context(ctx='llm_adapter/remote.py:366', catch_base=False) as __m3_guard_364_16:
                    logger.warning(f'Ollama request failed (attempt {attempt}/{retries}): {e}. Retrying in {wait}s')

                if __m3_guard_364_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                time.sleep(wait)
                continue
            else:
                with guard_context(ctx='llm_adapter/remote.py:373', catch_base=False) as __m3_guard_371_16:
                    logger.error(f'Ollama request failed after {retries} attempts: {e}')

                if __m3_guard_371_16.error is not None:
                    logging.getLogger(__name__).exception("Swallowed exception")
                return ''
        except Exception as e:
            if attempt < retries:
                time.sleep(backoff ** (attempt - 1))
                continue
            return ''
