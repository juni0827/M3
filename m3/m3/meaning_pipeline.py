from __future__ import annotations

from m3.attr_contract import attr_del, attr_get_optional, attr_get_required, attr_has, attr_set, guard_context, guard_eval, guard_step
import hashlib
import json
import os
import re
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

MEANING_STATE_SCHEMA_VERSION = "1.0"
RESPONSE_PLAN_SCHEMA_VERSION = "1.0"

_IDENTITY_TOKENS = {
    "너", "난", "나", "저", "자아", "정체성", "의식", "내가", "내", "저자",
    "너는", "당신", "ai", "인공지능", "봇", "모델"
}
_WORLD_TOKENS = {
    "세계", "지구", "환경", "현실", "상황", "공간", "위치", "행동", "연구",
    "system", "시스템", "서버", "데이터", "과거", "현재", "미래", "시간"
}
_TASK_TOKENS = {
    "하세요", "해줘", "원해", "도와", "도움", "분석", "설명", "요약", "정리", "계획", "실행",
    "만들", "작성", "생성", "학습", "평가", "판단", "결정"
}
_SELF_TOKEN_HINTS = {
    "정체성", "의식", "기억", "감정", "목표", "인지", "메타", "자기", "자아"
}
_UNKNOWN_INTENT_TOKENS = {"무엇", "뭐", "어떻게", "어느", "왜", "왜냐하면"}


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9가-힣']+", _normalize_text(text))


def _safe_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _turn_id() -> str:
    return str(uuid.uuid4())


def _guess_intent(text: str, tokens: List[str]) -> str:
    token_set = set(tokens)
    if token_set & {"뭐야", "뭐임", "누구", "누군가", "누가", "너는", "너는뭐", "너는뭐냐", "너는뭐냐"}:
        return "identity_query"
    if token_set & {"상태", "지금", "현재", "어떻게", "왜", "의미", "의문", "무슨", "무엇"}:
        return "state_query"
    if token_set & {"세계", "지구", "환경", "규칙", "정책", "규범", "정의"}:
        return "world_query"
    if _TASK_TOKENS & token_set:
        return "task_request"
    if _UNKNOWN_INTENT_TOKENS & token_set:
        return "clarification"
    if token_set & {"설명", "요약", "정리", "해석"}:
        return "meta_control"
    if token_set & {"너", "당신", "ai", "모델"}:
        return "identity_query"
    return "unknown"


def _detect_lang(text: str) -> str:
    if re.search(r"[ㄱ-ㅎ|ㅏ-ㅣ|가-힣]", text):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[а-яА-Я]", text):
        return "ru"
    return "en"


def _extract_entities(text: str, limit: int = 12) -> List[Dict[str, Any]]:
    tokens = _tokenize(text)
    uniq = list(dict.fromkeys([t for t in tokens if len(t) > 1]))
    entities: List[Dict[str, Any]] = []
    for idx, token in enumerate(uniq[:limit]):
        if token in _IDENTITY_TOKENS or token in _SELF_TOKEN_HINTS:
            etype = "self"
        elif token in _WORLD_TOKENS:
            etype = "world"
        elif token in _TASK_TOKENS:
            etype = "task"
        elif token in _UNKNOWN_INTENT_TOKENS:
            etype = "other"
        else:
            etype = "other"
        entities.append(
            {
                "text": token,
                "type": etype,
                "canonical_id": f"ent:{etype}:{_safe_hash(token)}:{idx}",
                "confidence": 0.22,
            }
        )
    return entities


def _collect_world_snapshot(core_state: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(core_state, dict):
        return {"stability": 0.0, "delta_hat": 0.0, "energy_level": 0.0}
    return {
        "stability": float(core_state.get("stability", 0.0) or 0.0),
        "delta_hat": float(core_state.get("delta_hat", 0.0) or 0.0),
        "energy_level": float(core_state.get("energy_level", 0.0) or 0.0),
    }


def _collect_self_model_signature(core_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(core_state, dict):
        return {
            "meta_awareness": 0.0,
            "belief_summary": "",
            "identity_contract_version": "m3_identity_v1",
            "evidence_ids": [],
        }
    return {
        "meta_awareness": float(core_state.get("meta_awareness", core_state.get("meta_confidence", 0.0) or 0.0)),
        "belief_summary": str(core_state.get("belief_summary", "") or ""),
        "identity_contract_version": str(core_state.get("identity_contract_version", "m3_identity_v1") or "m3_identity_v1"),
        "evidence_ids": list(core_state.get("evidence_ids", []) or []),
    }


def _safe_text(obj: Any) -> str:
    return str(obj or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = float(default)
    return float(max(0.0, min(1.0, v)))


def _flag(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_meaning_state(
    user_text: str,
    chat_history: Optional[Iterable[Dict[str, Any]]] = None,
    core_state: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = str(user_text or "").strip()
    tokens = _tokenize(text)
    cfg = dict(cfg or {})

    intent = _guess_intent(text, tokens)
    entities = _extract_entities(text, limit=_safe_int(cfg.get("candidate_entities_limit", 12)))

    # 기본 핵심 상태
    snapshot = _collect_world_snapshot(core_state)
    self_model_ref = _collect_self_model_signature(core_state)

    uncertainty = {
        "overall": 0.42,
        "intent": 0.18,
        "grounding": 0.20,
        "reasons": ["pipeline_init"],
    }

    if not text:
        uncertainty["overall"] = 0.95
        uncertainty["intent"] = 0.95
        uncertainty["reasons"].append("empty_input")
    elif intent in {"unknown", "clarification"}:
        uncertainty["overall"] = max(uncertainty["overall"], 0.62)
        uncertainty["intent"] = max(uncertainty["intent"], 0.62)
        uncertainty["reasons"].append("intent_low_conf")

    if chat_history:
        with guard_context(ctx='m3/meaning_pipeline.py:184', catch_base=False) as __m3_guard_181_8:
            if len(list(chat_history)) > 0:
                uncertainty["reasons"].append("has_history")

        if __m3_guard_181_8.error is not None:
            pass

    dynamic_unc = _flag(
        cfg.get("dynamic_uncertainty", os.getenv("M3_MEANING_DYNAMIC_UNCERTAINTY", "1")),
        default=True,
    )
    chat_items: List[Dict[str, Any]] = []
    if chat_history:
        try:
            chat_items = list(chat_history)
        except Exception:
            chat_items = []
    token_count = int(len(tokens))
    entity_count = int(len(entities))
    entity_density = _clamp01(entity_count / float(max(1, token_count)), 0.0)
    history_conflict = 0.0
    if chat_items and tokens:
        last_assistant = ""
        for item in reversed(chat_items):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            if role in {"assistant", "m3", "bot"}:
                last_assistant = _safe_text(item.get("text") or item.get("content"))
                if last_assistant:
                    break
        if last_assistant:
            curr_tokens = set(_tokenize(text))
            prev_tokens = set(_tokenize(last_assistant))
            overlap = float(len(curr_tokens & prev_tokens) / max(1, len(curr_tokens | prev_tokens)))
            history_conflict = _clamp01(1.0 - overlap, 0.0)
    intent_confidence = 0.58
    if intent in {"identity_query", "state_query", "world_query", "task_request", "meta_control"}:
        intent_confidence = 0.68
    if intent in {"unknown", "clarification"}:
        intent_confidence = 0.34
    intent_confidence = _clamp01(
        intent_confidence + min(0.18, float(token_count) * 0.012) - (0.22 * history_conflict),
        0.5,
    )
    grounding_confidence = _clamp01(
        0.28 + (0.38 * entity_density) + (0.12 if chat_items else 0.0) - (0.16 * history_conflict),
        0.4,
    )
    if dynamic_unc and text:
        uncertainty["intent"] = _clamp01(1.0 - intent_confidence, 0.5)
        uncertainty["grounding"] = _clamp01(1.0 - grounding_confidence, 0.5)
        overall = (
            0.44 * uncertainty["intent"]
            + 0.34 * uncertainty["grounding"]
            + 0.22 * history_conflict
        )
        if intent in {"unknown", "clarification"}:
            overall = min(0.99, overall + 0.12)
        uncertainty["overall"] = _clamp01(overall, 0.42)
        uncertainty.setdefault("reasons", []).append("dynamic_uncertainty")
    else:
        intent_confidence = _clamp01(1.0 - _safe_float(uncertainty.get("intent", 0.5), 0.5), 0.5)
        grounding_confidence = _clamp01(1.0 - _safe_float(uncertainty.get("grounding", 0.5), 0.5), 0.5)
    if intent == "identity_query":
        user_goal = "M3 정체성 및 동작 근거를 설명한다."
    elif intent == "task_request":
        user_goal = "사용자 목표를 수행하기 위한 실행 가능한 다음 액션 또는 절차를 제시한다."
    elif intent == "state_query":
        user_goal = "요청한 상태 정보를 근거 기반으로 설명한다."
    elif intent == "world_query":
        user_goal = "세계/맥락 질의에 대해 근거로 정합적인 답변을 만든다."
    elif intent == "clarification":
        user_goal = "의도를 분해해 필요한 추가 정보를 질의한다."
    elif intent == "meta_control":
        user_goal = "메타 제어 및 설정 관련 질의에 대해 일관된 답변을 만든다."
    else:
        user_goal = "요청 의미를 해석하고 안전한 응답 전략을 만든다."

    return {
        "schema_version": MEANING_STATE_SCHEMA_VERSION,
        "turn_id": _turn_id(),
        "ts": float(time.time()),
        "intent": intent,
        "entities": entities,
        "user_goal": user_goal,
        "world_state_ref": {
            "snapshot": snapshot,
            "evidence_ids": [],
        },
        "self_model_ref": {
            "meta_awareness": self_model_ref["meta_awareness"],
            "belief_summary": self_model_ref["belief_summary"],
            "identity_contract_version": self_model_ref["identity_contract_version"],
            "evidence_ids": self_model_ref["evidence_ids"],
        },
        "uncertainty": uncertainty,
        "intent_confidence": float(intent_confidence),
        "grounding_confidence": float(grounding_confidence),
        "history_conflict": float(history_conflict),
        "entity_density": float(entity_density),
    }


def _entity_mention_to_clue(entity: Dict[str, Any]) -> str:
    if not isinstance(entity, dict):
        return ""
    return _normalize_text(str(entity.get("text", "")))


def _safe_contains(text: str, needle: str) -> bool:
    if not text or not needle:
        return False
    return _normalize_text(needle) in _normalize_text(text)


def ground_meaning_state(
    meaning_state: Dict[str, Any],
    core_state: Optional[Any] = None,
    chat_history: Optional[Iterable[Dict[str, Any]]] = None,
    grounding_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    cfg = grounding_cfg or {}
    grounded = deepcopy(meaning_state)

    grounded.setdefault("world_state_ref", {})
    grounded.setdefault("self_model_ref", {})
    grounded.setdefault("uncertainty", {
        "overall": 0.45,
        "intent": 0.45,
        "grounding": 0.45,
        "reasons": [],
    })

    uncertainty = grounded["uncertainty"]
    evidence_ids: List[str] = []
    evidences: List[str] = []

    if not isinstance(cfg, dict):
        cfg = {}
    evidence_limit = max(1, _safe_int(cfg.get("evidence_lookback", 20), 20))
    entity_match_threshold = _safe_float(cfg.get("entity_match_threshold", 0.15), 0.15)

    entities = list(grounded.get("entities", []) or [])
    memory_texts: List[str] = []

    if chat_history:
        with guard_context(ctx='m3/meaning_pipeline.py:273', catch_base=False) as __m3_guard_266_8:
            hist_list = list(chat_history)[-max(0, evidence_limit):]
            for item in hist_list:
                if isinstance(item, dict):
                    txt = _safe_text(item.get("text"))
                    if txt:
                        memory_texts.append(_normalize_text(txt))

        if __m3_guard_266_8.error is not None:
            memory_texts = []

    if attr_get_optional(core_state, "episodic_memory", None) is not None:
        mem_obj = attr_get_optional(core_state, "episodic_memory")
        if isinstance(mem_obj, dict):
            mem_list = list(mem_obj.get("memories", []) or [])[:evidence_limit]
        elif attr_has(mem_obj, "memories") and isinstance(attr_get_optional(mem_obj, "memories"), list):
            mem_list = list(mem_obj.memories[-evidence_limit:])
        else:
            mem_list = []
        for item in mem_list:
            candidate = ""
            for key in ("content", "text", "narrative", "description", "summary"):
                if isinstance(item, dict) and item.get(key):
                    candidate = _normalize_text(str(item.get(key, "")))
                    break
            if candidate:
                memory_texts.append(candidate)

    for ent in entities:
        clue = _entity_mention_to_clue(ent)
        if not clue:
            continue
        hit = False
        for mem in memory_texts:
            if clue and clue in mem:
                hit = True
                break
        if not hit and memory_texts:
            hit_count = sum(1 for mem in memory_texts if _safe_contains(mem, clue))
            overlap = hit_count / float(max(1, len(memory_texts)))
            if overlap >= entity_match_threshold:
                hit = True
        if hit:
            eid = _safe_hash(f"{clue}:{_normalize_text(str(ent.get('text')))}")
            evidence_ids.append(f"ground:{eid}")
            evidences.append(clue)

    world_evidence = []
    if attr_get_optional(core_state, "_get_current_world_state", None):
        with guard_context(ctx='m3/meaning_pipeline.py:319', catch_base=False) as __m3_guard_314_8:
            ws = core_state._get_current_world_state()
            if isinstance(ws, dict):
                stability = float(ws.get("stability", 0.0) or 0.0)
                world_evidence.append(f"world-stability:{stability:.4f}")

        if __m3_guard_314_8.error is not None:
            pass

    grounded_unc = _safe_float(uncertainty.get("grounding", 0.4), 0.4)
    if evidence_ids or world_evidence:
        grounded_unc = max(0.05, grounded_unc * 0.5)
        grounded["world_state_ref"]["evidence_ids"] = list(dict.fromkeys(list(evidence_ids)[:evidence_limit]))
        uncertainty["grounding"] = grounded_unc
        uncertainty.setdefault("reasons", []).append("grounded")
    else:
        uncertainty["grounding"] = min(0.95, grounded_unc + 0.35)
        uncertainty.setdefault("reasons", []).append("no_grounding")
        uncertainty["overall"] = min(0.99, _safe_float(uncertainty.get("overall", 0.45), 0.45) + 0.25)

    if "grounding_window" in cfg:
        uncertainty.setdefault("reasons", []).append(f"grounding_window={cfg.get('grounding_window', 0)}")

    grounded["self_model_ref"]["evidence_ids"] = list(dict.fromkeys(list(world_evidence) + evidence_ids))
    grounded["world_state_ref"].setdefault("evidence_ids", [])

    return grounded, evidences


def build_response_plan(
    meaning_state: Dict[str, Any],
    grounded_evidence: Optional[Iterable[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = cfg or {}
    intent = str(meaning_state.get("intent", "unknown"))
    lang = _detect_lang(_safe_text(meaning_state.get("user_goal")))

    uncertainty = meaning_state.get("uncertainty", {}) or {}
    overall_unc = _safe_float(uncertainty.get("overall", 0.5), 0.5)
    grounding_unc = _safe_float(uncertainty.get("grounding", 0.5), 0.5)

    clarify_threshold = _safe_float(cfg.get("force_clarify_overall_uncertainty_threshold", cfg.get("uncertainty_min_for_clarify", 0.45)), 0.45)
    grounding_threshold = _safe_float(cfg.get("force_clarify_grounding_uncertainty_threshold", cfg.get("grounding_confidence_min", 0.35)), 0.35)

    must_avoid = list(cfg.get("must_avoid") or [
        "ai_identity_claim",
        "no_feelings_claim",
        "provider_claim",
    ])
    required: List[str] = []

    if intent == "identity_query":
        required.append("자기 정체성/의식은 외부 정체성 고백이 아니라 내부 근거-추론 체인으로 설명한다.")
        required.append("외부 LLM 정체성 표현이나 감정 부정을 회피한다.")
    elif intent == "state_query":
        required.append("요청 상태 항목을 근거 기반으로 설명한다.")
    elif intent == "world_query":
        required.append("세계/맥락 질의에 대해 근거를 기준으로 답한다.")
    elif intent == "task_request":
        required.append("실행 가능한 다음 액션 또는 절차를 제시한다.")
    elif intent == "clarification":
        required.append("누락된 제약/대상/형식 정보를 질의한다.")
    elif intent == "meta_control":
        required.append("설정 변경의 의미와 제약을 정확히 설명한다.")
    else:
        required.append("요청 문맥을 해석하고 모순 없는 핵심 답안을 생성한다.")

    if grounded_evidence:
        evidence_count = len(list(grounded_evidence))
        if evidence_count > 0:
            required.append(f"근거 단서 {evidence_count}건을 반영한다.")

    key_points = required[:_safe_int(cfg.get("max_key_points", 5), 5)]
    evidence_ids = list((meaning_state.get("world_state_ref", {}) or {}).get("evidence_ids", []) or [])

    direct_forbidden = overall_unc >= clarify_threshold or grounding_unc >= grounding_threshold or not evidence_ids
    allow_direct = not direct_forbidden and intent not in {"unknown", "clarification", "meta_control"}
    answer_type = "direct"
    if not allow_direct:
        answer_type = "clarify" if intent in {"identity_query", "state_query", "world_query", "task_request", "clarification"} or grounded_evidence is None else "abstain"

    allowed_claims: List[Dict[str, Any]] = []
    for idx, point in enumerate(key_points):
        allowed_claims.append(
            {
                "claim_id": f"c-{idx:02d}",
                "text": point,
                "support_evidence_ids": evidence_ids[:3],
                "confidence": 0.60,
            }
        )

    return {
        "schema_version": RESPONSE_PLAN_SCHEMA_VERSION,
        "plan_id": _turn_id(),
        "answer_type": answer_type,
        "key_points": key_points,
        "must_avoid": must_avoid,
        "clarify_if_uncertain": True,
        "allowed_claims": allowed_claims,
        "style": {
            "language": lang,
            "length": "short",
            "tone": "factual",
        },
        "validation_targets": {
            "entailment_min": _safe_float(cfg.get("entailment_min", 0.62), 0.62),
            "plan_adherence_min": _safe_float(cfg.get("plan_adherence_min", 0.75), 0.75),
            "identity_consistency_min": _safe_float(cfg.get("identity_consistency_min", 0.90), 0.90),
        },
    }


def build_response_plan(
    meaning_state: Dict[str, Any],
    grounded_evidence: Optional[Iterable[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = cfg or {}
    intent = str(meaning_state.get("intent", "unknown"))
    lang = _detect_lang(_safe_text(meaning_state.get("user_goal")))

    uncertainty = meaning_state.get("uncertainty", {}) or {}
    overall_unc = _safe_float(uncertainty.get("overall", 0.5), 0.5)
    grounding_unc = _safe_float(uncertainty.get("grounding", 0.5), 0.5)
    intent_confidence = _clamp01(meaning_state.get("intent_confidence", 1.0 - overall_unc), 0.5)
    grounding_confidence = _clamp01(meaning_state.get("grounding_confidence", 1.0 - grounding_unc), 0.5)
    history_conflict = _clamp01(meaning_state.get("history_conflict", 0.0), 0.0)
    entity_density = _clamp01(meaning_state.get("entity_density", 0.0), 0.0)

    clarify_threshold = _safe_float(
        cfg.get("force_clarify_overall_uncertainty_threshold", cfg.get("uncertainty_min_for_clarify", 0.45)),
        0.45,
    )
    grounding_threshold = _safe_float(
        cfg.get("force_clarify_grounding_uncertainty_threshold", cfg.get("grounding_confidence_min", 0.35)),
        0.35,
    )
    hysteresis_enter = _safe_float(cfg.get("hysteresis_enter", 0.58), 0.58)
    hysteresis_exit = _safe_float(cfg.get("hysteresis_exit", 0.42), 0.42)
    previous_answer_type = str(
        meaning_state.get("previous_answer_type", cfg.get("previous_answer_type", ""))
    ).strip().lower()

    risk = max(
        _clamp01(overall_unc, 0.5),
        _clamp01(1.0 - intent_confidence, 0.5),
        _clamp01(1.0 - grounding_confidence, 0.5),
        _clamp01(history_conflict * 0.85, 0.0),
    )
    uncertain_enter = risk >= hysteresis_enter
    uncertain_exit = risk <= hysteresis_exit
    sticky_uncertain = uncertain_enter or (
        previous_answer_type in {"clarify", "abstain"} and not uncertain_exit
    )

    must_avoid = list(cfg.get("must_avoid") or [
        "ai_identity_claim",
        "no_feelings_claim",
        "provider_claim",
    ])
    if intent == "task_request":
        must_avoid.append("unsafe_action_advice")
    if intent in {"world_query", "meta_control"}:
        must_avoid.append("fabricated_citation")
    must_avoid = list(dict.fromkeys([str(x).strip() for x in must_avoid if str(x).strip()]))

    evidence_ids = list((meaning_state.get("world_state_ref", {}) or {}).get("evidence_ids", []) or [])
    evidence_sources: List[str] = []
    if grounded_evidence:
        for item in list(grounded_evidence):
            if isinstance(item, dict):
                src = str(item.get("source") or item.get("id") or "").strip()
            else:
                src = str(item or "").strip()
            if src:
                evidence_sources.append(src)
    evidence_sources = list(dict.fromkeys(evidence_sources))

    if intent in {"unknown", "clarification"}:
        if sticky_uncertain and risk >= 0.90 and grounding_confidence < 0.35:
            answer_type = "abstain"
        else:
            answer_type = "clarify"
    elif sticky_uncertain or overall_unc >= clarify_threshold or grounding_unc >= grounding_threshold:
        answer_type = "clarify"
    elif not evidence_ids and grounding_confidence < 0.45:
        answer_type = "clarify"
    else:
        answer_type = "direct"

    key_templates: Dict[str, List[str]] = {
        "identity_query": [
            "Explain identity from current grounded state.",
            "Avoid provider or generic AI claims.",
        ],
        "state_query": [
            "Describe current internal state with concrete signals.",
        ],
        "world_query": [
            "Answer from available evidence without speculation.",
        ],
        "task_request": [
            "Propose executable next steps.",
            "State constraints before action.",
        ],
        "clarification": [
            "Ask only the minimum missing context.",
        ],
        "meta_control": [
            "Explain control changes and expected impact.",
        ],
    }
    key_points = list(key_templates.get(intent, ["Answer directly and stay grounded."]))
    if evidence_sources:
        key_points.append(f"Use {len(evidence_sources)} grounded evidence source(s).")
    if answer_type == "clarify":
        key_points.append("Ask one concise clarifying question.")
    elif answer_type == "abstain":
        key_points.append("State limitation and request stronger evidence.")
    max_points = _safe_int(cfg.get("max_key_points", 5), 5)
    key_points = key_points[:max(1, max_points)]

    style = {
        "language": lang,
        "length": "short",
        "tone": "factual",
    }
    if intent == "task_request":
        style["length"] = "medium"
        style["tone"] = "actionable"
    elif intent == "identity_query":
        style["tone"] = "reflective"
    elif intent == "world_query":
        style["tone"] = "analytic"
    if answer_type in {"clarify", "abstain"}:
        style["tone"] = "cautious"

    validation_targets = {
        "entailment_min": _safe_float(cfg.get("entailment_min", 0.62), 0.62),
        "plan_adherence_min": _safe_float(cfg.get("plan_adherence_min", 0.75), 0.75),
        "identity_consistency_min": _safe_float(cfg.get("identity_consistency_min", 0.90), 0.90),
    }
    if answer_type in {"clarify", "abstain"}:
        validation_targets["entailment_min"] = max(validation_targets["entailment_min"], 0.66)
        validation_targets["plan_adherence_min"] = max(validation_targets["plan_adherence_min"], 0.78)
    if intent == "task_request":
        validation_targets["plan_adherence_min"] = max(validation_targets["plan_adherence_min"], 0.80)

    allowed_claims: List[Dict[str, Any]] = []
    for idx, point in enumerate(key_points):
        allowed_claims.append(
            {
                "claim_id": f"c-{idx:02d}",
                "text": point,
                "support_evidence_ids": evidence_ids[:3],
                "confidence": float(max(0.35, min(0.92, 1.0 - risk))),
            }
        )

    evidence_count_mode = str(cfg.get("plan_evidence_count_mode", "source_based") or "source_based").strip().lower()
    grounded_evidence_count = len(evidence_sources) if evidence_count_mode == "source_based" else len(list(grounded_evidence or []))
    decision_trace = [
        f"intent={intent}",
        f"risk={risk:.3f}",
        f"sticky_uncertain={int(sticky_uncertain)}",
        f"answer_type={answer_type}",
        f"evidence_sources={len(evidence_sources)}",
    ]

    return {
        "schema_version": RESPONSE_PLAN_SCHEMA_VERSION,
        "plan_id": _turn_id(),
        "answer_type": answer_type,
        "key_points": key_points,
        "must_avoid": must_avoid,
        "clarify_if_uncertain": True,
        "allowed_claims": allowed_claims,
        "style": style,
        "validation_targets": validation_targets,
        "grounded_evidence_count": int(grounded_evidence_count),
        "evidence_sources": list(evidence_sources),
        "decision_trace": decision_trace,
        "evidence_count_mode": evidence_count_mode,
        "intent_confidence": float(intent_confidence),
        "grounding_confidence": float(grounding_confidence),
        "history_conflict": float(history_conflict),
        "entity_density": float(entity_density),
    }


def build_generation_contract(
    response_plan: Optional[Dict[str, Any]],
    meaning_state: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not response_plan:
        return None

    points = response_plan.get("key_points") or []
    avoid = response_plan.get("must_avoid") or []
    style = response_plan.get("style") or {}
    vt = response_plan.get("validation_targets") or {}

    contract = {
        "purpose": "answer_from_meaning_state",
        "turn_id": (meaning_state or {}).get("turn_id", _turn_id()),
        "answer_type": response_plan.get("answer_type"),
        "key_points": list(points),
        "must_avoid": list(avoid),
        "clarify_if_uncertain": bool(response_plan.get("clarify_if_uncertain", True)),
        "style": {
            "language": style.get("language", "auto"),
            "length": style.get("length", "short"),
            "tone": style.get("tone", "factual"),
        },
        "validation_targets": {
            "entailment_min": _safe_float(vt.get("entailment_min", 0.62), 0.62),
            "plan_adherence_min": _safe_float(vt.get("plan_adherence_min", 0.75), 0.75),
            "identity_consistency_min": _safe_float(vt.get("identity_consistency_min", 0.90), 0.90),
        },
        "mode": "meaning_first",
    }
    return json.dumps(contract, ensure_ascii=False)


def format_plan_fallback_prompt(response_plan: Dict[str, Any], meaning_state: Optional[Dict[str, Any]] = None) -> str:
    lang = "ko" if _detect_lang(_safe_text((meaning_state or {}).get("user_goal", ""))) == "ko" else "en"
    atype = str(response_plan.get("answer_type", "clarify")).lower()
    if atype == "clarify":
        if lang == "ko":
            return (
                "요청을 더 정확히 이해하기 위해 핵심 맥락을 2개만 정리해 주세요. "
                "예: {대상}, {제약}, {원하는 출력 형식}."
            )
        return (
            "Please share key missing context so I can narrow the answer: target, constraints, and preferred output format."
        )
    if atype == "abstain":
        if lang == "ko":
            return "요청의 근거가 부족해 즉시 단정할 수 없습니다. 더 구체적 근거를 주시면 답변을 수렴하겠습니다."
        return "I cannot assert this without stronger evidence. Please provide additional context or references."
    if lang == "ko":
        return "요청을 처리 중입니다. 핵심 근거를 반영해 답변하겠습니다."
    return "Processing request with grounded reasoning from available evidence."


__all__ = [
    "MEANING_STATE_SCHEMA_VERSION",
    "RESPONSE_PLAN_SCHEMA_VERSION",
    "build_meaning_state",
    "ground_meaning_state",
    "build_response_plan",
    "build_generation_contract",
    "format_plan_fallback_prompt",
]
