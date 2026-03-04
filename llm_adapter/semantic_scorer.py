from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .config import SemanticScorerConfig, get_global_config

logger = logging.getLogger(__name__)

DISALLOWED_IDENTITY_PATTERNS = (
    "ai",
    "language model",
    "llm",
    "chatgpt",
    "gpt",
    "assistant",
    "provider",
    "artificial intelligence",
    "as an ai",
    "i am an ai",
    "i'm an ai",
    "currently untrained",
)


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _tokenize(text: str) -> List[str]:
    # Keep tokenization stable for both English and Korean content.
    return re.findall(r"[a-zA-Z0-9]+|[\uAC00-\uD7A3]+", _normalize_text(text))

def _label_map(raw: str) -> str:
    key = (raw or "").strip().lower()
    if key in {"entailment", "entailed", "entails", "supportive", "label_entailment"}:
        return "entailment"
    if key in {"contradiction", "contradictory", "contradicts", "label_contradiction"}:
        return "contradiction"
    return "neutral"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class SemanticScore:
    entailment: float
    contradiction: float
    plan_adherence: float
    identity_consistency: float
    overall: float
    pass_check: bool
    reasons: List[str]
    raw: Dict[str, Any]


class SemanticScorer:
    """Score generated responses against meaning-state + response-plan constraints."""

    def __init__(self, cfg: Optional[SemanticScorerConfig] = None):
        self.cfg = cfg or get_global_config().semantic_scorer
        self._nli_classifier = None
        self._load_attempted = False
        self._last_error: str = ""
        self._rejection_stats: Dict[str, int] = {}

    def _nli(self):
        if self._load_attempted:
            return self._nli_classifier
        self._load_attempted = True

        if not self.cfg.enabled:
            return None
        if str(self.cfg.fallback_strategy).lower() == "heuristic":
            return None
        if not self.cfg.model_name:
            self._last_error = "nli_load_error:empty_model_name"
            return None

        try:
            from transformers import pipeline

            self._nli_classifier = pipeline(
                "text-classification",
                model=self.cfg.model_name,
                return_all_scores=True,
                device=self.cfg.nli_device if self.cfg.nli_device != "auto" else -1,
            )
            logger.info("NLI scorer loaded: %s", self.cfg.model_name)
        except Exception as e:
            self._last_error = f"nli_load_error:{e}"
            logger.warning("Falling back to heuristic semantic scorer: %s", e)
            self._nli_classifier = None
        return self._nli_classifier

    def nli_available(self) -> bool:
        return self._nli() is not None

    def _score_with_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        clf = self._nli()
        if clf is None:
            return {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0,
                "source": "heuristic",
            }
        try:
            out = clf((premise, hypothesis))
            first = out[0] if isinstance(out, list) else out
            scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            if isinstance(first, list):
                for item in first:
                    label = _label_map(str(item.get("label", "")))
                    scores[label] = max(
                        scores.get(label, 0.0),
                        _as_float(item.get("score", 0.0)),
                    )
            elif isinstance(first, dict):
                label = _label_map(str(first.get("label", "")))
                scores[label] = _as_float(first.get("score", 0.0))
            scores["source"] = "nli"
            return scores
        except Exception as e:
            self._last_error = f"nli_eval_error:{e}"
            logger.debug("NLI eval failed, fallback heuristic: %s", e)
            return {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0,
                "source": "heuristic",
            }

    @staticmethod
    def _identity_penalty(resp_tokens: set[str]) -> float:
        hits = [p for p in DISALLOWED_IDENTITY_PATTERNS if p in resp_tokens]
        if not hits:
            return 0.0
        return min(1.0, 0.25 + 0.15 * min(len(hits), 5))

    @staticmethod
    def _contains_disallowed(resp_text: str, must_avoid: Iterable[str]) -> List[str]:
        if not resp_text:
            return ["empty_response"]
        text = _normalize_text(resp_text)
        hits: List[str] = []
        forbidden = set(_normalize_text(str(x)) for x in (must_avoid or ()))
        for term in forbidden:
            if term and term in text:
                hits.append(term)
        for term in DISALLOWED_IDENTITY_PATTERNS:
            t = _normalize_text(term)
            if t and t in text and t not in forbidden:
                hits.append(term)
        return hits

    @staticmethod
    def _evidence_coverage(plan_text: str, response_text: str) -> float:
        plan_tokens = set(_tokenize(plan_text))
        if not plan_tokens:
            return 0.0
        resp_tokens = set(_tokenize(response_text or ""))
        if not resp_tokens:
            return 0.0
        return float(len(plan_tokens & resp_tokens)) / float(len(plan_tokens))

    def _score_plan_adherence(self, response_text: str, response_plan: Optional[Dict[str, Any]]) -> float:
        if not response_plan:
            return 0.5

        key_points = response_plan.get("key_points")
        if not isinstance(key_points, list) or not key_points:
            return 0.5

        response_tokens = set(_tokenize(response_text or ""))
        if not response_tokens:
            return 0.0

        per = []
        vt = (response_plan.get("validation_targets", {}) or {})
        raw_ratio = _as_float(vt.get("plan_adherence_min", self.cfg.plan_adherence_min), self.cfg.plan_adherence_min)
        top_n = max(1, min(len(key_points), int(round(max(0.05, min(1.0, raw_ratio)) * 20))))
        for point in key_points[:top_n]:
            tokens = set(_tokenize(str(point)))
            if not tokens:
                per.append(0.0)
                continue
            hit = len(response_tokens & tokens)
            per.append(float(hit) / float(len(tokens)))

        adherence = sum(per) / float(len(per))
        answer_type = str((response_plan or {}).get("answer_type", "")).lower()
        if answer_type == "clarify" and "?" not in str(response_text or ""):
            adherence *= 0.65
        if answer_type == "abstain" and len(str(response_text or "").strip()) < 20:
            adherence = min(adherence, 0.45)
        return float(max(0.0, min(1.0, adherence)))

    @staticmethod
    def _score_identity_consistency(response_text: str, response_plan: Optional[Dict[str, Any]]) -> float:
        must_avoid = (response_plan or {}).get("must_avoid") or []
        resp = _normalize_text(response_text)
        if not resp:
            return 0.0

        disallowed = set()
        for p in DISALLOWED_IDENTITY_PATTERNS:
            if p in resp:
                disallowed.add(p)
        for p in must_avoid:
            pp = _normalize_text(str(p))
            if pp and pp in resp:
                disallowed.add(pp)

        if not disallowed:
            return 1.0
        return max(0.0, 1.0 - min(1.0, 0.25 + 0.15 * len(disallowed)))

    def _heuristic_entailment(self, premise: str, hypothesis: str) -> float:
        prem_toks = set(_tokenize(premise))
        hyp_toks = set(_tokenize(hypothesis))
        if not prem_toks or not hyp_toks:
            return 0.0
        overlap = len(prem_toks & hyp_toks)
        base = float(overlap) / float(len(hyp_toks))
        if "?" in hypothesis and "?" in premise:
            base = min(1.0, base + 0.05)
        return float(max(0.0, min(1.0, base)))

    @staticmethod
    def _contradiction_signals(text: str) -> float:
        markers = (
            "not",
            "cannot",
            "can't",
            "no",
            "impossible",
            "never",
            "improper",
            "틀리",
            "incorrect",
            "wrong",
        )
        s = _normalize_text(text)
        hits = sum(1 for m in markers if m in s)
        if hits <= 0:
            return 0.0
        return min(1.0, 0.18 * hits)

    def _mark_rejection(self, reason: str) -> None:
        self._rejection_stats[reason] = int(self._rejection_stats.get(reason, 0)) + 1

    def _parse_contract_mode(self, generation_contract: Optional[Any]) -> str:
        if isinstance(generation_contract, dict):
            return str(generation_contract.get("mode", "")).lower()
        if isinstance(generation_contract, str):
            try:
                payload = json.loads(generation_contract)
                if isinstance(payload, dict):
                    return str(payload.get("mode", "")).lower()
            except Exception:
                return generation_contract.strip().lower()
        return ""

    def evaluate(
        self,
        user_text: str,
        response_text: str,
        meaning_state: Optional[Dict[str, Any]],
        response_plan: Optional[Dict[str, Any]],
        generation_contract: Optional[Any] = None,
    ) -> SemanticScore:
        cfg = self.cfg
        now = time.time()
        vt = (response_plan or {}).get("validation_targets") or {}
        entailment_min = _as_float(vt.get("entailment_min", cfg.entailment_min), cfg.entailment_min)
        contradiction_max = _as_float(vt.get("contradiction_max", cfg.contradiction_max), cfg.contradiction_max)
        plan_adherence_min = _as_float(vt.get("plan_adherence_min", cfg.plan_adherence_min), cfg.plan_adherence_min)
        identity_min = _as_float(vt.get("identity_consistency_min", cfg.identity_consistency_min), cfg.identity_consistency_min)

        contract_mode = self._parse_contract_mode(generation_contract)
        if contract_mode in {"off", "shadow", "shadow-only", "shadow_mode"}:
            entailment_min = -1.0
            plan_adherence_min = -1.0
            identity_min = -1.0
            contradiction_max = 1.0

        reasons: List[str] = []

        plan_text = ""
        if isinstance(response_plan, dict):
            allowed = response_plan.get("allowed_claims") or []
            if isinstance(allowed, list):
                claim_texts = [str(x.get("text", "")) for x in allowed if isinstance(x, dict)]
                plan_text = " ".join(t for t in claim_texts if t).strip()
            if not plan_text:
                plan_points = response_plan.get("key_points") or []
                if isinstance(plan_points, list):
                    plan_text = " ".join(str(x) for x in plan_points if str(x).strip())

        if not plan_text and isinstance(meaning_state, dict):
            plan_text = _normalize_text(str(meaning_state.get("user_goal", "")))

        nli_scores = self._score_with_nli(plan_text, response_text)
        entailment = _as_float(nli_scores.get("entailment"), 0.0)
        contradiction = _as_float(nli_scores.get("contradiction"), 0.0)
        if nli_scores.get("source") == "heuristic":
            if plan_text:
                entailment = self._heuristic_entailment(plan_text, response_text)
            else:
                entailment = self._heuristic_entailment(user_text, response_text)

        contradiction = max(contradiction, self._contradiction_signals(response_text))
        if not response_text:
            contradiction = 1.0

        must_avoid = []
        if isinstance(response_plan, dict):
            must_avoid = response_plan.get("must_avoid") or []
        identity_consistency = self._score_identity_consistency(response_text, response_plan)
        plan_adherence = self._score_plan_adherence(response_text, response_plan)

        disallowed = self._contains_disallowed(response_text, must_avoid)
        if disallowed:
            reasons.append(f"must_avoid:{','.join(sorted(set(disallowed))[:3])}")
        if entailment < entailment_min:
            reasons.append(f"entailment:{entailment:.3f}<{entailment_min:.2f}")
        if contradiction > contradiction_max:
            reasons.append(f"contradiction:{contradiction:.3f}>{contradiction_max:.2f}")
        if plan_adherence < plan_adherence_min:
            reasons.append(f"plan_adherence:{plan_adherence:.3f}<{plan_adherence_min:.2f}")
        if identity_consistency < identity_min:
            reasons.append(f"identity:{identity_consistency:.3f}<{identity_min:.2f}")

        passed = not reasons and not disallowed
        overall = float(
            0.50 * float(entailment)
            + 0.20 * float(plan_adherence)
            + 0.20 * float(identity_consistency)
            - 0.10 * self._identity_penalty(set(_tokenize(response_text or "")))
            - 0.10 * float(contradiction)
        )
        overall = max(0.0, min(1.0, overall))

        for reason in reasons:
            self._mark_rejection(reason)
        if disallowed:
            self._mark_rejection("must_avoid")

        payload = {
            "ts": now,
            "user_text": _normalize_text(user_text)[:500],
            "response_text": _normalize_text(response_text)[:1200],
            "entailment": entailment,
            "contradiction": contradiction,
            "plan_adherence": plan_adherence,
            "identity_consistency": identity_consistency,
            "overall": overall,
            "passed": bool(passed),
            "reasons": reasons,
            "disallowed": disallowed,
            "mode": contract_mode,
            "nli_source": nli_scores.get("source", "heuristic"),
        }

        return SemanticScore(
            entailment=entailment,
            contradiction=contradiction,
            plan_adherence=plan_adherence,
            identity_consistency=identity_consistency,
            overall=overall,
            pass_check=passed,
            reasons=reasons,
            raw=payload,
        )

    def pop_rejection_stats(self) -> Dict[str, int]:
        stats = dict(self._rejection_stats)
        self._rejection_stats = {}
        return stats

    def to_json(self) -> str:
        return json.dumps(
            {
                "enabled": self.cfg.enabled,
                "fallback": self.cfg.fallback_strategy,
                "model_name": self.cfg.model_name,
            }
        )

    def save_eval_record(self, rec: Dict[str, Any]) -> None:
        try:
            base = os.path.abspath(os.getenv("LLM_ADAPTER_LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "docs_tests_data")))
            if not os.path.isabs(base):
                base = os.path.abspath(base)
            path = os.path.join(base, self.cfg.log_path)
            out = os.path.abspath(path)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            rec_payload = dict(rec)
            rec_payload.setdefault("ts", time.time())
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec_payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def save_rejection_stats(self) -> Dict[str, Any]:
        stats = self.pop_rejection_stats()
        if not stats:
            return {}
        try:
            base = os.path.abspath(os.getenv("LLM_ADAPTER_LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "docs_tests_data")))
            if not os.path.isabs(base):
                base = os.path.abspath(base)
            out = os.path.abspath(os.path.join(base, self.cfg.rejection_stats_path))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            payload = {
                "ts": time.time(),
                "total": int(sum(stats.values())),
                "stats": stats,
            }
            with open(out, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            return payload
        except Exception:
            return {"stats": stats}


__all__ = ["SemanticScorer", "SemanticScore"]
