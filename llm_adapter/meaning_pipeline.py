"""Compatibility shim for imports expecting llm_adapter.meaning_pipeline.

The primary meaning-pipeline implementation lives in ``m3.meaning_pipeline``.
``llm_core`` historically imported from ``llm_adapter.meaning_pipeline``, while
newer modules import from ``m3.meaning_pipeline``. Re-export both so both import
paths remain valid.
"""

from m3.meaning_pipeline import (  # noqa: F401
    MEANING_STATE_SCHEMA_VERSION,
    RESPONSE_PLAN_SCHEMA_VERSION,
    build_generation_contract,
    build_meaning_state,
    build_response_plan,
    format_plan_fallback_prompt,
    ground_meaning_state,
)

__all__ = [
    "MEANING_STATE_SCHEMA_VERSION",
    "RESPONSE_PLAN_SCHEMA_VERSION",
    "build_generation_contract",
    "build_meaning_state",
    "build_response_plan",
    "ground_meaning_state",
    "format_plan_fallback_prompt",
]
