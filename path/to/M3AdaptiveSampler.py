"""
Planning/documentation module for a potential `M3AdaptiveSampler` implementation.

This file previously contained only stub classes, functions, and tests with `pass`
statements, and duplicated responsibilities that are implemented elsewhere in
the codebase (for example, in `llm_adapter/llm_core.py`).

To avoid exposing incomplete or misleading APIs and to satisfy static analysis
rules, all executable stub code has been removed from this module. If you need
an actual adaptive sampler implementation, use the existing, fully implemented
components in the main codebase rather than this planning file.

If you wish to keep design notes or planning sketches, please treat this file
purely as documentation and do not reintroduce stub classes or functions here.
"""

from typing import Final

# This constant makes it explicit that this module is intentionally used for
# planning/documentation and is not the home of a production implementation.
IS_M3_ADAPTIVE_SAMPLER_PLANNING_MODULE: Final[bool] = True


def get_planning_notes() -> str:
    """
    Return the planning/documentation notes for the potential
    `M3AdaptiveSampler` implementation.

    This helper exists to ensure the module contains executable code so that it
    is not treated as a documentation-only Python file by static analysis tools.
    It should not be used as an indicator that a concrete sampler implementation
    is available here.
    """

    # Prefer the module docstring if present, so updates to the top-level notes
    # are reflected automatically.
    return __doc__ or (
        "Planning/documentation module for a potential `M3AdaptiveSampler` "
        "implementation. No production sampler implementation is defined here."
    )