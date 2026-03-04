from __future__ import annotations

import builtins
import logging
from typing import Any, Callable, Optional


logger = logging.getLogger("m3.attr_contract")


class AttrContractError(RuntimeError):
    """Raised when required attribute contract is violated."""


class guard_context:
    """
    Context manager that catches failures in a standardized way.

    - catch_base=False: catches Exception subclasses only.
    - catch_base=True: catches BaseException subclasses (bare-except semantics).
    """

    def __init__(
        self,
        ctx: str,
        *,
        catch_base: bool = False,
    ) -> None:
        self.ctx = str(ctx)
        self.catch_base = bool(catch_base)
        self.error: Optional[BaseException] = None

    def __enter__(self):
        self.error = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False
        if self.catch_base:
            should_catch = isinstance(exc, BaseException)
        else:
            should_catch = isinstance(exc, Exception)
        if not should_catch:
            return False
        self.error = exc
        logger.exception("guard_context failure | ctx=%s | err=%s", self.ctx, exc)
        return True


def _safe_log(event: str, ctx: Optional[str], obj: Any, name: str, err: BaseException) -> None:
    logger.exception(
        "attr_contract failure | event=%s | ctx=%s | obj_type=%s | name=%s | err=%s",
        event,
        str(ctx or ""),
        type(obj).__name__,
        str(name),
        str(err),
    )


def attr_get_optional(obj: Any, name: str, default: Any = None, ctx: Optional[str] = None) -> Any:
    try:
        return builtins.getattr(obj, name, default)
    except Exception as exc:
        _safe_log("get_optional", ctx, obj, name, exc)
        return default


def attr_get_required(obj: Any, name: str, ctx: Optional[str] = None) -> Any:
    try:
        value = builtins.getattr(obj, name)
    except Exception as exc:
        _safe_log("get_required", ctx, obj, name, exc)
        raise AttrContractError(f"required attribute missing: {name}") from exc
    return value


def attr_has(obj: Any, name: str, ctx: Optional[str] = None) -> bool:
    try:
        return bool(builtins.hasattr(obj, name))
    except Exception as exc:
        _safe_log("has", ctx, obj, name, exc)
        return False


def attr_set(obj: Any, name: str, value: Any, ctx: Optional[str] = None) -> None:
    try:
        builtins.setattr(obj, name, value)
    except Exception as exc:
        _safe_log("set", ctx, obj, name, exc)


def attr_del(obj: Any, name: str, ctx: Optional[str] = None) -> None:
    try:
        builtins.delattr(obj, name)
    except Exception as exc:
        _safe_log("del", ctx, obj, name, exc)


def guard_eval(
    fn: Callable[[], Any],
    default: Any,
    ctx: str,
    on_error: Optional[Callable[[BaseException], Any]] = None,
) -> Any:
    try:
        return fn()
    except Exception as exc:
        logger.exception("guard_eval failure | ctx=%s | err=%s", str(ctx), exc)
        if on_error is not None:
            try:
                on_error(exc)
            except Exception as hook_exc:
                logger.exception(
                    "guard_eval on_error failure | ctx=%s | err=%s",
                    str(ctx),
                    hook_exc,
                )
        return default


def guard_step(
    fn: Callable[[], Any],
    ctx: str,
    on_error: Optional[Callable[[BaseException], Any]] = None,
) -> bool:
    try:
        fn()
        return True
    except Exception as exc:
        logger.exception("guard_step failure | ctx=%s | err=%s", str(ctx), exc)
        if on_error is not None:
            try:
                on_error(exc)
            except Exception as hook_exc:
                logger.exception(
                    "guard_step on_error failure | ctx=%s | err=%s",
                    str(ctx),
                    hook_exc,
                )
        return False


__all__ = [
    "AttrContractError",
    "attr_del",
    "attr_get_optional",
    "attr_get_required",
    "attr_has",
    "attr_set",
    "guard_context",
    "guard_eval",
    "guard_step",
]
