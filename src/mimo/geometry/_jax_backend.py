from __future__ import annotations

from typing import Any, Literal, Callable, TypeAlias, TypeVar
from dataclasses import fields, replace


ArrayLike: TypeAlias = Any
_PYTREE_REGISTERED: set[type[Any]] = set()
_MOTION_TO_BATCH: dict[type[Any], type[Any]] = {}
B = TypeVar("B")

PYTREE_META_FIELDS = frozenset({"xp", "dtype"})

def register_motion_batch(motion_cls: type[Any]) -> Callable[[type[B]], type[B]]:
    """"""
    def decorator(cls: type[B]) -> type[B]:
        if motion_cls in _MOTION_TO_BATCH:
            raise ValueError(
                f"A MotionBatch is already registered for {motion_cls.__name__!r}."
            )
        
        _MOTION_TO_BATCH[motion_cls] = cls
        _maybe_register_pytree(cls)
        return cls
    
    return decorator


def _maybe_register_pytree(cls: type[Any]) -> None:
    if cls in _PYTREE_REGISTERED:
        return
    try:
        import jax
    except ImportError:
        return
    
    all_field_names = tuple(f.name for f in fields(cls))
    meta = tuple(name for name in all_field_names if name in PYTREE_META_FIELDS)
    data = tuple(name for name in all_field_names if name not in PYTREE_META_FIELDS)
    
    jax.tree_util.register_dataclass(
        cls,
        data_fields=data,
        meta_fields=meta
    )
    _PYTREE_REGISTERED.add(cls)
    
        
def _ensure_all_pytrees_registered() -> None:
    for cls in _MOTION_TO_BATCH.values():
        _maybe_register_pytree(cls)
    

def batch_class_for(motion_cls: type[Any]) -> type[Any]:
    try:
        return _MOTION_TO_BATCH[motion_cls]
    except KeyError:
        raise ValueError() from None

    
# --- Registry for Steering Laws ---
_STEERING_TO_BATCH: dict[type[Any], type[Any]] = {}
_STEERING_PYTREE_REGISTERED: set[type[Any]] = set()
PYTREE_META_FIELDS = frozenset({"xp", "dtype"})

def register_steering_batch(law_cls: type[Any]) -> Callable[[type[Any]], type[Any]]:
    def decorator(cls: type[Any]) -> type[Any]:
        if law_cls in _STEERING_TO_BATCH:
            raise ValueError(f"SteeringLawBatch already registered for {law_cls.__name__!r}.")
        _STEERING_TO_BATCH[law_cls] = cls
        _maybe_register_steering_pytree(cls)
        return cls
    return decorator

def _maybe_register_steering_pytree(cls: type[Any]) -> None:
    if cls in _STEERING_PYTREE_REGISTERED:
        return
    try:
        import jax
    except ImportError:
        return
    
    all_field_names = tuple(f.name for f in fields(cls))
    meta = tuple(name for name in all_field_names if name in PYTREE_META_FIELDS)
    data = tuple(name for name in all_field_names if name not in PYTREE_META_FIELDS)
    
    jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)
    _STEERING_PYTREE_REGISTERED.add(cls)

def steering_batch_class_for(law_cls: type[Any]) -> type[Any]:
    try:
        return _STEERING_TO_BATCH[law_cls]
    except KeyError:
        raise ValueError(f"No SteeringLawBatch registered for {law_cls.__name__!r}") from None