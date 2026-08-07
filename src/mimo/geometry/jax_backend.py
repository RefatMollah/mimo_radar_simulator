from typing import Any, Literal, Callable, TypeAlias, TypeVar
from dataclasses import fields, replace

from ..scene.scene import CompiledScene
from .motion_model import (
    MotionBatch,
    Motion,
)
from .spatial_engine import State, state_at, densify, state_at_dense

ArrayLike: TypeAlias = Any
_PYTREE_REGISTERED: set[type[Any]] = set()
_MOTION_TO_BATCH: dict[type[Motion], type[MotionBatch]] = {}
B = TypeVar("B", bound=MotionBatch)


def register_motion_batch(motion_cls: type[Motion]) -> Callable[[type[B]], type[B]]:
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
    
    field_names = tuple(f.name for f in fields(cls))
    
    def flatten(obj: Any, names: tuple[str, ...] = field_names) -> Any:
        return tuple(getattr(obj, n) for n in names), None
    
    def unflatten(
        aux: None, 
        children: tuple[Any, ...], 
        cls: type[Any] = cls,
        names: tuple[str, ...] = field_names,
    ) -> Any:
        del aux
        return cls(**dict(zip(names, children)))
    
    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    _PYTREE_REGISTERED.add(cls)
    

def _ensure_all_pytrees_registered() -> None:
    for cls in _MOTION_TO_BATCH.values():
        _maybe_register_pytree(cls)
    _maybe_register_pytree(State)


def batch_class_for(motion_cls: type[Motion]) -> type[MotionBatch]:
    try:
        return _MOTION_TO_BATCH[motion_cls]
    except KeyError:
        raise ValueError() from None