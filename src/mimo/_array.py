"""Internal NumPy/JAX interoperability helpers."""

from __future__ import annotations

from types import ModuleType
from typing import Any, TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike as NumpyArrayLike
from numpy.typing import NDArray

# Represents the subset of dtype-like specifications supported across both
# NumPy and JAX Array API `asarray` entrypoints. Excludes dictionary-based
# structured dtypes supported solely by native NumPy.
DTypeLike: TypeAlias = str | type[Any] | np.dtype[Any]

if TYPE_CHECKING:
    from jax import Array as JaxArray
    from jax.typing import ArrayLike as JaxArrayLike

    Array: TypeAlias = NDArray[np.generic] | JaxArray

    # Boundary type for intermediate values across heterogeneous array engines.
    # Left unconstrained (`Any`) because NumPy and JAX lack a unified static
    # interface covering the full Array API specification. Public entrypoints
    # should accept `ArrayInput` instead.
    ArrayLike: TypeAlias = Any
    ArrayInput: TypeAlias = NumpyArrayLike | JaxArrayLike
else:
    # Runtime fallbacks avoiding a hard dependency on JAX.
    Array: TypeAlias = NDArray[np.generic]
    ArrayLike: TypeAlias = Any
    ArrayInput: TypeAlias = NumpyArrayLike

BackendName: TypeAlias = Literal["numpy", "jax"]
ArrayNamespace: TypeAlias = Any


def array_namespace(*arrays: ArrayLike) -> ModuleType:
    """Return the shared Array API namespace for the provided arrays.

    Raises:
        ValueError: If input arrays belong to mixed or unsupported backends.
    """
    from array_api_compat import array_namespace as get_namespace

    return get_namespace(*arrays)


def is_jax_namespace(namespace: ModuleType) -> bool:
    """Determine whether a given namespace object corresponds to the JAX backend."""
    from array_api_compat import is_jax_namespace as check_namespace

    return check_namespace(namespace)


__all__ = ["Array", "ArrayInput", "ArrayLike", "ArrayNamespace", "BackendName", "DTypeLike", "array_namespace", "is_jax_namespace"]
