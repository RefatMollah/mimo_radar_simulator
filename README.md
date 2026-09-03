# mimo-radar

`mimo-radar` is an experimental Python library for describing MIMO radar
scenes and evaluating their kinematics and geometry. The public API is still
pre-alpha; breaking changes may occur before the first stable release.

## Install

The default installation supports NumPy:

```bash
python -m pip install mimo-radar
```

For JAX acceleration and `jit`-compiled scene evaluation, install the optional
extra:

```bash
python -m pip install "mimo-radar[jax]"
```

JAX hardware packages are chosen by the user for their own platform. For
example, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
before using an NVIDIA GPU.

## Quick start

```python
import numpy as np

from mimo import Entity, Scene, StaticMotion, state_at

scene = Scene()
scene.add_entity(
    Entity(
        StaticMotion(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )
)

state = state_at(scene.compile(), 0.0)
print(state.positions)
```

Use `Scene(backend=BackendContext.jax())` when JAX is installed and the
compiled scene should use JAX arrays. Array-level quaternion functions infer
their namespace from their input arrays and reject mixed NumPy/JAX inputs.

## Public API

The supported starting points are exported from `mimo`:

- scene construction: `Scene`, `BackendContext`, `Entity`;
- motion models: `StaticMotion`, `ConstantVelocityMotion`, and
  `ConstantAccelerationMotion`;
- radar components: `RadarSensor`, `RadarTarget`, `TxElement`, and
  `RxElement`;
- scene state evaluation: `state_at`.

Lower-level modules remain available for experimentation but are not yet a
stable compatibility surface.

## Development

```bash
python -m pip install -e ".[dev,jax]"
python -m pytest
python -m pyright
python -m build
```

Building creates a source distribution and a universal pure-Python wheel. This
repository does not publish to TestPyPI, PyPI, Conda, or any other package
index automatically.
