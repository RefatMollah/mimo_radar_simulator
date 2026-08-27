from __future__ import annotations

from typing import Iterable, cast, TypeVar, TYPE_CHECKING

import uuid
import numpy as np

from ..geometry.motion_model import Motion, StaticMotion

if TYPE_CHECKING:
    from .radar_component import Component

C = TypeVar("C", bound="Component")
        
class Entity:
    """
    A scene object/platform.

    An Entity can be a target, a radar host, a jammer, a clutter patch,
    a receiver-only node, or any combination of those.
    """

    __slots__ = (
        "_id",
        "name",
        "motion",
        "enabled",
        "tags",
        "metadata",
        "_components",
        "_scene_index",
    )

    def __init__(
        self,
        motion: Motion,
        components: Iterable[Component] | None = None,
        *,
        entity_id: str | None = None,
        name: str | None = None,
        enabled: bool = True,
        tags: Iterable[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._id = entity_id or str(uuid.uuid4())
        self.name = name
        self.motion = motion
        self.enabled = enabled
        self.tags = set(tags or ())
        self.metadata = dict(metadata or {})

        self._scene_index: int | None = None
        self._components: dict[type[Component], list[Component]] = {}

        if components is not None:
            for component in components:
                self.add_component(component)

    @property
    def id(self) -> str:
        return self._id

    @property
    def scene_index(self) -> int:
        if self._scene_index is None:
            raise RuntimeError(f"Entity {self._id} has not been added to a Scene.")
        return self._scene_index

    @property
    def scene_index_or_none(self) -> int | None:
        return self._scene_index

    
    @property
    def is_static(self) -> bool:
        return isinstance(self.motion, StaticMotion)

    def _assign_scene_index(self, index: int) -> None:
        # Intended to be called only by Scene.
        self._scene_index = index

    def add_component(
        self,
        component: Component,
        *,
        replace: bool = False,
    ) -> Component:
        """
        Add a component.

        If replace=False, adding a second component of the exact same type
        raises. If replace=True, existing components of the exact same type
        are detached and replaced.
        """
        if component.is_attached:
            raise ComponentAlreadyAttachedError(
                f"{type(component).__name__} is already attached to an Entity."
            )

        component_type = type(component)
        existing = self._components.get(component_type, [])

        if existing and not replace:
            raise ComponentAlreadyAttachedError(
                f"Entity {self._id} already has component {component_type.__name__}."
            )

        if existing and replace:
            for old in existing:
                self._detach_component(old)
            existing.clear()
        elif not existing:
            self._components[component_type] = []

        self._components[component_type].append(component)
        component._attach(self)
        return component

    def has_component(self, component_type: type[Component]) -> bool:
        """
        Returns True if this entity has at least one component that is an
        instance of component_type.
        """
        return any(
            issubclass(existing_type, component_type)
            for existing_type in self._components
        )

    def get_components(self, component_type: type[C]) -> tuple[C, ...]:
        """
        Return all components assignable to component_type.

        This uses isinstance semantics, so subclasses are included.
        """
        return tuple(
            component
            for components in self._components.values()
            for component in components
            if isinstance(component, component_type)
        )

    def get_component(self, component_type: type[C], index: int = 0) -> C:
        """
        Return one component assignable to component_type.

        If multiple components exist, use get_components() or index.
        """
        components = self.get_components(component_type)
        if not components:
            raise ComponentNotFoundError(
                f"Entity {self._id} has no component of type {component_type.__name__}."
            )

        try:
            return components[index]
        except IndexError:
            raise ComponentNotFoundError(
                f"Entity {self._id} has no {component_type.__name__} at index {index}."
            ) from None

    def try_get_component(
        self,
        component_type: type[C],
        index: int = 0,
    ) -> C | None:
        components = self.get_components(component_type)
        if index < 0 or index >= len(components):
            return None
        return components[index]

    def remove_component(
        self,
        component_type: type[C],
        index: int = 0,
    ) -> C:
        components = self.get_components(component_type)
        if not components:
            raise ComponentNotFoundError(
                f"Entity {self._id} has no component of type {component_type.__name__}."
            )

        if index < 0 or index >= len(components):
            raise ComponentNotFoundError(
                f"Entity {self._id} has no {component_type.__name__} at index {index}."
            )

        component = components[index]
        self._detach_component(component)

        exact_type = type(component)
        self._components[exact_type].remove(component)

        if not self._components[exact_type]:
            del self._components[exact_type]

        return component

    def _detach_component(self, component: Component) -> None:
        component._detach()

    @property
    def components(self) -> tuple[Component, ...]:
        return tuple(
            component
            for components in self._components.values()
            for component in components
        )
    

##################################################################################################################
# Exceptions
##################################################################################################################   
    
class ComponentAlreadyAttachedError(Exception):
    def __init__(self, component_name: str):
        super().__init__(f"Component {component_name} already attached to this entity")

class ComponentNotFoundError(Exception):
    def __init__(self, component_name):
        super().__init__(f"Component {component_name} not attached.")