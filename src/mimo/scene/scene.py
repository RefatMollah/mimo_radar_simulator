from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Tuple
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from ..entity.radar_component import TransmitterElement, ReceiverElement, RadarNode, RadarComponent, TargetProperties

if TYPE_CHECKING:
    from ..entity.entity import Entity
    
logger = logging.getLogger(__name__)

class Scene:
    
    def __init__(self) -> None:
        self._static_entities: dict[int, Entity] = {}
        self._dynamic_entities: dict[int, Entity] = {}
        
        self._entity_ids: dict[str, Entity] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self._topology_version = 0
    
    @property
    def topology_version(self) -> int:
        return self._topology_version
    
    @property
    def max_slots(self) -> int:
        return self._next_slot
    
    def _bump_topology(self) -> None:
        self._topology_version += 1
         
    def add_entity(self, entity:Entity, static: bool=False) -> None:
        
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            slot = self._next_slot
            self._next_slot +=1
        
        entity._slot = slot
        self._entity_ids[entity.id] = entity

        if static:
            self._static_entities[slot] = entity
        else:
            self._dynamic_entities[slot] = entity
            
        self._bump_topology()
    
    def remove_entity(self, entity_id: str) -> None:
        entity = self._entity_ids.pop(entity_id, None)
        if not entity:
            raise EntityNotFoundError(entity_id)
        
        slot = entity.slot
        self._static_entities.pop(slot, None)
        self._dynamic_entities.pop(slot, None)
        
        self._free_slots.append(slot)
        entity._slot = -1
        self._bump_topology()

    def get_entity(self, entity_id: str) -> Entity:
        try:
            return self._entity_ids[entity_id]
        except KeyError:
            raise EntityNotFoundError(entity_id)
    
    def iter_static(self) -> Iterator[Entity]:
        yield from self._static_entities.values()
    
    def iter_dynamic(self) -> Iterator[Entity]:
        yield from self._dynamic_entities.values()
    
    
    def iter_all(self) -> Iterator[Entity]:
        yield from self._static_entities.values()
        yield from self._dynamic_entities.values()
    
    @property
    def entity_count(self) -> int:
        return len(self._static_entities) + len(self._dynamic_entities)
    
    def __repr__(self) -> str:
        return (
            f"Scene(static={len(self._static_entities)}, "
            f"dynamic={len(self._dynamic_entities)})"
        )  


_EMPTY_INT = np.empty(0, dtype=np.int_)

@dataclass
class ChannelLink:
    """A transmit/receive pair that can be toggled on or off."""
    tx: TransmitterElement
    rx: ReceiverElement
    active: bool

@dataclass(frozen=True, slots=True)
class EngagementIndices:
    """
    Every valid (transmitter, target, receiver) triple in the network.

    A triple exists for each active ChannelLink x each scene target, minus
    any triple where the target is one of the two radars in the link (a
    radar can't illuminate or receive reflections from itself).

    All three arrays are the same length; row i is one engagement.
    """
    tx_slots:     NDArray[np.int_]
    target_slots: NDArray[np.int_]
    rx_slots:     NDArray[np.int_]
    link_idx:     NDArray[np.int_]

@dataclass(frozen=True, slots=True)
class SensorOffsets:
    """
    Per-link mounting offsets, row-aligned with CompiledChannels.link_slots such that
    positions[link_slot] += pos_offsets
    """
    pos_offsets: NDArray[np.float64]   # (n_links, 2, 3) -> [:, 0]=tx, [:, 1]=rx
    rot_offsets: NDArray[np.float64]   # (n_links, 2, 4) -> [:, 0]=tx, [:, 1]=rx

@dataclass
class CompiledChannels:
    """Active channel links with sensor offsets flattened into aligned NumPy arrays."""
    link_slots:     NDArray[np.int_]   # [tx_slot, rx_slot]
    sensor_offsets: SensorOffsets

@dataclass
class RadarEngagements:
    """Radar Engagement Indices with with channel sensor offsets."""
    channel: CompiledChannels
    indices: EngagementIndices

@dataclass
class _EngagementsCache:
    scene_version:   int
    network_version: int
    compiled_links:  CompiledChannels
    engagements:     EngagementIndices
    
class RadarNetwork:
    """
    Owns the set of tx/rx ChannelLinks in a scene and derives, from them,
    every valid (tx, target, rx) engagement. Results are cached and only
    recomputed when the scene topology or the link set changes.
    """

    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self._links: list[ChannelLink] = []

        self._network_version = 0
        self._cache: _EngagementsCache | None = None

    @property
    def links(self) -> list[ChannelLink]:
        return self._links

    def add_link(self, link: ChannelLink) -> None:
        self._links.append(link)
        self._network_version += 1

    def remove_link(self, link: ChannelLink) -> None:
        self._links.remove(link)
        self._network_version += 1

    def set_link_active(self, link: ChannelLink, active: bool) -> None:
        if link.active != active:
            link.active = active
            self._network_version += 1

    def compile_channel_links(self) -> CompiledChannels:
        """Flatten the active links into aligned arrays for vectorized math."""
        active_links = [link for link in self._links if link.active]

        link_slots = np.asarray(
            [(link.tx.slot, link.rx.slot) for link in active_links],
            dtype=np.int_,
        )
        pos_offsets = np.asarray(
            [(link.tx.pos_offset, link.rx.pos_offset) for link in active_links]
        )
        rot_offsets = np.asarray(
            [(link.tx.rot_offset, link.rx.rot_offset) for link in active_links]
        )

        return CompiledChannels(
            link_slots=link_slots,
            sensor_offsets=SensorOffsets(pos_offsets, rot_offsets),
        )

    def get_engagements(self) -> RadarEngagements:
        """Return the cached engagements, recomputing if the scene or link set changed."""
        scene_version = self._scene.topology_version
        network_version = self._network_version

        cache_is_stale = (
            self._cache is None
            or self._cache.scene_version != scene_version
            or self._cache.network_version != network_version
        )

        if cache_is_stale:
            compiled = self.compile_channel_links()
            target_slots = self._collect_target_slots()
            engagements = self._cross_join_links_and_targets(compiled.link_slots, target_slots)

            self._cache = _EngagementsCache(
                scene_version=scene_version,
                network_version=network_version,
                compiled_links=compiled,
                engagements=engagements,
            )
        assert self._cache is not None        
        return RadarEngagements(
            channel=self._cache.compiled_links,
            indices=self._cache.engagements,
        )

    def _collect_target_slots(self) -> NDArray[np.int_]:
        """Slot numbers of every entity in the scene that carries TargetProperties."""
        return np.fromiter(
            (
                entity.slot
                for entity in self._scene.iter_all()
                if hasattr(entity, 'has_component') and entity.has_component(TargetProperties)
            ),
            dtype=np.int_,
        )

    @staticmethod
    def _cross_join_links_and_targets(link_slots: NDArray[np.int_], target_slots: NDArray[np.int_],
    ) -> EngagementIndices:
        """
        Pair every active (tx, rx) link with every target,
        excluding every pair where the target is one of the illuminators.
        """
        if link_slots.size == 0 or target_slots.size == 0:
            return EngagementIndices(_EMPTY_INT, _EMPTY_INT, _EMPTY_INT, _EMPTY_INT)

        n_targets = len(target_slots)
        n_links = len(link_slots)

        # Row i of each array below is one (link, target) combination:
        # repeat each link once per target, tile targets once per link.
        links_per_engagement = np.repeat(link_slots, n_targets, axis=0)
        targets_per_engagement = np.tile(target_slots, n_links)
        link_idx_per_engagement = np.repeat(np.arange(n_links), n_targets)

        tx_slots = links_per_engagement[:, 0]
        rx_slots = links_per_engagement[:, 1]

        is_self_engagement = (tx_slots == targets_per_engagement) | (rx_slots == targets_per_engagement)
        valid = ~is_self_engagement

        return EngagementIndices(
            tx_slots[valid],
            targets_per_engagement[valid],
            rx_slots[valid],
            link_idx_per_engagement[valid],
        )


#################################################################################
# Exceptions
#################################################################################

class DuplicateEntityError(Exception):
    """Raised when an entity UUID is already registered in the scene."""
    
    def __init__(self, entity_id: str) -> None:
        super().__init__(
            f"Entity '{entity_id}' is already registered in this scene."
        )

class EntityNotFoundError(Exception):
    """Raised when a requested entity UUID is absent from the scene."""
    
    def __init__(self, entity_id: str):
        super().__init__(
            f"Entity '{entity_id}' not found in this scene."
        ) 