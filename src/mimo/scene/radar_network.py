from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Any, TypeAlias

import numpy as np

from ..entity.radar_component import TransmitterElement, ReceiverElement, TargetProperties
from .scene import Scene

ArrayLike: TypeAlias = Any


@dataclass(slots=True)
class ChannelLink:
    tx: TransmitterElement
    rx: ReceiverElement
    active: bool


@dataclass(slots=True, frozen=True)
class EngagementIndices:
    tx_slots:   ArrayLike
    rx_slots:   ArrayLike
    tgt_slots:  ArrayLike
    link_idx:   ArrayLike
    

@dataclass
class _EngagementCache:
    scene_version:   int
    network_version: int
    compiled_channels: EngagementIndices

class RadarNetwork:
    
    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self._links: list[ChannelLink] = []
        
        self._network_version = 0
        self._cache: _EngagementCache | None = None
    

    @property
    def links(self) -> list[ChannelLink]:
        return self._links
    
    def add_link(self, link: ChannelLink) -> None:
        self._links.append(link)
        self._network_version += 1
        
    def remove_link(self, link: ChannelLink) -> None:
        self._links.remove(link)
        self._network_version += 1
        
    
    def get_engagements(self) -> EngagementIndices:
        xp = self._scene._backend.xp
        scene_version = self._scene.topology_version
        network_version = self._network_version
        
        cache_is_stale = (
            self._cache is None 
            or self._cache.scene_version != scene_version
            or self._cache.network_version != network_version
        )
        
        active_links = [link for link in self._links if link.active]
        
        link_slots = xp.asarray(
            [(link.tx.pos_offset, link.rx.pos_offset) for link in active_links],
            dtype=int
        )
        tgt_slots = self._collect_target_slots()
        
        if link_slots.size == 0 or tgt_slots.size == 0:
            return EngagementIndices(
                xp.empty(0, dtype=int),
                xp.empty(0, dtype=int),
                xp.empty(0, dtype=int),
                xp.empty(0, dtype=int),
            )
        
        n_targets = len(tgt_slots)
        n_links = len(link_slots)
        
        links_per_engagement = xp.repeat(link_slots, n_targets, axis=0)
        targets_per_engagement = xp.tile(tgt_slots, n_links)
        link_idx_per_engagement = np.repeat(np.arange(n_links), n_targets)
        
        tx_slots = links_per_engagement[:, 0]
        rx_slots = links_per_engagement[:, 1]
        
        is_self_engagement = (tx_slots == targets_per_engagement)
        valid = ~is_self_engagement
        
        return EngagementIndices(
            tx_slots=tx_slots[valid],
            tgt_slots=targets_per_engagement[valid],
            rx_slots=rx_slots[valid],
            link_idx=link_idx_per_engagement[valid],
        )
    
    def _collect_target_slots(self) -> ArrayLike:
        return np.fromiter(
            (
                entity.slot
                for entity in self._scene.iter_all()
                if hasattr(entity, 'has_component') and entity.has_component(TargetProperties)
            ),
            dtype=np.int_
        )
        
    