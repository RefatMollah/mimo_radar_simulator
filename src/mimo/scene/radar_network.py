from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Any, TypeAlias

import numpy as np

from ..entity.radar_component import TxElement, RxElement, RadarTarget
from .scene import Scene

ArrayLike: TypeAlias = Any
Backend: TypeAlias = Any

@dataclass(slots=True)
class ChannelLink:
    """A transmit/receive pair that can be toggled on or off."""
    tx: TxElement
    rx: RxElement
    active: bool
    

@dataclass(slots=True, frozen=True)
class EngagementIndices:
    """
    Every valid (transmitter, target, receiver) triple in the network.

    A triple exists for each active ChannelLink x each scene target, minus
    any triple where the target is one of the two radars in the link (a
    radar can't illuminate or receive reflections from itself).

    All three arrays are the same length; row i is one engagement.
    """
    tx_slots:   ArrayLike
    rx_slots:   ArrayLike
    tgt_slots:  ArrayLike    


@dataclass
class _EngagementCache:
    scene_version:   int
    network_version: int
    engagement_indices: EngagementIndices


class RadarNetwork:
    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self._xp = scene._backend.module
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
    
    def toggle_link(self, link: ChannelLink, active: bool) -> None:
        if link.active != active:
            link.active = active
            self._network_version += 1

    
    def compile_channel_links(self) -> ArrayLike:
        """Compile the relevant entity slots of each tx-rx pair in all links."""
        xp = self._xp
        if not self._links:
            return xp.empty((0,2), dtype=int)
        
        active_links = [l for l in self._links if l.active]
        if not active_links:
            return xp.empty((0,2), dtype=int)
        
        link_slots = xp.asarray(
            [(link.tx.entity.scene_index, link.rx.entity.scene_index) for link in active_links],
            dtype=int
        )
        return link_slots

    def _collect_target_slots(self) -> ArrayLike:
        return np.fromiter(
            (
                entity.scene_index
                for entity in self._scene.iter_all()
                if hasattr(entity, 'has_component') and entity.has_component(RadarTarget)
            ),
            dtype=int
        )
            
    def get_engagements(self) -> EngagementIndices:
        xp = self._scene._backend.xp
        scene_version = self._scene.topology_version
        network_version = self._network_version
        
        cache_is_stale = (
            self._cache is None 
            or self._cache.scene_version != scene_version
            or self._cache.network_version != network_version
        )
        
        if cache_is_stale:
            link_slots = self.compile_channel_links()
            tgt_slots = self._collect_target_slots()
        else:
            assert self._cache is not None
            return self._cache.engagement_indices
        
        if link_slots.size == 0 or tgt_slots.size == 0:
            engagement_indices = EngagementIndices(
                                    xp.empty(0, dtype=int),
                                    xp.empty(0, dtype=int),
                                    xp.empty(0, dtype=int),
                                )
            self._cache = _EngagementCache(
                scene_version=scene_version,
                network_version=network_version,
                engagement_indices=engagement_indices,
            )
            return engagement_indices
        
        engagement_indices = self.cross_join_links_and_targets(link_slots, tgt_slots, xp)
        self._cache = _EngagementCache(
            scene_version=scene_version,
            network_version=network_version,
            engagement_indices=engagement_indices,
        )
        return engagement_indices


    @staticmethod
    def cross_join_links_and_targets(link_slots: ArrayLike, tgt_slots: ArrayLike, xp):

        n_links = len(link_slots)
        n_targets = len(tgt_slots)

        links_per_engagement = xp.repeat(link_slots, n_targets, axis=0)
        targets_per_engagement = xp.tile(tgt_slots, n_links)
        
        tx_slots = links_per_engagement[:, 0]
        rx_slots = links_per_engagement[:, 1]
        
        # Remove tx-tgt-rx triples where the transmitter/receiver is also the target
        is_self_engagement = (tx_slots == targets_per_engagement) | (rx_slots == targets_per_engagement)
        valid = ~is_self_engagement
        
        return EngagementIndices(
            tx_slots=tx_slots[valid],
            tgt_slots=targets_per_engagement[valid],
            rx_slots=rx_slots[valid],
        )

        
    