from __future__ import annotations

import numpy as np
import pytest

from mimo.scene.radar_network import RadarNetwork, ChannelLink, EngagementIndices
from mimo.geometry.motion_model import StaticMotion
from mimo.scene.scene import Scene, BackendContext
from mimo.entity.entity import Entity
from mimo.entity.radar_component import(
    TxElement,
    RxElement,
    RadarTarget,
    RadarSensor,
)

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    HAS_JAX = False

requires_jax = pytest.mark.skipif(not HAS_JAX, reason="JAX is not installed.")

def static_motion():
    return StaticMotion(
        position=np.zeros(3),
        orientation=np.array([1, 0, 0, 0]),
    )


def make_radar_entity(node_name="node", n_tx=1, n_rx=1, extra_components=()):
    """Create an Entity carrying a RadarNode with `n_tx` transmitters and
    `n_rx` receivers. Returns (entity, node, tx_list, rx_list)."""
    txs = [TxElement() for _ in range(n_tx)]
    rxs = [RxElement() for _ in range(n_rx)]
    sensor = RadarSensor(node_name, transmitters=txs, receivers=rxs)
    entity = Entity(motion=static_motion(),components=[sensor, *extra_components])
    return entity, sensor, txs, rxs


def make_target_entity(rcs=1.0):
    props = RadarTarget(rcs=rcs)
    entity = Entity(motion=static_motion(), components=[props])
    return entity, props


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
 
@pytest.fixture
def scene():
    return Scene()
 
 
@pytest.fixture
def network(scene):
    return RadarNetwork(scene)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCrossJoinAlgorithm:
    def test_simple_cross_product_no_filtering(self):
        link_slots = np.array([[0, 1]])     # one link: tx=0, rx=1
        tgt_slots = np.array([2, 3])
        
        result = RadarNetwork.cross_join_links_and_targets(link_slots, tgt_slots, np)
    
        np.testing.assert_array_equal(result.tx_slots, [0, 0])
        np.testing.assert_array_equal(result.rx_slots, [1, 1])
        np.testing.assert_array_equal(result.tgt_slots, [2, 3])
    
    def test_filters_target_equal_to_tx_or_rx(self):
        link_slots = np.array([[0, 1]])
        tgt_slots = np.array([0, 1, 2])
        
        result = RadarNetwork.cross_join_links_and_targets(link_slots, tgt_slots, np)
        
        # Only target 2 survives; the radar can't illuminate/receive itself.
        np.testing.assert_array_equal(result.tx_slots, [0])
        np.testing.assert_array_equal(result.rx_slots, [1])
        np.testing.assert_array_equal(result.tgt_slots, [2])    
    
    def test_multiple_links_full_cross_product(self):
        link_slots = np.array([[0, 1], [2, 3]])
        tgt_slots = np.array([4, 5])
 
        result = RadarNetwork.cross_join_links_and_targets(link_slots, tgt_slots, np)
 
        np.testing.assert_array_equal(result.tx_slots, [0, 0, 2, 2])
        np.testing.assert_array_equal(result.rx_slots, [1, 1, 3, 3])
        np.testing.assert_array_equal(result.tgt_slots, [4, 5, 4, 5])
    
    def test_filtering_is_per_link_not_global(self):
        # Target 0 coincides with link0's tx, but link1 doesn't involve
        # slot 0 at all, so link1's engagement with target 0 must survive.
        link_slots = np.array([[0, 1], [2, 3]])
        tgt_slots = np.array([0, 4])
 
        result = RadarNetwork.cross_join_links_and_targets(link_slots, tgt_slots, np)
 
        np.testing.assert_array_equal(result.tx_slots, [0, 2, 2])
        np.testing.assert_array_equal(result.rx_slots, [1, 3, 3])
        np.testing.assert_array_equal(result.tgt_slots, [4, 0, 4])
    
    def test_all_targets_filtered_returns_empty(self):
        link_slots = np.array([[0, 1]])
        tgt_slots = np.array([0, 1])
 
        result = RadarNetwork.cross_join_links_and_targets(link_slots, tgt_slots, np)
 
        assert len(result.tx_slots) == 0
        assert len(result.rx_slots) == 0
        assert len(result.tgt_slots) == 0
        
    def test_output_lengths_are_consistent(self):
        link_slots = np.array([[0, 1], [2, 3], [4, 5]])
        tgt_slots = np.array([6, 7, 8, 9])
 
        result = RadarNetwork.cross_join_links_and_targets(link_slots, tgt_slots, np)
 
        assert len(result.tx_slots) == len(result.rx_slots) == len(result.tgt_slots)
        assert len(result.tx_slots) == 3 * 4  # no overlaps to filter here
    

# ---------------------------------------------------------------------------
# compile_channel_links
# ---------------------------------------------------------------------------

class TestChannelLinkCompilation:
    def test_no_links_returns_empty(self, scene, network):
        result = network.compile_channel_links()
        assert result.shape == (0, 2)

    def test_links_present_but_none_active_returns_empty(self, scene, network):
        tx_entity, _, txs, _ = make_radar_entity("txnode", n_tx=1, n_rx=0)
        rx_entity, _, _, rxs = make_radar_entity("rxnode", n_tx=0, n_rx=1)
        scene.add_entity(tx_entity)
        scene.add_entity(rx_entity)
 
        network.add_link(ChannelLink(tx=txs[0], rx=rxs[0], active=False))
 
        result = network.compile_channel_links()
        assert result.shape == (0, 2)
    
    def test_active_links_compiled_with_correct_slots_and_order(self, scene, network):
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        c, _, c_tx, _ = make_radar_entity("c", n_tx=1, n_rx=0)
        d, _, _, d_rx = make_radar_entity("d", n_tx=0, n_rx=1)
        for e in (a, b, c, d):
            scene.add_entity(e)
 
        # Inactive link sandwiched between two active links.
        network.add_link(ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True))
        network.add_link(ChannelLink(tx=c_tx[0], rx=d_rx[0], active=False))
        network.add_link(ChannelLink(tx=c_tx[0], rx=b_rx[0], active=True))
 
        result = network.compile_channel_links()
 
        expected = np.array([
            [a.scene_index, b.scene_index],
            [c.scene_index, b.scene_index],
        ])
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _collect_target_slots
# ---------------------------------------------------------------------------

class TestTargetSlotCollection:
    def test_no_entities_returns_empty(self, network):
        result = network._collect_target_slots()
        assert len(result) == 0
    
    def test_collects_only_entities_that_are_targets(self, scene, network):
        t1, _ = make_target_entity()
        t2, _ = make_target_entity()
        radar_entity, _, _, _ = make_radar_entity("radar")
        scene.add_entity(t1)
        scene.add_entity(radar_entity)
        scene.add_entity(t2)
        
        result = network._collect_target_slots()
        
        assert set(result.tolist()) == {t1.scene_index, t2.scene_index}
    
    def test_ignores_entities_without_has_component(self, scene, network):
        t1, _ = make_target_entity()
        bare = Entity(motion=static_motion())
        scene.add_entity(t1)
        scene.add_entity(bare)
 
        # Should not raise, and should skip `bare` via the hasattr guard.
        result = network._collect_target_slots()
 
        assert set(result.tolist()) == {t1.scene_index}
 
    def test_entity_that_is_both_radar_and_target_is_collected(self, scene, network):
        entity, _, _, _ = make_radar_entity(
            "dual", n_tx=1, n_rx=1, extra_components=[RadarTarget(rcs=2.0)]
        )
        scene.add_entity(entity)
 
        result = network._collect_target_slots()
 
        assert set(result.tolist()) == {entity.scene_index}
        
 
# ---------------------------------------------------------------------------
# get_engagements: end-to-end correctness
# ---------------------------------------------------------------------------

class TestGetEngagementsCorrectness:
    def test_empty_network_and_scene(self, network):
        result = network.get_engagements()
        assert len(result.tx_slots) == 0
        assert len(result.rx_slots) == 0
        assert len(result.tgt_slots) == 0
    
    def test_links_without_targets_yields_no_engagements(self, scene, network):
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        scene.add_entity(a)
        scene.add_entity(b)
        network.add_link(ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True))
 
        result = network.get_engagements()
        assert len(result.tx_slots) == 0
    
    def test_basic_bistatic_link_against_two_targets(self, scene, network):
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        t1, _ = make_target_entity()
        t2, _ = make_target_entity()
        for e in (a, b, t1, t2):
            scene.add_entity(e)
        network.add_link(ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True))
 
        result = network.get_engagements()
 
        got = set(zip(result.tx_slots.tolist(), result.rx_slots.tolist(), result.tgt_slots.tolist()))
        expected = {(a.scene_index, b.scene_index, t1.scene_index), (a.scene_index, b.scene_index, t2.scene_index)}
        assert got == expected
        
    def test_radar_that_is_also_a_target_is_excluded_from_its_own_link(self, scene, network):
        # `a` transmits and is itself trackable by other radars.
        a, _, a_tx, _ = make_radar_entity(
            "a", n_tx=1, n_rx=0, extra_components=[RadarTarget(rcs=5.0)]
        )
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        t1, _ = make_target_entity()
        for e in (a, b, t1):
            scene.add_entity(e)
        network.add_link(ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True))
 
        result = network.get_engagements()
 
        got = set(zip(result.tx_slots.tolist(), result.rx_slots.tolist(), result.tgt_slots.tolist()))
        # (a, b, a) is excluded because the target coincides with the tx.
        assert got == {(a.scene_index, b.scene_index, t1.scene_index)}
 
    def test_inactive_links_excluded_from_engagements(self, scene, network):
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        t1, _ = make_target_entity()
        for e in (a, b, t1):
            scene.add_entity(e)
        network.add_link(ChannelLink(tx=a_tx[0], rx=b_rx[0], active=False))
 
        result = network.get_engagements()
        assert len(result.tx_slots) == 0

# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

class TestEngagementCaching:
    def _build_basic_link(self, scene, network):
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        t1, _ = make_target_entity()
        for e in (a, b, t1):
            scene.add_entity(e)
        link = ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True)
        network.add_link(link)
        return link, a, b, t1
 
    def test_repeated_calls_without_changes_return_same_object(self, scene, network):
        self._build_basic_link(scene, network)
 
        result1 = network.get_engagements()
        result2 = network.get_engagements()
 
        assert result1 is result2
 
    def test_cache_invalidated_by_adding_entity(self, scene, network):
        self._build_basic_link(scene, network)
        result1 = network.get_engagements()
 
        t2, _ = make_target_entity()
        scene.add_entity(t2)
        result2 = network.get_engagements()
 
        assert result1 is not result2
        assert t2.scene_index in result2.tgt_slots.tolist()
 
    def test_cache_invalidated_by_removing_entity(self, scene, network):
        _, a, b, t1 = self._build_basic_link(scene, network)
        result1 = network.get_engagements()
 
        scene.remove_entity(t1.id)
        result2 = network.get_engagements()
 
        assert result1 is not result2
        assert len(result2.tx_slots) == 0
 
    def test_cache_invalidated_by_adding_link(self, scene, network):
        _, a, b, t1 = self._build_basic_link(scene, network)
        result1 = network.get_engagements()
 
        c, _, c_tx, _ = make_radar_entity("c", n_tx=1, n_rx=0)
        scene.add_entity(c)
        network.add_link(ChannelLink(tx=c_tx[0], rx=b.get_component(RadarSensor).rx_elements[0], active=True))
        result2 = network.get_engagements()
 
        assert result1 is not result2
        assert len(result2.tx_slots) == 2
 
    def test_cache_invalidated_by_removing_link(self, scene, network):
        self._build_basic_link(scene, network)
        result1 = network.get_engagements()
 
        link = network.links[0]
        network.remove_link(link)
        result2 = network.get_engagements()
 
        assert result1 is not result2
        assert len(result2.tx_slots) == 0
        
    def test_toggling_links_invalidate_cache(self, scene, network):
        link, a, b, t1 = self._build_basic_link(scene, network)
        result1 = network.get_engagements()
        assert len(result1.tx_slots) == 1
        
        network.toggle_link(link, False)
        result2 = network.get_engagements()
        
        assert result1 is not result2
        assert len(result2.tx_slots) == 0
        


# ---------------------------------------------------------------------------
# Link management
# ---------------------------------------------------------------------------
 
class TestLinkManagement:
    def test_add_link_appends_and_bumps_version(self, scene, network):
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        scene.add_entity(a)
        scene.add_entity(b)
 
        version_before = network._network_version
        link = ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True)
        network.add_link(link)
 
        assert network.links == [link]
        assert network._network_version == version_before + 1
 
    def test_remove_link_removes_and_bumps_version(self, scene, network):
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        scene.add_entity(a)
        scene.add_entity(b)
        link = ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True)
        network.add_link(link)
 
        version_before = network._network_version
        network.remove_link(link)
 
        assert network.links == []
        assert network._network_version == version_before + 1
 
    def test_remove_unknown_link_raises(self, scene, network):
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        scene.add_entity(a)
        scene.add_entity(b)
        unregistered_link = ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True)
 
        with pytest.raises(ValueError):
            network.remove_link(unregistered_link)
        
# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------
 
@requires_jax
class TestJaxBackend:
    
    @pytest.fixture
    def jax_scene(self):
        return Scene(backend=BackendContext.jax())
 
    @pytest.fixture
    def jax_network(self, jax_scene):
        return RadarNetwork(jax_scene)
 
    def test_network_uses_jax_module(self, jax_scene, jax_network):
        assert jax_network._xp is jnp
 
    def test_empty_engagements_are_jax_arrays(self, jax_scene, jax_network):
        assert jax is not None
        result = jax_network.get_engagements()
        assert isinstance(result.tx_slots, jax.Array)
        assert isinstance(result.rx_slots, jax.Array)
        assert isinstance(result.tgt_slots, jax.Array)
 
    def test_nonempty_engagements_are_jax_arrays(self, jax_scene, jax_network):
        assert jax is not None
        a, _, a_tx, _ = make_radar_entity("a", n_tx=1, n_rx=0)
        b, _, _, b_rx = make_radar_entity("b", n_tx=0, n_rx=1)
        t1, _ = make_target_entity()
        for e in (a, b, t1):
            jax_scene.add_entity(e)
        jax_network.add_link(ChannelLink(tx=a_tx[0], rx=b_rx[0], active=True))
 
        result = jax_network.get_engagements()
 
        assert isinstance(result.tx_slots, jax.Array)
        assert isinstance(result.rx_slots, jax.Array)
        assert isinstance(result.tgt_slots, jax.Array)
        assert result.tx_slots.tolist() == [a.scene_index]
 
    def test_compile_channel_links_empty_branch_uses_jax(self, jax_scene, jax_network):
        assert jax is not None
        result = jax_network.compile_channel_links()
        assert isinstance(result, jax.Array)