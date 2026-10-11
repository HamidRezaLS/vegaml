"""Adapter routing used one threshold for every adapter.

`route_adapters(gates, threshold)` takes a single threshold, and both the engine and the image path
called it with `self.adapters[0]["threshold"]` -- adapter 0's own setting, applied to every adapter.

Each adapter's metadata carries its own `gate_threshold` (`add_adapter` reads it and stores it as
`adapters[i]["threshold"]`). So a checkpoint shipping two adapters with different thresholds had
adapter 0's threshold deciding adapter 1's routing. When that sends a question through the wrong
adapter, it is calibrated with the wrong `theta` and the wrong conformal table: every probability
for that question is wrong, and nothing in the answer says so.

This is a calibration-correctness bug, and it is silent -- the answer looks completely normal.
"""
from __future__ import annotations

import numpy as np
import pytest

from vegaml._common import route_adapters, route_per_adapter


def test_each_adapter_is_gated_by_its_own_threshold():
    """Adapter 1 is the most open gate but its own threshold is 0.99, so it must not be used.

    Adapter 0 qualifies on its own threshold (0.90 >= 0.5). Adapter 1 does not (0.95 < 0.99), even
    though 0.95 is the larger of the two numbers.
    """
    gates = np.array([[0.90, 0.95]])
    assert list(route_per_adapter(gates, [0.5, 0.99])) == [0]


def test_a_question_can_go_to_the_base_engine_when_no_adapter_qualifies():
    gates = np.array([[0.10, 0.20], [0.30, 0.40]])
    assert list(route_per_adapter(gates, [0.5, 0.99])) == [-1, -1], \
        "below every threshold means the base calibration, not the least-bad adapter"


def test_the_single_threshold_call_got_this_wrong():
    """The behaviour that was actually shipped, pinned so the fix is visibly a fix.

    `route_adapters` with one threshold picks the largest gate that clears it. With adapter 0's
    threshold of 0.5, gate 0.95 on adapter 1 clears comfortably and the question is routed there --
    even though adapter 1's own threshold is 0.99 and it should have been refused.
    """
    gates = np.array([[0.90, 0.95]])
    assert list(route_adapters(gates, 0.5)) == [1], "this is the misroute the fix removes"
    assert list(route_per_adapter(gates, [0.5, 0.99])) == [0]


def test_an_adapter_can_be_shut_entirely_by_its_own_threshold():
    """A threshold of 1.0 means "never route here", which is a legitimate per-adapter setting.

    Row 2 has adapter 1 at 0.99 -- its most open possible gate -- and it still does not qualify,
    because its own threshold is 1.0. With a single shared threshold of 0.5 that question would have
    gone to adapter 1.
    """
    gates = np.array([[0.99, 0.99], [0.10, 0.99]])
    assert list(route_per_adapter(gates, [0.5, 1.0])) == [0, -1], \
        "adapter 1 is shut, so the question must not route there"
    assert list(route_adapters(gates, 0.5)) == [0, 1], "the old shared threshold sent row 2 to adapter 1"


def test_more_adapters_than_thresholds_falls_back_to_a_half():
    """A gate column for an adapter this checkpoint never loaded must not default to 0, which would
    close it. 0.5 is the same default the single-threshold path used."""
    gates = np.array([[0.1, 0.1, 0.9]])
    assert list(route_per_adapter(gates, [0.5, 0.5])) == [2]


def test_more_thresholds_than_gates_is_harmless():
    """A stale threshold list longer than the gate matrix must not raise or overflow."""
    gates = np.array([[0.9, 0.9]])
    assert list(route_per_adapter(gates, [0.5, 0.5, 0.5, 0.5])) == [0]


def test_no_gates_means_no_routing_and_is_handed_on():
    assert route_per_adapter(None, [0.5, 0.5]) is None
    assert route_adapters(None, 0.5) is None


def test_routing_is_deterministic_when_two_adapters_both_qualify():
    """Ties must not depend on argmax implementation or array order."""
    gates = np.array([[0.9, 0.9]])
    assert list(route_per_adapter(gates, [0.5, 0.5])) == [0], "ties go to the lowest index"
    assert list(route_per_adapter(gates, [0.5, 0.5])) == list(route_per_adapter(gates, [0.5, 0.5]))


def test_no_adapters_loaded_keeps_the_original_behaviour():
    """With nothing loaded there is nothing to route to, so a single 0.5 threshold is correct."""
    gates = np.array([[0.6, 0.4]])
    assert list(route_per_adapter(gates, [])) == [0, ]
    assert list(route_adapters(gates, 0.5)) == [0]


def test_a_one_dimensional_gate_vector_is_refused_rather_than_silently_misread():
    """g.max(1) on a 1-d array would raise; returning None keeps the caller on the base calibration."""
    assert route_per_adapter(np.array([0.9, 0.9]), [0.5, 0.5]) is None


# ------------------------------------------------------------------ the engine wiring
def test_the_engine_routes_with_per_adapter_thresholds(monkeypatch):
    """_api.Vega._route must consult each adapter's own threshold, not adapters[0]."""
    import inspect

    from vegaml._api import Vega

    src = inspect.getsource(Vega._view)
    assert 'self.adapters[0]["threshold"]' not in src, \
        "adapter 0's threshold must not decide every adapter's routing"
    assert "self._route(gates)" in src

    # and _route must actually read each adapter's own value
    src_route = inspect.getsource(Vega._route)
    assert 'a["threshold"] for a in self.adapters' in src_route


def test_the_image_path_routes_with_per_adapter_thresholds():
    """The same one-threshold bug, in the same form, on the image path."""
    import inspect

    from vegaml._vision import VegaVision

    src = inspect.getsource(VegaVision._decide)
    assert 'self.d.adapters[0]["threshold"]' not in src, \
        "adapter 0's threshold must not decide every adapter's routing"


def test_routing_end_to_end_through_the_engine():
    """The whole path: gates in, the calibration index out.

    Two adapters with thresholds 0.5 and 0.99. A gate of 0.95 on adapter 1 clears adapter 0's
    threshold but not its own, so it must be routed to adapter 0.
    """
    from vegaml._api import Vega

    class _Engine:
        """Just enough of _api.Vega to call the real _route."""
        adapters = [{"threshold": 0.5, "name": "a"}, {"threshold": 0.99, "name": "b"}]

    route = Vega._route(_Engine(), np.array([[0.90, 0.95], [0.10, 0.20], [0.60, 0.995]]))
    assert list(route) == [0, -1, 1], \
        "row 1: adapter 1's 0.95 is under its own 0.99, so adapter 0 wins. row 2: nothing qualifies. \
row 3: adapter 1 clears 0.99, so it wins."


def test_routing_with_a_single_adapter_matches_the_old_behaviour():
    """One adapter is the common case; nothing about it should change."""
    from vegaml._api import Vega

    class _Engine:
        adapters = [{"threshold": 0.7, "name": "only"}]

    gates = np.array([[0.8], [0.6], [0.7]])
    assert list(Vega._route(_Engine(), gates)) == list(route_adapters(gates, 0.7)) == [0, -1, 0]


def test_no_gates_still_routes_to_the_base():
    from vegaml._api import Vega

    class _Engine:
        adapters = [{"threshold": 0.5, "name": "a"}]

    assert Vega._route(_Engine(), None) is None


@pytest.mark.parametrize("n_adapters", [1, 2, 3, 5])
def test_every_gate_column_is_reachable(n_adapters):
    """Each adapter must be able to win for itself, or its threshold is doing nothing."""
    thresholds = [0.5] * n_adapters
    for j in range(n_adapters):
        gates = np.zeros((1, n_adapters))
        gates[0, j] = 0.9
        assert list(route_per_adapter(gates, thresholds)) == [j]
