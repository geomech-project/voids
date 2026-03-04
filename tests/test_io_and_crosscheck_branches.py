from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from voids.benchmarks.crosscheck import (
    _get_openpnm_pressure,
    _openpnm_phase_factory,
    crosscheck_singlephase_with_openpnm,
)
from voids.examples.mesh import make_cartesian_mesh_network
from voids.io.openpnm import to_openpnm_dict, to_openpnm_network
from voids.io.porespy import ensure_cartesian_boundary_labels, from_porespy, scale_porespy_geometry
from voids.physics.singlephase import FluidSinglePhase, PressureBC


def test_scale_porespy_geometry_scales_and_derives_common_fields() -> None:
    """Test common geometric scaling rules and derived throat volume."""

    raw = {
        "pore.coords": np.array([[0.0, 2.0], [4.0, 6.0]]),
        "pore.cross_sectional_area": np.array([2.0, 3.0]),
        "pore.perimeter": np.array([4.0, 5.0]),
        "pore.region_volume": np.array([6.0, 7.0]),
        "throat.cross_sectional_area": np.array([8.0]),
        "throat.total_length": np.array([9.0]),
        "nonnumeric": np.array(["skip"], dtype=object),
    }

    scaled = scale_porespy_geometry(raw, voxel_size=2.0)

    assert np.allclose(scaled["pore.coords"], [[0.0, 4.0], [8.0, 12.0]])
    assert np.allclose(scaled["pore.cross_sectional_area"], [8.0, 12.0])
    assert np.allclose(scaled["pore.perimeter"], [8.0, 10.0])
    assert np.allclose(scaled["pore.volume"], [48.0, 56.0])
    assert np.allclose(scaled["throat.volume"], [576.0])
    assert scaled["nonnumeric"].dtype == object


def test_scale_porespy_geometry_requires_positive_voxel_size() -> None:
    """Test rejection of nonpositive voxel size during PoreSpy scaling."""

    with pytest.raises(ValueError, match="voxel_size must be positive"):
        scale_porespy_geometry({}, voxel_size=0.0)


def test_ensure_cartesian_boundary_labels_validates_inputs_and_preserves_existing_labels() -> None:
    """Test input validation and label preservation in Cartesian boundary inference."""

    with pytest.raises(ValueError, match="pore.coords must have shape"):
        ensure_cartesian_boundary_labels({"pore.coords": np.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="tol_fraction must be nonnegative"):
        ensure_cartesian_boundary_labels({"pore.coords": np.zeros((2, 2))}, tol_fraction=-1.0)
    with pytest.raises(ValueError, match="axes entries must be drawn"):
        ensure_cartesian_boundary_labels({"pore.coords": np.zeros((2, 2))}, axes=("q",))
    with pytest.raises(ValueError, match="axis 'z' is not available"):
        ensure_cartesian_boundary_labels({"pore.coords": np.zeros((2, 2))}, axes=("z",))

    existing = {
        "pore.coords": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        "pore.inlet_xmin": np.array([False, True, False]),
    }
    updated = ensure_cartesian_boundary_labels(existing)

    assert updated["pore.inlet_xmin"].tolist() == [False, True, False]
    assert updated["pore.outlet_xmax"].tolist() == [False, False, True]
    assert updated["pore.boundary"].tolist() == [True, True, True]


def test_from_porespy_handles_strict_and_non_strict_paths() -> None:
    """Test strict and non-strict missing-topology paths in the PoreSpy importer."""

    with pytest.raises(KeyError, match="must include 'throat.conns' and 'pore.coords'"):
        from_porespy({}, strict=True)

    partial = {
        "pore.volume": np.array([1.0, 2.0]),
        "pore.left": np.array([True, False]),
        "pore.right": np.array([False, True]),
    }
    with pytest.raises(KeyError, match="Required keys 'throat.conns' and/or 'pore.coords' missing"):
        from_porespy(partial, strict=False)


def test_from_porespy_derives_geometry_aliases_and_warns_for_unsupported_size_factors() -> None:
    """Test alias handling, geometry derivation, and unsupported-factor warnings."""

    net_dict = {
        "throat.conns": np.array([[0, 1]], dtype=int),
        "pore.coords": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        "pore.left": np.array([True, False]),
        "pore.right": np.array([False, True]),
        "pore.radius": np.array([0.5, 0.25]),
        "pore.perimeter": np.array([4.0, 2.0]),
        "throat.radius": np.array([0.2]),
        "throat.perimeter": np.array([1.0]),
        "throat.conduit_lengths.pore1": np.array([0.1]),
        "throat.conduit_lengths.throat": np.array([0.8]),
        "throat.conduit_lengths.pore2": np.array([0.1]),
        "throat.hydraulic_size_factors": np.array([[1.0, 2.0, 3.0]]),
        "meta": {"source": "test"},
    }

    with pytest.warns(RuntimeWarning, match="Stored throat.hydraulic_size_factors"):
        net = from_porespy(net_dict, strict=True)

    assert net.pore_coords.shape == (2, 3)
    assert np.allclose(net.pore["diameter_inscribed"], [1.0, 0.5])
    assert np.allclose(net.pore["area"], [np.pi * 0.25, np.pi * 0.25**2])
    assert np.allclose(net.throat["area"], [np.pi * 0.2**2])
    assert np.allclose(net.throat["length"], [1.0])
    assert net.pore_labels["inlet_xmin"].tolist() == [True, False]
    assert net.pore_labels["outlet_xmax"].tolist() == [False, True]
    assert "throat.hydraulic_size_factors" in net.extra
    assert net.extra["meta"] == {"source": "test"}


def test_from_porespy_handles_dotted_passthrough_fields_and_diameter_based_area() -> None:
    """Test dotted key normalization and diameter-based area derivation."""

    net = from_porespy(
        {
            "throat.conns": np.array([[0, 1]], dtype=int),
            "pore.coords": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            "pore.volume": np.array([1.0, 1.0]),
            "throat.volume": np.array([0.2]),
            "throat.length": np.array([1.0]),
            "pore.diameter": np.array([2.0, 4.0]),
            "throat.boundary": np.array([True]),
            "throat.foo.bar": np.array([7.0]),
            "pore.foo.bar": np.array([5.0, 6.0]),
        },
        strict=True,
    )

    assert np.allclose(net.pore["area"], [np.pi, 4.0 * np.pi])
    assert np.array_equal(net.throat_labels["boundary"], np.array([True]))
    assert np.array_equal(net.pore["foo_bar"], np.array([5.0, 6.0]))
    assert np.array_equal(net.throat["foo_bar"], np.array([7.0]))


class _DictNetwork(dict):
    """Dictionary subclass used to mimic OpenPNM network objects in tests."""

    pass


def test_to_openpnm_dict_includes_aliases_and_extra() -> None:
    """Test OpenPNM-style dict export, including aliases and extra metadata."""

    net = make_cartesian_mesh_network((3, 3))
    net.extra["pore.extra_field"] = np.arange(net.Np, dtype=float)
    net.throat_labels["boundary_throat"] = np.zeros(net.Nt, dtype=bool)
    net.throat_labels["boundary_throat"][0] = True

    exported = to_openpnm_dict(net, include_extra=True)

    assert np.array_equal(exported["throat.conduit_lengths.pore1"], net.throat["pore1_length"])
    assert np.array_equal(exported["throat.conduit_lengths.throat"], net.throat["core_length"])
    assert np.array_equal(exported["throat.conduit_lengths.pore2"], net.throat["pore2_length"])
    assert np.array_equal(exported["pore.coords"], net.pore_coords)
    assert np.array_equal(exported["throat.boundary_throat"], net.throat_labels["boundary_throat"])
    assert np.array_equal(exported["pore.extra_field"], np.arange(net.Np, dtype=float))


def test_to_openpnm_network_handles_constructor_fallback_and_tolerates_bad_extra(
    monkeypatch, line_network
) -> None:
    """Test OpenPNM network construction fallback and tolerant extra-field copying."""

    class BadExtra:
        """Object that intentionally fails NumPy array conversion."""

        def __array__(self, *_args, **_kwargs):
            """Raise a conversion error to exercise tolerant extra handling."""

            raise TypeError("cannot convert")

    class FakeNetwork(_DictNetwork):
        """Minimal fake OpenPNM network that rejects keyword construction."""

        def __init__(self, *args, **kwargs):
            """Initialize the fake network and reject keyword arguments."""

            if kwargs:
                raise TypeError("kwargs unsupported")
            super().__init__()

    fake_openpnm = types.SimpleNamespace(network=types.SimpleNamespace(Network=FakeNetwork))
    monkeypatch.setitem(sys.modules, "openpnm", fake_openpnm)

    line_network.extra["pore.good"] = np.array([10.0, 11.0, 12.0])
    line_network.extra["throat.bad"] = BadExtra()
    pn = to_openpnm_network(
        line_network,
        copy_properties=False,
        copy_labels=False,
        include_extra=True,
    )

    assert np.array_equal(pn["pore.coords"], line_network.pore_coords)
    assert np.array_equal(pn["throat.conns"], line_network.throat_conns)
    assert "pore.volume" not in pn
    assert "pore.inlet_xmin" not in pn
    assert np.array_equal(pn["pore.good"], np.array([10.0, 11.0, 12.0]))
    assert "throat.bad" not in pn


def test_to_openpnm_network_copies_properties_and_labels(monkeypatch, line_network) -> None:
    """Test copying of pore/throat properties and labels into an OpenPNM object."""

    class FakeNetwork(_DictNetwork):
        """Minimal fake OpenPNM network accepting arbitrary construction."""

        def __init__(self, *args, **kwargs):
            """Initialize the fake network."""

            super().__init__()

    line_network.throat_labels["boundary_throat"] = np.array([True, False])
    fake_openpnm = types.SimpleNamespace(network=types.SimpleNamespace(Network=FakeNetwork))
    monkeypatch.setitem(sys.modules, "openpnm", fake_openpnm)

    pn = to_openpnm_network(line_network, copy_properties=True, copy_labels=True)

    assert np.array_equal(pn["pore.volume"], line_network.pore["volume"])
    assert np.array_equal(pn["throat.length"], line_network.throat["length"])
    assert np.array_equal(pn["pore.inlet_xmin"], line_network.pore_labels["inlet_xmin"])
    assert np.array_equal(pn["throat.boundary_throat"], np.array([True, False]))


def test_openpnm_phase_factory_uses_fallback_and_errors_cleanly() -> None:
    """Test phase-constructor fallback logic for multiple OpenPNM APIs."""

    fake_op = types.SimpleNamespace(
        phase=types.SimpleNamespace(
            Phase=lambda network: (_ for _ in ()).throw(RuntimeError("nope"))
        ),
        phases=types.SimpleNamespace(GenericPhase=lambda network: {"network": network}),
    )
    assert _openpnm_phase_factory(fake_op, {"pn": 1}) == {"network": {"pn": 1}}

    failing_op = types.SimpleNamespace(
        phase=types.SimpleNamespace(
            Phase=lambda network: (_ for _ in ()).throw(RuntimeError("nope"))
        ),
        phases=types.SimpleNamespace(
            GenericPhase=lambda network: (_ for _ in ()).throw(RuntimeError("still nope"))
        ),
    )
    with pytest.raises(RuntimeError, match="Unable to construct OpenPNM phase object"):
        _openpnm_phase_factory(failing_op, {})


def test_get_openpnm_pressure_supports_both_access_patterns_and_errors_cleanly() -> None:
    """Test pressure extraction from multiple OpenPNM result access patterns."""

    class MappingOnly(dict):
        """Fake result object exposing mapping-style pressure access only."""

        soln = {}

    assert np.array_equal(
        _get_openpnm_pressure(MappingOnly({"pore.pressure": [1.0, 0.0]})), [1.0, 0.0]
    )

    class SolnOnly(dict):
        """Fake result object exposing solution-container pressure access only."""

        soln = {"pore.pressure": [2.0, 1.0]}

        def __getitem__(self, key):
            """Reject direct item access to force the fallback path."""

            raise KeyError(key)

    assert np.array_equal(_get_openpnm_pressure(SolnOnly()), [2.0, 1.0])

    class BadResult(dict):
        """Fake result object exposing malformed pressure data."""

        soln = {"pore.pressure": [[1.0, 0.0]]}

        def __getitem__(self, key):
            """Reject direct item access to force malformed fallback data."""

            raise KeyError(key)

    with pytest.raises(RuntimeError, match="Unable to extract pore pressures"):
        _get_openpnm_pressure(BadResult())


def test_crosscheck_singlephase_with_openpnm_supports_set_bc_compatibility(
    monkeypatch, line_network
) -> None:
    """Test OpenPNM crosscheck compatibility with the legacy ``set_BC`` API."""

    class FakePhase(dict):
        """Minimal fake OpenPNM phase container."""

        def __init__(self, network):
            """Store the owning network for later inspection."""

            super().__init__()
            self.network = network

    class FakeStokesFlow:
        """Minimal fake OpenPNM StokesFlow algorithm using the ``set_BC`` API."""

        def __init__(self, network, phase):
            """Initialize fake state, synthetic pressures, and BC call storage."""

            self.network = network
            self.phase = phase
            self.soln = {"pore.pressure": np.array([1.0, 0.5, 0.0])}
            self.bc_calls = []

        def set_BC(self, *, pores, bctype, bcvalues):
            """Record legacy boundary-condition calls."""

            self.bc_calls.append((tuple(pores.tolist()), bctype, float(bcvalues)))

        def run(self):
            """Pretend to run the OpenPNM solver."""

            return None

        def rate(self, *, pores):
            """Return a synthetic inlet flow rate."""

            assert tuple(pores.tolist()) == (0,)
            return np.array([0.5])

    fake_op = types.SimpleNamespace(
        __version__="fake-openpnm",
        phase=types.SimpleNamespace(Phase=FakePhase),
        algorithms=types.SimpleNamespace(StokesFlow=FakeStokesFlow),
    )
    monkeypatch.setitem(sys.modules, "openpnm", fake_op)
    monkeypatch.setattr(
        "voids.benchmarks.crosscheck.to_openpnm_network", lambda *args, **kwargs: {}
    )

    summary = crosscheck_singlephase_with_openpnm(
        line_network,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
        axis="x",
    )

    assert summary.reference == "openpnm_stokesflow"
    assert summary.permeability_abs_diff == pytest.approx(0.0)
    assert summary.total_flow_abs_diff == pytest.approx(0.0)
    assert summary.details["openpnm_version"] == "fake-openpnm"
    assert summary.details["n_inlet_pores"] == 1
    assert summary.details["p_ref_min"] == pytest.approx(0.0)


def test_crosscheck_singlephase_with_openpnm_requires_nonzero_pressure_drop(
    monkeypatch, line_network
) -> None:
    """Test zero-pressure-drop rejection in the OpenPNM crosscheck adapter."""

    class FakePhase(dict):
        """Minimal fake OpenPNM phase container."""

        def __init__(self, network):
            """Initialize the fake phase."""

            super().__init__()

    class FakeStokesFlow:
        """Minimal fake OpenPNM StokesFlow algorithm using the ``set_value_BC`` API."""

        def __init__(self, network, phase):
            """Initialize fake pressure results."""

            self.soln = {"pore.pressure": np.array([1.0, 1.0, 1.0])}

        def set_value_BC(self, *, pores, values):
            """Accept fixed-value boundary conditions without further action."""

            return None

        def run(self):
            """Pretend to run the OpenPNM solver."""

            return None

        def rate(self, *, pores):
            """Return a synthetic zero flow rate."""

            return np.array([0.0])

    fake_op = types.SimpleNamespace(
        phase=types.SimpleNamespace(Phase=FakePhase),
        algorithms=types.SimpleNamespace(StokesFlow=FakeStokesFlow),
    )
    monkeypatch.setitem(sys.modules, "openpnm", fake_op)
    monkeypatch.setattr(
        "voids.benchmarks.crosscheck.to_openpnm_network", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        "voids.benchmarks.crosscheck.solve",
        lambda *args, **kwargs: types.SimpleNamespace(
            throat_conductance=np.array([1.0, 1.0]),
            total_flow_rate=0.0,
            permeability={"x": 0.0},
        ),
    )

    with pytest.raises(ValueError, match="Pressure drop pin-pout must be nonzero"):
        crosscheck_singlephase_with_openpnm(
            line_network,
            fluid=FluidSinglePhase(viscosity=1.0),
            bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=1.0),
            axis="x",
        )
