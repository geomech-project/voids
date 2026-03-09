from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import sparse

from voids.core.network import Network
from voids.geom.hydraulic import (
    _get_entity_area,
    _segment_conductance_from_agl,
    generic_poiseuille_conductance,
    throat_conductance,
    valvatne_blunt_conductance,
    valvatne_blunt_baseline_conductance,
)
from voids.graph.connectivity import spanning_component_ids
from voids.io.hdf5 import _read_json_attr, load_hdf5, save_hdf5
from voids.linalg.assemble import assemble_pressure_system
from voids.linalg.bc import apply_dirichlet_rowcol
from voids.physics.petrophysics import absolute_porosity, connectivity_metrics, effective_porosity
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    _make_dirichlet_vector,
    solve,
)


def test_hydraulic_geometry_fallbacks_and_errors(line_network: Network) -> None:
    """Test geometric fallbacks and missing-data errors in hydraulic helpers."""

    net = line_network.copy()
    net.throat.clear()
    net.pore.clear()

    net.throat["radius_inscribed"] = np.array([0.5, 0.25])
    assert np.allclose(_get_entity_area(net, "throat"), [np.pi * 0.5**2, np.pi * 0.25**2])

    del net.throat["radius_inscribed"]
    with pytest.raises(KeyError, match="Need throat.area or throat.diameter_inscribed"):
        _get_entity_area(net, "throat")


@pytest.mark.parametrize(
    ("area", "shape_factor", "length", "viscosity", "message"),
    [
        (np.array([1.0]), np.array([0.1]), np.array([1.0]), 0.0, "viscosity must be positive"),
        (np.array([-1.0]), np.array([0.1]), np.array([1.0]), 1.0, "area contains negative values"),
        (
            np.array([1.0]),
            np.array([0.1]),
            np.array([-1.0]),
            1.0,
            "length contains negative values",
        ),
        (
            np.array([1.0]),
            np.array([-0.1]),
            np.array([1.0]),
            1.0,
            "shape_factor contains negative values",
        ),
    ],
)
def test_segment_conductance_input_validation(
    area: np.ndarray, shape_factor: np.ndarray, length: np.ndarray, viscosity: float, message: str
) -> None:
    """Test input validation for area-shape-length conductance calculation."""

    with pytest.raises(ValueError, match=message):
        _segment_conductance_from_agl(area, shape_factor, length, viscosity)


def test_generic_poiseuille_validation_and_area_fallback(line_network: Network) -> None:
    """Test validation and area-based fallback paths in Poiseuille conductance."""

    with pytest.raises(ValueError, match="viscosity must be positive"):
        generic_poiseuille_conductance(line_network, viscosity=0.0)

    negative = line_network.copy()
    negative.throat["hydraulic_conductance"][0] = -1.0
    with pytest.raises(ValueError, match="contains negative values"):
        generic_poiseuille_conductance(negative, viscosity=1.0)

    missing = line_network.copy()
    missing.throat.pop("hydraulic_conductance")
    missing.throat.pop("length")
    with pytest.raises(KeyError, match="Missing required throat fields"):
        generic_poiseuille_conductance(missing, viscosity=1.0)

    area_based = line_network.copy()
    area_based.throat.pop("hydraulic_conductance")
    area_based.throat.pop("diameter_inscribed", None)
    area_based.throat["area"] = np.pi * np.array([0.5, 0.25]) ** 2
    g = generic_poiseuille_conductance(area_based, viscosity=1.0)
    expected = np.pi * np.array([0.5, 0.25]) ** 4 / (8.0 * area_based.throat["length"])
    assert np.allclose(g, expected)

    no_geom = line_network.copy()
    no_geom.throat.pop("hydraulic_conductance")
    no_geom.throat.pop("area", None)
    no_geom.throat.pop("diameter_inscribed", None)
    with pytest.raises(KeyError, match="Need throat.diameter_inscribed or throat.area"):
        generic_poiseuille_conductance(no_geom, viscosity=1.0)


def test_valvatne_shape_factor_branches_and_model_selection(line_network: Network) -> None:
    """Test shape-factor conductance branches and model-dispatch behavior."""

    with pytest.raises(ValueError, match="viscosity must be positive"):
        valvatne_blunt_baseline_conductance(line_network, viscosity=0.0)

    trusted = valvatne_blunt_baseline_conductance(line_network, viscosity=1.0)
    assert np.allclose(trusted, line_network.throat["hydraulic_conductance"])

    throat_only = line_network.copy()
    throat_only.throat.pop("hydraulic_conductance")
    throat_only.throat["radius_inscribed"] = np.array([0.5, 0.25])
    throat_only.throat["perimeter"] = 2.0 * np.pi * throat_only.throat["radius_inscribed"]
    g = valvatne_blunt_baseline_conductance(throat_only, viscosity=1.0)
    # Circle gives g = pi r^4 / (8 mu L)
    expected = (
        np.pi * throat_only.throat["radius_inscribed"] ** 4 / (8.0 * throat_only.throat["length"])
    )
    assert np.allclose(g, expected)

    via_wrapper = throat_conductance(throat_only, viscosity=1.0, model="valvatne_blunt_baseline")
    assert np.allclose(via_wrapper, expected)
    via_new_name = throat_conductance(throat_only, viscosity=1.0, model="valvatne_blunt")
    assert np.allclose(via_new_name, expected)
    via_throat_only = throat_conductance(throat_only, viscosity=1.0, model="valvatne_blunt_throat")
    assert np.allclose(via_throat_only, expected)
    direct_new = valvatne_blunt_conductance(throat_only, viscosity=1.0)
    assert np.allclose(direct_new, expected)

    with pytest.raises(ValueError, match="Unknown conductance model"):
        throat_conductance(throat_only, viscosity=1.0, model="unsupported")


def test_spanning_component_ids_validates_axis_and_autocomputes_labels(
    branched_network: Network,
) -> None:
    """Test spanning-component identification and axis validation."""

    ids = spanning_component_ids(branched_network, axis="x")
    assert np.array_equal(ids, np.array([0]))

    with pytest.raises(ValueError, match="Unsupported axis"):
        spanning_component_ids(branched_network, axis="q")


def test_hdf5_helpers_cover_defaults_bytes_and_throat_labels(
    tmp_path: Path, line_network: Network
) -> None:
    """Test HDF5 helper handling of defaults, raw bytes, and throat labels."""

    with h5py.File(tmp_path / "attrs.h5", "w") as handle:
        grp = handle.create_group("meta")
        assert _read_json_attr(grp, "missing", {"fallback": True}) == {"fallback": True}
        grp.attrs["payload"] = json.dumps({"value": 3}).encode("utf-8")
        assert _read_json_attr(grp, "payload") == {"value": 3}

    net = line_network.copy()
    net.throat_labels["boundary_throat"] = np.array([True, False])
    path = tmp_path / "with-throat-labels.h5"
    save_hdf5(net, path)
    roundtrip = load_hdf5(path)

    assert np.array_equal(roundtrip.throat_labels["boundary_throat"], np.array([True, False]))


def test_hdf5_read_json_attr_decodes_raw_bytes() -> None:
    """Test raw-byte JSON attribute decoding without an actual HDF5 file."""

    class FakeGroup:
        """Minimal fake HDF5 group exposing an ``attrs`` mapping."""

        def __init__(self):
            """Initialize a single raw-byte JSON attribute."""

            self.attrs = {"payload": json.dumps({"value": 7}).encode("utf-8")}

    assert _read_json_attr(FakeGroup(), "payload") == {"value": 7}


def test_assemble_pressure_system_input_validation(line_network: Network) -> None:
    """Test input validation in pressure-system assembly."""

    with pytest.raises(ValueError, match="shape \\(Nt,\\)"):
        assemble_pressure_system(line_network, np.ones(line_network.Nt + 1))
    with pytest.raises(ValueError, match="must be nonnegative"):
        assemble_pressure_system(line_network, np.array([1.0, -1.0]))


def test_apply_dirichlet_rowcol_shape_validation_and_noop() -> None:
    """Test Dirichlet elimination shape checks and the no-op path."""

    A = sparse.csr_matrix(np.eye(2))
    b = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same shape"):
        apply_dirichlet_rowcol(A, b, values=np.array([1.0]), mask=np.array([True, False]))

    A2, b2 = apply_dirichlet_rowcol(
        A, b, values=np.array([0.0, 0.0]), mask=np.array([False, False])
    )
    assert np.array_equal(A2.toarray(), np.eye(2))
    assert np.array_equal(b2, b)


def test_petrophysics_error_and_wrapper_paths(
    branched_network: Network, line_network: Network
) -> None:
    """Test petrophysics error paths and connectivity-summary wrapper behavior."""

    missing = line_network.copy()
    missing.throat.pop("volume")
    with pytest.raises(KeyError, match="pore.region_volume, or both pore.volume and throat.volume"):
        absolute_porosity(missing)

    with pytest.raises(ValueError, match="Unsupported effective porosity mode"):
        effective_porosity(line_network, mode="invalid")

    summary = connectivity_metrics(branched_network)
    assert summary.n_components == 2


def test_singlephase_dirichlet_vector_and_solver_branches(line_network: Network) -> None:
    """Test boundary-condition validation and solver edge branches."""

    with pytest.raises(KeyError, match="Missing pore label 'missing_inlet'"):
        _make_dirichlet_vector(line_network, PressureBC("missing_inlet", "outlet_xmax", 1.0, 0.0))
    with pytest.raises(KeyError, match="Missing pore label 'missing_outlet'"):
        _make_dirichlet_vector(line_network, PressureBC("inlet_xmin", "missing_outlet", 1.0, 0.0))

    empty = line_network.copy()
    empty.pore_labels["inlet_xmin"] = np.zeros(empty.Np, dtype=bool)
    with pytest.raises(ValueError, match="at least one pore each"):
        _make_dirichlet_vector(empty, PressureBC("inlet_xmin", "outlet_xmax", 1.0, 0.0))

    with pytest.raises(ValueError, match="Fluid viscosity must be positive"):
        solve(
            line_network,
            fluid=FluidSinglePhase(viscosity=0.0),
            bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
            axis="x",
        )

    regularized = line_network.copy()
    regularized.throat.pop("hydraulic_conductance")
    regularized.throat["diameter_inscribed"] = np.array([1.0, 1.0])
    result = solve(
        regularized,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
        axis="x",
        options=SinglePhaseOptions(regularization=1e-9, check_mass_balance=False),
    )
    assert np.isnan(result.mass_balance_error)
    assert result.permeability is not None

    with pytest.raises(ValueError, match="Pressure drop pin-pout must be nonzero"):
        solve(
            line_network,
            fluid=FluidSinglePhase(viscosity=1.0),
            bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=1.0),
            axis="x",
        )


def test_singlephase_ignores_floating_components(branched_network: Network) -> None:
    """Solve remains well-posed when disconnected pores lack boundary conditions."""

    result = solve(
        branched_network,
        fluid=FluidSinglePhase(viscosity=1.0),
        bc=PressureBC("inlet_xmin", "outlet_xmax", pin=1.0, pout=0.0),
        axis="x",
    )

    assert np.isfinite(result.total_flow_rate)
    assert np.isfinite(result.permeability["x"])
    assert np.isnan(result.pore_pressure[4])
    assert np.isnan(result.throat_flux).sum() == 0
    assert np.isclose(result.throat_flux[2], 0.0)
