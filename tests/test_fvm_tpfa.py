from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning

import voids.fvm.singlephase.tpfa as tpfa_mod
from voids.fvm.singlephase import (
    solve_tpfa,
    upscale_permeability_tpfa,
    upscale_principal_permeabilities_tpfa,
)
from voids.fvm.singlephase.upscaling import _default_axes
from voids.image.porosity import PermeabilityMap


def test_tpfa_constant_2d_map_recovers_input_permeability() -> None:
    permeability = PermeabilityMap(
        np.full((4, 3), 2.5),
        cell_size=(0.5, 0.25),
        metadata={"case": "constant"},
    )

    result = solve_tpfa(permeability, flow_axis="x", viscosity=3.0)

    assert result.permeability == pytest.approx(2.5)
    assert result.flow_rate > 0.0
    assert result.mass_balance_error < 1.0e-12
    assert result.cell_size == (0.5, 0.25)
    assert result.metadata["case"] == "constant"
    assert result.matrix_nnz > 0
    assert result.solve_seconds >= 0.0
    assert result.solver_info["method"] == "direct"
    assert result.residual_relative < 1.0e-10


def test_tpfa_constant_3d_array_accepts_sequence_cell_size() -> None:
    result = solve_tpfa(
        np.full((3, 2, 4), 1.7),
        flow_axis="z",
        viscosity=2.0,
        cell_size=[0.2, 0.3, 0.4],
    )

    assert result.permeability == pytest.approx(1.7)
    assert result.mass_balance_error < 1.0e-12


def test_tpfa_vectorized_assembly_matches_reference_stencil() -> None:
    values = np.array(
        [
            [[1.0, 2.0], [0.5, 0.0]],
            [[4.0, 3.0], [2.5, 1.5]],
        ],
        dtype=float,
    )
    cell_size = (0.2, 0.3, 0.4)
    flow_axis = 2
    viscosity = 1.7
    pressure_inlet = 5.0
    pressure_outlet = 1.0
    shape = values.shape
    n_cells = int(values.size)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n_cells, dtype=float)
    diagonal = np.zeros(n_cells, dtype=float)

    def flat(index: tuple[int, ...]) -> int:
        return int(np.ravel_multi_index(index, shape, order="C"))

    for index in np.ndindex(shape):
        row = flat(index)
        k_cell = float(values[index])
        if index[flow_axis] == 0 and k_cell > 0.0:
            trans = (
                k_cell
                * tpfa_mod._face_area(cell_size, flow_axis)
                / (viscosity * (cell_size[flow_axis] / 2.0))
            )
            diagonal[row] += trans
            rhs[row] += trans * pressure_inlet
        if index[flow_axis] == shape[flow_axis] - 1 and k_cell > 0.0:
            trans = (
                k_cell
                * tpfa_mod._face_area(cell_size, flow_axis)
                / (viscosity * (cell_size[flow_axis] / 2.0))
            )
            diagonal[row] += trans
            rhs[row] += trans * pressure_outlet

        for direction in range(values.ndim):
            neighbor_index = list(index)
            neighbor_index[direction] += 1
            if neighbor_index[direction] >= shape[direction]:
                continue
            neighbor = tuple(neighbor_index)
            neighbor_row = flat(neighbor)
            k_face = tpfa_mod._harmonic_face_permeability(k_cell, float(values[neighbor]))
            if k_face <= 0.0:
                continue
            trans = (
                k_face
                * tpfa_mod._face_area(cell_size, direction)
                / (viscosity * cell_size[direction])
            )
            diagonal[row] += trans
            diagonal[neighbor_row] += trans
            rows.extend((row, neighbor_row))
            cols.extend((neighbor_row, row))
            data.extend((-trans, -trans))

    rows.extend(range(n_cells))
    cols.extend(range(n_cells))
    data.extend(diagonal.tolist())
    expected_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    matrix, assembled_rhs = tpfa_mod._assemble_tpfa_system(
        values,
        cell_size=cell_size,
        flow_axis_index=flow_axis,
        viscosity=viscosity,
        pressure_inlet=pressure_inlet,
        pressure_outlet=pressure_outlet,
    )

    assert np.allclose(matrix.toarray(), expected_matrix.toarray())
    assert np.allclose(assembled_rhs, rhs)


def test_tpfa_vectorized_assembly_handles_zero_transmissibility_directions() -> None:
    matrix, rhs = tpfa_mod._assemble_tpfa_system(
        np.zeros((2, 2, 2), dtype=float),
        cell_size=(1.0, 1.0, 1.0),
        flow_axis_index=0,
        viscosity=1.0,
        pressure_inlet=1.0,
        pressure_outlet=0.0,
    )

    assert np.allclose(matrix.toarray(), 0.0)
    assert np.allclose(rhs, 0.0)


def test_tpfa_array_accepts_scalar_cell_size() -> None:
    result = solve_tpfa(np.full((2, 2), 2.0), cell_size=0.5)

    assert result.cell_size == (0.5, 0.5)
    assert result.permeability == pytest.approx(2.0)


def test_tpfa_upscaling_solves_requested_axes() -> None:
    permeability = PermeabilityMap(np.full((3, 4, 2), 4.2), cell_size=1.0)

    result = upscale_permeability_tpfa(permeability, axes=("x", "y"))

    assert set(result.results) == {"x", "y"}
    assert result.permeability == {"x": pytest.approx(4.2), "y": pytest.approx(4.2)}
    assert set(result.mass_balance_error) == {"x", "y"}
    assert set(result.solve_seconds) == {"x", "y"}
    assert upscale_principal_permeabilities_tpfa(permeability, axes=("z",)) == {
        "z": pytest.approx(4.2)
    }


def test_tpfa_upscaling_defaults_to_all_supported_axes() -> None:
    permeability = PermeabilityMap(np.full((2, 3), 1.4), cell_size=1.0)

    result = upscale_permeability_tpfa(permeability)

    assert set(result.results) == {"x", "y"}
    assert _default_axes(3) == ("x", "y", "z")
    with pytest.raises(ValueError, match="permeability maps must be 2D or 3D"):
        _default_axes(1)


def test_tpfa_accepts_iterative_solver_controls() -> None:
    result = solve_tpfa(
        np.full((4, 4), 3.1),
        flow_axis="y",
        solver_method="cg",
        solver_parameters={"rtol": 1.0e-12, "atol": 0.0, "maxiter": 200},
    )

    assert result.permeability == pytest.approx(3.1)
    assert result.solver_method == "cg"
    assert result.solver_info["info"] == 0
    assert result.residual_relative < 1.0e-10


def test_tpfa_umfpack_solver_matches_direct() -> None:
    """TPFA can reuse the shared UMFPACK direct sparse backend."""

    permeability = np.full((4, 4), 3.1)
    direct = solve_tpfa(permeability, flow_axis="y", solver_method="direct")

    try:
        umfpack = solve_tpfa(permeability, flow_axis="y", solver_method="umfpack")
    except ImportError as exc:
        assert "umfpack" in str(exc).lower()
        pytest.skip("UMFPACK solver not available in this environment")

    assert np.allclose(umfpack.pressure, direct.pressure, rtol=1.0e-12, atol=1.0e-14)
    assert umfpack.permeability == pytest.approx(direct.permeability, rel=1.0e-12)
    assert umfpack.flow_rate == pytest.approx(direct.flow_rate, rel=1.0e-12)
    assert umfpack.solver_method == "umfpack"
    assert umfpack.solver_info["backend"] == "scikits.umfpack.spsolve"


def test_tpfa_rejects_nonpositive_pressure_drop() -> None:
    with pytest.raises(ValueError, match="pressure_inlet must be greater"):
        solve_tpfa(np.ones((2, 2)), pressure_inlet=0.0, pressure_outlet=1.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"permeability": np.ones((2,)), "cell_size": None}, "2D or 3D"),
        ({"permeability": np.ones((2, 2)), "cell_size": (1.0,)}, "cell_size dimensionality"),
        ({"permeability": np.ones((2, 2)), "cell_size": (1.0, 0.0)}, "positive and finite"),
        (
            {"permeability": np.array([[1.0, np.nan], [1.0, 1.0]]), "cell_size": None},
            "only finite",
        ),
        (
            {"permeability": np.array([[1.0, -1.0], [1.0, 1.0]]), "cell_size": None},
            "non-negative",
        ),
    ],
)
def test_tpfa_rejects_invalid_map_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_tpfa(**kwargs)


def test_tpfa_defensively_revalidates_permeability_map_dimensionality() -> None:
    malformed = object.__new__(PermeabilityMap)
    malformed.values = np.ones((2,))
    malformed.cell_size = (1.0,)
    malformed.metadata = {}

    with pytest.raises(ValueError, match="permeability must be a 2D or 3D field"):
        solve_tpfa(malformed)


def test_tpfa_rejects_invalid_axis_and_physical_controls() -> None:
    with pytest.raises(ValueError, match="flow_axis must be one of"):
        solve_tpfa(np.ones((2, 2)), flow_axis="z")
    with pytest.raises(ValueError, match="viscosity must be positive"):
        solve_tpfa(np.ones((2, 2)), viscosity=0.0)
    with pytest.raises(ValueError, match="pressure values must be finite"):
        solve_tpfa(np.ones((2, 2)), pressure_inlet=float("nan"))


def test_tpfa_zero_permeability_faces_make_the_current_system_singular() -> None:
    with pytest.raises(RuntimeError, match="pressure system is singular"):
        solve_tpfa(np.array([[1.0, 0.0], [1.0, 1.0]]), flow_axis="x")


def test_tpfa_reports_singular_matrix_as_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_solve_linear_system(
        matrix: sparse.spmatrix,
        rhs: np.ndarray,
        *,
        method: str = "direct",
        solver_parameters: object = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        raise MatrixRankWarning("singular")

    monkeypatch.setattr(tpfa_mod, "solve_linear_system", fake_solve_linear_system)

    with pytest.raises(RuntimeError, match="pressure system is singular"):
        solve_tpfa(np.ones((2, 2)))


def test_tpfa_reports_solver_nonconvergence_and_nonfinite_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_nonconverged(
        matrix: sparse.spmatrix,
        rhs: np.ndarray,
        *,
        method: str = "direct",
        solver_parameters: object = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        return np.zeros(rhs.shape, dtype=float), {"method": method, "info": 7}

    monkeypatch.setattr(tpfa_mod, "solve_linear_system", fake_nonconverged)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_tpfa(np.ones((2, 2)))

    def fake_nonfinite(
        matrix: sparse.spmatrix,
        rhs: np.ndarray,
        *,
        method: str = "direct",
        solver_parameters: object = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        return np.full(rhs.shape, np.nan, dtype=float), {"method": method, "info": 0}

    monkeypatch.setattr(tpfa_mod, "solve_linear_system", fake_nonfinite)
    with pytest.raises(RuntimeError, match="non-finite pressures"):
        solve_tpfa(np.ones((2, 2)))
