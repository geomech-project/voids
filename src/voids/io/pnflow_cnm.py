from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voids.core.network import Network
from voids.core.provenance import Provenance
from voids.core.sample import SampleGeometry
from voids.core.validation import validate_network

_BOUNDARY_LENGTH_EPS = 1.0e-300
_TRIANGLE_MAX_G = np.sqrt(3.0) / 36.0
_SQUARE_G = 1.0 / 16.0
_CIRCLE_G = 1.0 / (4.0 * np.pi)


@dataclass(slots=True)
class PnflowCNMImportResult:
    """Container for an imported Imperial College CNM network.

    Attributes
    ----------
    net :
        Imported network ready for `voids` single-phase calculations.
    prefix :
        File prefix used to locate the CNM text files.
    box_lengths :
        Physical sample lengths encoded in the CNM header.
    n_physical_pores :
        Number of pores listed in `*_node*.dat`, excluding mirrored
        inlet/outlet helper pores inserted during import.
    n_boundary_mirror_pores :
        Number of helper pores added to mimic `pnflow` reservoir semantics.
    """

    net: Network
    prefix: Path
    box_lengths: dict[str, float]
    n_physical_pores: int
    n_boundary_mirror_pores: int


def _split_numeric_line(line: str, *, expected_min_tokens: int, label: str) -> list[str]:
    """Split a whitespace-formatted CNM line and validate a minimal token count."""

    tokens = line.split()
    if len(tokens) < expected_min_tokens:
        raise ValueError(
            f"Malformed {label}: expected at least {expected_min_tokens} fields, got {len(tokens)}"
        )
    return tokens


def _pnflow_effective_shape_factor(shape_factor: np.ndarray) -> np.ndarray:
    """Return the effective shape factor used internally by `pnflow`.

    Notes
    -----
    The Imperial code uses the exported shape factor to classify elements, but
    square and circular element models solve with canonical shape factors.
    Triangle elements retain the exported value, capped slightly below the
    triangular upper bound in the same way as `Element.cpp`.
    """

    g = np.asarray(shape_factor, dtype=float)
    out = g.copy()
    tri = out <= _TRIANGLE_MAX_G + 1.0e-5
    sq = (out > _TRIANGLE_MAX_G + 1.0e-5) & (out < 0.07)
    cir = out >= 0.07
    out[tri] = np.minimum(out[tri], _TRIANGLE_MAX_G - 5.0e-5)
    out[sq] = _SQUARE_G
    out[cir] = _CIRCLE_G
    return out


def load_pnflow_cnm(
    prefix: str | Path,
    *,
    boundary_axis: str = "x",
    length_unit: str = "m",
    pressure_unit: str = "Pa",
    boundary_length_epsilon: float = _BOUNDARY_LENGTH_EPS,
    boundary_radius_scale: float = 1.1,
    pnflow_solver_box_compat: bool = False,
) -> PnflowCNMImportResult:
    """Import an Imperial College `pnextract` / `pnflow` CNM text network.

    Parameters
    ----------
    prefix :
        File prefix for the four CNM text files. For a benchmark case stored as
        `case_dir/case_name_node1.dat`, pass `case_dir/case_name`.
    boundary_axis :
        Axis along which the inlet/outlet reservoir labels are attached. The
        committed Imperial CNM format is x-directed, so `"x"` is the default
        and the only axis currently supported.
    length_unit, pressure_unit :
        Unit metadata attached to the resulting `SampleGeometry`.
    boundary_length_epsilon :
        Small positive reservoir-side pore length used to reproduce the
        near-zero boundary resistance applied internally by `pnflow`.
    boundary_radius_scale :
        Scale factor used for mirrored inlet/outlet helper pores. This follows
        the `InOutBoundary::prepare2()` construction in the Imperial code.
    pnflow_solver_box_compat :
        If ``True``, reproduce the Imperial CNM preprocessing quirk that
        excludes the first physical pore from the solver box when
        ``nBSs_ = 2`` is hard-coded in `FlowDomain.cpp`. The excluded pore is
        then treated as an inlet or outlet solver-boundary pore based on its
        x-position relative to the sample mid-plane. This is kept opt-in
        because it reproduces checked-in `pnflow` behavior rather than a
        generic physical boundary rule. Enabling this option is required for
        near machine-precision single-phase parity with the saved `pnflow`
        benchmark cases.

    Returns
    -------
    PnflowCNMImportResult
        Imported network together with import metadata.

    Notes
    -----
    The CNM text files store internal pores only. To match `pnflow`'s
    single-phase boundary treatment more closely, this importer inserts one
    zero-volume mirrored pore for each inlet/outlet connection throat and
    collapses the reservoir-side pore segment length to a tiny positive value.
    """

    if boundary_axis != "x":
        raise ValueError("Imperial CNM import currently supports only boundary_axis='x'")
    if boundary_length_epsilon <= 0.0:
        raise ValueError("boundary_length_epsilon must be positive")
    if boundary_radius_scale <= 0.0:
        raise ValueError("boundary_radius_scale must be positive")

    prefix_path = Path(prefix)
    node1_path = prefix_path.with_name(f"{prefix_path.name}_node1.dat")
    node2_path = prefix_path.with_name(f"{prefix_path.name}_node2.dat")
    link1_path = prefix_path.with_name(f"{prefix_path.name}_link1.dat")
    link2_path = prefix_path.with_name(f"{prefix_path.name}_link2.dat")

    node1_lines = node1_path.read_text().splitlines()
    node2_lines = node2_path.read_text().splitlines()
    link1_lines = link1_path.read_text().splitlines()
    link2_lines = link2_path.read_text().splitlines()

    if not node1_lines:
        raise ValueError(f"CNM file is empty: {node1_path}")
    header = _split_numeric_line(node1_lines[0], expected_min_tokens=4, label=str(node1_path))
    n_physical_pores = int(header[0])
    lx, ly, lz = map(float, header[1:4])
    box_lengths = {"x": lx, "y": ly, "z": lz}
    cross_sections = {"x": ly * lz, "y": lx * lz, "z": lx * ly}

    if len(node1_lines) != n_physical_pores + 1:
        raise ValueError(
            f"{node1_path} header declares {n_physical_pores} pores but file contains "
            f"{len(node1_lines) - 1} pore rows"
        )
    if len(node2_lines) != n_physical_pores:
        raise ValueError(
            f"{node2_path} should contain {n_physical_pores} pore rows, got {len(node2_lines)}"
        )
    if not link1_lines:
        raise ValueError(f"CNM file is empty: {link1_path}")
    n_throats = int(
        _split_numeric_line(link1_lines[0], expected_min_tokens=1, label=str(link1_path))[0]
    )
    link1_rows = link1_lines[1:]
    if len(link1_rows) != n_throats or len(link2_lines) != n_throats:
        raise ValueError(
            f"Throat-row mismatch for prefix {prefix_path}: "
            f"header={n_throats}, link1_rows={len(link1_rows)}, link2_rows={len(link2_lines)}"
        )

    pore_coords = np.zeros((n_physical_pores, 3), dtype=float)
    pore_volume = np.zeros(n_physical_pores, dtype=float)
    pore_radius = np.zeros(n_physical_pores, dtype=float)
    pore_shape_factor_raw = np.zeros(n_physical_pores, dtype=float)
    pore_connected_inlet = np.zeros(n_physical_pores, dtype=bool)
    pore_connected_outlet = np.zeros(n_physical_pores, dtype=bool)

    for line in node1_lines[1:]:
        tokens = _split_numeric_line(line, expected_min_tokens=6, label=str(node1_path))
        idx = int(tokens[0]) - 1
        x, y, z = map(float, tokens[1:4])
        conn_number = int(tokens[4])
        inlet_pos = 5 + conn_number
        outlet_pos = inlet_pos + 1
        if len(tokens) < outlet_pos + 1 + conn_number:
            raise ValueError(f"Malformed pore-connectivity row in {node1_path}: {line}")
        pore_coords[idx] = (x, y, z)
        pore_connected_inlet[idx] = bool(int(tokens[inlet_pos]))
        pore_connected_outlet[idx] = bool(int(tokens[outlet_pos]))

    for line in node2_lines:
        tokens = _split_numeric_line(line, expected_min_tokens=5, label=str(node2_path))
        idx = int(tokens[0]) - 1
        pore_volume[idx] = float(tokens[1])
        pore_radius[idx] = max(float(tokens[2]), boundary_length_epsilon)
        pore_shape_factor_raw[idx] = max(float(tokens[3]), boundary_length_epsilon)

    coords_list = pore_coords.tolist()
    volume_list = pore_volume.tolist()
    radius_list = pore_radius.tolist()
    shape_factor_raw_list = pore_shape_factor_raw.tolist()
    inlet_label = np.zeros(n_physical_pores, dtype=bool).tolist()
    outlet_label = np.zeros(n_physical_pores, dtype=bool).tolist()

    throat_conns: list[list[int]] = []
    throat_radius = np.zeros(n_throats, dtype=float)
    throat_shape_factor = np.zeros(n_throats, dtype=float)
    throat_volume = np.zeros(n_throats, dtype=float)
    throat_core_length = np.zeros(n_throats, dtype=float)
    throat_pore1_length = np.zeros(n_throats, dtype=float)
    throat_pore2_length = np.zeros(n_throats, dtype=float)

    n_boundary_mirror_pores = 0
    for throat_idx, (line1, line2) in enumerate(zip(link1_rows, link2_lines, strict=True)):
        tokens1 = _split_numeric_line(line1, expected_min_tokens=6, label=str(link1_path))
        tokens2 = _split_numeric_line(line2, expected_min_tokens=8, label=str(link2_path))

        pore1_idx_raw = int(tokens1[1])
        pore2_idx_raw = int(tokens1[2])
        throat_radius[throat_idx] = max(float(tokens1[3]), boundary_length_epsilon)
        throat_shape_factor[throat_idx] = max(float(tokens1[4]), boundary_length_epsilon)
        throat_pore1_length[throat_idx] = max(float(tokens2[3]), boundary_length_epsilon)
        throat_pore2_length[throat_idx] = max(float(tokens2[4]), boundary_length_epsilon)
        throat_core_length[throat_idx] = max(float(tokens2[5]), boundary_length_epsilon)
        throat_volume[throat_idx] = float(tokens2[6])

        left = pore1_idx_raw - 1 if pore1_idx_raw > 0 else None
        right = pore2_idx_raw - 1 if pore2_idx_raw > 0 else None

        if pore1_idx_raw in {-1, 0}:
            if right is None:
                raise ValueError(
                    f"Boundary throat without an internal neighbor in {link1_path}: {line1}"
                )
            x_boundary = 0.0 if pore1_idx_raw == -1 else lx
            coords_list.append([x_boundary, pore_coords[right, 1], pore_coords[right, 2]])
            volume_list.append(0.0)
            radius_list.append(throat_radius[throat_idx] * boundary_radius_scale)
            shape_factor_raw_list.append(throat_shape_factor[throat_idx])
            inlet_label.append(pore1_idx_raw == -1)
            outlet_label.append(pore1_idx_raw == 0)
            left = len(coords_list) - 1
            throat_pore1_length[throat_idx] = boundary_length_epsilon
            n_boundary_mirror_pores += 1

        if pore2_idx_raw in {-1, 0}:
            if left is None:
                raise ValueError(
                    f"Boundary throat without an internal neighbor in {link1_path}: {line1}"
                )
            x_boundary = 0.0 if pore2_idx_raw == -1 else lx
            left_coords = coords_list[left]
            coords_list.append([x_boundary, left_coords[1], left_coords[2]])
            volume_list.append(0.0)
            radius_list.append(throat_radius[throat_idx] * boundary_radius_scale)
            shape_factor_raw_list.append(throat_shape_factor[throat_idx])
            inlet_label.append(pore2_idx_raw == -1)
            outlet_label.append(pore2_idx_raw == 0)
            right = len(coords_list) - 1
            throat_pore2_length[throat_idx] = boundary_length_epsilon
            n_boundary_mirror_pores += 1

        if left is None or right is None:
            raise ValueError(f"Unresolved throat endpoints while importing {link1_path}: {line1}")
        throat_conns.append([left, right])

    pore_coords_arr = np.asarray(coords_list, dtype=float)
    pore_volume_arr = np.asarray(volume_list, dtype=float)
    pore_radius_arr = np.asarray(radius_list, dtype=float)
    pore_shape_factor_raw_arr = np.asarray(shape_factor_raw_list, dtype=float)
    pore_shape_factor_arr = _pnflow_effective_shape_factor(pore_shape_factor_raw_arr)
    inlet_label_arr = np.asarray(inlet_label, dtype=bool)
    outlet_label_arr = np.asarray(outlet_label, dtype=bool)
    if pnflow_solver_box_compat and n_physical_pores > 0:
        if pore_coords_arr[0, 0] < 0.5 * lx:
            inlet_label_arr[0] = True
        else:
            outlet_label_arr[0] = True
    pore_area_arr = pore_radius_arr**2 / (4.0 * pore_shape_factor_arr)
    throat_shape_factor_raw = throat_shape_factor.copy()
    throat_shape_factor = _pnflow_effective_shape_factor(throat_shape_factor_raw)
    throat_area = throat_radius**2 / (4.0 * throat_shape_factor)

    net = Network(
        throat_conns=np.asarray(throat_conns, dtype=np.int64),
        pore_coords=pore_coords_arr,
        sample=SampleGeometry(
            bulk_volume=lx * ly * lz,
            lengths=box_lengths,
            cross_sections=cross_sections,
            units={"length": length_unit, "pressure": pressure_unit},
        ),
        provenance=Provenance(
            source_kind="external_network",
            extraction_method="pnextract_cnm_text",
            user_notes={
                "prefix": str(prefix_path),
                "boundary_axis": boundary_axis,
                "boundary_length_epsilon": boundary_length_epsilon,
                "boundary_radius_scale": boundary_radius_scale,
                "pnflow_solver_box_compat": pnflow_solver_box_compat,
            },
        ),
        pore={
            "volume": pore_volume_arr,
            "radius_inscribed": pore_radius_arr,
            "diameter_inscribed": 2.0 * pore_radius_arr,
            "shape_factor": pore_shape_factor_arr,
            "shape_factor_raw": pore_shape_factor_raw_arr,
            "area": pore_area_arr,
        },
        throat={
            "volume": throat_volume,
            "radius_inscribed": throat_radius,
            "diameter_inscribed": 2.0 * throat_radius,
            "shape_factor": throat_shape_factor,
            "shape_factor_raw": throat_shape_factor_raw,
            "area": throat_area,
            "core_length": throat_core_length,
            "pore1_length": throat_pore1_length,
            "pore2_length": throat_pore2_length,
            "length": throat_core_length + throat_pore1_length + throat_pore2_length,
        },
        pore_labels={
            "inlet_xmin": inlet_label_arr,
            "outlet_xmax": outlet_label_arr,
            "boundary": inlet_label_arr | outlet_label_arr,
            "boundary_connected_inlet_xmin": np.pad(
                pore_connected_inlet,
                (0, n_boundary_mirror_pores),
                constant_values=False,
            ),
            "boundary_connected_outlet_xmax": np.pad(
                pore_connected_outlet,
                (0, n_boundary_mirror_pores),
                constant_values=False,
            ),
        },
        extra={
            "pnflow_cnm": {
                "prefix": str(prefix_path),
                "n_physical_pores": n_physical_pores,
                "n_boundary_mirror_pores": n_boundary_mirror_pores,
                "box_lengths": box_lengths,
                "pnflow_solver_box_compat": pnflow_solver_box_compat,
            }
        },
    )
    validate_network(net)
    return PnflowCNMImportResult(
        net=net,
        prefix=prefix_path,
        box_lengths=box_lengths,
        n_physical_pores=n_physical_pores,
        n_boundary_mirror_pores=n_boundary_mirror_pores,
    )


__all__ = ["PnflowCNMImportResult", "load_pnflow_cnm"]
