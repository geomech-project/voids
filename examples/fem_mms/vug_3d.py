"""Body-fitted 3D centered-vug field-gallery minimum working example."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voids.examples.mms import CenteredVugBenchmark, run_centered_vug_benchmark
from voids.fem.singlephase import FEniCSSolverOptions
from voids.visualization.fields import (
    sample_dolfinx_function_at_points,
    write_fem_result_xdmf,
)

_DEFAULT_UPSCALING_FRACTIONS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4)


def _l2_project_to_cg1(function: object, *, prefix: str) -> object:
    """Return the continuous CG1 L2 projection used only for visualization."""

    import basix.ufl as basix_ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem
    import ufl

    mesh = function.function_space.mesh
    element = basix_ufl.element("Lagrange", mesh.basix_cell(), 1)
    space = fem.functionspace(mesh, element)
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    problem = LinearProblem(
        ufl.inner(trial, test) * ufl.dx,
        ufl.inner(function, test) * ufl.dx,
        petsc_options_prefix=prefix,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    projected = problem.solve()
    projected.name = "pressure_l2_projected_cg1"
    return projected


def _pressure_jump_l2(function: object) -> float:
    """Return the raw interior-facet pressure-jump diagnostic."""

    from dolfinx import fem
    from mpi4py import MPI
    import ufl

    local_square = fem.assemble_scalar(fem.form(ufl.jump(function) ** 2 * ufl.dS))
    global_square = function.function_space.mesh.comm.allreduce(local_square, op=MPI.SUM)
    return float(np.sqrt(max(global_square, 0.0)))


def _midplane_points(samples: int) -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    coordinates = np.linspace(1.0e-6, 1.0 - 1.0e-6, samples)
    first, second = np.meshgrid(coordinates, coordinates, indexing="xy")
    half = np.full(first.size, 0.5)
    return {
        "xy, z=0.5": (
            np.column_stack((first.ravel(), second.ravel(), half)),
            first.shape,
        ),
        "xz, y=0.5": (
            np.column_stack((first.ravel(), half, second.ravel())),
            first.shape,
        ),
        "yz, x=0.5": (
            np.column_stack((half, first.ravel(), second.ravel())),
            first.shape,
        ),
    }


def _plot_gallery(
    fields: dict[str, dict[str, np.ndarray]],
    *,
    title: str,
    colorbar_label: str,
    path: Path,
) -> None:
    display_methods = {
        "taylor_hood": "Taylor–Hood P2/P1",
        "usfem_p1dg1": "USFEM P1/DG1",
    }
    methods = tuple(fields)
    planes = tuple(next(iter(fields.values())))
    all_values = np.concatenate(
        [np.ravel(fields[method][plane]) for method in methods for plane in planes]
    )
    finite_values = all_values[np.isfinite(all_values)]
    lower, upper = float(np.min(finite_values)), float(np.max(finite_values))
    figure, axes = plt.subplots(
        len(methods),
        len(planes),
        figsize=(11.3, 6.7),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row, method in enumerate(methods):
        for column, plane in enumerate(planes):
            axis = axes[row, column]
            image = axis.imshow(
                fields[method][plane],
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="viridis",
                vmin=lower,
                vmax=upper,
            )
            axis.set_title(f"{display_methods[method]}\n{plane}")
            axis.set_xlabel("first in-plane coordinate")
            axis.set_ylabel("second in-plane coordinate")
            axis.set_aspect("equal")
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.82, label=colorbar_label)
    figure.suptitle(title)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _plot_component_gallery(fields: dict[str, dict[str, np.ndarray]], *, path: Path) -> None:
    methods = tuple(fields)
    components = ("u_x", "u_y", "u_z")
    display_methods = {
        "taylor_hood": "Taylor–Hood P2/P1",
        "usfem_p1dg1": "USFEM P1/DG1",
    }
    figure, axes = plt.subplots(
        len(methods),
        len(components),
        figsize=(11.3, 6.7),
        constrained_layout=True,
        squeeze=False,
    )
    for column, component in enumerate(components):
        limit = max(float(np.nanmax(np.abs(fields[method][component]))) for method in methods)
        image = None
        for row, method in enumerate(methods):
            axis = axes[row, column]
            image = axis.imshow(
                fields[method][component],
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_title(f"{display_methods[method]}\n${component}$")
            axis.set_xlabel(r"$x$")
            axis.set_ylabel(r"$y$")
            axis.set_aspect("equal")
        assert image is not None
        figure.colorbar(
            image,
            ax=axes[:, column],
            shrink=0.8,
            label=f"{component} (normalized)",
        )
    figure.suptitle(r"3D body-fitted centered vug: velocity components on $z=1/2$")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _sphere_radius_from_volume_fraction(volume_fraction: float) -> float:
    """Return the centered-sphere radius for a unit-cube volume fraction."""

    maximum_fraction = np.pi / 6.0
    if (
        not np.isfinite(volume_fraction)
        or volume_fraction < 0.0
        or volume_fraction >= maximum_fraction
    ):
        raise ValueError(
            "volume fractions must be finite and lie in [0, pi/6) for a "
            "centered sphere contained in the unit cube"
        )
    return float(np.cbrt(3.0 * volume_fraction / (4.0 * np.pi)))


def _plot_upscaling_study(rows: list[dict[str, Any]], *, path: Path) -> None:
    display_methods = {
        "taylor_hood": "Taylor–Hood P2/P1",
        "usfem_p1dg1": "USFEM P1/DG1",
    }
    methods = tuple(display_methods)
    by_method = {
        method: sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["analytic_vug_fraction"]),
        )
        for method in methods
    }
    fractions = np.asarray(
        [float(row["analytic_vug_fraction"]) for row in by_method["taylor_hood"]]
    )
    permeability_ratios = {
        method: np.asarray(
            [float(row["effective_permeability_over_matrix"]) for row in by_method[method]]
        )
        for method in methods
    }
    difference_percent = (
        100.0
        * np.abs(permeability_ratios["usfem_p1dg1"] - permeability_ratios["taylor_hood"])
        / permeability_ratios["taylor_hood"]
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    for method in methods:
        axes[0].plot(
            fractions,
            permeability_ratios[method],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=display_methods[method],
        )
    axes[0].set_title("Flow-based permeability upscaling")
    axes[0].set_xlabel(r"spherical vug volume fraction, $f_v$")
    axes[0].set_ylabel(r"$K_{\mathrm{eff}}/K_m$")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        fractions,
        difference_percent,
        color="tab:purple",
        marker="o",
        linewidth=1.8,
        markersize=4.5,
    )
    axes[1].set_title("Discretization difference")
    axes[1].set_xlabel(r"spherical vug volume fraction, $f_v$")
    axes[1].set_ylabel(r"$100|K_{\mathrm{USFEM}}-K_{\mathrm{TH}}|/K_{\mathrm{TH}}$ [%]")
    axes[1].grid(alpha=0.3)
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _run_upscaling_study(
    output_dir: Path,
    *,
    resolution: int,
    fractions: tuple[float, ...],
    options: FEniCSSolverOptions,
) -> list[dict[str, Any]]:
    matrix_viscosity = 1.0e-2
    matrix_drag = 1.0e7
    matrix_permeability = matrix_viscosity / matrix_drag
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        benchmark = CenteredVugBenchmark(
            dimension=3,
            resolution=resolution,
            radius=_sphere_radius_from_volume_fraction(fraction),
            viscosity=matrix_viscosity,
            matrix_drag=matrix_drag,
            vug_drag=1.0,
            pressure_inlet=1.0,
            pressure_outlet=-1.0,
            mesh_representation="body_fitted",
        )
        fraction_results = {
            method: run_centered_vug_benchmark(
                benchmark,
                method=method,
                options=options,
                facet_size_mode="facet_measure",
            )
            for method in ("taylor_hood", "usfem_p1dg1")
        }
        permeability_by_method = {
            method: float(result.permeability) for method, result in fraction_results.items()
        }
        relative_difference_percent = (
            100.0
            * abs(permeability_by_method["usfem_p1dg1"] - permeability_by_method["taylor_hood"])
            / permeability_by_method["taylor_hood"]
        )
        for method, result in fraction_results.items():
            rows.append(
                {
                    "method": method,
                    "resolution": resolution,
                    "num_cells": result.metadata["num_cells"],
                    "analytic_vug_fraction": benchmark.analytic_fraction,
                    "represented_vug_fraction": result.metadata["represented_vug_fraction"],
                    "sphere_radius": benchmark.radius,
                    "flow_rate": result.flow_rate,
                    "effective_permeability": result.permeability,
                    "matrix_permeability": matrix_permeability,
                    "effective_permeability_over_matrix": (
                        result.permeability / matrix_permeability
                    ),
                    "usfem_vs_taylor_hood_difference_percent": (relative_difference_percent),
                    "solve_seconds": result.solve_seconds,
                }
            )
    with (output_dir / "vug_3d_upscaling_summary.csv").open(
        "w",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_upscaling_study(
        rows,
        path=output_dir / "vug_3d_effective_permeability.png",
    )
    return rows


def main(
    output_dir: Path,
    *,
    resolution: int,
    upscaling_resolution: int,
    upscaling_fractions: tuple[float, ...],
    fields: bool,
    upscaling: bool,
    export_xdmf: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = CenteredVugBenchmark(
        dimension=3,
        resolution=resolution,
        radius=0.25,
        viscosity=1.0e-2,
        matrix_drag=1.0e7,
        vug_drag=1.0,
        pressure_inlet=1.0,
        pressure_outlet=-1.0,
        mesh_representation="body_fitted",
    )
    options = FEniCSSolverOptions.superlu_direct()
    if upscaling:
        upscaling_rows = _run_upscaling_study(
            output_dir,
            resolution=upscaling_resolution,
            fractions=upscaling_fractions,
            options=options,
        )
        for row in upscaling_rows:
            print(row)
    if not fields:
        return
    results = {
        method: run_centered_vug_benchmark(
            benchmark,
            method=method,
            options=options,
            facet_size_mode="facet_measure",
        )
        for method in ("taylor_hood", "usfem_p1dg1")
    }
    rows = [
        {
            "method": method,
            "resolution": benchmark.resolution,
            "num_cells": result.metadata["num_cells"],
            "analytic_vug_fraction": result.metadata["analytic_vug_fraction"],
            "represented_vug_fraction": result.metadata["represented_vug_fraction"],
            "flow_rate": result.flow_rate,
            "permeability_diagnostic": result.permeability,
            "solve_seconds": result.solve_seconds,
            "facet_law": result.metadata["facet_law"],
            "facet_size_mode": result.metadata["facet_size_mode"],
            "raw_pressure_jump_l2": _pressure_jump_l2(result.pressure),
        }
        for method, result in results.items()
    ]
    with (output_dir / "vug_3d_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    planes = _midplane_points(101)
    raw_pressure_fields: dict[str, dict[str, np.ndarray]] = {}
    projected_pressure_fields: dict[str, dict[str, np.ndarray]] = {}
    velocity_magnitude_fields: dict[str, dict[str, np.ndarray]] = {}
    velocity_components: dict[str, dict[str, np.ndarray]] = {}
    for method, result in results.items():
        raw_pressure_fields[method] = {}
        projected_pressure_fields[method] = {}
        velocity_magnitude_fields[method] = {}
        velocity_components[method] = {}
        projected_pressure = _l2_project_to_cg1(
            result.pressure,
            prefix=f"voids_vug_3d_pressure_projection_{method}_",
        )
        for plane, (points, shape) in planes.items():
            raw_pressure = sample_dolfinx_function_at_points(result.pressure, points)
            display_pressure = sample_dolfinx_function_at_points(projected_pressure, points)
            velocity = sample_dolfinx_function_at_points(result.velocity, points)
            raw_pressure_fields[method][plane] = raw_pressure.reshape(shape)
            projected_pressure_fields[method][plane] = display_pressure.reshape(shape)
            velocity_magnitude_fields[method][plane] = np.linalg.norm(velocity, axis=0).reshape(
                shape
            )
            if plane == "xy, z=0.5":
                for component, name in enumerate(("u_x", "u_y", "u_z")):
                    velocity_components[method][name] = velocity[component].reshape(shape)

    _plot_gallery(
        projected_pressure_fields,
        title="3D body-fitted centered vug: CG1 L2-projected pressure",
        colorbar_label="projected pressure (normalized)",
        path=output_dir / "vug_3d_pressure_midplanes.png",
    )
    _plot_gallery(
        raw_pressure_fields,
        title="3D body-fitted centered vug: raw discrete pressure",
        colorbar_label="raw pressure (normalized)",
        path=output_dir / "vug_3d_pressure_raw_midplanes.png",
    )
    _plot_gallery(
        velocity_magnitude_fields,
        title="3D body-fitted centered vug: velocity magnitude on three midplanes",
        colorbar_label="velocity magnitude",
        path=output_dir / "vug_3d_velocity_midplanes.png",
    )
    _plot_component_gallery(
        velocity_components,
        path=output_dir / "vug_3d_velocity_components.png",
    )

    coordinates = np.linspace(0.0, 1.0, 301)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="xy")
    geometry = (x_grid - 0.5) ** 2 + (y_grid - 0.5) ** 2 <= benchmark.radius**2
    figure, axes = plt.subplots(1, 3, figsize=(9.8, 3.15), constrained_layout=True)
    for axis, label in zip(axes, ("xy section", "xz section", "yz section"), strict=True):
        axis.imshow(
            geometry,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="cividis",
            interpolation="nearest",
        )
        axis.set_title(label)
        axis.set_xlabel("first coordinate")
        axis.set_ylabel("second coordinate")
        axis.set_aspect("equal")
    figure.suptitle("Centered sphere (yellow) in the porous matrix (blue)")
    figure.savefig(output_dir / "vug_3d_geometry_sections.png", dpi=190)
    plt.close(figure)

    if export_xdmf:
        for method, result in results.items():
            write_fem_result_xdmf(result, output_dir / f"vug_3d_{method}.xdmf")
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/outputs/fem_mms/vug_3d"),
    )
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--upscaling-resolution", type=int, default=16)
    parser.add_argument(
        "--upscaling-fractions",
        type=float,
        nargs="+",
        default=_DEFAULT_UPSCALING_FRACTIONS,
    )
    parser.add_argument("--skip-fields", action="store_true")
    parser.add_argument("--skip-upscaling", action="store_true")
    parser.add_argument("--export-xdmf", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.output_dir,
        resolution=arguments.resolution,
        upscaling_resolution=arguments.upscaling_resolution,
        upscaling_fractions=tuple(arguments.upscaling_fractions),
        fields=not arguments.skip_fields,
        upscaling=not arguments.skip_upscaling,
        export_xdmf=arguments.export_xdmf,
    )
