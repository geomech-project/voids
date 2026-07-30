"""Physical 2D centered-vug family minimum working example."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voids.examples.mms import CenteredVugFlowCase2D, run_centered_vug_flow_case
from voids.fem.singlephase import FEniCSSolverOptions
from voids.visualization.fields import (
    sample_dolfinx_function_on_grid,
    write_fem_result_xdmf,
)


def _gallery(
    fields: dict[tuple[str, float], np.ndarray],
    *,
    title: str,
    label: str,
    output: Path,
) -> None:
    models = ("darcy_brinkman", "darcy_darcy")
    fractions = sorted({fraction for _, fraction in fields})
    values = np.concatenate([field.ravel() for field in fields.values()])
    finite = values[np.isfinite(values)]
    lower, upper = float(np.min(finite)), float(np.max(finite))
    figure, axes = plt.subplots(
        len(models),
        len(fractions),
        figsize=(3.0 * len(fractions), 5.8),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row, model in enumerate(models):
        for column, fraction in enumerate(fractions):
            image = axes[row, column].imshow(
                fields[(model, fraction)].T,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="viridis",
                vmin=lower,
                vmax=upper,
            )
            axes[row, column].set_title(f"{model}\n$f_v={fraction:g}$")
            axes[row, column].set_xlabel(r"$x/L$")
            axes[row, column].set_ylabel(r"$y/L$")
            axes[row, column].set_aspect("equal")
    assert image is not None
    figure.colorbar(image, ax=axes, shrink=0.82, label=label)
    figure.suptitle(title)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(
    output_dir: Path,
    *,
    resolution: int,
    full_family: bool,
    export_xdmf: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fractions = (0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.7) if full_family else (0.0, 0.1, 0.4, 0.7)
    models = ("darcy_brinkman", "darcy_darcy")
    options = FEniCSSolverOptions.superlu_direct()
    rows: list[dict[str, object]] = []
    pressure_fields: dict[tuple[str, float], np.ndarray] = {}
    velocity_magnitude_fields: dict[tuple[str, float], np.ndarray] = {}
    for fraction in fractions:
        case = CenteredVugFlowCase2D(
            area_fraction=fraction,
            image_shape=(500, 500),
            voxel_size_m=15.0e-6,
            matrix_porosity=0.2,
            matrix_permeability_md=200.0,
            vug_permeability_m2=1.0e-8,
            dynamic_viscosity_pa_s=1.0e-3,
            pressure_inlet_pa=1.0,
            pressure_outlet_pa=0.0,
            mesh_resolution=resolution,
        )
        for model in models:
            result = run_centered_vug_flow_case(case, model=model, options=options)
            pressure = sample_dolfinx_function_on_grid(
                result.pressure,
                shape=(121, 121),
                cell_size=1.0 / 121.0,
            )
            velocity = sample_dolfinx_function_on_grid(
                result.velocity,
                shape=(121, 121),
                cell_size=1.0 / 121.0,
            )
            pressure_fields[(model, fraction)] = pressure
            velocity_magnitude_fields[(model, fraction)] = np.linalg.norm(velocity, axis=0)
            rows.append(
                {
                    "model": model,
                    "area_fraction": fraction,
                    "mesh_resolution": resolution,
                    "num_cells": result.metadata["num_cells"],
                    "represented_area_fraction": result.metadata["represented_vug_area_fraction"],
                    "flow_rate_m2_per_s": result.flow_rate,
                    "effective_permeability_m2": result.permeability,
                    "effective_permeability_over_matrix": (
                        result.permeability / case.matrix_permeability_m2
                    ),
                    "solve_seconds": result.solve_seconds,
                }
            )
            if export_xdmf:
                suffix = str(fraction).replace(".", "p")
                write_fem_result_xdmf(
                    result,
                    output_dir / f"vug_2d_{model}_f{suffix}.xdmf",
                )
    with (output_dir / "vug_2d_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _gallery(
        pressure_fields,
        title="Physical 2D centered-vug family: pressure",
        label="pressure (Pa, zero-mean representation)",
        output=output_dir / "vug_2d_pressure_gallery.png",
    )
    _gallery(
        velocity_magnitude_fields,
        title="Physical 2D centered-vug family: velocity magnitude",
        label=r"velocity magnitude (m s$^{-1}$)",
        output=output_dir / "vug_2d_velocity_gallery.png",
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/outputs/fem_mms/vug_2d"),
    )
    parser.add_argument("--resolution", type=int, default=40)
    parser.add_argument("--full-family", action="store_true")
    parser.add_argument("--export-xdmf", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.output_dir,
        resolution=arguments.resolution,
        full_family=arguments.full_family,
        export_xdmf=arguments.export_xdmf,
    )
