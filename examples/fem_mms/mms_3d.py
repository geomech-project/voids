"""Five-level 3D Brinkman manufactured-solution minimum working example."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from voids.examples.mms import (
    available_mms_methods,
    bubble_case_3d,
    run_mms_convergence,
)
from voids.fem.singlephase import FEniCSSolverOptions


def _slope_triangle(
    axis: plt.Axes,
    *,
    h_fine: float,
    h_coarse: float,
    y: float,
    rate: float,
    color: str,
    label: str,
    label_above: bool,
) -> None:
    top = y * (h_coarse / h_fine) ** rate
    axis.plot(
        [h_fine, h_coarse, h_coarse, h_fine],
        [y, y, top, y],
        color=color,
        linewidth=1.2,
    )
    text_x = np.sqrt(h_fine * h_coarse) if label_above else h_coarse * 1.06
    text_y = top * 1.18 if label_above else y * 1.05
    axis.annotate(
        f"{label}: r={rate:.2f}",
        (text_x, text_y),
        color=color,
        fontsize=8,
        ha="center" if label_above else "left",
        va="bottom" if label_above else "center",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
        clip_on=False,
    )


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    viscosity = 1.0e-2
    reaction = 1.0
    resolutions = (4, 6, 8, 10, 12)
    case = bubble_case_3d(viscosity=viscosity, reaction=reaction)
    studies = {
        method: run_mms_convergence(
            case,
            method=method,
            resolutions=resolutions,
            options=FEniCSSolverOptions.superlu_direct(),
            facet_size_mode="representative",
            face_refinement=24,
            keep_solution=False,
        )
        for method in available_mms_methods()
    }
    for study in studies.values():
        study.assert_expected_rates(absolute_tolerance=0.35)

    rows = [
        {"method": method, **row} for method, study in studies.items() for row in study.as_dicts()
    ]
    with (output_dir / "mms_3d_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    coordinates = np.linspace(0.0, 1.0, 241)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.vstack((x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, 0.5)))
    velocity, pressure = case.evaluate(points)
    exact_fields = (
        velocity[0].reshape(x_grid.shape),
        velocity[1].reshape(x_grid.shape),
        velocity[2].reshape(x_grid.shape),
        pressure.reshape(x_grid.shape),
    )
    exact_labels = (r"$u_x$", r"$u_y$", r"$u_z$", r"$p$")
    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.25), constrained_layout=True)
    for axis, field, label in zip(axes, exact_fields, exact_labels, strict=True):
        image = axis.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
        axis.set_title(label)
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(r"3D exact fields on the midplane $z=1/2$")
    figure.savefig(output_dir / "mms_3d_exact_midplane.png", dpi=190)
    plt.close(figure)

    metrics = (
        ("velocity_l2_error", r"$\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$"),
        ("velocity_h1_error", r"$\|\mathbf{u}-\mathbf{u}_h\|_{H^1}$"),
        ("pressure_l2_error", r"$\|p-p_h\|_{L^2}$"),
    )
    method_labels = {
        "taylor_hood": "Taylor–Hood P2/P1",
        "usfem_p1dg0": "USFEM P1/DG0",
        "usfem_p1dg1": "USFEM P1/DG1",
    }
    colors = {
        "taylor_hood": "#0072B2",
        "usfem_p1dg0": "#D55E00",
        "usfem_p1dg1": "#009E73",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.05), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, metrics, strict=True):
        metric_name = metric.removesuffix("_error")
        for method, study in studies.items():
            h_values = np.array([level.h for level in study.levels])
            errors = np.array([getattr(level, metric) for level in study.levels])
            axis.loglog(
                h_values,
                errors,
                "o-",
                color=colors[method],
                label=method_labels[method],
                linewidth=1.7,
                markersize=4.5,
            )
        for index, method in enumerate(("taylor_hood", "usfem_p1dg1")):
            study = studies[method]
            rate = study.last_rates[metric_name]
            values = np.array([getattr(level, metric) for level in study.levels])
            finest_pair_peak = max(
                getattr(candidate.levels[level_index], metric)
                for candidate in studies.values()
                for level_index in (-2, -1)
            )
            _slope_triangle(
                axis,
                h_fine=study.levels[-1].h,
                h_coarse=study.levels[-2].h,
                y=(float(values[-1]) * 0.22 if index == 0 else float(finest_pair_peak) * 1.8),
                rate=rate,
                color=colors[method],
                label="TH" if method == "taylor_hood" else "P1/DG1",
                label_above=index == 1,
            )
        axis.set_xlabel(r"$h=1/n$")
        axis.set_ylabel(ylabel)
        tick_resolutions = tuple(reversed(resolutions))
        axis.set_xticks([1.0 / value for value in tick_resolutions])
        axis.set_xticklabels(
            [f"1/{value}" for value in tick_resolutions],
            rotation=32,
            ha="right",
        )
        axis.xaxis.set_minor_locator(NullLocator())
        axis.grid(True, which="both", alpha=0.28)
    axes[0].legend(fontsize=8)
    figure.suptitle("3D Brinkman MMS — five mesh levels and measured finest-pair slopes")
    figure.savefig(output_dir / "mms_3d_convergence.png", dpi=190)
    plt.close(figure)

    for method, study in studies.items():
        rates = ", ".join(f"{name}={value:.3f}" for name, value in study.last_rates.items())
        print(f"{method}: {rates}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/outputs/fem_mms/3d"),
    )
    main(parser.parse_args().output_dir)
