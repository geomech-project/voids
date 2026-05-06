# %% [markdown]
# # Connected 3D Blobs Realizations (`phi = 0.20`) with Pore-Network Statistics and `Kx`
#
# Objective:
# - generate 30 connected **3D blobs** porous-media realizations at target porosity `phi=0.20`
# - extract one pore network per realization
# - characterize:
#   - pore size distribution (PSD)
#   - throat size distribution
#   - coordination number distribution
# - estimate absolute permeability in the x-direction (`Kx`) for each realization
# - plot the `Kx` distribution over all realizations
#
# Scientific notes:
# - `phi=0.20` in 3D can still produce occasional non-spanning samples, so we use a retry loop
#   and keep only successful x-spanning realizations.
# - This is a synthetic morphology benchmark; quantitative transfer to a specific rock requires
#   calibration/validation against image-based or laboratory data.
#

# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

from voids.generators import (
    generate_spanning_blobs_matrix,
    generate_spanning_multiscale_blobs_matrix,
)
from voids.image import extract_spanning_pore_network
from voids.geom import characteristic_size
from voids.graph.metrics import coordination_numbers
from voids.physics.petrophysics import absolute_porosity, effective_porosity
from voids.physics.singlephase import (
    FluidSinglePhase,
    PressureBC,
    SinglePhaseOptions,
    solve,
)


# %%
def iter_progress(iterable, *, desc: str, enabled: bool = True):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, desc=desc, dynamic_ncols=True, leave=False)
    return iterable


# %%
# Study controls (3D synthetic)
N_REALIZATIONS = 30
TARGET_POROSITY = 0.20

SHAPE_3D = (256, 256, 256)  # (nx, ny, nz)
FLOW_AXIS = "x"
FLOW_AXIS_INDEX = 0
VOXEL_SIZE_M = 10.0e-6

# Generator selector
GENERATOR_NAME = "blobs"  # "blobs" or "multiscale_blobs"

# Single-scale blobs control
BLOBINESS = 2.2

# Multiscale blobs controls
MULTISCALE_BLOBINESS_PRIMARY = (0.8, 0.8, 1.5)
MULTISCALE_BLOBINESS_SECONDARY = (3.5, 3.5, 7.0)
MULTISCALE_PRIMARY_WEIGHT = 0.70

BASE_SEED = 20261408
ATTEMPT_BUDGET_MULTIPLIER = 8

USE_TQDM = True
N_PREVIEW_VOLUMES = 3

fluid = FluidSinglePhase(viscosity=1.0e-3)
bc = PressureBC("inlet_xmin", "outlet_xmax", pin=2.0e5, pout=1.0e5)
solve_options = SinglePhaseOptions(conductance_model="valvatne_blunt", solver="direct")

print("N realizations:", N_REALIZATIONS)
print("Target porosity:", TARGET_POROSITY)
print("Shape:", SHAPE_3D)
print("Flow axis:", FLOW_AXIS)
print("Generator:", GENERATOR_NAME)
print("Base seed:", BASE_SEED)
if GENERATOR_NAME == "blobs":
    print("Blobiness:", BLOBINESS)
elif GENERATOR_NAME == "multiscale_blobs":
    print("Primary blobiness:", MULTISCALE_BLOBINESS_PRIMARY)
    print("Secondary blobiness:", MULTISCALE_BLOBINESS_SECONDARY)
    print("Primary weight:", MULTISCALE_PRIMARY_WEIGHT)
else:
    raise ValueError(f"Unsupported GENERATOR_NAME: {GENERATOR_NAME}")
max_total_attempts = max(int(N_REALIZATIONS * ATTEMPT_BUDGET_MULTIPLIER), int(N_REALIZATIONS))
print("Attempt budget:", max_total_attempts)
print(
    "Image reproducibility rule: for this exact configuration, the notebook scans "
    "the same seed stream starting at BASE_SEED and keeps the first successful realizations."
)


# %%
rows: list[dict[str, float | int | str]] = []
payloads: list[dict[str, object]] = []
preview_volumes: list[tuple[int, np.ndarray]] = []
failures: list[dict[str, object]] = []
accepted_seed_starts: list[int] = []
seed_schedule = [BASE_SEED + attempt for attempt in range(max_total_attempts)]
iterator = enumerate(
    iter_progress(seed_schedule, desc="Generating 3D realizations", enabled=USE_TQDM),
    start=1,
)

for loop_id, seed_start in iterator:
    if len(rows) >= N_REALIZATIONS:
        break

    realization_id = len(rows) + 1

    try:
        if GENERATOR_NAME == "blobs":
            binary_void, seed_used = generate_spanning_blobs_matrix(
                shape=SHAPE_3D,
                porosity=TARGET_POROSITY,
                blobiness=BLOBINESS,
                axis_index=FLOW_AXIS_INDEX,
                seed_start=seed_start,
                max_tries=1,
            )
        elif GENERATOR_NAME == "multiscale_blobs":
            binary_void, seed_used = generate_spanning_multiscale_blobs_matrix(
                shape=SHAPE_3D,
                porosity=TARGET_POROSITY,
                blobiness_primary=MULTISCALE_BLOBINESS_PRIMARY,
                blobiness_secondary=MULTISCALE_BLOBINESS_SECONDARY,
                primary_weight=MULTISCALE_PRIMARY_WEIGHT,
                axis_index=FLOW_AXIS_INDEX,
                seed_start=seed_start,
                max_tries=1,
            )
        else:  # pragma: no cover - guarded above
            raise ValueError(f"Unsupported GENERATOR_NAME: {GENERATOR_NAME}")

        extract = extract_spanning_pore_network(
            binary_void.astype(int),
            voxel_size=VOXEL_SIZE_M,
            flow_axis=FLOW_AXIS,
            length_unit="m",
            provenance_notes={
                "study": "connected_porous_media_3d_blobs_phi020",
                "realization": realization_id,
                "generator_name": GENERATOR_NAME,
                "seed_start": int(seed_start),
                "seed_used": int(seed_used),
                "target_porosity": TARGET_POROSITY,
                "blobiness": BLOBINESS if GENERATOR_NAME == "blobs" else None,
                "blobiness_primary": (
                    list(MULTISCALE_BLOBINESS_PRIMARY)
                    if GENERATOR_NAME == "multiscale_blobs"
                    else None
                ),
                "blobiness_secondary": (
                    list(MULTISCALE_BLOBINESS_SECONDARY)
                    if GENERATOR_NAME == "multiscale_blobs"
                    else None
                ),
                "primary_weight": (
                    MULTISCALE_PRIMARY_WEIGHT
                    if GENERATOR_NAME == "multiscale_blobs"
                    else None
                ),
            },
        )
        net = extract.net

        res = solve(net, fluid=fluid, bc=bc, axis=FLOW_AXIS, options=solve_options)
        kx = float(res.permeability[FLOW_AXIS])

        pore_size_m, pore_field = characteristic_size(net.pore, expected_shape=(net.Np,))
        throat_size_m, throat_field = characteristic_size(
            net.throat, expected_shape=(net.Nt,)
        )
        pore_size_um = 1.0e6 * pore_size_m
        throat_size_um = 1.0e6 * throat_size_m
        coordination = coordination_numbers(net).astype(int)

        phi_image = float(binary_void.mean())
        phi_abs = float(absolute_porosity(net))
        phi_eff_x = float(effective_porosity(net, axis=FLOW_AXIS))

        rows.append(
            {
                "realization": realization_id,
                "seed_start": int(seed_start),
                "seed_used": int(seed_used),
                "phi_image": phi_image,
                "phi_abs_network": phi_abs,
                "phi_eff_x_network": phi_eff_x,
                "Np": int(net.Np),
                "Nt": int(net.Nt),
                "pore_size_field": pore_field,
                "throat_size_field": throat_field,
                "pore_size_mean_um": float(np.mean(pore_size_um)),
                "pore_size_p50_um": float(np.median(pore_size_um)),
                "throat_size_mean_um": float(np.mean(throat_size_um)),
                "throat_size_p50_um": float(np.median(throat_size_um)),
                "coordination_mean": float(np.mean(coordination)),
                "kabs_x_m2": kx,
                "mass_balance_error": float(res.mass_balance_error),
            }
        )
        payloads.append(
            {
                "realization": realization_id,
                "binary_void": binary_void,
                "pore_size_um": pore_size_um,
                "throat_size_um": throat_size_um,
                "coordination": coordination,
            }
        )

        if len(preview_volumes) < N_PREVIEW_VOLUMES:
            preview_volumes.append((realization_id, binary_void))
        accepted_seed_starts.append(int(seed_start))
    except Exception as exc:
        failures.append({"attempt": loop_id, "seed_start": seed_start, "error": str(exc)})
        continue

if len(rows) < N_REALIZATIONS:
    raise RuntimeError(
        f"Collected {len(rows)} successful realizations out of {N_REALIZATIONS} requested "
        f"after {len(seed_schedule)} attempts. For the same configuration, increasing "
        "ATTEMPT_BUDGET_MULTIPLIER preserves reproducibility because the seed stream stays fixed."
    )

summary_df = pd.DataFrame(rows).sort_values("realization").reset_index(drop=True)
summary_df


# %%
print("Successful realizations:", len(summary_df))
print("Accepted seed starts:")
print(accepted_seed_starts)
print("Failed attempts:", len(failures))
print("First 5 failures (if any):")
for item in failures[:5]:
    print(item)

summary_stats = summary_df[
    [
        "phi_image",
        "phi_abs_network",
        "phi_eff_x_network",
        "Np",
        "Nt",
        "pore_size_mean_um",
        "throat_size_mean_um",
        "coordination_mean",
        "kabs_x_m2",
        "mass_balance_error",
    ]
].describe(percentiles=[0.1, 0.5, 0.9])
summary_stats


# %%
# Visualize a few representative realizations with orthogonal middle slices.
fig, axes = plt.subplots(len(preview_volumes), 3, figsize=(12.5, 3.8 * len(preview_volumes)))
axes = np.atleast_2d(axes)

for row, (rid, vol) in enumerate(preview_volumes):
    mx, my, mz = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2

    axes[row, 0].imshow(vol[mx, :, :], cmap="gray", origin="lower")
    axes[row, 0].set_title(f"R{rid} | YZ slice (x={mx})")
    axes[row, 0].set_xlabel("z")
    axes[row, 0].set_ylabel("y")

    axes[row, 1].imshow(vol[:, my, :], cmap="gray", origin="lower")
    axes[row, 1].set_title(f"R{rid} | XZ slice (y={my})")
    axes[row, 1].set_xlabel("z")
    axes[row, 1].set_ylabel("x")

    axes[row, 2].imshow(vol[:, :, mz], cmap="gray", origin="lower")
    axes[row, 2].set_title(f"R{rid} | XY slice (z={mz})")
    axes[row, 2].set_xlabel("y")
    axes[row, 2].set_ylabel("x")

fig.suptitle("Representative 3D binary-void realizations", fontsize=13)
plt.tight_layout()
plt.show()


# %%
# Aggregate distributions over all extracted networks.
all_pore_sizes_um = np.concatenate([p["pore_size_um"] for p in payloads])
all_throat_sizes_um = np.concatenate([p["throat_size_um"] for p in payloads])
all_coordination = np.concatenate([p["coordination"] for p in payloads]).astype(int)
all_kx = summary_df["kabs_x_m2"].to_numpy(dtype=float)

coord_vals, coord_counts = np.unique(all_coordination, return_counts=True)

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

axes[0, 0].hist(all_pore_sizes_um, color="tab:blue", edgecolor="black", alpha=0.8)
axes[0, 0].set_xlabel("Pore characteristic size [um]")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Pore Size Distribution (pooled over 30 networks)")
axes[0, 0].grid(alpha=0.3, linestyle=":")

axes[0, 1].hist(
    all_throat_sizes_um,
    color="tab:orange",
    edgecolor="black",
    alpha=0.8,
)
axes[0, 1].set_xlabel("Throat characteristic size [um]")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Throat Size Distribution (pooled over 30 networks)")
axes[0, 1].grid(alpha=0.3, linestyle=":")

axes[1, 0].bar(
    coord_vals,
    coord_counts,
    width=0.85,
    color="tab:green",
    edgecolor="black",
    alpha=0.85,
)
axes[1, 0].set_xlabel("Coordination number")
axes[1, 0].set_ylabel("Pore count")
axes[1, 0].set_title("Coordination Number Distribution (pooled)")
axes[1, 0].grid(alpha=0.3, linestyle=":", axis="y")

axes[1, 1].hist(all_kx, color="tab:red", edgecolor="black", alpha=0.85)
axes[1, 1].set_xlabel("Absolute permeability Kx [m2]")
axes[1, 1].set_ylabel("Network count")
axes[1, 1].set_title("Absolute Permeability Distribution (30 realizations)")
axes[1, 1].grid(alpha=0.3, linestyle=":")

fig.suptitle("3D blobs study: statistical characterization and Kx distribution", fontsize=14)
plt.tight_layout()
plt.show()


# %%
# Optional: inspect one specific realization in detail.
def plot_single_realization_distributions(realization: int) -> None:
    item = next(p for p in payloads if int(p["realization"]) == int(realization))
    pore_size_um = np.asarray(item["pore_size_um"], dtype=float)
    throat_size_um = np.asarray(item["throat_size_um"], dtype=float)
    coordination = np.asarray(item["coordination"], dtype=int)
    coord_vals_local, coord_counts_local = np.unique(coordination, return_counts=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].hist(pore_size_um, color="tab:blue", edgecolor="black", alpha=0.85)
    axes[0].set_title(f"Realization {realization} PSD")
    axes[0].set_xlabel("Pore size [um]")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.3, linestyle=":")

    axes[1].hist(
        throat_size_um, color="tab:orange", edgecolor="black", alpha=0.85
    )
    axes[1].set_title(f"Realization {realization} throat sizes")
    axes[1].set_xlabel("Throat size [um]")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3, linestyle=":")

    axes[2].bar(
        coord_vals_local,
        coord_counts_local,
        width=0.85,
        color="tab:green",
        edgecolor="black",
        alpha=0.85,
    )
    axes[2].set_title(f"Realization {realization} coordination")
    axes[2].set_xlabel("Coordination number")
    axes[2].set_ylabel("Pore count")
    axes[2].grid(alpha=0.3, linestyle=":", axis="y")

    plt.tight_layout()
    plt.show()


plot_single_realization_distributions(realization=1)


# %% [markdown]
# ## Assumptions and possible weak points
#
# 1. We assume `blobs` with `blobiness=2.2` at this voxel resolution provides a meaningful synthetic 3D ensemble.
#    This can be incorrect if your target material has very different texture/connectivity scales.
# 2. Conditioning on x-spanning realizations is useful for transport studies, but it can bias morphology statistics
#    relative to the unconditional `phi=0.20` blobs ensemble.
# 3. `Kx` is estimated on extracted pore networks (`snow2` + conduit model), not from full voxel-scale DNS.
#    Extraction/model uncertainty can shift absolute values and distribution width.
# 4. If you need lower Monte Carlo uncertainty on the tails of `Kx`, increase realization count beyond 30 and
#    monitor convergence of quantiles (e.g., p10, p50, p90).
