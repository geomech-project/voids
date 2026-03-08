# Benchmarks

The `voids.benchmarks` sub-package provides utilities for cross-checking `voids`
results against reference implementations such as OpenPNM and XLB.

The two high-level segmented-volume benchmark wrappers now share the same
physical pressure convention:

- the preferred public input is the physical pressure drop `delta_p`, typically
  in Pa
- optional `pin` and `pout` values can also be supplied when the user wants to
  preserve a particular absolute pressure reference level
- for the current incompressible permeability benchmark, only the pressure drop
  `Δp = pin - pout` affects the reported permeability
- the applied `p_inlet_physical`, `p_outlet_physical`, and `dp_physical` values
  are recorded in the benchmark result tables

So `delta_p=1.0`, `pin=1.0`/`pout=0.0`, and `delta_p=1.0` with
`pin=101326.0`/`pout=101325.0` all represent the same current benchmark
driving condition.

The XLB benchmark API now has two distinct layers:

- `solve_binary_volume_with_xlb` is the low-level direct-image solver. It works
  in lattice units and accepts lattice pressure boundary conditions through
  `pressure_inlet_lattice`, `pressure_outlet_lattice`, or
  `pressure_drop_lattice`.
- `benchmark_segmented_volume_with_xlb` is the high-level verification wrapper.
  It resolves a physical pressure drop from `delta_p` and optional `pin` /
  `pout`, then maps that same physical `Δp` into lattice units before calling
  XLB on the original binary image.

For the high-level XLB benchmark, `fluid.density` must be provided because the
shared physical pressure drop must be converted into lattice pressure units.

---

## Cross-Check

::: voids.benchmarks.crosscheck

## Segmented Volume Benchmarks

::: voids.benchmarks.segmented_volume

::: voids.benchmarks.xlb
