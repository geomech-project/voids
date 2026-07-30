# FEM MMS and centered-vug minimum working examples

These scripts are the executable companions to the FEM verification reports:

- `mms_2d.py`: five-level 2D Brinkman MMS study;
- `mms_3d.py`: five-level 3D Brinkman MMS study;
- `vug_2d.py`: physical 2D Darcy--Brinkman/Darcy--Darcy vug family;
- `vug_3d.py`: 3D body-fitted centered-vug formulation and flow-based
  upscaling comparison.

Run them from the repository root in the Pixi default environment. Each script
accepts `--output-dir`; use a disposable directory unless intentionally
refreshing documentation assets.

```bash
pixi run python examples/fem_mms/mms_2d.py
pixi run python examples/fem_mms/mms_3d.py
pixi run python examples/fem_mms/vug_2d.py
pixi run python examples/fem_mms/vug_3d.py
```

The MMS scripts use five mesh levels and annotate measured finest-pair slopes.
The vug scripts do not claim an exact solution: they report integral flux and
permeability diagnostics and plot the computed pressure and velocity fields.
The 3D MWE sweeps seven feasible spherical volume fractions by default; use
`--skip-fields` for the upscaling study alone and `--upscaling-resolution` for
mesh sensitivity.
