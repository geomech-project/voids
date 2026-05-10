# Image Processing

The `voids.image` sub-package provides utilities for segmented image processing,
connectivity analysis, and pore network extraction used in vug sensitivity studies.

---

## Maximal-Ball Extraction

::: voids.image.maximal_ball

---

## PREGO Extraction

::: voids.image.prego

---

## Porosity Maps

For conceptual background, `block_shape` interpretation, and synthetic
verification context, see [Porosity Maps](../porosity_maps.md).

::: voids.image.porosity

---

## Morphometry

The morphometry helpers compute local-thickness diameter maps for binary 2D/3D
phase images. This API operates on an explicit phase mask supplied by the
caller, requires isotropic voxel spacing, returns diameter-valued maps in the
requested physical units, and summarizes values over the selected phase only.

For the scientific definition, radius-to-diameter conversion, backend method
choices, and verification notes, see
[Local Thickness Morphometry](../local_thickness_morphometry.md).

::: voids.image.morphometry

---

## Network Extraction

::: voids.image.network_extraction

---

## Segmentation

::: voids.image.segmentation

---

## Image Connectivity

::: voids.image.connectivity
