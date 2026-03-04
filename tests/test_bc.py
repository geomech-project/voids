from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from voids.linalg.bc import apply_dirichlet_rowcol
from voids.physics.singlephase import PressureBC, _make_dirichlet_vector


def test_apply_dirichlet_rowcol_simple():
    """Test Dirichlet elimination on a two-node linear system."""

    A = sparse.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    b = np.array([0.0, 0.0])
    values = np.array([1.0, 0.0])
    mask = np.array([True, False])
    A2, b2 = apply_dirichlet_rowcol(A, b, values=values, mask=mask)
    x = np.linalg.solve(A2.toarray(), b2)
    assert np.allclose(x, [1.0, 0.5])


def test_bc_overlap_raises(line_network):
    """Test that overlapping inlet and outlet labels are rejected."""

    line_network.pore_labels["same"] = np.array([True, False, True])
    with pytest.raises(ValueError, match="overlap"):
        _make_dirichlet_vector(line_network, PressureBC("same", "same", 1.0, 0.0))
