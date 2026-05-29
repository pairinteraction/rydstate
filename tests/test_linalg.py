from __future__ import annotations

import numpy as np
import pytest
from rydstate.utils.linalg import find_roots


def test_find_roots_detects_roots_on_grid_samples() -> None:
    roots = find_roots(lambda x: x * (x - 0.5) * (x - 1), 0, 1, min_dx=0.3)

    assert roots == pytest.approx([0, 0.5, 1])


def test_find_roots_detects_integer_endpoint_roots() -> None:
    func = lambda nu: np.sin(np.pi * nu)  # noqa: E731
    reference_roots = [30, 31, 32, 33, 34, 35]

    roots1 = find_roots(func, 30, 35, min_dx=0.5)
    assert roots1 == pytest.approx(reference_roots)

    roots2 = find_roots(func, 30, 35, min_dx=1)
    assert roots2 == pytest.approx(reference_roots)
