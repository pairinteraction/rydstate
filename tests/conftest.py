from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from rydstate.angular.utils import CouplingScheme

COUPLING_SCHEMES: tuple[CouplingScheme, ...] = ("LS", "JJ", "FJ")


@pytest.fixture(params=COUPLING_SCHEMES)
def coupling_scheme(request: pytest.FixtureRequest) -> CouplingScheme:
    """Parametrized fixture yielding each of the three angular coupling schemes ("LS", "JJ", "FJ").

    Any test that takes ``coupling_scheme`` as an argument is automatically run once per scheme.
    """
    return request.param  # type: ignore[no-any-return]
