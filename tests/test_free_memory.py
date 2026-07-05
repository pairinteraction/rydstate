# ruff: noqa: SLF001

from __future__ import annotations

import contextlib
import gc
import weakref
from typing import TYPE_CHECKING

from rydstate import RydbergStateSQDTAlkali
from rydstate.basis import BasisMQDT
from rydstate.radial import RadialDummy, RadialKet

if TYPE_CHECKING:
    from rydstate.radial.radial_base import Radial


def _cache_key(radial: Radial) -> tuple[tuple[str, object], ...]:
    """Return the key under which the given radial ket is stored in RadialKet._instances."""
    with contextlib.suppress(StopIteration):
        return next(key for key, value in RadialKet._instances.items() if value is radial)
    with contextlib.suppress(StopIteration):
        return next(key for key, value in RadialDummy._instances.items() if value is radial)
    raise ValueError("Radial ket is not cached in RadialKet._instances or RadialDummy._instances")


def test_free_memory_releases_radial_kets_sqdt() -> None:
    RadialKet.clear_cached_instances()

    state = RydbergStateSQDTAlkali("Rb", n=50, l=0, j=0.5)

    # access radial and angular kets to ensure they are created (and the radial ket is cached)
    radial_refs: list[weakref.ref[Radial]] = []
    cache_keys = []
    for ket in state.rydberg_kets:
        radial, _angular = ket.radial, ket.angular
        radial_refs.append(weakref.ref(radial))
        cache_keys.append(_cache_key(radial))
    del radial, _angular, ket

    assert all(ref() is not None for ref in radial_refs)
    assert all(key in RadialKet._instances for key in cache_keys)

    state._free_memory()
    gc.collect()

    # the radial kets are no longer referenced and have been evicted from the cache
    assert all(ref() is None for ref in radial_refs)
    assert all(key not in RadialKet._instances for key in cache_keys)


def test_free_memory_releases_radial_kets_mqdt() -> None:
    RadialKet.clear_cached_instances()

    basis = BasisMQDT("Yb174", nu=(50, 52), l_r=(0, 2), m=(0, 0))
    # pick a genuine multi-channel state so several radial kets are involved
    state = max(basis.states, key=lambda s: len(s.rydberg_kets))
    assert len(state.rydberg_kets) > 1

    # access radial and angular kets to ensure they are created (and the radial kets are cached)
    radial_refs: list[weakref.ref[Radial]] = []
    cache_keys = []
    for ket in state.rydberg_kets:
        radial, _angular = ket.radial, ket.angular
        radial_refs.append(weakref.ref(radial))
        cache_keys.append(_cache_key(radial))
    del radial, _angular, ket

    assert all(ref() is not None for ref in radial_refs)
    assert all(key in RadialKet._instances or key in RadialDummy._instances for key in cache_keys)

    # drop the basis so sibling states do not keep the (potentially shared) radial kets alive
    del basis
    state._free_memory()
    gc.collect()

    # the radial kets are no longer referenced and have been evicted from the cache
    assert all(ref() is None for ref in radial_refs)
    assert all(key not in RadialKet._instances and key not in RadialDummy._instances for key in cache_keys)
