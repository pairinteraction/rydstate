from __future__ import annotations

import threading
import weakref
from abc import ABCMeta
from inspect import Parameter, signature
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from inspect import Signature

CachedT = TypeVar("CachedT", bound="CachedABCMeta")


class CachedABCMeta(ABCMeta):
    """Metaclass that reuses live instances with equivalent constructor arguments."""

    # Per-class storage: declared here so type checkers know every cached class carries them,
    # but actually created per class in __init__ (see below).
    _instances: weakref.WeakValueDictionary[tuple[tuple[str, object], ...], object]
    _signature: Signature | None
    _instances_lock: threading.RLock

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, object], **kwargs: object) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        # Each cached class gets its own cache, signature and lock.
        cls._instances = weakref.WeakValueDictionary()
        cls._signature = None
        cls._instances_lock = threading.RLock()

    def clear_cached_instances(cls) -> None:
        """Clear currently cached instances of this class."""
        with cls._instances_lock:
            cls._instances.clear()

    def __call__(cls: type[CachedT], *args: object, **kwargs: object) -> CachedT:  # type: ignore [misc]
        with cls._instances_lock:
            if cls._signature is None:
                cls._signature = signature(cls.__init__)

            bound_arguments = cls._signature.bind(None, *args, **kwargs)
            bound_arguments.apply_defaults()
            constructor_arguments_list: list[tuple[str, object]] = []
            for name, value in list(bound_arguments.arguments.items())[1:]:
                parameter_kind = cls._signature.parameters[name].kind
                normalized_value = value
                if parameter_kind is Parameter.VAR_POSITIONAL:
                    normalized_value = tuple(value)
                elif parameter_kind is Parameter.VAR_KEYWORD:
                    normalized_value = tuple(sorted(value.items()))
                constructor_arguments_list.append((name, normalized_value))
            key = tuple(constructor_arguments_list)
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError(
                    f"Arguments to cached class {cls.__name__} must be hashable, but received {key!r}."
                ) from exc

            instance = cls._instances.get(key)
            if instance is None:
                instance = ABCMeta.__call__(cls, *args, **kwargs)
                cls._instances[key] = instance
            return instance  # type: ignore [return-value]
