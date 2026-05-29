from __future__ import annotations

from typing import Any, TypeVar

T = TypeVar("T")


class RegistrySingletonMeta(type):
    """Metaclass for root-class factories with registered singleton subclasses."""

    _registry_owner: type[Any]
    _registry: dict[str, type[Any]]
    _instances: dict[type[Any], Any]

    def __new__(
        mcls,
        name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
    ) -> RegistrySingletonMeta:
        cls = super().__new__(mcls, name, bases, namespace)

        registry_owner = next(
            (base._registry_owner for base in bases if isinstance(base, RegistrySingletonMeta)),  # noqa: SLF001
            None,
        )

        # Initialize root class (e.g. ElementProperties)
        if registry_owner is None:
            cls._registry_owner = cls
            cls._registry = {}
            cls._instances = {}
            return cls

        # Initialize child classes (e.g. ElementPropertiesRubidium)
        cls._registry_owner = registry_owner

        try:
            species = namespace["species"]
        except KeyError:
            raise TypeError(f"{name} must define class attribute 'species'") from None

        identifier = get_identifier(species, namespace.get("tag"))
        if identifier in registry_owner._registry:  # noqa: SLF001
            raise TypeError(f"Duplicate identifier name: {identifier!r}")
        registry_owner._registry[identifier] = cls  # noqa: SLF001

        is_default = namespace.get("is_default", False)
        if is_default:
            default_identifier = get_identifier(species, "default")
            if default_identifier in registry_owner._registry:  # noqa: SLF001
                raise TypeError(f"Cannot have multiple default identifiers for the same species: {species!r}")
            registry_owner._registry[default_identifier] = cls  # noqa: SLF001

        return cls

    def __call__(cls: type[T], species: str | None = None, tag: str | None = None) -> T:
        registry_owner = cls._registry_owner  # type: ignore [attr-defined]

        # Factory call from root class: ElementProperties("Rb")
        if cls is registry_owner:
            if species is None:
                raise TypeError(f"{cls.__name__}() missing required argument: 'species'")

            identifier = get_identifier(species, tag)
            subclass = registry_owner._registry.get(identifier)  # type: ignore [attr-defined] # noqa: SLF001
            if subclass is None:
                if tag is None:
                    default_identifier = get_identifier(species, "default")
                    subclass = registry_owner._registry.get(default_identifier)  # type: ignore [attr-defined] # noqa: SLF001
                if subclass is None:
                    raise ValueError(f"Unknown identifier: {identifier!r}") from None

            return subclass()  # type: ignore [no-any-return]

        # Direct subclass call: ElementPropertiesRubidium()
        if species is not None or tag is not None:
            raise TypeError(f"{cls.__name__}() takes no arguments")

        if cls not in registry_owner._instances:  # noqa: SLF001
            registry_owner._instances[cls] = super().__call__()  # type: ignore [misc] # noqa: SLF001

        return registry_owner._instances[cls]  # type: ignore [no-any-return] # noqa: SLF001


def get_identifier(species: str, tag: str | None) -> str:
    return f"{species}:{tag}" if tag else species
