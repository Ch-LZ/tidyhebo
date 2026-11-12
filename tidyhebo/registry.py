from typing import ClassVar, TypeVar, cast

R = TypeVar("R", bound="Registrable")


class Registrable:
    """Mixin providing named registration and construction from a config.

    Each direct subclass starts an independent registry. Its descendants
    share that registry and register themselves with _target_="name".
    Subclasses without a _target_ are not registered.

    from_config() selects a class by config._target_ and calls its constructor
    with config=config and the supplied dependencies. Registered constructors
    must accept this calling convention. Configs are passed without unpacking.

    Registration happens when the implementation's module is imported.
    Duplicate targets within one family and inheritance across distinct
    registry families are rejected. This mixin does not import plugins.
    """

    # Shared storage: family base class -> {implementation name: class}.
    _registries: ClassVar[dict[type, dict[str, type]]] = {}
    # Points to the base class of this family. The annotation alone does not
    # create an attribute; its value is assigned in __init_subclass__.
    _registry_root: ClassVar[type]

    def __init_subclass__(cls, *, _target_: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        # Find the families of the immediate parent classes.
        # cls.__bases__ contains only direct subclasses of a cls
        roots = {base._registry_root for base in cls.__bases__ if hasattr(base, "_registry_root")}
        # Multiple parents from the same family are allowed. Parents from
        # different families would make registry ownership ambiguous.
        if len(roots) > 1:
            raise TypeError("Cannot inherit from different registry families")

        # Inherit the existing family. If none exists, this class becomes
        # the root of a new family with its own registry.
        cls._registry_root = next(iter(roots), cls)

        registry = Registrable._registries.setdefault(cls._registry_root, {})
        # Without a _target_, the class only inherits the mechanism. This suits
        # interfaces and intermediate classes that need no name-based lookup.
        if _target_ is not None:
            if not isinstance(_target_, str) or not _target_:
                raise ValueError("_target_ must be a nonempty string")
            if _target_ in registry:
                raise ValueError(f"{cls._registry_root.__name__}: _target_ already registered: {_target_}")
            registry[_target_] = cls

    @classmethod
    def from_config(cls: type[R], config, **dependencies) -> R:
        """Construct a registered implementation using config and explicit dependencies."""
        registry = Registrable._registries[cls._registry_root]
        try:
            implementation = registry[config._target_]
        except KeyError:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown {cls.__name__} _target_: {config._target_!r}. Available: {available}"
            ) from None

        # The method may also be called on a concrete subclass. In that case,
        # returning a sibling implementation from the same family is invalid:
        # the result must be an instance of cls or one of its descendants.
        # So, this check is places to guarantee, that returned is of type R
        if not issubclass(implementation, cls):
            raise TypeError(f"Target {config._target_!r} is not a subclass of {cls.__name__}")
        # Pass the config intact and dependencies as separate keyword arguments.
        # Keep construction outside try so a KeyError inside the constructor
        # is not reported as an unknown _target_.
        # cast only informs the type checker; at runtime it returns the same
        # object without conversion or copying.
        return cast(R, implementation(config=config, **dependencies))
