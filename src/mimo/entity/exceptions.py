"""Exceptions raised by entity and component operations."""


class ComponentError(RuntimeError):
    """Base class for errors involving entity components."""


class ComponentAlreadyAttachedError(ComponentError):
    """Raised when an attached component is attached again."""


class ComponentNotFoundError(ComponentError):
    """Raised when a requested component is not present."""