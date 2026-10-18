"""Community fabric primitives for mlx2."""


class FabricError(Exception):
    """The fabric configuration or a fabric operation is invalid."""


class FabricUnavailable(FabricError):
    """An optional fabric facility is unavailable."""
