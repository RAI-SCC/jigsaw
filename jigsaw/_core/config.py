"""Check that a kernel configuration fits the layer it is passed to."""

from typing import Any


def check_config(config: Any, expected: dict[str, Any]) -> None:
    """Check that ``config`` was built for a layer with the properties ``expected``.

    A mismatched config selects the wrong kernels; a missing bias, for example,
    leaves the bias without a gradient, and a config for another group size
    leaves the ranks waiting on different peers.

    Parameters
    ----------
    config : Any
        Config passed to the layer. Its ``_layer_key`` holds the properties it
        was built for.
    expected : dict[str, Any]
        The same properties, read from the layer.

    Raises
    ------
    ValueError
        If ``config`` was built for a layer with other properties.
    """
    diff = [
        f"{k}={config._layer_key[k]!r} (layer: {v!r})"
        for k, v in expected.items()
        if config._layer_key[k] != v
    ]
    if diff:
        raise ValueError(
            "config was built for a different layer: " + ", ".join(diff) + "."
        )
