"""Helpers for global boundary-mode switches."""


def _unwrap_scalar(value):
    """Best-effort scalar extraction for bool-like config values."""
    data = value._data if hasattr(value, "_data") else value
    if hasattr(data, "item"):
        try:
            return data.item()
        except Exception:
            pass
    return data


def is_zero_flux_enabled(sim):
    """Return True if global closed-box transport mode is enabled."""
    ini = getattr(sim, "ini", None)
    if ini is None:
        return False
    boundary = getattr(ini, "boundary", None)
    if boundary is None:
        return False
    return bool(_unwrap_scalar(getattr(boundary, "zeroFlux", False)))
