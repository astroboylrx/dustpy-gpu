"""Private backend-neutral helpers for dust evolution."""

from types import SimpleNamespace

from dustpy.utils.backend import to_numpy
from dustpy.utils.boundary_modes import is_dust_inner_outflow_only_enabled
from simframe.backends.api import xp
from simframe.frame import field_data as _field_data


_GAS_FLOOR_FREEZE_CONFIG = {
    "enabled": True,
    "ratio": 1.0e3,
}
_DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG = {
    "enabled": False,
    "ratio": 1.0,
}
_RUNTIME_STATES = {}
_ACTIVE_RUNTIME_TOKEN = None

_RUNTIME_STATE_VARS = (
    "_GAS_FLOOR_FREEZE_CONFIG",
    "_DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG",
)


def _fresh_runtime_state():
    return {
        "_GAS_FLOOR_FREEZE_CONFIG": {
            "enabled": True,
            "ratio": 1.0e3,
        },
        "_DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG": {
            "enabled": False,
            "ratio": 1.0,
        },
    }


def _capture_runtime_state():
    return {name: globals()[name] for name in _RUNTIME_STATE_VARS}


def _restore_runtime_state(state):
    for name, value in state.items():
        globals()[name] = value


def switch_runtime_state(runtime_token):
    global _ACTIVE_RUNTIME_TOKEN
    if runtime_token is None or runtime_token == _ACTIVE_RUNTIME_TOKEN:
        return
    if _ACTIVE_RUNTIME_TOKEN is not None:
        _RUNTIME_STATES[_ACTIVE_RUNTIME_TOKEN] = _capture_runtime_state()
    state = _RUNTIME_STATES.get(runtime_token)
    if state is None:
        state = _fresh_runtime_state()
        _RUNTIME_STATES[runtime_token] = state
    _restore_runtime_state(state)
    _ACTIVE_RUNTIME_TOKEN = runtime_token


def configure_runtime(gas_floor_freeze_config, dust_transport_inactive_floor_config):
    global _GAS_FLOOR_FREEZE_CONFIG, _DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG
    _GAS_FLOOR_FREEZE_CONFIG = gas_floor_freeze_config
    _DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG = dust_transport_inactive_floor_config


def _gas_floor_freeze_mask_from_arrays(SigmaGas, SigmaGasFloor):
    cfg = _GAS_FLOOR_FREEZE_CONFIG
    if not cfg.get("enabled", True):
        return xp.zeros(SigmaGas.shape, dtype=bool)
    ratio = float(cfg.get("ratio", 0.0))
    if ratio <= 0.0:
        return xp.zeros(SigmaGas.shape, dtype=bool)
    return xp.asarray(SigmaGas <= ratio * SigmaGasFloor, dtype=bool)


def _gas_floor_freeze_mask(sim, SigmaGas=None):
    SigmaGas = _field_data(sim.gas.Sigma if SigmaGas is None else SigmaGas)
    SigmaGasFloor = _field_data(sim.gas.SigmaFloor)
    return _gas_floor_freeze_mask_from_arrays(SigmaGas, SigmaGasFloor)


def _gas_floor_freeze_interface_mask(sim, SigmaGas=None):
    radial_mask = _gas_floor_freeze_mask(sim, SigmaGas=SigmaGas)
    iface_mask = xp.zeros((int(radial_mask.shape[0]) + 1,), dtype=bool)
    iface_mask[:-1] = iface_mask[:-1] | radial_mask
    iface_mask[1:] = iface_mask[1:] | radial_mask
    return iface_mask


def _dust_transport_inactive_mask_from_arrays(SigmaDust, SigmaDustFloor):
    cfg = _DUST_TRANSPORT_INACTIVE_FLOOR_CONFIG
    if not cfg.get("enabled", False):
        return xp.zeros(SigmaDust.shape, dtype=bool)
    ratio = float(cfg.get("ratio", 0.0))
    if ratio <= 0.0:
        return xp.zeros(SigmaDust.shape, dtype=bool)
    return xp.asarray(SigmaDust <= ratio * SigmaDustFloor, dtype=bool)


def _dust_transport_inactive_mask(sim, SigmaDust=None):
    SigmaDust = _field_data(sim.dust.Sigma if SigmaDust is None else SigmaDust)
    SigmaDustFloor = _field_data(sim.dust.SigmaFloor)
    return _dust_transport_inactive_mask_from_arrays(SigmaDust, SigmaDustFloor)


def _interface_drain_mask_from_flux(Fi, inactive):
    Fi = _field_data(Fi)
    inactive = xp.asarray(inactive, dtype=bool)
    iface_mask = xp.zeros(Fi.shape, dtype=bool)
    if Fi.shape[0] != inactive.shape[0] + 1 or Fi.shape[1] != inactive.shape[1]:
        return iface_mask
    iface_mask[1:-1, :] = (
        ((Fi[1:-1, :] > 0.0) & inactive[:-1, :])
        | ((Fi[1:-1, :] < 0.0) & inactive[1:, :])
    )
    iface_mask[0, :] = (Fi[0, :] < 0.0) & inactive[0, :]
    iface_mask[-1, :] = (Fi[-1, :] > 0.0) & inactive[-1, :]
    return iface_mask


def _dust_transport_drain_interface_mask(sim, SigmaDust=None, Fi_adv=None, Fi_diff=None):
    inactive = _dust_transport_inactive_mask(sim, SigmaDust=SigmaDust)
    if int(to_numpy(inactive.sum())) == 0:
        nr = int(inactive.shape[0]) + 1
        nm = int(inactive.shape[1])
        return xp.zeros((nr, nm), dtype=bool)
    iface_mask = xp.zeros((int(inactive.shape[0]) + 1, int(inactive.shape[1])), dtype=bool)
    if Fi_adv is not None:
        iface_mask |= _interface_drain_mask_from_flux(Fi_adv, inactive)
    if Fi_diff is not None:
        iface_mask |= _interface_drain_mask_from_flux(Fi_diff, inactive)
    return iface_mask


def _transport_velocity_freeze_interface_mask(sim, SigmaDust=None, SigmaGas=None, Fi_adv=None):
    gas_mask = _gas_floor_freeze_interface_mask(sim, SigmaGas=SigmaGas)[:, None]
    dust_mask = _dust_transport_drain_interface_mask(sim, SigmaDust=SigmaDust, Fi_adv=Fi_adv, Fi_diff=None)
    return gas_mask | dust_mask


def _transport_diffusion_freeze_interface_mask(sim, SigmaDust=None, SigmaGas=None):
    gas_mask = _gas_floor_freeze_interface_mask(sim, SigmaGas=SigmaGas)[:, None]
    return gas_mask


def _inner_boundary_advective_drain_mask(sim, SigmaDust=None, Fi_adv=None):
    inactive = _dust_transport_inactive_mask(sim, SigmaDust=SigmaDust)
    if inactive.ndim != 2 or inactive.shape[0] == 0:
        return None
    inner_inactive = xp.asarray(inactive[0, :], dtype=bool)
    if int(to_numpy(inner_inactive.sum())) == 0:
        return None
    if Fi_adv is not None:
        fi_adv = _field_data(Fi_adv)
        if fi_adv.shape[0] > 1:
            adv_mask = inner_inactive & xp.asarray(fi_adv[1, :] > 0.0, dtype=bool)
            if int(to_numpy(adv_mask.sum())) > 0:
                return adv_mask
    return None


def _apply_gas_floor_freeze_to_radial_field(sim, arr, SigmaGas=None):
    mask = _gas_floor_freeze_mask(sim, SigmaGas=SigmaGas)
    if int(to_numpy(mask.sum())) == 0:
        return arr
    arr = _field_data(arr)
    arr[mask, ...] = 0.0
    return arr


def _apply_gas_floor_freeze_to_flux(sim, Fi, SigmaGas=None):
    iface_mask = _gas_floor_freeze_interface_mask(sim, SigmaGas=SigmaGas)
    if int(to_numpy(iface_mask.sum())) == 0:
        return Fi
    Fi = _field_data(Fi)
    Fi[iface_mask, ...] = 0.0
    return Fi


def _ensure_dust_floor_topup_state(dust):
    state = getattr(dust, "floor_topup", None)
    if state is None:
        state = SimpleNamespace(total_mearth=0.0, last_mearth=0.0, count=0)
        dust.floor_topup = state
    return state


def _ensure_dust_floor_retry_state(dust):
    state = getattr(dust, "floor_retry", None)
    if state is None:
        state = SimpleNamespace(total_mearth=0.0, last_mearth=0.0, count=0, step_count=0, giveup_count=0, giveup_last_mearth=0.0)
        dust.floor_retry = state
    return state


def _record_implicit_floor_retry(owner, added_mearth):
    state = _ensure_dust_floor_retry_state(owner.dust)
    state.total_mearth += float(added_mearth)
    state.last_mearth = float(added_mearth)
    state.count += 1
    state.step_count += 1


def _record_implicit_floor_retry_exhausted(owner, added_mearth):
    state = _ensure_dust_floor_retry_state(owner.dust)
    state.giveup_count += 1
    state.giveup_last_mearth = float(added_mearth)


def _inner_diode_block_mask(sim):
    """Return per-mass-bin mask for inner-edge outward drift (potential artificial feed)."""
    if not is_dust_inner_outflow_only_enabled(sim):
        return None
    block = None
    try:
        v0 = _field_data(sim.dust.v.rad)[0, :]
        block = xp.asarray(v0 > 0.0, dtype=bool)
    except Exception:
        block = None
    inactive = None
    try:
        inactive = _dust_transport_inactive_mask(sim)[0, :]
        inactive = xp.asarray(inactive, dtype=bool)
    except Exception:
        inactive = None
    floor_mask = None
    negative_vb_mask = None
    try:
        _refresh_dust_boundary_views(sim)
        inner = getattr(sim.dust.boundary, "inner", None)
        if inner is not None:
            vb = inner._getboundary()
            if vb is not None:
                negative_vb_mask = xp.asarray(vb < 0.0, dtype=bool)
    except Exception:
        negative_vb_mask = None
    merged = None
    for mask in (block, inactive, floor_mask, negative_vb_mask):
        if mask is None:
            continue
        merged = mask if merged is None else (merged | mask)
    return merged


def _refresh_dust_boundary_views(sim):
    """Refresh boundary helper views so boundary formulas use the current dust field."""
    inner = getattr(sim.dust.boundary, "inner", None)
    if inner is not None:
        inner._r = sim.grid.r[:3]
        inner._ri = sim.grid.ri[:3]
        inner._S = sim.dust.Sigma[:3]

    outer = getattr(sim.dust.boundary, "outer", None)
    if outer is not None:
        outer._r = sim.grid.r[::-1][:3]
        outer._ri = sim.grid.ri[::-1][:3]
        outer._S = sim.dust.Sigma[::-1][:3]


def _apply_inner_boundary_selective(sim, block_mask=None):
    """Apply the configured inner boundary condition only for bins not blocked by the diode."""
    _refresh_dust_boundary_views(sim)
    boundary = getattr(sim.dust.boundary, "inner", None)
    if boundary is None:
        return

    vb = boundary._getboundary()
    if vb is None:
        return

    sigma0 = sim.dust.Sigma[0]
    if block_mask is None:
        sigma0[...] = vb
        return

    mask = xp.asarray(block_mask, dtype=bool)
    if sigma0.ndim != 1 or mask.ndim != 1 or sigma0.shape[0] != mask.shape[0]:
        sigma0[...] = vb
        return

    open_mask = ~mask
    if bool(to_numpy(xp.any(open_mask))):
        sigma0[open_mask] = vb[open_mask]
