"""Factory functions for domain-specific :class:`PybsmPerturber` presets.

The presets capture sensor and scenario parameter combinations that are typical
of common operational domains. They are intentionally opinionated starting
points, not calibrated models -- callers are expected to override individual
parameters via ``**overrides`` when tuning to a specific platform.

All factory functions accept the same ``seed`` / ``is_static`` keyword arguments
as :class:`PybsmPerturber` and forward any additional ``**overrides`` through to
the constructor, letting callers tweak specific parameters without duplicating
the whole parameter list.

Spatial domain presets:

* :func:`maritime_perturber` -- low altitude over water, high humidity
  (``ihaze=2``), moderate slant range. Intended for ship / small-boat imagery.
* :func:`automotive_perturber` -- ground-level sensor with short ground range.
  Intended for roadside / vehicle-borne cameras.
* :func:`overhead_wami_perturber` -- Wide-Area Motion Imagery-style high
  altitude collection with large ground range.

Spectral domain presets (override ``opt_trans_wavelengths``, ``qe_wavelengths``,
and ``qe`` to reflect non-visible sensors):

* :func:`ir_perturber` -- mid-wave infrared (MWIR) placeholder centered on the
  3 -- 5 um band.
* :func:`multispectral_perturber` -- coarse visible + near-infrared (VIS+NIR)
  sampling, roughly matching the 0.45 -- 0.90 um window covered by common
  earth-observation multispectral sensors.

The spectral presets are approximate: ``qe`` curves are linear interpolations
between representative endpoints rather than vendor-calibrated response curves.
"""

from __future__ import annotations

__all__ = [
    "automotive_perturber",
    "ir_perturber",
    "maritime_perturber",
    "multispectral_perturber",
    "overhead_wami_perturber",
]

from typing import Any

import numpy as np

from nrtk.impls.perturb_image.optical._pybsm_perturber import PybsmPerturber


def _build_perturber(
    *,
    seed: int | None,
    is_static: bool,
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> PybsmPerturber:
    """Merge ``defaults`` with ``overrides`` (overrides win) and instantiate."""
    params = dict(defaults)
    params.update(overrides)
    return PybsmPerturber(seed=seed, is_static=is_static, **params)


def maritime_perturber(
    *,
    seed: int | None = None,
    is_static: bool = False,
    **overrides: Any,
) -> PybsmPerturber:
    """Maritime / over-water preset.

    Rationale:
    * ``altitude=500`` m and ``ground_range=2000`` m model a low-level patrol
      sensor imaging surface targets at modest slant range.
    * ``ihaze=2`` selects a high-humidity pyBSM atmosphere typical of a
      maritime boundary layer.
    * ``target_reflectance`` / ``background_reflectance`` are nudged to reflect
      bright wakes against a darker sea surface, and the wind speed is raised.
    """
    defaults: dict[str, Any] = {
        "sensor_name": "maritime_preset",
        "scenario_name": "maritime",
        "ihaze": 2,
        "altitude": 500.0,
        "ground_range": 2000.0,
        "aircraft_speed": 60.0,
        "target_reflectance": 0.40,
        "background_reflectance": 0.05,
        "ha_wind_speed": 10.0,
    }
    return _build_perturber(seed=seed, is_static=is_static, defaults=defaults, overrides=overrides)


def automotive_perturber(
    *,
    seed: int | None = None,
    is_static: bool = False,
    **overrides: Any,
) -> PybsmPerturber:
    """Ground-level / automotive preset.

    Rationale:
    * ``altitude=2.0`` m and ``ground_range=100`` m approximate a windshield-
      or bumper-mounted camera imaging nearby objects (``100`` m is the
      smallest non-zero ``ground_range`` pyBSM supports).
    * ``aircraft_speed=20`` m/s (~72 km/h) models motion blur for a moving
      vehicle.
    * ``ihaze=1`` keeps a standard clear atmosphere; callers can override to
      simulate fog or rain.
    """
    defaults: dict[str, Any] = {
        "sensor_name": "automotive_preset",
        "scenario_name": "automotive",
        "ihaze": 1,
        "altitude": 2.0,
        "ground_range": 100.0,
        "aircraft_speed": 20.0,
        "int_time": 1.0 / 60.0,
    }
    return _build_perturber(seed=seed, is_static=is_static, defaults=defaults, overrides=overrides)


def overhead_wami_perturber(
    *,
    seed: int | None = None,
    is_static: bool = False,
    **overrides: Any,
) -> PybsmPerturber:
    """Wide-Area Motion Imagery (WAMI) / high-altitude overhead preset.

    Rationale:
    * ``altitude=20_000`` m and ``ground_range=30_000`` m approximate a
      stratospheric overhead collect with a large slant range.
    * ``ihaze=1`` assumes a standard (non-maritime) atmosphere.
    * Aperture ``D=400e-3`` and focal length ``f=8`` are characteristic of a
      long-focal-length WAMI objective.
    """
    defaults: dict[str, Any] = {
        "sensor_name": "overhead_wami_preset",
        "scenario_name": "overhead_wami",
        "ihaze": 1,
        "altitude": 20_000.0,
        "ground_range": 30_000.0,
        "D": 400e-3,
        "f": 8.0,
        "aircraft_speed": 120.0,
    }
    return _build_perturber(seed=seed, is_static=is_static, defaults=defaults, overrides=overrides)


# ---------------------------------------------------------------------------
# Spectral / modality presets
# ---------------------------------------------------------------------------

# Approximate MWIR band edges (3 -- 5 um) in meters. Represented as a coarse
# sampling; pyBSM performs spectral integration between these wavelengths.
_MWIR_WAVELENGTHS: np.ndarray = np.linspace(3.0e-6, 5.0e-6, num=5)

# Indicative HgCdTe / InSb MWIR quantum-efficiency curve: roughly flat across
# the band with gentle roll-off at the edges. Values are illustrative and
# should be replaced with vendor-calibrated curves for quantitative work.
_MWIR_QE: np.ndarray = np.array([0.55, 0.70, 0.75, 0.70, 0.55])


def ir_perturber(
    *,
    seed: int | None = None,
    is_static: bool = False,
    **overrides: Any,
) -> PybsmPerturber:
    """Mid-wave infrared (MWIR, ~3 -- 5 um) placeholder preset.

    The optical transmission band, quantum-efficiency wavelengths, and quantum
    efficiencies are set to coarsely sample the 3 -- 5 um window. ``eta`` is
    raised to ``0.1`` to emulate a central obscuration typical of reflective
    optics, and a small dark current and read noise are added since MWIR focal
    planes are noisier than visible CMOS.

    Intended as a scaffold for IR-domain experiments; callers should supply
    their sensor-specific QE curve via ``**overrides`` for quantitative use.
    """
    defaults: dict[str, Any] = {
        "sensor_name": "ir_mwir_preset",
        "scenario_name": "ir",
        "opt_trans_wavelengths": _MWIR_WAVELENGTHS,
        "qe_wavelengths": _MWIR_WAVELENGTHS,
        "qe": _MWIR_QE,
        "eta": 0.1,
        "dark_current": 1e-9,
        "read_noise": 30.0,
    }
    return _build_perturber(seed=seed, is_static=is_static, defaults=defaults, overrides=overrides)


# Coarse VIS + NIR sampling roughly matching common multispectral
# earth-observation bands (e.g. Blue / Green / Red / NIR).
_MULTISPECTRAL_WAVELENGTHS: np.ndarray = np.array(
    [0.45e-6, 0.55e-6, 0.65e-6, 0.75e-6, 0.85e-6],
)

# Indicative silicon CCD / CMOS quantum efficiency: peaks in the visible and
# rolls off into the NIR. Values are illustrative.
_MULTISPECTRAL_QE: np.ndarray = np.array([0.35, 0.55, 0.60, 0.45, 0.25])


def multispectral_perturber(
    *,
    seed: int | None = None,
    is_static: bool = False,
    **overrides: Any,
) -> PybsmPerturber:
    """Visible + near-infrared multispectral preset (~0.45 -- 0.85 um).

    Optical transmission wavelengths and QE wavelengths span a coarse VIS+NIR
    grid; the QE curve is an illustrative silicon-detector response that peaks
    in the visible and tapers toward the NIR.

    As with :func:`ir_perturber`, this preset is a scaffold for experiments and
    should be replaced by a vendor-calibrated response curve before drawing
    quantitative conclusions.
    """
    defaults: dict[str, Any] = {
        "sensor_name": "multispectral_preset",
        "scenario_name": "multispectral",
        "opt_trans_wavelengths": _MULTISPECTRAL_WAVELENGTHS,
        "qe_wavelengths": _MULTISPECTRAL_WAVELENGTHS,
        "qe": _MULTISPECTRAL_QE,
    }
    return _build_perturber(seed=seed, is_static=is_static, defaults=defaults, overrides=overrides)
