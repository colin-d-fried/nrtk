Phase 3 domain and modality PybsmPerturber presets
---------------------------------------------------

Additions
~~~~~~~~~

* Added ``nrtk.impls.perturb_image.optical._pybsm_presets`` with opinionated
  factory functions returning pre-configured :class:`PybsmPerturber` instances
  for common operational domains:

  - ``maritime_perturber``: low altitude, high-humidity atmosphere
    (``ihaze=2``), moderate slant range.
  - ``automotive_perturber``: ground-level sensor, short ground range, short
    integration time for motion-blur realism.
  - ``overhead_wami_perturber``: stratospheric-altitude WAMI-style collection
    with large ground range and long-focal-length optics.

* Added spectral / modality presets in the same module:

  - ``ir_perturber``: mid-wave infrared (3 -- 5 um) placeholder with a coarse
    MWIR-shaped QE curve, elevated dark current, and nominal read noise.
  - ``multispectral_perturber``: visible + near-infrared sampling
    (~0.45 -- 0.85 um) with an indicative silicon-detector QE curve.

  Both spectral presets document their assumptions and are intended as
  scaffolds -- callers should override the QE curve with vendor-calibrated
  values for quantitative work.

* All factories forward ``**overrides`` through to :class:`PybsmPerturber`,
  so callers can tweak individual parameters without duplicating the full
  default set.

* Added unit tests covering preset instantiation, preset-specific invariants
  (e.g., maritime ``ihaze == 2``, spectral band coverage), and override
  precedence.
