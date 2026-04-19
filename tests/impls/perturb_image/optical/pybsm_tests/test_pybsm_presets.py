"""Unit tests for the domain / modality presets in :mod:`_pybsm_presets`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from nrtk.impls.perturb_image.optical._pybsm_perturber import PybsmPerturber
from nrtk.impls.perturb_image.optical._pybsm_presets import (
    automotive_perturber,
    ir_perturber,
    maritime_perturber,
    multispectral_perturber,
    overhead_wami_perturber,
)

PresetFactory = Callable[..., PybsmPerturber]

_SPATIAL_FACTORIES: list[tuple[str, PresetFactory]] = [
    ("maritime", maritime_perturber),
    ("automotive", automotive_perturber),
    ("overhead_wami", overhead_wami_perturber),
]

_SPECTRAL_FACTORIES: list[tuple[str, PresetFactory]] = [
    ("ir", ir_perturber),
    ("multispectral", multispectral_perturber),
]


@pytest.mark.pybsm
class TestSpatialDomainPresets:
    @pytest.mark.parametrize(("name", "factory"), _SPATIAL_FACTORIES)
    def test_instantiates(self, name: str, factory: PresetFactory) -> None:  # noqa: ARG002
        inst = factory(seed=7)
        assert isinstance(inst, PybsmPerturber)
        # ``seed`` should be propagated through to the base perturber.
        assert inst.seed == 7

    def test_maritime_high_humidity_and_low_altitude(self) -> None:
        inst = maritime_perturber()
        cfg = inst.get_config()
        assert cfg["ihaze"] == 2
        assert cfg["altitude"] < 1000.0
        assert cfg["scenario_name"] == "maritime"

    def test_automotive_ground_level(self) -> None:
        inst = automotive_perturber()
        cfg = inst.get_config()
        assert cfg["altitude"] < 10.0
        assert cfg["ground_range"] <= 500.0
        assert cfg["scenario_name"] == "automotive"

    def test_overhead_wami_long_range(self) -> None:
        inst = overhead_wami_perturber()
        cfg = inst.get_config()
        assert cfg["altitude"] >= 10_000.0
        assert cfg["ground_range"] >= 10_000.0
        assert cfg["scenario_name"] == "overhead_wami"

    def test_overrides_take_precedence(self) -> None:
        """User-supplied ``**overrides`` must replace preset defaults."""
        inst = maritime_perturber(ihaze=1, altitude=1000.0)
        cfg = inst.get_config()
        assert cfg["ihaze"] == 1
        assert cfg["altitude"] == 1000.0


@pytest.mark.pybsm
class TestSpectralDomainPresets:
    @pytest.mark.parametrize(("name", "factory"), _SPECTRAL_FACTORIES)
    def test_instantiates(self, name: str, factory: PresetFactory) -> None:  # noqa: ARG002
        inst = factory(seed=3)
        assert isinstance(inst, PybsmPerturber)
        assert inst.seed == 3

    def test_ir_uses_mwir_band(self) -> None:
        inst = ir_perturber()
        cfg = inst.get_config()
        wavelengths = np.asarray(cfg["opt_trans_wavelengths"])
        qe = np.asarray(cfg["qe"])
        qe_wavelengths = np.asarray(cfg["qe_wavelengths"])
        assert wavelengths.min() >= 3.0e-6
        assert wavelengths.max() <= 5.0e-6
        assert qe.shape == qe_wavelengths.shape
        assert np.all(qe > 0.0)
        assert np.all(qe <= 1.0)

    def test_multispectral_spans_vis_and_nir(self) -> None:
        inst = multispectral_perturber()
        cfg = inst.get_config()
        wavelengths = np.asarray(cfg["opt_trans_wavelengths"])
        qe = np.asarray(cfg["qe"])
        qe_wavelengths = np.asarray(cfg["qe_wavelengths"])
        assert wavelengths.min() <= 0.5e-6  # covers blue
        assert wavelengths.max() >= 0.8e-6  # covers NIR
        assert qe.shape == qe_wavelengths.shape
        assert np.all(qe > 0.0)
        assert np.all(qe <= 1.0)

    def test_spectral_overrides(self) -> None:
        """Custom QE curves via ``**overrides`` must be respected."""
        custom_wavelengths = np.array([4.0e-6, 4.5e-6, 5.0e-6])
        custom_qe = np.array([0.5, 0.6, 0.55])
        inst = ir_perturber(
            opt_trans_wavelengths=custom_wavelengths,
            qe_wavelengths=custom_wavelengths,
            qe=custom_qe,
        )
        cfg = inst.get_config()
        np.testing.assert_allclose(cfg["opt_trans_wavelengths"], custom_wavelengths)
        np.testing.assert_allclose(cfg["qe"], custom_qe)
