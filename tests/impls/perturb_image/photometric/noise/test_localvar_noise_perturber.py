from __future__ import annotations

from collections.abc import Hashable, Iterable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.perturb_image.photometric.noise import LocalvarNoisePerturber
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.photometric.noise.noise_perturber_test_utils import seed_assertions

test_rng = np.random.default_rng()


@pytest.mark.skimage
class TestLocalvarNoisePerturber(PerturberTestsMixin):
    impl_class = LocalvarNoisePerturber

    def test_non_deterministic_default(self) -> None:
        """Verify different results when seed=None (default)."""
        dummy_image = test_rng.integers(low=0, high=255, size=(64, 64, 3), dtype=np.uint8)
        out1, _ = LocalvarNoisePerturber(var=0.05)(image=dummy_image)
        out2, _ = LocalvarNoisePerturber(var=0.05)(image=dummy_image)
        assert not np.array_equal(out1, out2)

    @pytest.mark.parametrize("seed", [2])
    def test_seed_reproducibility(self, seed: int) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        seed_assertions(perturber=LocalvarNoisePerturber, seed=seed)

    def test_is_static(self) -> None:
        """Verify is_static resets RNG each call."""
        dummy_image = test_rng.integers(low=0, high=255, size=(64, 64, 3), dtype=np.uint8)
        inst = LocalvarNoisePerturber(seed=42, is_static=True, var=0.05)
        out1, _ = inst(image=dummy_image)
        out2, _ = inst(image=dummy_image)
        assert np.array_equal(out1, out2)

    @pytest.mark.parametrize(
        ("seed", "is_static", "var", "clip"),
        [(42, False, 0.1, True), (None, False, 0.0, False)],
    )
    def test_configuration(self, seed: int | None, is_static: bool, var: float, clip: bool) -> None:
        """Test configuration stability."""
        inst = LocalvarNoisePerturber(seed=seed, is_static=is_static, var=var, clip=clip)
        for i in configuration_test_helper(inst):
            assert i.seed == seed
            assert i.is_static == is_static
            assert i.var == var
            assert i.clip == clip
            # ndarray-valued local_vars is intentionally not serialized.
            assert i.local_vars is None

    @pytest.mark.parametrize(
        ("kwargs", "expectation"),
        [
            ({"var": 0.5}, does_not_raise()),
            ({"var": 0.0}, does_not_raise()),
            ({"var": -1.0}, pytest.raises(ValueError, match=r"LocalvarNoisePerturber invalid var")),
            (
                {"local_vars": np.zeros((32, 32, 3), dtype=np.float64)},
                pytest.raises(ValueError, match=r"strictly positive"),
            ),
        ],
    )
    def test_configuration_bounds(
        self,
        kwargs: dict[str, Any],
        expectation: AbstractContextManager,
    ) -> None:
        """Validate guard conditions on constructor parameters."""
        with expectation:
            LocalvarNoisePerturber(**kwargs)

    def test_explicit_local_vars_shape_mismatch(self) -> None:
        """Mismatched local_vars shape raises at perturb time."""
        local_vars = np.full((16, 16, 3), 0.01, dtype=np.float64)
        inst = LocalvarNoisePerturber(seed=0, local_vars=local_vars)
        with pytest.raises(ValueError, match=r"does not match image shape"):
            inst(image=np.ones((32, 32, 3), dtype=np.uint8))

    def test_explicit_local_vars_runs(self) -> None:
        """Explicit per-pixel variance map produces a valid output."""
        shape = (32, 32, 3)
        local_vars = test_rng.uniform(low=0.001, high=0.05, size=shape).astype(np.float64)
        image = test_rng.integers(low=0, high=255, size=shape, dtype=np.uint8)
        inst = LocalvarNoisePerturber(seed=0, local_vars=local_vars)
        out, _ = inst(image=image)
        assert out.shape == image.shape
        assert out.dtype == image.dtype

    def test_output_shape_and_dtype(self) -> None:
        """Output preserves shape and dtype via the scalar ``var`` fallback."""
        image = test_rng.integers(low=0, high=255, size=(32, 32, 3), dtype=np.uint8)
        out, _ = LocalvarNoisePerturber(seed=0, var=0.01)(image=image)
        assert out.shape == image.shape
        assert out.dtype == image.dtype

    @pytest.mark.parametrize(
        "boxes",
        [
            None,
            [(AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0})],
            [
                (AxisAlignedBoundingBox(min_vertex=(0, 0), max_vertex=(1, 1)), {"test": 0.0}),
                (AxisAlignedBoundingBox(min_vertex=(2, 2), max_vertex=(3, 3)), {"test2": 1.0}),
            ],
        ],
    )
    def test_perturb_with_boxes(
        self,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
    ) -> None:
        """Test that bounding boxes do not change during perturb."""
        inst = LocalvarNoisePerturber(seed=42, var=0.05)
        _, out_boxes = inst.perturb(image=np.ones((64, 64, 3), dtype=np.uint8), boxes=boxes)
        assert boxes == out_boxes
