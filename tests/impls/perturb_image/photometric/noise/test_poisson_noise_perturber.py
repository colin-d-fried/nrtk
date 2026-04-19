from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import pytest
from smqtk_core.configuration import configuration_test_helper
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.perturb_image.photometric.noise import PoissonNoisePerturber
from tests.impls.perturb_image.perturber_tests_mixin import PerturberTestsMixin
from tests.impls.perturb_image.photometric.noise.noise_perturber_test_utils import seed_assertions

test_rng = np.random.default_rng()


@pytest.mark.skimage
class TestPoissonNoisePerturber(PerturberTestsMixin):
    impl_class = PoissonNoisePerturber

    def test_non_deterministic_default(self) -> None:
        """Verify different results when seed=None (default)."""
        dummy_image = test_rng.integers(low=0, high=255, size=(64, 64, 3), dtype=np.uint8)
        out1, _ = PoissonNoisePerturber()(image=dummy_image)
        out2, _ = PoissonNoisePerturber()(image=dummy_image)
        assert not np.array_equal(out1, out2)

    @pytest.mark.parametrize("seed", [2])
    def test_seed_reproducibility(self, seed: int) -> None:
        """Ensure results are reproducible when explicit seed is provided."""
        seed_assertions(perturber=PoissonNoisePerturber, seed=seed)

    def test_is_static(self) -> None:
        """Verify is_static resets RNG each call."""
        dummy_image = test_rng.integers(low=0, high=255, size=(64, 64, 3), dtype=np.uint8)
        inst = PoissonNoisePerturber(seed=42, is_static=True)
        out1, _ = inst(image=dummy_image)
        out2, _ = inst(image=dummy_image)
        assert np.array_equal(out1, out2)

    @pytest.mark.parametrize(("seed", "is_static", "clip"), [(42, False, True), (None, False, False)])
    def test_configuration(self, seed: int | None, is_static: bool, clip: bool) -> None:
        """Test configuration stability."""
        inst = PoissonNoisePerturber(seed=seed, is_static=is_static, clip=clip)
        for i in configuration_test_helper(inst):
            assert i.seed == seed
            assert i.is_static == is_static
            assert i.clip == clip

    def test_output_shape_and_dtype(self) -> None:
        """Output preserves shape and dtype."""
        image = test_rng.integers(low=0, high=255, size=(32, 32, 3), dtype=np.uint8)
        out, _ = PoissonNoisePerturber(seed=0)(image=image)
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
        inst = PoissonNoisePerturber(seed=42)
        _, out_boxes = inst.perturb(image=np.ones((64, 64, 3), dtype=np.uint8), boxes=boxes)
        assert boxes == out_boxes
