"""Tests for the optional ``masks`` / :meth:`perturb_with_masks` support on geometric perturbers."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
import pytest
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.perturb_image.geometric.random import (
    RandomCropPerturber,
    RandomTranslationPerturber,
)
from nrtk.impls.perturb_image.photometric._noise.gaussian_noise_perturber import (
    GaussianNoisePerturber,
)
from nrtk.interfaces import PerturbImage

BoxesT = Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]


class _NopPerturber(PerturbImage):
    """Minimal concrete perturber for exercising the base ``perturb_with_masks`` default impl."""

    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: BoxesT | None = None,  # noqa: ARG002
        masks: np.ndarray[Any, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[np.ndarray[Any, Any], BoxesT | None]:
        return image.copy(), None


class TestPerturbImageMasksBase:
    """Default base-class behavior when a subclass does not override ``perturb_with_masks``."""

    def test_masks_pass_through_when_none(self) -> None:
        img = np.ones((4, 4), dtype=np.uint8)
        perturbed_img, _, perturbed_masks = _NopPerturber().perturb_with_masks(image=img)
        assert perturbed_img.shape == img.shape
        assert perturbed_masks is None

    def test_masks_pass_through_unchanged(self) -> None:
        img = np.ones((4, 4), dtype=np.uint8)
        masks = np.array([[0, 1, 0, 1]] * 4, dtype=np.int32)
        _, _, perturbed_masks = _NopPerturber().perturb_with_masks(image=img, masks=masks)
        assert perturbed_masks is not None
        np.testing.assert_array_equal(perturbed_masks, masks)
        # Must be a copy, not the same array
        assert perturbed_masks is not masks


class TestGaussianNoisePerturberMasks:
    """Photometric perturbers inherit the base pass-through behavior for masks."""

    def test_masks_unchanged(self) -> None:
        perturber = GaussianNoisePerturber(mean=0.0, var=0.01, seed=0)
        img = (np.ones((8, 8, 3)) * 127).astype(np.uint8)
        masks = np.zeros((8, 8), dtype=np.int32)
        masks[2:5, 2:5] = 7
        _, _, perturbed_masks = perturber.perturb_with_masks(image=img, masks=masks)
        assert perturbed_masks is not None
        np.testing.assert_array_equal(perturbed_masks, masks)


class TestRandomCropMasks:
    def test_2d_mask_is_cropped_to_output_shape(self) -> None:
        perturber = RandomCropPerturber(crop_size=(4, 4), seed=42)
        img = np.arange(8 * 8, dtype=np.uint8).reshape((8, 8))
        masks = np.arange(8 * 8, dtype=np.int32).reshape((8, 8))
        out_img, _, out_masks = perturber.perturb_with_masks(image=img, masks=masks)
        assert out_img.shape == (4, 4)
        assert out_masks is not None
        assert out_masks.shape == (4, 4)

    def test_3d_mask_multi_instance(self) -> None:
        perturber = RandomCropPerturber(crop_size=(3, 3), seed=0)
        img = np.zeros((6, 6, 3), dtype=np.uint8)
        masks = np.zeros((2, 6, 6), dtype=np.int32)
        out_img, _, out_masks = perturber.perturb_with_masks(image=img, masks=masks)
        assert out_img.shape[:2] == (3, 3)
        assert out_masks is not None
        assert out_masks.shape == (2, 3, 3)

    def test_invalid_mask_ndim_raises(self) -> None:
        perturber = RandomCropPerturber(crop_size=(3, 3), seed=0)
        img = np.zeros((6, 6, 3), dtype=np.uint8)
        masks = np.zeros((1, 1, 6, 6), dtype=np.int32)
        with pytest.raises(ValueError, match=r"Unsupported mask ndim"):
            perturber.perturb_with_masks(image=img, masks=masks)

    def test_image_and_mask_share_crop_rectangle(self) -> None:
        # Use a 2D image == mask so the crop of each must be identical.
        perturber = RandomCropPerturber(crop_size=(4, 4), seed=123)
        img = np.arange(8 * 8, dtype=np.int32).reshape((8, 8))
        masks = img.copy()
        out_img, _, out_masks = perturber.perturb_with_masks(image=img, masks=masks)
        assert out_masks is not None
        np.testing.assert_array_equal(out_img, out_masks)


class TestRandomTranslationMasks:
    def test_mask_shape_preserved(self) -> None:
        perturber = RandomTranslationPerturber(seed=7, color_fill=[0, 0, 0])
        img = (np.arange(8 * 8 * 3, dtype=np.uint8) % 255).reshape((8, 8, 3))
        masks = np.arange(8 * 8, dtype=np.int32).reshape((8, 8))
        _, _, out_masks = perturber.perturb_with_masks(
            image=img,
            masks=masks,
            max_translation_limit=(2, 2),
        )
        assert out_masks is not None
        assert out_masks.shape == masks.shape

    def test_masks_and_image_share_shift(self) -> None:
        # A 2D image == mask path: the translation should yield identical outputs for both.
        perturber = RandomTranslationPerturber(seed=7)
        img = np.arange(8 * 8, dtype=np.int32).reshape((8, 8))
        masks = img.copy()
        out_img, _, out_masks = perturber.perturb_with_masks(
            image=img,
            masks=masks,
            max_translation_limit=(2, 2),
        )
        assert out_masks is not None
        np.testing.assert_array_equal(out_img, out_masks)
