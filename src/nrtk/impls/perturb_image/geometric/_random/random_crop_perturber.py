"""Defines RandomCropPerturber for random image crops with bounding box adjustment for labeled datasets.

Classes:
    RandomCropPerturber: A perturbation class for randomly cropping images and modifying bounding boxes.

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.PerturbImage: Base class for perturbation algorithms.

Example usage:
    >>> image = np.ones((256, 256, 3))
    >>> crop_size = (image.shape[0] // 2, image.shape[1] // 2)
    >>> perturber = RandomCropPerturber(crop_size=crop_size, seed=42)
    >>> perturbed_image, _ = perturber(image=image)
"""

from __future__ import annotations

__all__ = ["RandomCropPerturber"]

import warnings
from collections.abc import Hashable, Iterable
from copy import deepcopy
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image._base import NumpyRandomPerturbImage


class RandomCropPerturber(NumpyRandomPerturbImage):
    """RandomCropPerturber randomly crops an image and adjusts bounding boxes accordingly.

    Attributes:
        crop_size (tuple[int, int]): Target crop dimensions for the input image.
        seed (int | None): Random seed for reproducibility. None for non-deterministic behavior.
        is_static (bool): If True, resets RNG after each call for consistent results.
    """

    def __init__(
        self,
        *,
        crop_size: tuple[int, int] | None = None,
        seed: int | None = None,
        is_static: bool = False,
    ) -> None:
        """RandomCropPerturber applies a random cropping perturbation to an input image.

        It ensures that bounding boxes are adjusted correctly to reflect the new cropped region.

        Args:
            crop_size:
                Target crop size as (crop_height, crop_width). If crop_size is None, it defaults
                to the size of the input image. If crop_size is greater than or equal to image size,
                the original image is returned.
            seed:
                Random seed for reproducible results. Defaults to None for non-deterministic
                behavior.
            is_static:
                If True and seed is provided, resets RNG after each perturb call for consistent
                results across multiple calls (useful for video frame processing).
        """
        super().__init__(seed=seed, is_static=is_static)
        self.crop_size = crop_size

    @staticmethod
    def _compute_bboxes(
        *,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]],
        crop_x: int,
        crop_y: int,
        crop_w: int,
        crop_h: int,
    ) -> Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]:
        """Compute the intersect-shifted bbox coordinates."""
        adjusted_bboxes = []
        for bbox, metadata in boxes:
            crop_box = AxisAlignedBoundingBox(
                min_vertex=(crop_y, crop_x),
                max_vertex=(crop_y + crop_h, crop_x + crop_w),
            )
            # Calculate intersection of the bounding box with the crop region
            intersected_box = bbox.intersection(crop_box)
            if intersected_box:
                # Shift the intersected bounding box to align with the cropped image coordinates
                shifted_min = (
                    intersected_box.min_vertex[0] - crop_y,
                    intersected_box.min_vertex[1] - crop_x,
                )
                shifted_max = (
                    intersected_box.max_vertex[0] - crop_y,
                    intersected_box.max_vertex[1] - crop_x,
                )
                adjusted_box = AxisAlignedBoundingBox(min_vertex=shifted_min, max_vertex=shifted_max)
                adjusted_bboxes.append((adjusted_box, deepcopy(metadata)))
        return adjusted_bboxes

    def _sample_crop_params(
        self,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        """Sample a random crop rectangle for the given image shape.

        Returns ``(crop_x, crop_y, crop_h, crop_w)`` or ``None`` when no crop should be
        applied (``crop_size`` is ``None``, equal to the image shape, or larger than it).
        """
        orig_h, orig_w = image_shape[:2]
        crop_size = self.crop_size if self.crop_size is not None else (orig_h, orig_w)

        if any(crop > img for crop, img in zip(crop_size, image_shape[:2], strict=False)):
            warnings.warn(
                f"crop_size of {crop_size} is larger than image size of ({orig_h}, {orig_w}). Returning original image",
                UserWarning,
                stacklevel=2,
            )
            return None

        if all(crop == img for crop, img in zip(crop_size, image_shape[:2], strict=False)):
            return None

        crop_h, crop_w = crop_size
        crop_x = int(self._rng.integers(low=0, high=(orig_w - crop_w)))
        crop_y = int(self._rng.integers(low=0, high=(orig_h - crop_h)))
        return crop_x, crop_y, crop_h, crop_w

    @staticmethod
    def _apply_crop(
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        params: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        if params is None:
            return image.copy(), boxes
        crop_x, crop_y, crop_h, crop_w = params
        cropped = image[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
        if boxes is None:
            return cropped, []
        cropped_boxes = RandomCropPerturber._compute_bboxes(
            boxes=boxes,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
        )
        return cropped, cropped_boxes

    @staticmethod
    def _apply_crop_to_masks(
        *,
        masks: np.ndarray[Any, Any],
        params: tuple[int, int, int, int] | None,
    ) -> np.ndarray[Any, Any]:
        """Apply the same crop rectangle to a ``(H, W)`` or ``(N, H, W)`` mask array."""
        if params is None:
            return masks.copy()
        crop_x, crop_y, crop_h, crop_w = params
        if masks.ndim == 2:
            return masks[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
        if masks.ndim == 3:
            return masks[:, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w].copy()
        raise ValueError(
            f"Unsupported mask ndim {masks.ndim}; expected (H, W) or (N, H, W).",
        )

    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **_: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Randomly crops an image and adjusts bounding boxes.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.

        Returns:
            Cropped image with the modified bounding boxes.
        """
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes)
        params = self._sample_crop_params(perturbed_image.shape)
        return RandomCropPerturber._apply_crop(image=perturbed_image, boxes=perturbed_boxes, params=params)

    @override
    def perturb_with_masks(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        masks: np.ndarray[Any, Any] | None = None,
        **_: Any,
    ) -> tuple[
        np.ndarray[Any, Any],
        Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        np.ndarray[Any, Any] | None,
    ]:
        """Randomly crop an image, boxes, and optional segmentation masks with a shared transform."""
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes)
        params = self._sample_crop_params(perturbed_image.shape)
        cropped_image, cropped_boxes = RandomCropPerturber._apply_crop(
            image=perturbed_image,
            boxes=perturbed_boxes,
            params=params,
        )
        if masks is not None:
            cropped_masks = RandomCropPerturber._apply_crop_to_masks(masks=masks, params=params)
        else:
            cropped_masks = None
        return cropped_image, cropped_boxes, cropped_masks

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RandomCropPerturber instance."""
        cfg = super().get_config()
        cfg["crop_size"] = self.crop_size
        return cfg
