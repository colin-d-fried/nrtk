"""Defines RandomTranslationPerturber for random image shifts with bounding box adjustment for labeled datasets.

Classes:
    RandomTranslationPerturber: A perturbation class for applying random translation
    on images and the corresponding bounding boxes.

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.PerturbImage: Base class for perturbation algorithms.

Example usage:
    >>> perturber = RandomTranslationPerturber(seed=42)
    >>> image = np.ones((256, 256, 3))
    >>> max_translation_limit = (image.shape[0] // 2, image.shape[1] // 2)
    >>> perturbed_image, _ = perturber(image=image, max_translation_limit=max_translation_limit)
"""

from __future__ import annotations

__all__ = ["RandomTranslationPerturber"]

from collections.abc import Hashable, Iterable, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image._base import NumpyRandomPerturbImage


class RandomTranslationPerturber(NumpyRandomPerturbImage):
    """RandomTranslationPerturber randomly translates an image and adjusts bounding boxes accordingly.

    Attributes:
        seed (int | None):
            Random seed for reproducibility. None for non-deterministic behavior.
        is_static (bool):
            If True, resets RNG after each call for consistent results.
        color_fill (numpy.array):
            Background color fill for RGB image.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        is_static: bool = False,
        color_fill: Sequence[int] | None = [0, 0, 0],
    ) -> None:
        """RandomTranslationPerturber applies a random translation perturbation to an input image.

        It ensures that bounding boxes are adjusted correctly to reflect the translated
        image coordinates.

        Args:
            seed:
                Random seed for reproducible results. Defaults to None for non-deterministic
                behavior.
            is_static:
                If True and seed is provided, resets RNG after each perturb call for consistent
                results across multiple calls (useful for video frame processing).
            color_fill:
                Background color fill for RGB image. Defaults to [0, 0, 0] (black).

        """
        if color_fill is None:
            color_fill = [0, 0, 0]
        super().__init__(seed=seed, is_static=is_static)
        self.color_fill: np.ndarray[np.int64, Any] = np.array(color_fill)

    def _sample_translate(
        self,
        *,
        image_shape: tuple[int, ...],
        max_translation_limit: tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Sample a random ``(translate_y, translate_x)`` shift honoring the optional limit."""
        if max_translation_limit is None:
            translate_h, translate_w = (image_shape[0], image_shape[1])
        else:
            translate_h, translate_w = max_translation_limit

        if abs(translate_h) > image_shape[0] or abs(translate_w) > image_shape[1]:
            raise ValueError(f"Max translation limit should be less than or equal to {image_shape[:2]}")

        translate_x, translate_y = (0, 0)
        if translate_w > 0:
            translate_x = int(self._rng.integers(low=-translate_w, high=translate_w))
        if translate_h > 0:
            translate_y = int(self._rng.integers(low=-translate_h, high=translate_h))
        return translate_y, translate_x

    @staticmethod
    def _apply_shift(
        *,
        image: np.ndarray[Any, Any],
        translate_y: int,
        translate_x: int,
        fill: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Roll ``image`` by ``(translate_y, translate_x)`` and fill the exposed border with ``fill``."""
        if image.ndim == 3 and fill is not None:
            final_image = np.full_like(image, fill.astype(image.dtype), dtype=image.dtype)
        else:
            final_image = np.zeros_like(image, dtype=image.dtype)

        rolled = np.roll(image.copy(), (translate_y, translate_x), axis=(0, 1))

        if translate_x >= 0 and translate_y >= 0:
            final_image[translate_y:, translate_x:, ...] = rolled[translate_y:, translate_x:, ...]
        elif translate_x < 0 and translate_y >= 0:
            final_image[translate_y:, :translate_x, ...] = rolled[translate_y:, :translate_x, ...]
        elif translate_x >= 0 and translate_y < 0:
            final_image[:translate_y, translate_x:, ...] = rolled[:translate_y, translate_x:, ...]
        else:
            final_image[:translate_y, :translate_x, ...] = rolled[:translate_y, :translate_x, ...]
        return final_image

    @staticmethod
    def _apply_shift_to_masks(
        *,
        masks: np.ndarray[Any, Any],
        translate_y: int,
        translate_x: int,
    ) -> np.ndarray[Any, Any]:
        """Shift segmentation ``masks`` by ``(translate_y, translate_x)`` with a zero-fill border.

        Supports both 2D single-class ``(H, W)`` masks and 3D multi-instance ``(N, H, W)``
        stacks. For the 3D case, the instance axis (``0``) is left untouched and the spatial
        axes ``(1, 2)`` are shifted, mirroring the convention used by
        :meth:`RandomCropPerturber._apply_crop_to_masks`.
        """
        if masks.ndim not in (2, 3):
            msg = f"Expected masks of ndim 2 (H, W) or 3 (N, H, W); got ndim={masks.ndim}."
            raise ValueError(msg)

        # For (H, W) the spatial axes are (0, 1). For (N, H, W) they are (1, 2).
        spatial_axes: tuple[int, int] = (0, 1) if masks.ndim == 2 else (1, 2)
        rolled = np.roll(masks.copy(), (translate_y, translate_x), axis=spatial_axes)
        final = np.zeros_like(masks, dtype=masks.dtype)

        # Build a pair of (final, rolled) slice tuples that operate on the spatial axes only.
        y_slice = slice(translate_y, None) if translate_y >= 0 else slice(None, translate_y)
        x_slice = slice(translate_x, None) if translate_x >= 0 else slice(None, translate_x)
        if masks.ndim == 2:
            index: tuple[slice, ...] = (y_slice, x_slice)
        else:
            index = (slice(None), y_slice, x_slice)
        final[index] = rolled[index]
        return final

    @staticmethod
    def _clamp_shifted_vertex(
        *,
        vertex_x: float,
        vertex_y: float,
        max_x: float,
        max_y: float,
    ) -> tuple[float, float]:
        if vertex_x < 0:
            vertex_x = 0
        elif vertex_x > max_x:
            vertex_x = max_x
        if vertex_y < 0:
            vertex_y = 0
        elif vertex_y > max_y:
            vertex_y = max_y
        return vertex_x, vertex_y

    @staticmethod
    def _shift_boxes(
        *,
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        translate_y: int,
        translate_x: int,
    ) -> list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]:
        perturbed_boxes: list[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] = []
        if boxes is None:
            return perturbed_boxes
        for bbox, metadata in boxes:
            shifted_min = RandomTranslationPerturber._clamp_shifted_vertex(
                vertex_x=bbox.min_vertex[0] + translate_x,
                vertex_y=bbox.min_vertex[1] + translate_y,
                max_x=bbox.max_vertex[0],
                max_y=bbox.max_vertex[1],
            )
            shifted_max = RandomTranslationPerturber._clamp_shifted_vertex(
                vertex_x=bbox.max_vertex[0] + translate_x,
                vertex_y=bbox.max_vertex[1] + translate_y,
                max_x=bbox.max_vertex[0],
                max_y=bbox.max_vertex[1],
            )
            adjusted_box = AxisAlignedBoundingBox(min_vertex=shifted_min, max_vertex=shifted_max)
            perturbed_boxes.append((adjusted_box, deepcopy(metadata)))
        return perturbed_boxes

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        max_translation_limit: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Randomly translates an image and adjusts bounding boxes.

        Args:
            image:
                Input image as a numpy array of shape (H, W, C).
            boxes:
                List of bounding boxes in AxisAlignedBoundingBox format and their corresponding classes.
            max_translation_limit:
                Max translation magnitude (translate_h, translate_w) lesser than or equal to the size of the input
                image.
            kwargs:
                Additional perturbation keyword arguments (currently unused).

        Returns:
            Translated image with the modified bounding boxes.
        """
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)
        translate_y, translate_x = self._sample_translate(
            image_shape=perturbed_image.shape,
            max_translation_limit=max_translation_limit,
        )
        final_image = RandomTranslationPerturber._apply_shift(
            image=perturbed_image,
            translate_y=translate_y,
            translate_x=translate_x,
            fill=self.color_fill,
        )
        perturbed_boxes = RandomTranslationPerturber._shift_boxes(
            boxes=boxes,
            translate_y=translate_y,
            translate_x=translate_x,
        )
        return final_image, perturbed_boxes

    @override
    def perturb_with_masks(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        masks: np.ndarray[Any, Any] | None = None,
        max_translation_limit: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> tuple[
        np.ndarray[Any, Any],
        Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        np.ndarray[Any, Any] | None,
    ]:
        """Randomly translate an image, boxes, and optional segmentation masks with a shared shift.

        Masks are shifted with a zero-fill border (background class 0) regardless of ``color_fill``.
        """
        perturbed_image, _ = super().perturb(image=image, boxes=boxes, **kwargs)
        translate_y, translate_x = self._sample_translate(
            image_shape=perturbed_image.shape,
            max_translation_limit=max_translation_limit,
        )
        final_image = RandomTranslationPerturber._apply_shift(
            image=perturbed_image,
            translate_y=translate_y,
            translate_x=translate_x,
            fill=self.color_fill,
        )
        perturbed_boxes = RandomTranslationPerturber._shift_boxes(
            boxes=boxes,
            translate_y=translate_y,
            translate_x=translate_x,
        )
        if masks is not None:
            shifted_masks = RandomTranslationPerturber._apply_shift_to_masks(
                masks=masks,
                translate_y=translate_y,
                translate_x=translate_x,
            )
        else:
            shifted_masks = None
        return final_image, perturbed_boxes, shifted_masks

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the RandomTranslationPerturber instance."""
        cfg = super().get_config()
        cfg["color_fill"] = self.color_fill.tolist()
        return cfg
