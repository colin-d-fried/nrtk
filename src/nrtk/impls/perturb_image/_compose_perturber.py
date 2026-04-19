"""Defines ComposePerturber to apply multiple PerturbImage instances sequentially for combined image perturbations.

Classes:
    ComposePerturber: A perturbation class for applying perturbations from Albumentations

Dependencies:
    - numpy: For numerical operations and random number generation.
    - smqtk_image_io.AxisAlignedBoundingBox: For handling and adjusting bounding boxes.
    - nrtk.interfaces.PerturbImage: Base class for perturbation algorithms.

Example usage:
    >>> from nrtk.impls.perturb_image.photometric.enhance import BrightnessPerturber
    >>> from nrtk.impls.perturb_image.geometric.random import RandomCropPerturber
    >>> image = np.ones((256, 256, 3))
    >>> perturbers = [RandomCropPerturber(), BrightnessPerturber(factor=0.5)]
    >>> perturber = ComposePerturber(perturbers=perturbers)
    >>> perturbed_image, _ = perturber(image=image)
"""

from __future__ import annotations

__all__ = ["ComposePerturber"]

import copy
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_core.configuration import (
    from_config_dict,
    to_config_dict,
)
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import Self, override

from nrtk.interfaces import PerturbImage

_VALID_COMPOSE_MODES: tuple[str, ...] = ("sequential", "blend")


class ComposePerturber(PerturbImage):
    """Composes multiple image perturbations by applying a list of perturbers to an input image.

    Two composition modes are supported:

    - ``"sequential"`` (default): perturbers are chained — each perturber receives the
      output of the previous one. This is the original behavior.
    - ``"blend"``: every perturber is applied independently to the original image and
      the resulting images are averaged. This is a simple approximation of effects
      occurring concurrently rather than in a fixed order. All perturbers must produce
      outputs with the same shape; bounding boxes are passed through unchanged because
      averaging box coordinates across independent perturbations is ambiguous.

    Attributes:
        perturbers (list[PerturbImage]):
            List of perturbers to apply.
        mode (str):
            Composition mode, either ``"sequential"`` or ``"blend"``.

    Note:
        This class has not been tested with perturber factories and is not expected
        to work with perturber factories.
    """

    def __init__(
        self,
        perturbers: list[PerturbImage] | None = None,
        *,
        mode: str = "sequential",
    ) -> None:
        """Initializes the ComposePerturber.

        This has not been tested with perturber factories and is not expected to work with perturber factories.

        Args:
            perturbers:
                List of perturbers to apply.
            mode:
                Composition mode. ``"sequential"`` (default) chains perturbers — each
                perturber sees the output of the previous one. ``"blend"`` applies
                each perturber independently to the original image and averages the
                results.

        Raises:
            ValueError: If ``mode`` is not one of ``"sequential"`` or ``"blend"``.
        """
        super().__init__()
        if perturbers is None:
            perturbers = []
        if mode not in _VALID_COMPOSE_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Must be one of {_VALID_COMPOSE_MODES}.",
            )
        self.perturbers = perturbers
        self.mode = mode

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Apply the configured perturbers to the input image.

        Args:
            image:
                The input image to perturb.
            boxes:
                The bounding boxes for the input image. This is the single image
                output from DetectImageObjects.detect_objects.
            kwargs:
                Additional perturbation keyword arguments.

        Returns:
            The perturbed image and the corresponding bounding boxes. In
            ``"sequential"`` mode the boxes are whatever the final perturber in the
            chain produces. In ``"blend"`` mode the original boxes are passed through
            unchanged.

        Raises:
            ValueError: In ``"blend"`` mode, if any two perturbers produce output
                images with mismatched shapes.
        """
        if not self.perturbers:
            return copy.deepcopy(image), copy.deepcopy(boxes)

        if self.mode == "blend":
            return self._perturb_blend(image=image, boxes=boxes, **kwargs)

        return self._perturb_sequential(image=image, boxes=boxes, **kwargs)

    def _perturb_sequential(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        perturbed_image = image
        perturbed_boxes = boxes
        for perturber in self.perturbers:
            perturbed_image, perturbed_boxes = perturber(
                image=perturbed_image,
                boxes=perturbed_boxes,
                **kwargs,
            )
        return perturbed_image, perturbed_boxes

    def _perturb_blend(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        outputs: list[np.ndarray[Any, Any]] = []
        reference_shape: tuple[int, ...] | None = None
        for perturber in self.perturbers:
            out_image, _ = perturber(image=image, boxes=boxes, **kwargs)
            if reference_shape is None:
                reference_shape = out_image.shape
            elif out_image.shape != reference_shape:
                raise ValueError(
                    "ComposePerturber 'blend' mode requires all perturbers to produce "
                    f"outputs with the same shape; got {reference_shape} and {out_image.shape}.",
                )
            outputs.append(out_image.astype(np.float64, copy=False))

        blended = np.mean(outputs, axis=0)
        blended = blended.astype(image.dtype, copy=False)
        return blended, copy.deepcopy(boxes)

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the configuration dictionary of the ComposePerturber instance."""
        cfg = super().get_config()
        cfg["perturbers"] = [to_config_dict(perturber) for perturber in self.perturbers]
        cfg["mode"] = self.mode
        return cfg

    @classmethod
    @override
    def from_config(
        cls,
        config_dict: dict[str, Any],
        merge_default: bool = True,
    ) -> Self:
        """Create a ComposePerturber instance from a configuration dictionary.

        Args:
            config_dict:
                Configuration dictionary with perturber details.
            merge_default:
                Whether to merge with the default configuration.

        Returns:
            An instance of ComposePerturber.
        """
        config_dict = dict(config_dict)

        config_dict["perturbers"] = [
            from_config_dict(config=perturber, type_iter=PerturbImage.get_impls())
            for perturber in config_dict["perturbers"]
        ]

        return super().from_config(config_dict, merge_default=merge_default)
