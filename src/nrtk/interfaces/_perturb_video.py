"""Defines PerturbVideo, an interface for perturbing a sequence of video frames.

Classes:
    PerturbVideo: Abstract interface for perturbation algorithms that operate on an ordered
        sequence of frames, with built-in hooks for maintaining random state across the
        sequence (temporal consistency).

Dependencies:
    - numpy for frame arrays.
    - nrtk.interfaces.PerturbImage for the per-frame analog.
    - smqtk_core for configurable plugin interface capabilities.

Usage:
    To create a custom video perturber, inherit from :class:`PerturbVideo` and implement
    :meth:`perturb`. If your perturber relies on random state and you want the same
    perturbation to be applied consistently to every frame, follow the ``is_static``
    pattern from :class:`RandomPerturbImage`: seed the RNG once in ``__init__`` and
    reset it at the start of each :meth:`perturb` call.
"""

from __future__ import annotations

__all__ = ["PerturbVideo"]

import abc
from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.interfaces._plugfigurable import Plugfigurable

FrameBoxes = Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]]


class PerturbVideo(Plugfigurable):
    """Algorithm that generates perturbed frames for a given input video clip.

    A "video" here is simply an ordered :class:`~collections.abc.Sequence` of frames,
    each an ``np.ndarray`` shaped ``(H, W)`` or ``(H, W, C)``. Implementations are free
    to introduce temporal dependencies between frames (e.g. shared motion blur, shared
    noise realizations, rolling atmospheric turbulence), but the default contract is
    that :meth:`perturb` returns the same number of frames as it was given, in order.

    Implementations that use randomness should manage their RNG state in a way that
    mirrors :class:`~nrtk.interfaces._random_perturb_image.RandomPerturbImage`'s
    ``is_static`` semantics: if ``seed`` is set and ``is_static`` is True, calling
    :meth:`perturb` twice with the same input must return identical outputs.
    """

    def __init__(self) -> None:
        """Initializes the PerturbVideo."""

    @abc.abstractmethod
    def perturb(
        self,
        *,
        frames: Sequence[np.ndarray[Any, Any]],
        boxes: Sequence[FrameBoxes | None] | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray[Any, Any]], list[FrameBoxes | None] | None]:
        """Generate perturbed frames for the given sequence of input frames.

        Args:
            frames:
                Ordered sequence of input frames. Each frame is an ``np.ndarray`` of
                shape ``(H, W)`` or ``(H, W, C)``. Implementations must not mutate
                the input frames in place.
            boxes:
                Optional per-frame bounding boxes. If supplied, it must be a sequence
                with the same length as ``frames``; each element is either ``None`` or
                an iterable of ``(AxisAlignedBoundingBox, score_dict)`` tuples matching
                the single-image :class:`PerturbImage.perturb` contract.
            kwargs:
                Implementation-specific keyword arguments (forwarded to the per-frame
                perturber where appropriate).

        Returns:
            Tuple of:

            * A list of perturbed frames, one per input frame, in order. Each frame
              retains its original dtype.
            * A list of per-frame bounding boxes, or ``None`` when ``boxes`` was not
              provided. When supplied, the returned list has the same length as
              ``frames``.
        """

    def __call__(
        self,
        *,
        frames: Sequence[np.ndarray[Any, Any]],
        boxes: Sequence[FrameBoxes | None] | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray[Any, Any]], list[FrameBoxes | None] | None]:
        """Convenience wrapper for :meth:`perturb`."""
        return self.perturb(frames=frames, boxes=boxes, **kwargs)
