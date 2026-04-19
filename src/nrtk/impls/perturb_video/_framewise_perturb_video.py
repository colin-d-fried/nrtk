"""Implements :class:`FramewisePerturbVideo`, a simple per-frame video perturber.

This wraps any :class:`~nrtk.interfaces.PerturbImage` and applies it independently to
each frame. It is the simplest possible :class:`~nrtk.interfaces.PerturbVideo` and is
primarily intended as a baseline / sanity check for more sophisticated temporally-aware
perturbers.

Temporal-consistency note
~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapper itself introduces no temporal correlation between frames beyond what the
underlying per-frame perturber provides. If the wrapped perturber is a
:class:`~nrtk.interfaces._random_perturb_image.RandomPerturbImage` configured with
``is_static=True`` and a deterministic ``seed``, the same random realization will be
applied to every frame -- this is the "video-friendly" static mode.
"""

from __future__ import annotations

__all__ = ["FramewisePerturbVideo"]

from collections.abc import Sequence
from typing import Any

import numpy as np
from smqtk_core.configuration import from_config_dict, make_default_config, to_config_dict
from typing_extensions import override

from nrtk.interfaces import PerturbImage
from nrtk.interfaces._perturb_video import FrameBoxes, PerturbVideo


class FramewisePerturbVideo(PerturbVideo):
    """Apply a :class:`PerturbImage` independently to every frame of a clip.

    Args:
        perturber:
            A concrete :class:`PerturbImage` used for every frame. The same instance is
            reused for all frames, so any RNG state it owns is threaded across the
            sequence in the order the frames are supplied.
    """

    def __init__(self, perturber: PerturbImage) -> None:
        super().__init__()
        self.perturber = perturber

    @override
    def perturb(
        self,
        *,
        frames: Sequence[np.ndarray[Any, Any]],
        boxes: Sequence[FrameBoxes | None] | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray[Any, Any]], list[FrameBoxes | None] | None]:
        """Apply ``self.perturber`` to each frame in order.

        If ``boxes`` is ``None``, the returned box list is also ``None``. Otherwise it
        is a list of the per-frame bounding-box iterables returned by the underlying
        perturber, one per input frame.
        """
        if boxes is not None and len(boxes) != len(frames):
            raise ValueError(
                f"boxes must have the same length as frames (got {len(boxes)} boxes for {len(frames)} frames)",
            )

        out_frames: list[np.ndarray[Any, Any]] = []
        out_boxes: list[FrameBoxes | None] = []
        for i, frame in enumerate(frames):
            frame_boxes = boxes[i] if boxes is not None else None
            perturbed_frame, perturbed_boxes = self.perturber.perturb(
                image=frame,
                boxes=frame_boxes,
                **kwargs,
            )
            out_frames.append(perturbed_frame)
            out_boxes.append(perturbed_boxes)

        return out_frames, (out_boxes if boxes is not None else None)

    @override
    def get_config(self) -> dict[str, Any]:
        return {"perturber": to_config_dict(self.perturber)}

    @classmethod
    @override
    def get_default_config(cls) -> dict[str, Any]:
        return {"perturber": make_default_config(PerturbImage.get_impls())}

    @classmethod
    @override
    def from_config(
        cls,
        config_dict: dict[str, Any],
        merge_default: bool = True,
    ) -> FramewisePerturbVideo:
        cfg = dict(config_dict)
        cfg["perturber"] = from_config_dict(
            config=cfg["perturber"],
            type_iter=PerturbImage.get_impls(),
        )
        return super().from_config(cfg, merge_default=merge_default)
