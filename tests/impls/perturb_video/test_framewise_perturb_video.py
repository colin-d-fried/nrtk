"""Unit tests for :class:`FramewisePerturbVideo`."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from smqtk_image_io.bbox import AxisAlignedBoundingBox

from nrtk.impls.perturb_video import FramewisePerturbVideo
from nrtk.interfaces import PerturbImage


class _AddConstantPerturber(PerturbImage):
    """Toy perturber that adds a constant to the image and passes boxes through."""

    def __init__(self, constant: int = 1) -> None:
        super().__init__()
        self.constant = constant

    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        return np.copy(image) + self.constant, deepcopy(boxes)

    def get_config(self) -> dict[str, Any]:
        return {"constant": self.constant}


class TestFramewisePerturbVideo:
    def test_applies_perturber_to_each_frame(self) -> None:
        perturber = _AddConstantPerturber(constant=2)
        wrapper = FramewisePerturbVideo(perturber=perturber)
        frames = [np.zeros((4, 4), dtype=np.int32) + i for i in range(3)]

        out_frames, out_boxes = wrapper.perturb(frames=frames)

        assert out_boxes is None
        assert len(out_frames) == 3
        for i, out in enumerate(out_frames):
            assert np.all(out == i + 2)

    def test_returns_list_of_boxes_when_provided(self) -> None:
        perturber = _AddConstantPerturber(constant=0)
        wrapper = FramewisePerturbVideo(perturber=perturber)
        frames = [np.zeros((2, 2), dtype=np.uint8) for _ in range(2)]
        box = AxisAlignedBoundingBox(min_vertex=(0.0, 0.0), max_vertex=(1.0, 1.0))
        per_frame_boxes: list[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None] = [
            [(box, {"cat": 1.0})],
            None,
        ]

        _, out_boxes = wrapper.perturb(frames=frames, boxes=per_frame_boxes)

        assert out_boxes is not None
        assert len(out_boxes) == 2

    def test_boxes_length_mismatch_raises(self) -> None:
        perturber = _AddConstantPerturber(constant=0)
        wrapper = FramewisePerturbVideo(perturber=perturber)
        frames = [np.zeros((2, 2), dtype=np.uint8) for _ in range(3)]
        boxes: list[Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None] = [None, None]
        with pytest.raises(ValueError, match=r"same length as frames"):
            wrapper.perturb(frames=frames, boxes=boxes)

    def test_does_not_mutate_inputs(self) -> None:
        perturber = _AddConstantPerturber(constant=5)
        wrapper = FramewisePerturbVideo(perturber=perturber)
        frames = [np.zeros((3, 3), dtype=np.int32) for _ in range(2)]
        frames_copy = [f.copy() for f in frames]

        wrapper.perturb(frames=frames)

        for original, post in zip(frames_copy, frames, strict=True):
            assert np.array_equal(original, post)

    def test_config_roundtrip(self) -> None:
        perturber = _AddConstantPerturber(constant=7)
        wrapper = FramewisePerturbVideo(perturber=perturber)
        cfg = wrapper.get_config()
        assert "perturber" in cfg
        # Config serializes the wrapped perturber as a smqtk-core plugin payload.
        assert isinstance(cfg["perturber"], dict)
