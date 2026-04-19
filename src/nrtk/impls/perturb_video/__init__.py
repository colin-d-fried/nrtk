"""Concrete :class:`~nrtk.interfaces.PerturbVideo` implementations."""

from nrtk.impls.perturb_video._framewise_perturb_video import (
    FramewisePerturbVideo as FramewisePerturbVideo,
)

__all__ = ["FramewisePerturbVideo"]

FramewisePerturbVideo.__module__ = __name__
