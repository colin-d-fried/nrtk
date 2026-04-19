"""Implements PoissonNoisePerturber for applying Poisson (shot) noise.

Dependencies:
    - skimage (scikit-image) for noise application.
    - numpy for image data handling.

Example:
    >>> import numpy as np
    >>> image = np.ones((256, 256, 3))
    >>> poisson_noise_perturber = PoissonNoisePerturber()
    >>> noisy_image, _ = poisson_noise_perturber(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
    Poisson noise is signal-dependent (shot noise) and has no free parameters
    beyond the clipping behavior inherited from ``NoisePerturberMixin``.
"""

from __future__ import annotations

__all__ = ["PoissonNoisePerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.photometric._noise.noise_perturber_mixin import NoisePerturberMixin


class PoissonNoisePerturber(NoisePerturberMixin):
    """Adds Poisson-distributed (shot) noise to image stimulus.

    Poisson noise models the discrete nature of photon arrival and is inherently
    signal-dependent: brighter pixels carry higher variance. ``scikit-image``'s
    ``random_noise`` implementation does not expose tunable parameters for this
    mode, so the only configuration options inherited from the mixin are ``seed``,
    ``is_static``, and ``clip``.

    Attributes:
        seed (int | None):
            Random seed for reproducible results. None means non-deterministic.
        is_static (bool):
            If True and seed is set, resets RNG state after each perturb call.
        clip (bool):
            Whether to clip output into ``[-1, 1]`` / ``[0, 1]`` as appropriate
            for the underlying dtype.
    """

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus with Poisson noise."""
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)
        return self._perturb(image=perturbed_image, mode="poisson"), perturbed_boxes
