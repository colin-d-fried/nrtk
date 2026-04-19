"""Implements LocalvarNoisePerturber for applying Gaussian noise with a spatially-varying variance.

Dependencies:
    - skimage (scikit-image) for noise application.
    - numpy for image data handling.

Example:
    >>> import numpy as np
    >>> image = np.ones((256, 256, 3))
    >>> # Constant-variance fallback: equivalent to a zero-mean Gaussian perturber.
    >>> localvar_perturber = LocalvarNoisePerturber(var=0.01)
    >>> noisy_image, _ = localvar_perturber(image=image)

Note:
    The boxes returned from `perturb` are identical to the boxes passed in.
    When ``local_vars`` is explicitly provided it must share the shape of the
    input image and contain strictly positive values.
"""

from __future__ import annotations

__all__ = ["LocalvarNoisePerturber"]

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from smqtk_image_io.bbox import AxisAlignedBoundingBox
from typing_extensions import override

from nrtk.impls.perturb_image.photometric._noise.noise_perturber_mixin import NoisePerturberMixin


class LocalvarNoisePerturber(NoisePerturberMixin):
    """Adds zero-mean Gaussian noise with a spatially-varying variance.

    Wraps ``skimage.util.random_noise(mode="localvar")``, which adds
    Gaussian-distributed additive noise whose variance is specified *per pixel*
    via a ``local_vars`` array matching the image shape.

    Two usage patterns are supported:

    1. Pass an explicit ``local_vars`` ndarray describing the desired per-pixel
       variance. Its shape must match the input image.
    2. Pass a scalar ``var`` (default ``0.01``). A uniform variance map of that
       value is synthesized per call; this collapses to a standard zero-mean
       Gaussian perturbation and is useful as a baseline or placeholder.

    Attributes:
        seed (int | None):
            Random seed for reproducible results. None means non-deterministic.
        is_static (bool):
            If True and seed is set, resets RNG state after each perturb call.
        clip (bool):
            Whether to clip output into ``[-1, 1]`` / ``[0, 1]`` as appropriate
            for the underlying dtype.
        var (float):
            Fallback scalar variance used when ``local_vars`` is not provided.
        local_vars (np.ndarray | None):
            Optional explicit per-pixel variance map. Must be strictly positive
            and share the shape of the perturbed image.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        is_static: bool = False,
        var: float = 0.01,
        local_vars: np.ndarray[Any, Any] | None = None,
        clip: bool = True,
    ) -> None:
        """Initializes the LocalvarNoisePerturber.

        Args:
            seed:
                Random seed for reproducible results. Defaults to None for
                non-deterministic behavior.
            is_static:
                If True and seed is provided, resets the random state after each
                perturb call for identical results on repeated calls.
            var:
                Fallback scalar variance used when ``local_vars`` is not provided.
                Must be non-negative.
            local_vars:
                Optional explicit per-pixel variance map. When provided, must
                contain strictly positive values and share the shape of the
                input image.
            clip:
                Whether to clip output into the canonical value range for the
                image dtype.

        Raises:
            ValueError: If ``var`` is negative, or if ``local_vars`` contains
                non-positive entries.
        """
        super().__init__(seed=seed, is_static=is_static, clip=clip)

        if var < 0:
            raise ValueError(
                f"{type(self).__name__} invalid var ({var}). Must be >= 0.0",
            )
        if local_vars is not None and not np.all(local_vars > 0):
            raise ValueError(
                f"{type(self).__name__} local_vars must contain strictly positive values.",
            )

        self.var = var
        self.local_vars = local_vars

    def _resolve_local_vars(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return a per-pixel variance map matching the image shape."""
        if self.local_vars is not None:
            if self.local_vars.shape != image.shape:
                raise ValueError(
                    f"{type(self).__name__} local_vars shape {self.local_vars.shape} "
                    f"does not match image shape {image.shape}.",
                )
            return self.local_vars
        # ``skimage`` requires strictly positive entries; clamp the scalar floor.
        floor_var = max(self.var, np.finfo(np.float64).tiny)
        return np.full(image.shape, floor_var, dtype=np.float64)

    @override
    def perturb(
        self,
        *,
        image: np.ndarray[Any, Any],
        boxes: Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], Iterable[tuple[AxisAlignedBoundingBox, dict[Hashable, float]]] | None]:
        """Return image stimulus with spatially-varying Gaussian (localvar) noise."""
        perturbed_image, perturbed_boxes = super().perturb(image=image, boxes=boxes, **kwargs)
        local_vars = self._resolve_local_vars(perturbed_image)
        return (
            self._perturb(image=perturbed_image, mode="localvar", local_vars=local_vars),
            perturbed_boxes,
        )

    @override
    def get_config(self) -> dict[str, Any]:
        """Returns the current configuration of the LocalvarNoisePerturber instance."""
        cfg = super().get_config()
        cfg["var"] = self.var
        # ``local_vars`` is intentionally excluded from the serialized config: ndarrays
        # are not JSON-serializable through the default smqtk_core path, and tying a
        # preset to a specific array shape makes configs brittle. Users who need a
        # custom variance map should pass it programmatically after construction.
        cfg["local_vars"] = None
        return cfg
