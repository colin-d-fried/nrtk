"""Random noise perturbers using skimage."""

_SKIMAGE_CLASSES = [
    "GaussianNoisePerturber",
    "LocalvarNoisePerturber",
    "PepperNoisePerturber",
    "PoissonNoisePerturber",
    "SaltAndPepperNoisePerturber",
    "SaltNoisePerturber",
    "SpeckleNoisePerturber",
]

__all__: list[str] = []

_import_error: ImportError | None = None

try:
    from nrtk.impls.perturb_image.photometric._noise.gaussian_noise_perturber import (
        GaussianNoisePerturber as GaussianNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.localvar_noise_perturber import (
        LocalvarNoisePerturber as LocalvarNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.pepper_noise_perturber import (
        PepperNoisePerturber as PepperNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.poisson_noise_perturber import (
        PoissonNoisePerturber as PoissonNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.salt_and_pepper_noise_perturber import (
        SaltAndPepperNoisePerturber as SaltAndPepperNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.salt_noise_perturber import (
        SaltNoisePerturber as SaltNoisePerturber,
    )
    from nrtk.impls.perturb_image.photometric._noise.speckle_noise_perturber import (
        SpeckleNoisePerturber as SpeckleNoisePerturber,
    )

    # Override __module__ to reflect the public API path for plugin discovery
    GaussianNoisePerturber.__module__ = __name__
    LocalvarNoisePerturber.__module__ = __name__
    PepperNoisePerturber.__module__ = __name__
    PoissonNoisePerturber.__module__ = __name__
    SaltAndPepperNoisePerturber.__module__ = __name__
    SaltNoisePerturber.__module__ = __name__
    SpeckleNoisePerturber.__module__ = __name__

    __all__ += _SKIMAGE_CLASSES
except ImportError as _ex:
    _import_error = _ex


def __getattr__(name: str) -> None:
    if name in _SKIMAGE_CLASSES:
        msg = f"{name} requires the `skimage` extra. Install with: `pip install nrtk[skimage]`"
        if _import_error is not None:
            msg += (
                f"\n\nIf the extra is already installed, the following upstream error may be the cause:"
                f"\n  {type(_import_error).__name__}: {_import_error}"
            )
        raise ImportError(msg)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
