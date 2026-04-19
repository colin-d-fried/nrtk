r"""Compare NRTK-perturbed imagery against real degraded imagery.

Scaffold benchmark script (Phase 7 of the NRTK enhancement plan).

This script loads a directory of reference images and a directory of
"real-world degraded" counterparts (for example: clean satellite tiles
vs. hazy tiles, clean driving frames vs. rainy frames), applies a user-
selected NRTK perturber to the reference imagery, and reports how close
the synthetic degradation is to the real one using SSIM, PSNR, and
(optionally) FID.

It is intentionally written as a *scaffold*: it is fully runnable end-
to-end on any directory of aligned ``.png`` / ``.jpg`` pairs, but it
does not attempt to automatically download RarePlanes or BDD100k (each
comes with its own license / EULA). The README next to this file walks
through the dataset-acquisition steps.

Usage
-----

.. code-block:: bash

    python benchmarks/compare_real_degraded.py \
        --clean-dir data/clean \
        --degraded-dir data/real_degraded \
        --perturber brightness --severity 0.6 \
        --output-dir outputs/compare_real_degraded \
        --metrics ssim psnr

"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from nrtk.interfaces import PerturbImage

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark script."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--degraded-dir", type=Path, required=True)
    parser.add_argument(
        "--perturber",
        type=str,
        default="brightness",
        choices=sorted(_PERTURBER_BUILDERS),
    )
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["ssim", "psnr"],
        choices=sorted(_METRIC_BUILDERS),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/compare_real_degraded"))
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


@dataclass(frozen=True)
class ImagePair:
    """An aligned ``(clean, real_degraded)`` image pair loaded from disk."""

    stem: str
    clean: np.ndarray
    degraded: np.ndarray


def _scan_images(directory: Path) -> dict[str, Path]:
    """Return ``{stem: path}`` for every image file in ``directory``."""
    return {p.stem: p for p in sorted(directory.iterdir()) if p.suffix.lower() in _IMAGE_SUFFIXES}


def _read_pair(*, stem: str, clean_path: Path, degraded_path: Path) -> ImagePair | None:
    """Load a single aligned image pair, returning ``None`` on shape mismatch."""
    from PIL import Image

    clean = np.asarray(Image.open(clean_path).convert("RGB"))
    degraded = np.asarray(Image.open(degraded_path).convert("RGB"))
    if clean.shape != degraded.shape:
        print(
            f"[warn] shape mismatch for {stem}: {clean.shape} vs {degraded.shape}, skipping.",
            file=sys.stderr,
        )
        return None
    return ImagePair(stem=stem, clean=clean, degraded=degraded)


def _load_pairs(*, clean_dir: Path, degraded_dir: Path, max_pairs: int | None) -> list[ImagePair]:
    """Load aligned image pairs sharing the same file stem from the two directories."""
    clean_by_stem = _scan_images(clean_dir)
    degraded_by_stem = _scan_images(degraded_dir)

    shared = sorted(set(clean_by_stem) & set(degraded_by_stem))
    if max_pairs is not None:
        shared = shared[:max_pairs]
    if not shared:
        msg = f"No image pairs found between {clean_dir} and {degraded_dir} (matched by file stem)."
        raise RuntimeError(msg)

    pairs = [
        _read_pair(stem=stem, clean_path=clean_by_stem[stem], degraded_path=degraded_by_stem[stem]) for stem in shared
    ]
    return [p for p in pairs if p is not None]


def _build_brightness(*, severity: float, seed: int) -> PerturbImage:  # noqa: ARG001
    from nrtk.impls.perturb_image.photometric._enhance.brightness_perturber import BrightnessPerturber

    return BrightnessPerturber(factor=1.0 + 0.9 * severity)


def _build_gaussian_noise(*, severity: float, seed: int) -> PerturbImage:
    from nrtk.impls.perturb_image.photometric._noise.gaussian_noise_perturber import (
        GaussianNoisePerturber,
    )

    return GaussianNoisePerturber(mean=0.0, var=0.0001 + 0.05 * severity, seed=seed)


def _build_pybsm_maritime(*, severity: float, seed: int) -> PerturbImage:
    try:
        from nrtk.impls.perturb_image.optical._pybsm_presets import maritime_perturber
    except ImportError as exc:
        msg = "pybsm extra is required for --perturber pybsm_maritime; install 'nrtk[pybsm]'."
        raise ImportError(msg) from exc
    ihaze = 1 + int(round(severity * 2))
    return maritime_perturber(seed=seed, is_static=True, ihaze=ihaze)


def _build_random_crop(*, severity: float, seed: int) -> PerturbImage:  # noqa: ARG001
    try:
        from nrtk.impls.perturb_image.geometric.random import RandomCropPerturber
    except ImportError as exc:
        msg = "Geometric extras required for --perturber random_crop; install 'nrtk[graphics]'."
        raise ImportError(msg) from exc
    return RandomCropPerturber(crop_size=None, seed=seed)


_PERTURBER_BUILDERS: dict[str, Callable[..., PerturbImage]] = {
    "brightness": _build_brightness,
    "gaussian_noise": _build_gaussian_noise,
    "pybsm_maritime": _build_pybsm_maritime,
    "random_crop": _build_random_crop,
}


def _build_perturber(*, name: str, severity: float, seed: int) -> PerturbImage:
    """Instantiate a :class:`PerturbImage` from a short CLI-facing name and severity."""
    severity = float(np.clip(severity, 0.0, 1.0))
    try:
        builder = _PERTURBER_BUILDERS[name]
    except KeyError as exc:
        msg = f"Unknown perturber: {name!r}"
        raise ValueError(msg) from exc
    return builder(severity=severity, seed=seed)


def _ssim_metric() -> Callable[[np.ndarray, np.ndarray], float]:
    from skimage.metrics import structural_similarity

    def _ssim(a: np.ndarray, b: np.ndarray) -> float:  # noqa: PLR0917
        score = structural_similarity(im1=a, im2=b, channel_axis=-1, data_range=255)
        return float(cast("float", score))

    return _ssim


def _psnr_metric() -> Callable[[np.ndarray, np.ndarray], float]:
    from skimage.metrics import peak_signal_noise_ratio

    def _psnr(a: np.ndarray, b: np.ndarray) -> float:  # noqa: PLR0917
        return float(peak_signal_noise_ratio(image_true=a, image_test=b, data_range=255))

    return _psnr


def _fid_metric() -> Callable[[np.ndarray, np.ndarray], float]:
    def _fid_stub(a: np.ndarray, b: np.ndarray) -> float:  # noqa: ARG001, PLR0917
        return float("nan")

    return _fid_stub


_METRIC_BUILDERS: dict[str, Callable[[], Callable[[np.ndarray, np.ndarray], float]]] = {
    "ssim": _ssim_metric,
    "psnr": _psnr_metric,
    "fid": _fid_metric,
}


def _resolve_metric_fns(metrics: list[str]) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Return a mapping of requested metric name -> callable."""
    return {name: _METRIC_BUILDERS[name]() for name in metrics}


def _align_shapes(*, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center-crop both arrays to their common ``(H, W)`` shape."""
    if a.shape == b.shape:
        return a, b
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def _center(arr: np.ndarray) -> np.ndarray:
        oh, ow = arr.shape[:2]
        y0 = (oh - h) // 2
        x0 = (ow - w) // 2
        return arr[y0 : y0 + h, x0 : x0 + w]

    return _center(a), _center(b)


def _evaluate_pair(
    *,
    pair: ImagePair,
    perturber: PerturbImage,
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> dict[str, Any]:
    """Apply ``perturber`` to ``pair.clean`` and score against ``pair.degraded``."""
    synthetic, _ = perturber.perturb(image=pair.clean)
    synthetic = synthetic if synthetic.dtype == np.uint8 else synthetic.astype(np.uint8)
    synthetic_aligned, real_aligned = _align_shapes(a=synthetic, b=pair.degraded)
    scores: dict[str, Any] = {"stem": pair.stem}
    for name, fn in metric_fns.items():
        scores[name] = fn(synthetic_aligned, real_aligned)  # noqa: FKA100 - metric callables are positional (a, b)
    return scores


def _summarize_metric(*, rows: list[dict[str, Any]], key: str) -> str | None:
    """Return a formatted summary line for ``key``, or ``None`` if all values are NaN."""
    values = [row[key] for row in rows if isinstance(row[key], (int, float)) and not np.isnan(row[key])]
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return f"{key:>8}: mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}"


def _write_report(
    *,
    output_dir: Path,
    per_pair: list[dict[str, Any]],
    perturber_name: str,
    severity: float,
) -> None:
    """Write the per-image CSV and a plaintext summary to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_pair.csv"
    fieldnames = list(per_pair[0].keys()) if per_pair else ["stem"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_pair)

    summary_path = output_dir / "summary.txt"
    metric_keys = [k for k in fieldnames if k != "stem"]
    with summary_path.open("w") as f:
        f.write(f"Perturber: {perturber_name}\n")
        f.write(f"Severity:  {severity}\n")
        f.write(f"N pairs:   {len(per_pair)}\n")
        for key in metric_keys:
            line = _summarize_metric(rows=per_pair, key=key)
            if line is not None:
                f.write(line + "\n")
    print(f"Wrote {csv_path} and {summary_path}")


def main(argv: list[str] | None = None) -> int:
    """Script entrypoint.  Returns a process exit code."""
    args = _parse_args(argv)
    pairs = _load_pairs(clean_dir=args.clean_dir, degraded_dir=args.degraded_dir, max_pairs=args.max_pairs)
    perturber = _build_perturber(name=args.perturber, severity=args.severity, seed=args.seed)
    metric_fns = _resolve_metric_fns(args.metrics)

    per_pair = [_evaluate_pair(pair=pair, perturber=perturber, metric_fns=metric_fns) for pair in pairs]

    _write_report(
        output_dir=args.output_dir,
        per_pair=per_pair,
        perturber_name=args.perturber,
        severity=args.severity,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
