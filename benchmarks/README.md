# NRTK Operational Realism Benchmarks

This directory hosts scripts that compare NRTK-generated perturbations against
**real-world degraded imagery** so you can quantitatively estimate how close the
synthetic distribution is to the one a deployed sensor actually sees.

Phase 7 of the multi-phase enhancement plan ships a single scaffold entry point,
[`compare_real_degraded.py`](./compare_real_degraded.py). The script is
deliberately opinionated about directory layout (paired `stem.png` files) but is
dataset-agnostic — it does not download anything on your behalf.

## Contents

| File                       | Purpose                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `compare_real_degraded.py` | Applies an NRTK perturber to clean imagery and scores it against real degraded counterparts via SSIM / PSNR (and an FID stub). |
| `README.md`                | You are here.                                                                                                                  |

## Quickstart

```bash
# From the repo root
poetry install --extras graphics       # Pillow + scikit-image
# (add --extras pybsm for the maritime preset)

python benchmarks/compare_real_degraded.py \
    --clean-dir    data/clean \
    --degraded-dir data/real_degraded \
    --perturber    brightness \
    --severity     0.6 \
    --metrics      ssim psnr \
    --output-dir   outputs/compare_real_degraded
```

The script writes two files into `--output-dir`:

- `per_pair.csv` — one row per image pair with the raw metric values.
- `summary.txt` — mean / std / min / max across the corpus, plus the run config.

Pairs are matched by file **stem** across the two directories (so
`data/clean/00001.png` pairs with `data/real_degraded/00001.png`).

## Supported perturbers

| `--perturber`    | NRTK implementation                                                  | Extras required |
| ---------------- | -------------------------------------------------------------------- | --------------- |
| `brightness`     | `BrightnessPerturber(factor=1.0 + 0.9 * severity)`                   | none            |
| `gaussian_noise` | `GaussianNoisePerturber(mean=0.0, var=0.0001 + 0.05 * severity)`     | none            |
| `pybsm_maritime` | `maritime_perturber(ihaze=1 + round(severity * 2))` (Phase 3 preset) | `pybsm`         |

Geometric perturbers (e.g. `RandomCropPerturber`) are intentionally omitted from
this scaffold because they require the image dimensions at construction time to
produce a meaningful, severity-scaled output — the benchmark currently resolves
perturbers before loading any images. Wire them in after you've added an
"inspect the first pair, then build the perturber" step to `_evaluate_pair`.

## Supplying datasets

The scaffold does not download datasets; pick one of the recipes below.

### Option A — RarePlanes (overhead / synthetic-vs-real aircraft)

1. Review the terms at
   <https://www.cosmiqworks.org/rareplanes-public-user-guide/> and sign the EULA
   if it applies to your use case.
2. Download a `real_cleaned` tile subset plus the paired `synthetic`
   degradations. Aligned basenames are not guaranteed, so you may need to rename
   files so clean and degraded variants share a stem.
3. Rescale / crop tiles to a consistent size (the script center-crops to the
   common shape, but matching sizes up-front avoids silent ROI drift).

### Option B — BDD100k (driving imagery; clear-weather vs. rainy)

1. Register at <https://bdd-data.berkeley.edu/> and agree to the license.
2. Use the `weather` attribute in the label JSON to split a sample of frames
   into "clear" and "rainy" / "foggy" / "snowy" buckets.
3. Pair one clear frame per vehicle/location with one matching degraded frame
   and rename them to share a file stem.

### Option C — Any in-house dataset

Any two directories of **aligned, same-basename** `.png` / `.jpg` files work.
The scaffold script handles RGB 8-bit; grayscale and 16-bit inputs will need
minor adjustments to `_load_pairs`.

## Interpreting the results

- **Higher SSIM / PSNR** → the synthetic perturbation is *more* similar to the
  real degraded image. Very high values (SSIM > 0.95) typically mean the
  perturber is too mild at this severity.
- **FID** is a stub in this scaffold (returns `NaN`). Wire up an
  ImageNet-pretrained Inception feature extractor and the standard FID formula
  when you're ready for a distributional metric.
- The script does **not** report a single "passing" threshold. These numbers are
  *diagnostic*; couple them with task-level performance plots (e.g. mAP on a
  downstream detector) before drawing operational conclusions.

## Extending the scaffold

Common extensions, in rough order of effort:

1. **New perturbers** — add a branch to `_build_perturber(...)` with a
   severity-to-parameter mapping.
2. **New metrics** — add an entry to `_resolve_metric_fns(...)`. Follow the
   SSIM/PSNR pattern; the signature is `(synthetic, real) -> float`.
3. **FID** — load `torchvision.models.inception_v3(pretrained=True)`, extract
   pool3 features for each set, and compute the Fréchet distance between their
   empirical Gaussians.
4. **Per-class breakdowns** — pass along a class label per pair (e.g. by
   including it in the filename) and aggregate the summary over classes.
5. **Parameter sweeps** — wrap `main(...)` in a loop over `--severity` to
   produce a curve of metric vs. severity, similar to the Phase 6 embedding
   notebook.
