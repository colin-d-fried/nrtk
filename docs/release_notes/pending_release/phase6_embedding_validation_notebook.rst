Phase 6 embedding validation notebook scaffold
-----------------------------------------------

Additions
~~~~~~~~~

* Added ``docs/examples/embedding_validation.ipynb``, a runnable scaffold
  that documents how to validate NRTK perturbers in feature-embedding space.
  The notebook:

  - Loads a pretrained ResNet-50 via ``torchvision`` (with a hook for CLIP);
  - Defines a severity sweep for one photometric, one geometric, and one
    optical perturber as exemplars;
  - Extracts embeddings for original and perturbed images;
  - Plots cosine distance vs. severity and reports the Spearman rank
    correlation to flag non-monotone perturbers.

* The notebook is self-contained: when no image directory is supplied, it
  synthesizes gradient/noise tiles so the sweep still runs end-to-end in an
  unconfigured environment. Real deployments should point
  ``CONFIG["image_dir"]`` at representative imagery for the target domain.
* Geometric and optical sweeps are guarded with ``ImportError`` fallbacks so
  the notebook runs even when the relevant extras (``graphics`` / ``pybsm``)
  are not installed.
