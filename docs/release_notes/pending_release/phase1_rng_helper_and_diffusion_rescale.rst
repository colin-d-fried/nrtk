Phase 1 code quality and internal fixes
----------------------------------------

Updates
~~~~~~~

* Refactored ``PybsmPerturber._set_seed`` to delegate the private pyBSM
  ``_simulator._rng`` reassignment to a new ``_update_simulator_rng`` helper,
  isolating the upstream private-attribute access in a single place so it can
  be swapped for a public pyBSM API once one is available.

Fixes
~~~~~

* ``DiffusionPerturber.perturb`` now rescales bounding boxes from the original
  image dimensions to the (typically smaller) diffusion output dimensions using
  the inherited ``PerturbImage._rescale_boxes`` helper. Boxes may still be
  spatially inaccurate because the generative process can shift, remove, or
  introduce content independent of the original geometry, but their coordinates
  now correspond to the perturbed image's pixel grid.
