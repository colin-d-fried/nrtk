Phase 2 new noise models and composition modes
-----------------------------------------------

Additions
~~~~~~~~~

* Added ``PoissonNoisePerturber`` (wraps ``skimage.util.random_noise(mode="poisson")``),
  a parameter-free signal-dependent shot-noise model.

* Added ``LocalvarNoisePerturber`` (wraps ``skimage.util.random_noise(mode="localvar")``).
  Accepts either an explicit per-pixel ``local_vars`` ndarray or a scalar
  ``var`` fallback used to synthesize a uniform variance map at perturb time.

* Registered both new classes in ``nrtk.impls.perturb_image.photometric.noise``
  and extended the import-guard / public-import canary tests accordingly.

* Added a ``mode`` parameter to ``ComposePerturber`` with two values:

  - ``"sequential"`` (default, unchanged behavior): perturbers run in order and
    each receives the previous perturber's output.
  - ``"blend"``: every perturber is applied independently to the original image
    and the resulting images are averaged. All perturbers must produce outputs
    of matching shape; bounding boxes are passed through unchanged because
    averaging independent spatial transforms is ambiguous.

  The new parameter is serialized via ``get_config`` / ``from_config`` and
  validated at construction time.
