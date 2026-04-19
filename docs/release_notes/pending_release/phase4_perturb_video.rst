Phase 4 FMV / video support scaffolding
---------------------------------------

Additions
~~~~~~~~~

* Added a new abstract interface :class:`nrtk.interfaces.PerturbVideo` for
  perturbation algorithms that operate on an ordered sequence of frames. The
  docstring documents the expected temporal-consistency contract and refers
  implementers to :class:`RandomPerturbImage`'s ``is_static`` pattern for
  deterministic, frame-consistent random state.
* Registered :class:`PerturbVideo` in ``nrtk.interfaces.__init__``.
* Added :mod:`nrtk.impls.perturb_video` with a concrete
  :class:`FramewisePerturbVideo` implementation that wraps any
  :class:`PerturbImage` and applies it independently to every frame. A shared
  ``PerturbImage`` instance is reused across frames, so
  :class:`RandomPerturbImage` perturbers configured with
  ``is_static=True`` + a deterministic ``seed`` produce the same realization on
  every frame (video-friendly static mode).
* Added unit tests covering per-frame application, bounding-box round-trip,
  length-mismatch validation, input-non-mutation, and config serialization.
