Phase 7 operational-realism benchmark scaffold
----------------------------------------------

Additions
~~~~~~~~~

* Added ``benchmarks/compare_real_degraded.py``, a runnable scaffold script
  that applies an NRTK perturber to clean reference imagery and scores the
  result against a directory of real degraded counterparts using SSIM,
  PSNR, and an FID stub.
* The CLI accepts four perturber presets out of the box
  (``brightness``, ``gaussian_noise``, ``pybsm_maritime``, ``random_crop``)
  mapped to Phase 2 and Phase 3 implementations; the optical preset reuses
  the Phase 3 ``maritime_perturber`` factory.
* Writes ``per_pair.csv`` with one row per image pair and ``summary.txt``
  with mean / std / min / max per metric to the user-specified output
  directory.
* Added ``benchmarks/README.md`` documenting the expected on-disk layout
  (paired ``stem.png`` files across ``--clean-dir`` and ``--degraded-dir``)
  and the acquisition steps for RarePlanes and BDD100k. Dataset downloads
  are intentionally left to the operator because of license / EULA
  constraints.
