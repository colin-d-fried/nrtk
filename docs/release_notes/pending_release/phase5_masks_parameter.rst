Phase 5 optional segmentation-mask support
------------------------------------------

Additions
~~~~~~~~~

* :meth:`nrtk.interfaces.PerturbImage.perturb` now accepts an optional
  ``masks`` keyword (default ``None``). The base ``perturb`` implementation
  and all existing photometric, noise, blur, enhance, optical, generative,
  and environment perturbers leave ``masks`` untouched -- only geometric
  perturbers apply the transform to masks.
* Added a new :meth:`PerturbImage.perturb_with_masks` method that returns a
  three-tuple ``(image, boxes, masks)``. Default behavior calls
  :meth:`perturb` and returns a ``deepcopy`` of the provided masks unchanged.
  Geometric perturbers (:class:`RandomCropPerturber`,
  :class:`RandomTranslationPerturber`) override this method to apply the
  same spatial transform to image, boxes, and masks in a single RNG draw.
* Supports ``(H, W)`` single-class masks and ``(N, H, W)`` multi-instance
  mask stacks for crop; translation masks inherit the image shape and use a
  zero-fill border (background class 0) regardless of ``color_fill``.
* The :class:`RandomCropPerturber` and :class:`RandomTranslationPerturber`
  implementations were refactored to share transform-sampling helpers
  (``_sample_crop_params`` / ``_sample_translate``) between :meth:`perturb`
  and :meth:`perturb_with_masks` so a single stochastic draw is applied to
  every output channel.

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~

* :meth:`perturb` still returns the original two-tuple ``(image, boxes)``;
  masks flow only through the new :meth:`perturb_with_masks` method. No
  existing caller is affected.
