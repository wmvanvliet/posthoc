.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: workbench

Main classes that implement the post-hoc adaptation framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. autosummary::
   :toctree: generated/
   :template: class.rst

    Workbench
    WorkbenchOptimizer
    Beamformer

Ways of estimating the covariance matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: workbench.cov_estimators
.. autosummary::
   :toctree: generated/
   :template: class.rst

    Empirical
    L2
    L2Kernel
    Shrinkage
    ShrinkageKernel
    KroneckerKernel

Utilities for efficient leave-one-out crossvalidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: workbench.loo_utils
.. autosummary::
   :toctree: generated/
   :template: function.rst

   loo
   loo_kern_inv
   loo_mean_norm
   loo_ols_regression
   loo_ols_values
   loo_patterns_from_model
    

