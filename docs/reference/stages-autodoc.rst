Stages Reference
======================================

This documentation is automatically generated from source code docstrings.

The ``stages`` module provides composable device-side abstractions for building 
high-performance GEMM kernels in Triton.

.. note::

   All stages APIs are **device-side only** — they execute within ``@triton.jit`` 
   kernels on the GPU. They cannot be called from host Python code.

   Type annotations shown as ``MagicMock`` represent Triton types (``tl.tensor``, 
   ``tl.constexpr``) that are mocked during documentation generation.

Module Reference
----------------

.. automodule:: tritonblas.kernels.stages
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __dict__, __weakref__, __module__
