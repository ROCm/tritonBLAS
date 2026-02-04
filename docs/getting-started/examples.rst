Examples
========

This page shows the example scripts included with tritonBLAS.

Basic Matrix Multiplication
---------------------------

The ``example_matmul.py`` script demonstrates basic matrix multiplication using the simple API:

.. literalinclude:: ../../examples/example_matmul.py
   :language: python
   :caption: examples/example_matmul.py

**Usage:**

.. code-block:: bash

   cd examples
   python3 example_matmul.py
   python3 example_matmul.py --m 4096 --n 4096 --k 4096

Matrix Multiplication with Selector
-----------------------------------

The ``example_matmul_lt.py`` script demonstrates using the optimized API with a pre-computed selector:

.. literalinclude:: ../../examples/example_matmul_lt.py
   :language: python
   :caption: examples/example_matmul_lt.py

**Usage:**

.. code-block:: bash

   cd examples
   python3 example_matmul_lt.py
   python3 example_matmul_lt.py --m 4096 --n 4096 --k 4096
