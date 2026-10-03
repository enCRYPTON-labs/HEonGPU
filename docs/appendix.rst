.. _appendix:

Appendix
========

This appendix contains supplementary material, including detailed performance benchmarks and instructions for citing the HEonGPU library in academic work.

Performance Benchmarks
----------------------

All benchmarks listed below were performed on an **NVIDIA RTX 4090 GPU** and are sourced from the project's main repository. Performance is highly dependent on the chosen parameters and hardware configuration.

**TFHE Unsigned Integer Arithmetic**

This table compares the latency (in milliseconds) of homomorphic unsigned integer addition between HEonGPU, TFHE-rs, and results from recent academic literature.

.. list-table:: TFHE Unsigned Integer Arithmetic Latency (ms)
   :widths: 14 14 14 14 14 14 14
   :header-rows: 1

   * - Library
     - uint8
     - uint16
     - uint32
     - uint64
     - uint128
     - uint256
   * - `TFHE-rs <https://github.com/zama-ai/tfhe-rs>`_
     - 31.53
     - 31.54
     - 31.55
     - 32.03
     - 33.74
     - 58.32
   * - `Literature <https://tches.iacr.org/index.php/TCHES/article/view/11931>`_
     - 18.63
     - 18.61
     - 18.87
     - 24.23
     - 29.97
     - 58.30
   * - **HEonGPU**
     - **12.72**
     - **12.75**
     - **13.60**
     - **15.88**
     - **23.10**
     - **38.24**

**CKKS Bootstrapping**

This table details the execution times for various CKKS bootstrapping operations implemented in HEonGPU (dense secret key h=192 with a temporary sparse secret key h=32, median of 5 runs).

* **Remaining Level**: The number of levels left before the next bootstrapping.
* **Amortized Time**: The total execution time divided by the number of slots, providing a per-slot performance metric.

.. list-table:: CKKS Bootstrapping Performance
   :widths: 20 10 15 15 15 15
   :header-rows: 1

   * - Bootstrapping Type
     - :math:`N`
     - Slot Count
     - Remaining Level
     - Total Time
     - Amortized Time
   * - **Regular**
     - :math:`2^{15}`
     - :math:`2^{14}`
     - 9
     - 51.02 ms
     - 3.11 µs
   * - **Regular**
     - :math:`2^{16}`
     - :math:`2^{15}`
     - 9
     - 109.28 ms
     - 3.33 µs
   * - **Slim**
     - :math:`2^{15}`
     - :math:`2^{14}`
     - 0
     - 17.56 ms
     - 1.07 µs
   * - **Slim**
     - :math:`2^{15}`
     - :math:`2^{14}`
     - 9
     - 30.87 ms
     - 1.88 µs
   * - **Slim**
     - :math:`2^{16}`
     - :math:`2^{15}`
     - 0
     - 38.55 ms
     - 1.18 µs
   * - **Slim**
     - :math:`2^{16}`
     - :math:`2^{15}`
     - 9
     - 70.59 ms
     - 2.15 µs
   * - **Bit**
     - :math:`2^{15}`
     - :math:`2^{14}`
     - 0
     - 22.53 ms
     - 1.38 µs
   * - **Bit**
     - :math:`2^{15}`
     - :math:`2^{14}`
     - 8
     - 37.51 ms
     - 2.29 µs
   * - **Bit**
     - :math:`2^{16}`
     - :math:`2^{15}`
     - 0
     - 50.72 ms
     - 1.55 µs
   * - **Bit**
     - :math:`2^{16}`
     - :math:`2^{15}`
     - 8
     - 82.43 ms
     - 2.52 µs
   * - **Gate\***
     - :math:`2^{15}`
     - :math:`2^{14}`
     - 0
     - 29.96–31.85 ms
     - 1.83–1.94 µs
   * - **Gate\***
     - :math:`2^{15}`
     - :math:`2^{14}`
     - 7
     - 45.36–47.50 ms
     - 2.77–2.90 µs
   * - **Gate\***
     - :math:`2^{16}`
     - :math:`2^{15}`
     - 0
     - 62.90–66.55 ms
     - 1.92–2.03 µs
   * - **Gate\***
     - :math:`2^{16}`
     - :math:`2^{15}`
     - 7
     - 99.45–102.17 ms
     - 3.04–3.12 µs

\*For all logic gates.

How to Cite HEonGPU
-------------------

Please use the following BibTeX entries to cite HEonGPU and its associated research in academic papers.

Main Library Paper
^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

    @misc{cryptoeprint:2024/1543,
          author = {Ali Şah Özcan and Erkay Savaş},
          title = {{HEonGPU}: a {GPU}-based Fully Homomorphic Encryption Library 1.0},
          howpublished = {Cryptology {ePrint} Archive, Paper 2024/1543},
          year = {2024},
          url = {https://eprint.iacr.org/2024/1543}
    }

Key-Switching Optimizations Paper
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

    @misc{cryptoeprint:2025/124,
          author = {Ali Şah Özcan and Erkay Savaş},
          title = {{GPU} Implementations of Three Different Key-Switching Methods for Homomorphic Encryption Schemes},
          howpublished = {Cryptology {ePrint} Archive, Paper 2025/124},
          year = {2025},
          url = {https://eprint.iacr.org/2025/124}
    }
