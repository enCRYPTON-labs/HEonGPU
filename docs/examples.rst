.. _examples:

Library Examples
================

The HEonGPU repository contains a rich set of sample programs designed to help users understand and experiment with the library's functionalities. This guide provides a detailed walkthrough of each example, explaining its purpose and the key concepts demonstrated.

Basic Operations
----------------

These examples cover the fundamental workflows for the BFV, CKKS, and TFHE schemes.

1. Basic BFV Arithmetic (``1_basic_bfv.cpp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example demonstrates the core workflow for performing exact integer arithmetic with the BFV scheme.

    * **Workflow**: It shows how to choose the parameters with ``HEParams`` from the multiplicative depth and the plaintext modulus size, set up the ``HEContext`` for BFV, generate keys, encode a vector of integers into a plaintext, encrypt it, perform homomorphic multiplication and relinearization, and finally decrypt and decode the result.

    * **Key Concept**: **Noise Budget Management**. The example explicitly prints the noise budget remaining in the ciphertext before and after operations using ``decryptor.remainder_noise_budget()``.

2. Basic CKKS Arithmetic (``2_basic_ckks.cpp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example introduces the CKKS scheme for approximate arithmetic on real numbers.

    * **Workflow**: It covers choosing the parameters with ``HEParams`` from the multiplicative depth and the scale, setting up the CKKS context, encoding a vector of doubles, encrypting, performing homomorphic multiplication, and then applying the essential ``relinearize`` and ``rescale`` operations.
    * **Key Concept**: **Scaling and Rescaling**. It demonstrates the critical need to call ``operators.rescale_inplace()`` after a multiplication to manage the scale of the ciphertext and control the precision of the approximate numbers.

3. Memory Pool Configuration (``3_basic_memorypool_config.cpp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example shows how to configure HEonGPU's memory pool sizes before generating a context.

    * **Workflow**: It sets host and device pool sizes using either percentage-based limits or explicit byte values, then generates the context with the custom configuration.
    * **Key Concept**: Memory pooling is initialized once and reused across contexts, so the first configuration controls pool sizing for the process.

4. BFV Logic Operations (``11_basic_bfv_logic.cpp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example shows how to perform bitwise logic operations on encrypted binary data (0s and 1s) using the BFV scheme.

    * **Workflow**: It encrypts vectors of binary values and then uses the ``HELogicOperator`` class to perform homomorphic ``AND`` and ``XNOR`` operations.
    * **Key Concept**: The ``HELogicOperator`` provides a dedicated interface for boolean computations, which are fundamental in many privacy-preserving applications.

5. CKKS Logic Operations (``12_basic_ckks_logic.cpp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Similar to the BFV logic example, this demonstrates performing logic gates on binary data that has been encoded within the CKKS scheme.

    * **Workflow**: It encrypts vectors of 0.0s and 1.0s and uses the ``HELogicOperator`` to perform ``AND`` and ``XNOR`` operations.
    * **Key Concept**: This showcases the versatility of the CKKS scheme, which can handle both approximate arithmetic and boolean logic.

6. Basic TFHE Operations (``15_basic_tfhe.cpp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   This example provides a complete demonstration of the TFHE scheme for boolean circuit evaluation.

    * **Workflow**: It initializes a TFHE context, generates a secret key and a bootstrapping key, encrypts several boolean vectors, and then evaluates a series of logic gates (``NAND``, ``AND``, ``OR``, ``XOR``, ``NOT``, ``MUX``) on the ciphertexts.

    * **Key Concept**: **Gate Bootstrapping**. Every binary gate operation (except for NOT) in TFHE involves a bootstrapping procedure to refresh the ciphertext, allowing for the evaluation of arbitrarily complex boolean circuits. This is demonstrated by passing the ``boot_key`` to each logic function.

Advanced Features
-----------------

These examples demonstrate more advanced capabilities of the library, such as different key-switching methods and parallel execution with CUDA streams.

7. Key-Switching Methods in BFV (``4_switchkey_methods_bfv.cpp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example explores the various key-switching operations available in the BFV scheme.

    * **Workflow**: After a standard multiplication and relinearization, it demonstrates how to perform ``rotate_rows`` and ``rotate_columns`` using Galois keys. It also shows how to perform a ``keyswitch`` operation to change the secret key under which a ciphertext is encrypted.
    * **Key Concept**: **Galois Keys and Rotation**. It illustrates how to generate ``Galoiskey`` objects and use them to perform SIMD rotations on the encrypted vectors.

8. Key-Switching Methods in CKKS (``5_switchkey_methods_ckks.cpp``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example is the CKKS counterpart to the previous one, showcasing rotations and key-switching for approximate numbers.

    * **Workflow**: It demonstrates the use of ``rotate_rows_inplace`` with Galois keys and the ``keyswitch`` operation, highlighting that the core concepts are similar across both BFV and CKKS.
    * **Key Concept**: **Key-Switching Digits**. The example uses two P primes, so hybrid key switching groups the ciphertext primes into digits of two primes; with one P prime every digit is one prime (the classical method).

9. Multi-Stream Parallel Execution (``6_``, ``7_``, ``8_`` examples)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This set of examples demonstrates how to leverage CUDA streams to execute multiple independent homomorphic operations concurrently, maximizing GPU utilization.

    * ``8_default_stream_usage.cpp``: Serves as a baseline, performing a sequence of 64 operations serially in the default CUDA stream.
    * ``9_multi_stream_usage_way1.cpp``: Uses **OpenMP** to parallelize a loop across multiple CPU threads. Each thread is assigned its own CUDA stream, and operations within that thread are submitted to its dedicated stream.
    * ``10_multi_stream_usage_way2.cpp``: Shows an alternative approach where a single CPU thread manages multiple CUDA streams, assigning tasks to them in a round-robin fashion.
    * **Key Concept**: By passing a ``cudaStream_t`` object via ``ExecutionOptions``, users can direct HEonGPU to execute an operation on a specific stream, enabling powerful parallel execution patterns.

Serialization
-------------

10. Serialization in BFV and CKKS (``11_``, ``12_`` examples)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    These examples demonstrate how to serialize and deserialize all major HEonGPU objects, which is essential for saving state, persistence, and client-server communication.

    * ``13_bfv_serialization.cpp`` and ``14_ckks_serialization.cpp`` showcase the same workflow for their respective schemes.
    * **Workflow**: They demonstrate serializing and deserializing every object type: ``HEContext``, ``Secretkey``, ``Publickey``, ``Relinkey``, ``Galoiskey``, ``Plaintext``, and ``Ciphertext``.
    * **Key Concept**: The library provides two main ways to serialize:
        * Directly calling ``object.save(stream)`` and ``object.load(stream)`` for raw, uncompressed binary data.
        * Using the convenient ``heongpu::serializer`` helpers (e.g., ``save_to_file``, ``load_from_file``), which automatically apply Zlib compression to reduce object size by 50-60%.

Bootstrapping Examples
----------------------

These examples, located in the `example/bootstrapping` directory, demonstrate the use of the library's powerful bootstrapping capabilities.

* ``1_ckks_regular_bootstrapping.cpp``: Shows how to refresh a CKKS ciphertext containing complex numbers.
* ``2_ckks_slim_bootstrapping.cpp``: Demonstrates the more efficient slim bootstrapping variant for real numbers.
* ``3_ckks_bit_bootstrapping.cpp``: Illustrates the specialized bootstrapping for binary data.
* ``4_ckks_gate_bootstrapping.cpp``: Shows how to embed a logic gate within the bootstrapping process for maximum efficiency.

Multi-Party Computation Examples
--------------------------------

These examples, located in the `example/mpc` directory, cover the library's features for secure multi-party computation.

* ``1_multiparty_computation_bfv.cpp`` and ``2_multiparty_computation_ckks.cpp``: Demonstrate the full protocol for N-out-of-N threshold FHE. This includes each party generating key shares, a server aggregating them into a single public key, encryption with the collective key, and finally, each party generating a partial decryption which are then combined to get the final result.
* ``3_mpc_collective_bootstrapping_bfv.cpp`` and ``4_mpc_collective_bootstrapping_ckks.cpp``: Showcase the advanced collective bootstrapping protocol, where multiple parties can jointly refresh a ciphertext without any single party having access to the full secret key.
