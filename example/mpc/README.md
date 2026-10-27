# Multiparty Computation (MPC) Support in HEonGPU

HEonGPU now supports Multiparty Computation (MPC) protocols through its implementation of the Multiparty Homomorphic Encryption (MHE) scheme, as described in the paper [Multiparty Homomorphic Encryption from Ring-Learning-With-Errors](https://eprint.iacr.org/2020/304) and [POSEIDON:Privacy-Preserving Federated Neural Network Learning](https://arxiv.org/pdf/2009.00349). The MPC model employed in this implementation is based on N-out-of-N threshold encryption, which ensures that all participating parties must collaborate to decrypt or perform operations on the encrypted data. This integration has been carefully optimized for GPUs, enabling secure and efficient multiparty operations, tailored to meet the demands of modern computational workloads.

## MPC Supported Features

- Supports both the `BFV` and `CKKS` homomorphic encryption schemes.
- Uses hybrid key switching (`KEYSWITCHING_METHOD_II`), with any number of P primes.

<div align="center">
  <img src="../../figures/multiparty_computation.svg" alt="MPC Flow Diagram" width="100%">
</div>

### How It Works

1. **Collective Key Generation**  
   Parties collaboratively generate common public key and evaluation keys, which will be used for encrypting private data and evaluation. (See Stage 1 in the diagram above).
  **Note:**
  - Generating the common relinearization key required two consecutive interactive phases.

2. **Encrypt and Send**  
   Each party encrypts their private data using the common public key and sends the ciphertext to the server (See Stage 2).

3. **Server Processing**  
   The server performs the desired computations on the received ciphertexts without accessing the plaintext data and sends the results back to the parties (See Stage 3).

4. **Collective Decryption**  
   Parties collaborate to decrypt the server's results and securely retrieve the plaintext outputs. Each party partially decrypts the ciphertext using their own secret key and shares the partially decrypted results with the other participants. Once all partial decryptions are combined, the final plaintext output is obtained. Since this process follows the N-out-of-N threshold encryption model, every participant must contribute to the partial decryption for successful reconstruction of the result. (See Stage 4).

5. **Collective Bootstrapping**  
   Collective Bootstrapping—sometimes called Distributed Bootstrapping—is a single-round protocol that lets several parties refresh the noise of an encrypted value together, instead of handing the whole job to one node.

   - Each participant locally splits the ciphertext into a secret share,
   - re-encrypts its share under the collective public key,
   - broadcasts two short vectors; a simple sum of all broadcasts yields the fresh ciphertext.
   
   Because everybody works in parallel and only one exchange is required, CB turns the most expensive step of homomorphic evaluation into a lightweight, collaborative routine.

   How is this different from “normal” (centralised) bootstrapping?
   - *No heavy server:* compute is spread across the N parties.
   - *One message round:* Enc→Share and Share→Enc happen concurrently.
   - *Size-stable keys & ciphertexts:* they remain constant even as the number of parties grows.

### How to Use

To enable MPC in your project, follow these steps:

1. **Integrate HEonGPU into Your Project**  
   Begin by integrating HEonGPU into your project. Detailed instructions for adding HEonGPU as a dependency can be found in the primary [README.md](../README.md) file in the root directory of the repository.

2. **Define Parameters for Collective Key Generation**  
   Utilize the provided APIs to set up parameters for collective key generation, including the common public key and evaluation keys.

3. **Encrypt and Execute**  
   Encrypt private data using the common public key and securely execute multiparty operations with HEonGPU's optimized infrastructure.

4. **Decrypt and Retrieve Results**  
   Use the decryption APIs to securely retrieve plaintext outputs from the server's results.

For a practical demonstration, refer to the [1_multiparty_computation_bfv.cpp](1_multiparty_computation_bfv.cpp),[2_multiparty_computation_ckks.cpp](2_multiparty_computation_ckks.cpp),
[3_mpc_collective_bootstrapping_bfv.cpp](3_mpc_collective_bootstrapping_bfv.cpp) and
[4_mpc_collective_bootstrapping_ckks.cpp](4_mpc_collective_bootstrapping_ckks.cpp),
files in the repository. This examples showcase a complete workflow for using HEonGPU's MPC capabilities, including key generation, encryption, computation, decryption and collective bootstrapping. 

## Upcoming Feature:

Future updates will bring enhanced capabilities for multiparty computations, including `t-out-of-N threshold encryption`. This advanced feature will allow subsets of participants to collaboratively decrypt and perform operations, offering greater flexibility and fault tolerance in distributed systems. By enabling partial decryption and cooperation among a limited number of parties.

#

If you have further questions or require additional information, feel free to reach out at:
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)
