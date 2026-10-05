// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_SECRETKEY_H
#define HEONGPU_BFV_SECRETKEY_H

#include <heongpu/host/bfv/context.cuh>
#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/keygeneration.cuh>

namespace heongpu
{

    /**
     * @brief Secretkey represents a secret key used for decrypting data and
     * generating other keys in homomorphic encryption schemes.
     *
     * The Secretkey class is initialized with encryption parameters and
     * provides a method to access the underlying secret key data. This key is
     * essential for decryption operations as well as for generating other keys
     * like public keys, relinearization keys, and Galois keys.
     */
    template <> class Secretkey<Scheme::BFV>
    {
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HEDecryptor;
        template <Scheme S> friend class HEMultiPartyManager;

        template <typename T, typename F>
        friend void input_storage_manager(T& object, F function,
                                          ExecutionOptions options,
                                          bool check_initial_condition);
        template <typename T, typename F>
        friend void input_vector_storage_manager(std::vector<T>& objects,
                                                 F function,
                                                 ExecutionOptions options,
                                                 bool is_input_output_same);

      public:
        /**
         * @brief Constructs a new Secretkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Secretkey(HEContext<Scheme::BFV> context);

        /**
         * @brief Constructs a new Secretkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param hamming_weight Parameter defining hamming weight of secret
         * key, try to use it as (ring size / 2) for maximum security.
         */
        __host__ Secretkey(HEContext<Scheme::BFV> context,
                           const int hamming_weight);

        /**
         * @brief Constructs a new Secretkey object with specified parameters.
         *
         * @param secret_key Vector where the device data will be copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        __host__ Secretkey(const std::vector<int>& secret_key,
                           HEContext<Scheme::BFV> context,
                           cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Constructs a new Secretkey object with specified parameters.
         *
         * @param secret_key Vector where the device data will be copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        __host__ Secretkey(const HostVector<int>& secret_key,
                           HEContext<Scheme::BFV> context,
                           cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Returns a pointer to the underlying secret key data.
         *
         * @return Data64* Pointer to the secret key data.
         */
        Data64* data();

        /**
         * @brief Switches the secretkey CUDA stream.
         *
         * @param stream The new CUDA stream to be used.
         */
        void switch_stream(cudaStream_t stream)
        {
            device_locations_.set_stream(stream);
        }

        /**
         * @brief Retrieves the CUDA stream associated with the secretkey.
         *
         * This function returns the CUDA stream that was used to create or last
         * modify the secretkey.
         *
         * @return The CUDA stream associated with the secretkey.
         */
        cudaStream_t stream() const { return device_locations_.stream(); }

        /**
         * @brief Returns the size of the polynomial ring used in the
         * homomorphic encryption scheme.
         *
         * @return int Size of the polynomial ring.
         */
        inline int ring_size() const noexcept { return ring_size_; }

        /**
         * @brief Returns the number of coefficient modulus primes used in the
         * encryption parameters.
         *
         * @return int Number of coefficient modulus primes.
         */
        inline int coeff_modulus_count() const noexcept
        {
            return coeff_modulus_count_;
        }

        /**
         * @brief Stores the ciphertext in the device (GPU) memory.
         */
        void store_in_device(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Stores the ciphertext in the host (CPU) memory.
         */
        void store_in_host(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Checks whether the data is stored on the device (GPU) memory.
         */
        bool is_on_device() const noexcept
        {
            if (storage_type_ == storage_type::DEVICE)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /**
         * @brief Copy constructor for creating a new Secretkey object by
         * copying an existing one.
         *
         * This constructor performs a deep copy of the secret key data,
         * ensuring that the new object has its own independent copy of the
         * data. GPU memory operations are handled using `cudaMemcpyAsync` for
         * asynchronous data transfer.
         *
         * @param copy The source Secretkey object to copy from.
         */
        Secretkey(const Secretkey& copy)
            : context_(copy.context_), scheme_(copy.scheme_),
              ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_),
              n_power_(copy.n_power_), hamming_weight_(copy.hamming_weight_),
              in_ntt_domain_(copy.in_ntt_domain_),
              secret_key_generated_(copy.secret_key_generated_),
              storage_type_(copy.storage_type_)
        {
            if (copy.storage_type_ == storage_type::DEVICE)
            {
                device_locations_.resize(copy.device_locations_.size(),
                                         copy.device_locations_.stream());
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    device_locations_.data(), copy.device_locations_.data(),
                    copy.device_locations_.size() * sizeof(Data64),
                    cudaMemcpyDeviceToDevice, copy.device_locations_.stream()));
                HEONGPU_CUDA_CHECK(
                    cudaStreamSynchronize(copy.device_locations_.stream()));
            }
            else
            {
                host_locations_.resize(copy.host_locations_.size());
                std::memcpy(host_locations_.data(), copy.host_locations_.data(),
                            copy.host_locations_.size() * sizeof(Data64));
            }
        }

        /**
         * @brief Move constructor for transferring ownership of a Secretkey
         * object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the newly constructed object. The source object is left in a valid
         * but undefined state.
         *
         * @param assign The source Secretkey object to move from.
         */
        Secretkey(Secretkey&& assign) noexcept
            : context_(std::move(assign.context_)),
              scheme_(std::move(assign.scheme_)),
              ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              n_power_(std::move(assign.n_power_)),
              hamming_weight_(std::move(assign.hamming_weight_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              secret_key_generated_(std::move(assign.secret_key_generated_)),
              storage_type_(std::move(assign.storage_type_)),
              device_locations_(std::move(assign.device_locations_)),
              host_locations_(std::move(assign.host_locations_))
        {
            assign.secret_key_generated_ = false;
            assign.in_ntt_domain_ = false;
        }

        /**
         * @brief Copy assignment operator for assigning one Secretkey object to
         * another.
         *
         * Performs a deep copy of the secret key data, ensuring that the target
         * object has its own independent copy. GPU memory is copied using
         * `cudaMemcpyAsync`.
         *
         * @param copy The source Secretkey object to copy from.
         * @return Reference to the assigned object.
         */
        Secretkey& operator=(const Secretkey& copy)
        {
            if (this != &copy)
            {
                Secretkey temporary(copy);
                *this = std::move(temporary);
            }
            return *this;
        }

        /**
         * @brief Move assignment operator for transferring ownership of a
         * Secretkey object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the target object. The source object is left in a valid but
         * undefined state.
         *
         * @param assign The source Secretkey object to move from.
         * @return Reference to the assigned object.
         */
        Secretkey& operator=(Secretkey&& assign) noexcept
        {
            if (this != &assign)
            {
                context_ = std::move(assign.context_);
                scheme_ = std::move(assign.scheme_);
                ring_size_ = std::move(assign.ring_size_);
                coeff_modulus_count_ = std::move(assign.coeff_modulus_count_);
                n_power_ = std::move(assign.n_power_);

                hamming_weight_ = std::move(assign.hamming_weight_);
                in_ntt_domain_ = std::move(assign.in_ntt_domain_);
                secret_key_generated_ = std::move(assign.secret_key_generated_);
                storage_type_ = std::move(assign.storage_type_);

                device_locations_ = std::move(assign.device_locations_);
                host_locations_ = std::move(assign.host_locations_);
                assign.secret_key_generated_ = false;
                assign.in_ntt_domain_ = false;
            }
            return *this;
        }

        /**
         * @brief Default constructor for Secretkey.
         *
         * Initializes an empty Secretkey object. All members will have their
         * default values.
         */
        Secretkey() = default;

        void save(std::ostream& os) const;

        void load(std::istream& is);

        // Generated keys cannot be rebound to different encryption parameters.
        void set_context(HEContext<Scheme::BFV> context)
        {
            if (secret_key_generated_)
                validate_context(context);
            else
                *this = Secretkey(std::move(context));
        }

      private:
        HEContext<Scheme::BFV> context_;
        scheme_type scheme_ = scheme_type::bfv;
        int ring_size_ = 0;
        int coeff_modulus_count_ = 0;
        int n_power_ = 0;

        int hamming_weight_ = 0;
        bool in_ntt_domain_ = false;
        bool secret_key_generated_ = false;

        storage_type storage_type_ = storage_type::DEVICE;

        DeviceVector<Data64> device_locations_; // coefficients are RNS domain
        HostVector<Data64> host_locations_; // coefficients are RNS domain

        void validate_context(const HEContext<Scheme::BFV>& context);
        void validate_key_data(const HEContext<Scheme::BFV>& context) const;
        int memory_size();
        void memory_clear(cudaStream_t stream);
        void memory_set(DeviceVector<Data64>&& new_device_vector);

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

} // namespace heongpu
#endif // HEONGPU_BFV_SECRETKEY_H
