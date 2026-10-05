// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_PUBLICKEY_H
#define HEONGPU_BFV_PUBLICKEY_H

#include <heongpu/host/bfv/context.cuh>

namespace heongpu
{

    /**
     * @brief Publickey represents a public key used for encrypting data in
     * homomorphic encryption schemes.
     *
     * The Publickey class is initialized with encryption parameters and
     * provides a method to access the underlying public key data. This key is
     * used in conjunction with the HEEncryptor class to encrypt plaintexts,
     * making them suitable for homomorphic operations.
     */
    template <> class Publickey<Scheme::BFV>
    {
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HEEncryptor;
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
         * @brief Constructs a new Publickey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Publickey(HEContext<Scheme::BFV> context);

        /**
         * @brief Returns a pointer to the underlying public key data.
         *
         * @return Data64* Pointer to the public key data.
         */
        Data64* data();

        /**
         * @brief Switches the publickey CUDA stream.
         *
         * @param stream The new CUDA stream to be used.
         */
        void switch_stream(cudaStream_t stream)
        {
            device_locations_.set_stream(stream);
        }

        /**
         * @brief Retrieves the CUDA stream associated with the publickey.
         *
         * This function returns the CUDA stream that was used to create or last
         * modify the publickey.
         *
         * @return The CUDA stream associated with the publickey.
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
         * @brief Copy constructor for creating a new Publickey object by
         * copying an existing one.
         *
         * This constructor performs a deep copy of the public key data,
         * ensuring that the new object has its own independent copy of the
         * data. GPU memory operations are handled using `cudaMemcpyAsync` for
         * asynchronous data transfer.
         *
         * @param copy The source Publickey object to copy from.
         */
        Publickey(const Publickey& copy)
            : context_(copy.context_), scheme_(copy.scheme_),
              ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_),
              in_ntt_domain_(copy.in_ntt_domain_),
              public_key_generated_(copy.public_key_generated_),
              storage_type_(copy.storage_type_)
        {
            if (copy.storage_type_ == storage_type::DEVICE)
            {
                device_locations_.resize(copy.device_locations_.size(),
                                         copy.device_locations_.stream());
                cudaMemcpyAsync(device_locations_.data(),
                                copy.device_locations_.data(),
                                copy.device_locations_.size() * sizeof(Data64),
                                cudaMemcpyDeviceToDevice,
                                copy.device_locations_
                                    .stream()); // TODO: use cudaStreamPerThread
            }
            else
            {
                host_locations_.resize(copy.host_locations_.size());
                std::memcpy(host_locations_.data(), copy.host_locations_.data(),
                            copy.host_locations_.size() * sizeof(Data64));
            }
        }

        /**
         * @brief Move constructor for transferring ownership of a Publickey
         * object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the newly constructed object. The source object is left in a valid
         * but undefined state.
         *
         * @param assign The source Publickey object to move from.
         */
        Publickey(Publickey&& assign) noexcept
            : context_(std::move(assign.context_)),
              scheme_(std::move(assign.scheme_)),
              ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              public_key_generated_(std::move(assign.public_key_generated_)),
              storage_type_(std::move(assign.storage_type_)),
              device_locations_(std::move(assign.device_locations_)),
              host_locations_(std::move(assign.host_locations_))
        {
        }

        /**
         * @brief Copy assignment operator for assigning one Publickey object to
         * another.
         *
         * Performs a deep copy of the public key data, ensuring that the target
         * object has its own independent copy. GPU memory is copied using
         * `cudaMemcpyAsync`.
         *
         * @param copy The source Publickey object to copy from.
         * @return Reference to the assigned object.
         */
        Publickey& operator=(const Publickey& copy)
        {
            if (this != &copy)
            {
                context_ = copy.context_;
                scheme_ = copy.scheme_;
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;
                in_ntt_domain_ = copy.in_ntt_domain_;

                public_key_generated_ = copy.public_key_generated_;

                storage_type_ = copy.storage_type_;

                if (copy.storage_type_ == storage_type::DEVICE)
                {
                    device_locations_.resize(copy.device_locations_.size(),
                                             copy.device_locations_.stream());
                    cudaMemcpyAsync(
                        device_locations_.data(), copy.device_locations_.data(),
                        copy.device_locations_.size() * sizeof(Data64),
                        cudaMemcpyDeviceToDevice,
                        copy.device_locations_
                            .stream()); // TODO: use cudaStreamPerThread
                }
                else
                {
                    host_locations_.resize(copy.host_locations_.size());
                    std::memcpy(host_locations_.data(),
                                copy.host_locations_.data(),
                                copy.host_locations_.size() * sizeof(Data64));
                }
            }
            return *this;
        }

        /**
         * @brief Move assignment operator for transferring ownership of a
         * Publickey object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the target object. The source object is left in a valid but
         * undefined state.
         *
         * @param assign The source Publickey object to move from.
         * @return Reference to the assigned object.
         */
        Publickey& operator=(Publickey&& assign) noexcept
        {
            if (this != &assign)
            {
                context_ = std::move(assign.context_);
                scheme_ = std::move(assign.scheme_);
                ring_size_ = std::move(assign.ring_size_);
                coeff_modulus_count_ = std::move(assign.coeff_modulus_count_);
                in_ntt_domain_ = std::move(assign.in_ntt_domain_);

                public_key_generated_ = std::move(assign.public_key_generated_);

                storage_type_ = std::move(assign.storage_type_);

                device_locations_ = std::move(assign.device_locations_);
                host_locations_ = std::move(assign.host_locations_);
            }
            return *this;
        }

        /**
         * @brief Default constructor for Publickey.
         *
         * Initializes an empty Publickey object. All members will have their
         * default values.
         */
        Publickey() = default;

        void save(std::ostream& os) const;

        void load(std::istream& is);

        void set_context(HEContext<Scheme::BFV> context)
        {
            context_ = std::move(context);
        }

      private:
        HEContext<Scheme::BFV> context_;
        scheme_type scheme_;
        int ring_size_;
        int coeff_modulus_count_;
        bool in_ntt_domain_;

        bool public_key_generated_ = false;

        storage_type storage_type_;

        DeviceVector<Data64> device_locations_; // coefficients are RNS domain
        HostVector<Data64> host_locations_; // coefficients are RNS domain

        int memory_size();
        void memory_clear(cudaStream_t stream);
        void memory_set(DeviceVector<Data64>&& new_device_vector);

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

    /**
     * @brief MultipartyPublickey is a specialized class for managing public
     * keys in multiparty computation (MPC) settings.
     *
     * This class extends the `Publickey` class to include functionality
     * specific to MPC scenarios, such as managing a seed for deterministic key
     * generation across multiple participants. It integrates with the
     * `HEKeyGenerator` class to facilitate collaborative key generation.
     */

    template <>
    class MultipartyPublickey<Scheme::BFV> : public Publickey<Scheme::BFV>
    {
        template <Scheme S> friend class HEKeyGenerator;

      public:
        __host__ MultipartyPublickey(HEContext<Scheme::BFV> context,
                                     const RNGSeed seed);

        inline RNGSeed seed() const noexcept { return seed_; }

      private:
        RNGSeed seed_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_PUBLICKEY_H
