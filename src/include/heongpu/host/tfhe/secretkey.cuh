// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_SECRETKEY_H
#define HEONGPU_TFHE_SECRETKEY_H

#include <heongpu/host/tfhe/context.cuh>
#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/keygeneration.cuh>

namespace heongpu
{
    template <> class Secretkey<Scheme::TFHE>
    {
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HEEncryptor;
        template <Scheme S> friend class HEDecryptor;

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
        __host__ Secretkey(HEContext<Scheme::TFHE> context,
                           bool store_in_gpu = true);

        /**
         * @brief Stores the Secretkey in the device (GPU) memory.
         */
        void store_in_device(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Stores the Secretkey in the host (CPU) memory.
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
         * @brief Switches the Ciphertext CUDA stream.
         *
         * @param stream The new CUDA stream to be used.
         */
        void switch_stream(cudaStream_t stream)
        {
            lwe_key_device_location_.set_stream(stream);
            tlwe_key_device_location_.set_stream(stream);
        }

        /**
         * @brief Retrieves the CUDA stream associated with the ciphertext.
         *
         * This function returns the CUDA stream that was used to create or last
         * modify the ciphertext.
         *
         * @return The CUDA stream associated with the ciphertext.
         */
        cudaStream_t stream() const noexcept
        {
            return lwe_key_device_location_.stream();
        }

        /**
         * @brief Indicates whether the secret key has been generated.
         *
         * @return true if the secret key has been generated, false otherwise.
         */
        inline bool is_generated() const noexcept
        {
            return secret_key_generated_;
        }

        // Binary coefficients in the current storage location (see
        // is_on_device()).
        int32_t* lwe_data() noexcept
        {
            return is_on_device() ? lwe_key_device_location_.data()
                                  : lwe_key_host_location_.data();
        }
        int32_t* tlwe_data() noexcept
        {
            return is_on_device() ? tlwe_key_device_location_.data()
                                  : tlwe_key_host_location_.data();
        }
        int lwe_size() const noexcept { return n_; }
        int tlwe_size() const noexcept { return k_ * N_; }

        Secretkey() = default;

        void set_context(HEContext<Scheme::TFHE> context)
        {
            if (secret_key_generated_)
                validate_context(context);
            else
                *this = Secretkey(std::move(context));
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
            : context_(copy.context_), n_(copy.n_),
              lwe_alpha_min(copy.lwe_alpha_min),
              lwe_alpha_max(copy.lwe_alpha_max), N_(copy.N_), k_(copy.k_),
              tlwe_alpha_min(copy.tlwe_alpha_min),
              tlwe_alpha_max(copy.tlwe_alpha_max), bk_l_(copy.bk_l_),
              bk_bg_bit_(copy.bk_bg_bit_), bg_(copy.bg_),
              half_bg_(copy.half_bg_), mask_mod_(copy.mask_mod_),
              kpl_(copy.kpl_), h_(copy.h_), offset_(copy.offset_),
              secret_key_generated_(copy.secret_key_generated_),
              storage_type_(copy.storage_type_)
        {
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(copy.stream()));
            if (copy.secret_key_generated_)
            {
                if (copy.storage_type_ == storage_type::DEVICE)
                {
                    lwe_key_device_location_.resize(
                        copy.lwe_key_device_location_.size(),
                        copy.lwe_key_device_location_.stream());
                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        lwe_key_device_location_.data(),
                        copy.lwe_key_device_location_.data(),
                        copy.lwe_key_device_location_.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice,
                        copy.lwe_key_device_location_.stream()));
                    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(copy.stream()));

                    tlwe_key_device_location_.resize(
                        copy.tlwe_key_device_location_.size(),
                        copy.tlwe_key_device_location_.stream());
                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        tlwe_key_device_location_.data(),
                        copy.tlwe_key_device_location_.data(),
                        copy.tlwe_key_device_location_.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice,
                        copy.tlwe_key_device_location_.stream()));
                    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(copy.stream()));
                }
                else
                {
                    lwe_key_host_location_.resize(
                        copy.lwe_key_host_location_.size());
                    std::memcpy(lwe_key_host_location_.data(),
                                copy.lwe_key_host_location_.data(),
                                copy.lwe_key_host_location_.size() *
                                    sizeof(int32_t));

                    tlwe_key_host_location_.resize(
                        copy.tlwe_key_host_location_.size());
                    std::memcpy(tlwe_key_host_location_.data(),
                                copy.tlwe_key_host_location_.data(),
                                copy.tlwe_key_host_location_.size() *
                                    sizeof(int32_t));
                }
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
            : context_(std::move(assign.context_)), n_(std::move(assign.n_)),
              lwe_alpha_min(std::move(assign.lwe_alpha_min)),
              lwe_alpha_max(std::move(assign.lwe_alpha_max)),
              N_(std::move(assign.N_)), k_(std::move(assign.k_)),
              tlwe_alpha_min(std::move(assign.tlwe_alpha_min)),
              tlwe_alpha_max(std::move(assign.tlwe_alpha_max)),
              bk_l_(std::move(assign.bk_l_)),
              bk_bg_bit_(std::move(assign.bk_bg_bit_)),
              bg_(std::move(assign.bg_)), half_bg_(std::move(assign.half_bg_)),
              mask_mod_(std::move(assign.mask_mod_)),
              kpl_(std::move(assign.kpl_)), h_(std::move(assign.h_)),
              offset_(std::move(assign.offset_)),
              secret_key_generated_(std::move(assign.secret_key_generated_)),
              storage_type_(std::move(assign.storage_type_)),
              lwe_key_device_location_(
                  std::move(assign.lwe_key_device_location_)),
              lwe_key_host_location_(std::move(assign.lwe_key_host_location_)),
              tlwe_key_device_location_(
                  std::move(assign.tlwe_key_device_location_)),
              tlwe_key_host_location_(std::move(assign.tlwe_key_host_location_))
        {
            assign.secret_key_generated_ = false;
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
                n_ = std::move(assign.n_);
                lwe_alpha_min = std::move(assign.lwe_alpha_min);
                lwe_alpha_max = std::move(assign.lwe_alpha_max);
                N_ = std::move(assign.N_);
                k_ = std::move(assign.k_);
                tlwe_alpha_min = std::move(assign.tlwe_alpha_min);
                tlwe_alpha_max = std::move(assign.tlwe_alpha_max);
                bk_l_ = std::move(assign.bk_l_);
                bk_bg_bit_ = std::move(assign.bk_bg_bit_);
                bg_ = std::move(assign.bg_);
                half_bg_ = std::move(assign.half_bg_);
                mask_mod_ = std::move(assign.mask_mod_);
                kpl_ = std::move(assign.kpl_);
                h_ = std::move(assign.h_);
                offset_ = std::move(assign.offset_);
                secret_key_generated_ = std::move(assign.secret_key_generated_);
                storage_type_ = std::move(assign.storage_type_);
                lwe_key_device_location_ =
                    std::move(assign.lwe_key_device_location_);
                lwe_key_host_location_ =
                    std::move(assign.lwe_key_host_location_);
                tlwe_key_device_location_ =
                    std::move(assign.tlwe_key_device_location_);
                tlwe_key_host_location_ =
                    std::move(assign.tlwe_key_host_location_);
                assign.secret_key_generated_ = false;
            }
            return *this;
        }

      private:
        HEContext<Scheme::TFHE> context_;
        const scheme_type scheme_ = scheme_type::tfhe;

        // LWE Context
        int n_ = 0;
        double lwe_alpha_min = 0.0;
        double lwe_alpha_max = 0.0;

        DeviceVector<int32_t> lwe_key_device_location_;
        HostVector<int32_t> lwe_key_host_location_;

        // TLWE Context
        int N_ = 0;
        int k_ = 0;
        double tlwe_alpha_min = 0.0;
        double tlwe_alpha_max = 0.0;
        // extracted_lwe_params

        DeviceVector<int32_t> tlwe_key_device_location_;
        HostVector<int32_t> tlwe_key_host_location_;

        // TGSW Context
        int bk_l_ = 0;
        int bk_bg_bit_ = 0;
        int bg_ = 0;
        int half_bg_ = 0;
        int mask_mod_ = 0;
        // tlwe_params = TLWE Context
        int kpl_ = 0;
        std::vector<int> h_;
        int offset_ = 0;

        bool secret_key_generated_ = false;
        storage_type storage_type_ = storage_type::DEVICE;

        void validate_context(const HEContext<Scheme::TFHE>& context) const;
        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_SECRETKEY_H
