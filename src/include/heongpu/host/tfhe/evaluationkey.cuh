// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_EVALUATIONKEY_H
#define HEONGPU_TFHE_EVALUATIONKEY_H

#include <heongpu/host/tfhe/context.cuh>
#include <heongpu/kernel/keygeneration.cuh>

namespace heongpu
{
    template <> class Bootstrappingkey<Scheme::TFHE>
    {
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HELogicOperator;

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
        __host__ Bootstrappingkey(HEContext<Scheme::TFHE> context,
                                  bool store_in_gpu = true);

        /**
         * @brief Stores the Bootstrappingkey in the device (GPU) memory.
         */
        void store_in_device(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Stores the Bootstrappingkey in the host (CPU) memory.
         */
        void store_in_host(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Checks whether the data is stored on the device (GPU) memory.
         */
        bool is_on_device() const noexcept
        {
            if (storage_type_ == storage_type::device)
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
            boot_key_device_location_.set_stream(stream);
            switch_key_device_location_a_.set_stream(stream);
            switch_key_device_location_b_.set_stream(stream);
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
            return boot_key_device_location_.stream();
        }

        /**
         * @brief Indicates whether the boot key has been generated.
         *
         * @return true if the boot key has been generated, false otherwise.
         */
        inline bool is_generated() const noexcept
        {
            return boot_key_generated_;
        }

        Bootstrappingkey() = default;

        void set_context(HEContext<Scheme::TFHE> context)
        {
            if (!context)
                throw ParameterError("HEContext is not set! Pass a context "
                                     "created with GenHEContext.");
            if (boot_key_generated_)
            {
                if (bk_k_ != context->k_ ||
                    bk_base_bit_ != context->bk_bg_bit_ ||
                    bk_length_ != context->bk_l_ ||
                    bk_stdev_ != context->bk_stdev_ ||
                    ks_base_bit_ != context->ks_base_bit_ ||
                    ks_length_ != context->ks_length_)
                    throw std::invalid_argument(
                        "Bootstrappingkey context mismatch!");
                context_ = std::move(context);
            }
            else
                *this = Bootstrappingkey(std::move(context));
        }

        /**
         * @brief Copy constructor for creating a new Bootstrappingkey object by
         * copying an existing one.
         *
         * This constructor performs a deep copy of the secret key data,
         * ensuring that the new object has its own independent copy of the
         * data. GPU memory operations are handled using `cudaMemcpyAsync` for
         * asynchronous data transfer.
         *
         * @param copy The source Bootstrappingkey object to copy from.
         */
        Bootstrappingkey(const Bootstrappingkey& copy)
            : context_(copy.context_), bk_k_(copy.bk_k_),
              bk_base_bit_(copy.bk_base_bit_), bk_length_(copy.bk_length_),
              bk_stdev_(copy.bk_stdev_),
              boot_key_variances_(copy.boot_key_variances_),
              ks_base_bit_(copy.ks_base_bit_), ks_length_(copy.ks_length_),
              switch_key_variances_(copy.switch_key_variances_),
              boot_key_generated_(copy.boot_key_generated_),
              storage_type_(copy.storage_type_)
        {
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(copy.stream()));
            if (copy.boot_key_generated_)
            {
                if (copy.storage_type_ == storage_type::device)
                {
                    boot_key_device_location_.resize(
                        copy.boot_key_device_location_.size(),
                        copy.boot_key_device_location_.stream());
                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        boot_key_device_location_.data(),
                        copy.boot_key_device_location_.data(),
                        copy.boot_key_device_location_.size() * sizeof(Data64),
                        cudaMemcpyDeviceToDevice,
                        copy.boot_key_device_location_.stream()));
                    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(copy.stream()));

                    switch_key_device_location_a_.resize(
                        copy.switch_key_device_location_a_.size(),
                        copy.switch_key_device_location_a_.stream());
                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        switch_key_device_location_a_.data(),
                        copy.switch_key_device_location_a_.data(),
                        copy.switch_key_device_location_a_.size() *
                            sizeof(int32_t),
                        cudaMemcpyDeviceToDevice,
                        copy.switch_key_device_location_a_.stream()));
                    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(copy.stream()));

                    switch_key_device_location_b_.resize(
                        copy.switch_key_device_location_b_.size(),
                        copy.switch_key_device_location_b_.stream());
                    HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                        switch_key_device_location_b_.data(),
                        copy.switch_key_device_location_b_.data(),
                        copy.switch_key_device_location_b_.size() *
                            sizeof(int32_t),
                        cudaMemcpyDeviceToDevice,
                        copy.switch_key_device_location_b_.stream()));
                    HEONGPU_CUDA_CHECK(cudaStreamSynchronize(copy.stream()));
                }
                else
                {
                    boot_key_host_location_.resize(
                        copy.boot_key_host_location_.size());
                    std::memcpy(boot_key_host_location_.data(),
                                copy.boot_key_host_location_.data(),
                                copy.boot_key_host_location_.size() *
                                    sizeof(Data64));

                    switch_key_host_location_a_.resize(
                        copy.switch_key_host_location_a_.size());
                    std::memcpy(switch_key_host_location_a_.data(),
                                copy.switch_key_host_location_a_.data(),
                                copy.switch_key_host_location_a_.size() *
                                    sizeof(int32_t));

                    switch_key_host_location_b_.resize(
                        copy.switch_key_host_location_b_.size());
                    std::memcpy(switch_key_host_location_b_.data(),
                                copy.switch_key_host_location_b_.data(),
                                copy.switch_key_host_location_b_.size() *
                                    sizeof(int32_t));
                }
            }
        }

        /**
         * @brief Move constructor for transferring ownership of a
         * Bootstrappingkey object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the newly constructed object. The source object is left in a valid
         * but undefined state.
         *
         * @param assign The source Bootstrappingkey object to move from.
         */
        Bootstrappingkey(Bootstrappingkey&& assign) noexcept
            : context_(std::move(assign.context_)),
              bk_k_(std::move(assign.bk_k_)),
              bk_base_bit_(std::move(assign.bk_base_bit_)),
              bk_length_(std::move(assign.bk_length_)),
              bk_stdev_(std::move(assign.bk_stdev_)),
              boot_key_variances_(std::move(assign.boot_key_variances_)),
              ks_base_bit_(std::move(assign.ks_base_bit_)),
              ks_length_(std::move(assign.ks_length_)),
              switch_key_variances_(std::move(assign.switch_key_variances_)),
              boot_key_generated_(std::move(assign.boot_key_generated_)),
              storage_type_(std::move(assign.storage_type_)),
              boot_key_device_location_(
                  std::move(assign.boot_key_device_location_)),
              boot_key_host_location_(
                  std::move(assign.boot_key_host_location_)),
              switch_key_device_location_a_(
                  std::move(assign.switch_key_device_location_a_)),
              switch_key_host_location_a_(
                  std::move(assign.switch_key_host_location_a_)),
              switch_key_device_location_b_(
                  std::move(assign.switch_key_device_location_b_)),
              switch_key_host_location_b_(
                  std::move(assign.switch_key_host_location_b_))
        {
            assign.boot_key_generated_ = false;
        }

        /**
         * @brief Copy assignment operator for assigning one Bootstrappingkey
         * object to another.
         *
         * Performs a deep copy of the secret key data, ensuring that the target
         * object has its own independent copy. GPU memory is copied using
         * `cudaMemcpyAsync`.
         *
         * @param copy The source Bootstrappingkey object to copy from.
         * @return Reference to the assigned object.
         */
        Bootstrappingkey& operator=(const Bootstrappingkey& copy)
        {
            if (this != &copy)
            {
                Bootstrappingkey temporary(copy);
                *this = std::move(temporary);
            }
            return *this;
        }

        /**
         * @brief Move assignment operator for transferring ownership of a
         * Bootstrappingkey object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the target object. The source object is left in a valid but
         * undefined state.
         *
         * @param assign The source Bootstrappingkey object to move from.
         * @return Reference to the assigned object.
         */
        Bootstrappingkey& operator=(Bootstrappingkey&& assign) noexcept
        {
            if (this != &assign)
            {
                context_ = std::move(assign.context_);
                bk_k_ = std::move(assign.bk_k_);
                bk_base_bit_ = std::move(assign.bk_base_bit_);
                bk_length_ = std::move(assign.bk_length_);
                bk_stdev_ = std::move(assign.bk_stdev_);
                boot_key_variances_ = std::move(assign.boot_key_variances_);
                ks_base_bit_ = std::move(assign.ks_base_bit_);
                ks_length_ = std::move(assign.ks_length_);
                switch_key_variances_ = std::move(assign.switch_key_variances_);
                boot_key_generated_ = std::move(assign.boot_key_generated_);
                storage_type_ = std::move(assign.storage_type_);
                boot_key_device_location_ =
                    std::move(assign.boot_key_device_location_);
                boot_key_host_location_ =
                    std::move(assign.boot_key_host_location_);
                switch_key_device_location_a_ =
                    std::move(assign.switch_key_device_location_a_);
                switch_key_host_location_a_ =
                    std::move(assign.switch_key_host_location_a_);
                switch_key_device_location_b_ =
                    std::move(assign.switch_key_device_location_b_);
                switch_key_host_location_b_ =
                    std::move(assign.switch_key_host_location_b_);
                assign.boot_key_generated_ = false;
            }
            return *this;
        }

      private:
        HEContext<Scheme::TFHE> context_;
        const scheme_type scheme_ = scheme_type::tfhe;

        // Boot Key Context
        int bk_k_ = 0;
        int bk_base_bit_ = 0;
        int bk_length_ = 0;
        double bk_stdev_ = 0;

        DeviceVector<Data64> boot_key_device_location_;
        HostVector<Data64> boot_key_host_location_;

        std::vector<double> boot_key_variances_;

        // Switch Key Context
        int ks_base_bit_ = 0;
        int ks_length_ = 0;

        DeviceVector<int32_t> switch_key_device_location_a_;
        HostVector<int32_t> switch_key_host_location_a_;

        DeviceVector<int32_t> switch_key_device_location_b_;
        HostVector<int32_t> switch_key_host_location_b_;

        std::vector<double> switch_key_variances_;

        bool boot_key_generated_ = false;
        storage_type storage_type_ = storage_type::device;

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_EVALUATIONKEY_H
