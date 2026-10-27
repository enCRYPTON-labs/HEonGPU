// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_CIPHERTEXT_H
#define HEONGPU_TFHE_CIPHERTEXT_H

#include <heongpu/host/tfhe/context.cuh>

namespace heongpu
{
    template <> class Ciphertext<Scheme::TFHE>
    {
        template <Scheme S> friend class HEEncryptor;
        template <Scheme S> friend class HEDecryptor;
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
        /**
         * @brief Constructs a new Ciphertext object with specified parameters
         * and CUDA stream.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        __host__
        Ciphertext(HEContext<Scheme::TFHE> context,
                   const ExecutionOptions& options = ExecutionOptions());

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
            a_device_location_.set_stream(stream);
            b_device_location_.set_stream(stream);
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
            return a_device_location_.stream();
        }

        /**
         * @brief Returns the size of the lwe size used in ciphertext.
         *
         * @return int Size of the lwe. (n)
         */
        inline int lwe_size() const noexcept { return n_; }

        /**
         * @brief Returns the how many lwe stored in the ciphertext.
         *
         * @return int Size of the ciphertext.
         */
        inline int size() const noexcept { return shape_; }

        // LWE mask and body in the current storage location (see
        // is_on_device()).
        int32_t* a_data() noexcept
        {
            return is_on_device() ? a_device_location_.data()
                                  : a_host_location_.data();
        }
        int32_t* b_data() noexcept
        {
            return is_on_device() ? b_device_location_.data()
                                  : b_host_location_.data();
        }

        Ciphertext() = default;

        Ciphertext(const Ciphertext& copy)
            : n_(copy.n_), alpha_min_(copy.alpha_min_),
              alpha_max_(copy.alpha_max_), shape_(copy.shape_),
              variances_(copy.variances_), storage_type_(copy.storage_type_),
              ciphertext_generated_(copy.ciphertext_generated_)
        {
            if (copy.ciphertext_generated_)
            {
                if (copy.storage_type_ == storage_type::device)
                {
                    a_device_location_.resize(copy.a_device_location_.size(),
                                              copy.a_device_location_.stream());
                    cudaMemcpyAsync(
                        a_device_location_.data(),
                        copy.a_device_location_.data(),
                        copy.a_device_location_.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice,
                        copy.a_device_location_
                            .stream()); // TODO: use cudaStreamPerThread

                    b_device_location_.resize(copy.b_device_location_.size(),
                                              copy.b_device_location_.stream());
                    cudaMemcpyAsync(
                        b_device_location_.data(),
                        copy.b_device_location_.data(),
                        copy.b_device_location_.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice,
                        copy.b_device_location_
                            .stream()); // TODO: use cudaStreamPerThread
                }
                else
                {
                    a_host_location_.resize(copy.a_host_location_.size());
                    std::memcpy(a_host_location_.data(),
                                copy.a_host_location_.data(),
                                copy.a_host_location_.size() * sizeof(int32_t));

                    b_host_location_.resize(copy.b_host_location_.size());
                    std::memcpy(b_host_location_.data(),
                                copy.b_host_location_.data(),
                                copy.b_host_location_.size() * sizeof(int32_t));
                }
            }
        }

        Ciphertext(Ciphertext&& assign) noexcept
            : n_(std::move(assign.n_)),
              alpha_min_(std::move(assign.alpha_min_)),
              alpha_max_(std::move(assign.alpha_max_)),
              shape_(std::move(assign.shape_)),
              variances_(std::move(assign.variances_)),
              storage_type_(std::move(assign.storage_type_)),
              ciphertext_generated_(std::move(assign.ciphertext_generated_)),
              a_device_location_(std::move(assign.a_device_location_)),
              b_device_location_(std::move(assign.b_device_location_)),
              a_host_location_(std::move(assign.a_host_location_)),
              b_host_location_(std::move(assign.b_host_location_))
        {
        }

        Ciphertext& operator=(const Ciphertext& copy)
        {
            if (this != &copy)
            {
                n_ = copy.n_;
                alpha_min_ = copy.alpha_min_;
                alpha_max_ = copy.alpha_max_;
                shape_ = copy.shape_;
                variances_ = copy.variances_;
                storage_type_ = copy.storage_type_;
                ciphertext_generated_ = copy.ciphertext_generated_;

                if (copy.ciphertext_generated_)
                {
                    if (copy.storage_type_ == storage_type::device)
                    {
                        a_device_location_.resize(
                            copy.a_device_location_.size(),
                            copy.a_device_location_.stream());
                        cudaMemcpyAsync(
                            a_device_location_.data(),
                            copy.a_device_location_.data(),
                            copy.a_device_location_.size() * sizeof(int32_t),
                            cudaMemcpyDeviceToDevice,
                            copy.a_device_location_
                                .stream()); // TODO: use cudaStreamPerThread

                        b_device_location_.resize(
                            copy.b_device_location_.size(),
                            copy.b_device_location_.stream());
                        cudaMemcpyAsync(
                            b_device_location_.data(),
                            copy.b_device_location_.data(),
                            copy.b_device_location_.size() * sizeof(int32_t),
                            cudaMemcpyDeviceToDevice,
                            copy.b_device_location_
                                .stream()); // TODO: use cudaStreamPerThread
                    }
                    else
                    {
                        a_host_location_.resize(copy.a_host_location_.size());
                        std::memcpy(a_host_location_.data(),
                                    copy.a_host_location_.data(),
                                    copy.a_host_location_.size() *
                                        sizeof(int32_t));

                        b_host_location_.resize(copy.b_host_location_.size());
                        std::memcpy(b_host_location_.data(),
                                    copy.b_host_location_.data(),
                                    copy.b_host_location_.size() *
                                        sizeof(int32_t));
                    }
                }
            }
            return *this;
        }

        Ciphertext& operator=(Ciphertext&& assign) noexcept
        {
            if (this != &assign)
            {
                n_ = std::move(assign.n_);
                alpha_min_ = std::move(assign.alpha_min_);
                alpha_max_ = std::move(assign.alpha_max_);
                shape_ = std::move(assign.shape_);
                variances_ = std::move(assign.variances_);
                storage_type_ = std::move(assign.storage_type_);
                ciphertext_generated_ = std::move(assign.ciphertext_generated_);
                a_device_location_ = std::move(assign.a_device_location_);
                b_device_location_ = std::move(assign.b_device_location_);
                a_host_location_ = std::move(assign.a_host_location_);
                b_host_location_ = std::move(assign.b_host_location_);
            }
            return *this;
        }

      private:
        const scheme_type scheme_ = scheme_type::tfhe;

        int n_;
        double alpha_min_;
        double alpha_max_;

        int shape_;

        std::vector<double> variances_;

        storage_type storage_type_;
        bool ciphertext_generated_ = false;

        DeviceVector<int32_t> a_device_location_;
        DeviceVector<int32_t> b_device_location_;

        HostVector<int32_t> a_host_location_;
        HostVector<int32_t> b_host_location_;

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_CIPHERTEXT_H