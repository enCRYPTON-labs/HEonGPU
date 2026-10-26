// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/secretkey.cuh>

namespace heongpu
{
    __host__ Secretkey<Scheme::TFHE>::Secretkey(HEContext<Scheme::TFHE> context,
                                                bool store_in_gpu)
    {
        if (!context)
        {
            throw ParameterError("HEContext is not set! Pass a context "
                                 "created with GenHEContext.");
        }

        context_ = context;
        // LWE Context
        n_ = context->n_;
        lwe_alpha_min = context->ks_stdev_;
        lwe_alpha_max = context->max_stdev_;

        // TLWE Context
        N_ = context->N_;
        k_ = context->k_;
        tlwe_alpha_min = context->bk_stdev_;
        tlwe_alpha_max = context->max_stdev_;

        // TGSW Context
        bk_l_ = context->bk_l_;
        bk_bg_bit_ = context->bk_bg_bit_;
        bg_ = context->bg_;
        half_bg_ = context->half_bg_;
        mask_mod_ = context->mask_mod_;
        kpl_ = context->kpl_;
        h_ = context->h_;
        offset_ = context->offset_;

        storage_type_ =
            store_in_gpu ? storage_type::DEVICE : storage_type::HOST;
    }

    void Secretkey<Scheme::TFHE>::validate_context(
        const HEContext<Scheme::TFHE>& context) const
    {
        if (!context)
            throw ParameterError("HEContext is not set! Pass a context "
                                 "created with GenHEContext.");
        if (!context_)
            throw ContextMismatchError("Secretkey context mismatch! The "
                                       "secret key has no context.");
        if (n_ != context->n_ || N_ != context->N_ || k_ != context->k_ ||
            bk_l_ != context->bk_l_ || bk_bg_bit_ != context->bk_bg_bit_ ||
            lwe_alpha_min != context->ks_stdev_ ||
            tlwe_alpha_min != context->bk_stdev_)
            throw ContextMismatchError(error_message(
                "Secretkey context mismatch! The key has n=", n_, ", N=", N_,
                ", k=", k_, ", bk_l=", bk_l_, ", bk_bg_bit=", bk_bg_bit_,
                ", the context has n=", context->n_, ", N=", context->N_,
                ", k=", context->k_, ", bk_l=", context->bk_l_,
                ", bk_bg_bit=", context->bk_bg_bit_,
                " (the noise parameters must match as well)."));
        if (secret_key_generated_)
        {
            const size_t lwe_size = storage_type_ == storage_type::DEVICE
                                        ? lwe_key_device_location_.size()
                                        : lwe_key_host_location_.size();
            const size_t tlwe_size = storage_type_ == storage_type::DEVICE
                                         ? tlwe_key_device_location_.size()
                                         : tlwe_key_host_location_.size();
            if (lwe_size != n_ || tlwe_size != k_ * N_)
                throw std::invalid_argument("Invalid Secretkey storage!");
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        }
    }

    void Secretkey<Scheme::TFHE>::store_in_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (secret_key_generated_)
            {
                lwe_key_device_location_ =
                    DeviceVector<int32_t>(lwe_key_host_location_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                lwe_key_host_location_.resize(0);
                lwe_key_host_location_.shrink_to_fit();

                tlwe_key_device_location_ =
                    DeviceVector<int32_t>(tlwe_key_host_location_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                tlwe_key_host_location_.resize(0);
                tlwe_key_host_location_.shrink_to_fit();
            }
            else
            {
                // pass
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Secretkey<Scheme::TFHE>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if (secret_key_generated_)
            {
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(this->stream()));
                lwe_key_host_location_ =
                    HostVector<int32_t>(lwe_key_device_location_.size());
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    lwe_key_host_location_.data(),
                    lwe_key_device_location_.data(),
                    lwe_key_device_location_.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream));

                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                lwe_key_device_location_.resize(0, stream);
                lwe_key_device_location_.shrink_to_fit(stream);

                tlwe_key_host_location_ =
                    HostVector<int32_t>(tlwe_key_device_location_.size());
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    tlwe_key_host_location_.data(),
                    tlwe_key_device_location_.data(),
                    tlwe_key_device_location_.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream));

                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                tlwe_key_device_location_.resize(0, stream);
                tlwe_key_device_location_.shrink_to_fit(stream);
            }
            else
            {
                // pass
            }

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    void Secretkey<Scheme::TFHE>::copy_to_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (secret_key_generated_)
            {
                lwe_key_device_location_ =
                    DeviceVector<int32_t>(lwe_key_host_location_, stream);

                tlwe_key_device_location_ =
                    DeviceVector<int32_t>(tlwe_key_host_location_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            else
            {
                // pass
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Secretkey<Scheme::TFHE>::remove_from_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            lwe_key_device_location_.resize(0, stream);
            lwe_key_device_location_.shrink_to_fit(stream);

            tlwe_key_device_location_.resize(0, stream);
            tlwe_key_device_location_.shrink_to_fit(stream);

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    void Secretkey<Scheme::TFHE>::remove_from_host()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            lwe_key_host_location_.resize(0);
            lwe_key_host_location_.shrink_to_fit();

            tlwe_key_host_location_.resize(0);
            tlwe_key_host_location_.shrink_to_fit();

            storage_type_ = storage_type::DEVICE;
        }
    }

} // namespace heongpu
