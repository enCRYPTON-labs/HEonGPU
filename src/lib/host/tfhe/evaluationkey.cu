// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/evaluationkey.cuh>

namespace heongpu
{
    __host__ Bootstrappingkey<Scheme::TFHE>::Bootstrappingkey(
        HEContext<Scheme::TFHE> context, bool store_in_gpu)
    {
        if (!context)
        {
            throw std::invalid_argument("HEContext is not set!");
        }

        context_ = context;
        bk_k_ = context->k_;
        bk_base_bit_ = context->bk_bg_bit_;
        bk_length_ = context->bk_l_;
        bk_stdev_ = context->bk_stdev_;

        ks_length_ = context->ks_length_;
        ks_base_bit_ = context->ks_base_bit_;

        storage_type_ =
            store_in_gpu ? storage_type::DEVICE : storage_type::HOST;
    }

    void Bootstrappingkey<Scheme::TFHE>::store_in_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (boot_key_generated_)
            {
                boot_key_device_location_ =
                    DeviceVector<Data64>(boot_key_host_location_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                boot_key_host_location_.resize(0);
                boot_key_host_location_.shrink_to_fit();

                switch_key_device_location_a_ =
                    DeviceVector<int32_t>(switch_key_host_location_a_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                switch_key_host_location_a_.resize(0);
                switch_key_host_location_a_.shrink_to_fit();

                switch_key_device_location_b_ =
                    DeviceVector<int32_t>(switch_key_host_location_b_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                switch_key_host_location_b_.resize(0);
                switch_key_host_location_b_.shrink_to_fit();
            }
            else
            {
                // pass
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Bootstrappingkey<Scheme::TFHE>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if (boot_key_generated_)
            {
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(this->stream()));
                boot_key_host_location_ =
                    HostVector<Data64>(boot_key_device_location_.size());
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    boot_key_host_location_.data(),
                    boot_key_device_location_.data(),
                    boot_key_device_location_.size() * sizeof(Data64),
                    cudaMemcpyDeviceToHost, stream));

                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                boot_key_device_location_.resize(0, stream);
                boot_key_device_location_.shrink_to_fit(stream);

                switch_key_host_location_a_ =
                    HostVector<int32_t>(switch_key_device_location_a_.size());
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    switch_key_host_location_a_.data(),
                    switch_key_device_location_a_.data(),
                    switch_key_device_location_a_.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream));

                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                switch_key_device_location_a_.resize(0, stream);
                switch_key_device_location_a_.shrink_to_fit(stream);

                switch_key_host_location_b_ =
                    HostVector<int32_t>(switch_key_device_location_b_.size());
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    switch_key_host_location_b_.data(),
                    switch_key_device_location_b_.data(),
                    switch_key_device_location_b_.size() * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream));

                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                switch_key_device_location_b_.resize(0, stream);
                switch_key_device_location_b_.shrink_to_fit(stream);
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

    void Bootstrappingkey<Scheme::TFHE>::copy_to_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (boot_key_generated_)
            {
                boot_key_device_location_ =
                    DeviceVector<Data64>(boot_key_host_location_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

                switch_key_device_location_a_ =
                    DeviceVector<int32_t>(switch_key_host_location_a_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));

                switch_key_device_location_b_ =
                    DeviceVector<int32_t>(switch_key_host_location_b_, stream);
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            else
            {
                // pass
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Bootstrappingkey<Scheme::TFHE>::remove_from_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            boot_key_device_location_.resize(0, stream);
            boot_key_device_location_.shrink_to_fit(stream);

            switch_key_device_location_a_.resize(0, stream);
            switch_key_device_location_a_.shrink_to_fit(stream);

            switch_key_device_location_b_.resize(0, stream);
            switch_key_device_location_b_.shrink_to_fit(stream);

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    void Bootstrappingkey<Scheme::TFHE>::remove_from_host()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            boot_key_host_location_.resize(0);
            boot_key_host_location_.shrink_to_fit();

            switch_key_host_location_a_.resize(0);
            switch_key_host_location_a_.shrink_to_fit();

            switch_key_host_location_b_.resize(0);
            switch_key_host_location_b_.shrink_to_fit();

            storage_type_ = storage_type::DEVICE;
        }
    }

} // namespace heongpu
