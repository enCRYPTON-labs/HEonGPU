// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/plaintext.cuh>

namespace heongpu
{
    __host__ Plaintext<Scheme::BFV>::Plaintext(HEContext<Scheme::BFV> context,
                                               const ExecutionOptions& options)
    {
        if (!context || !context->context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context->scheme_;
        plain_size_ = context->n;
        storage_type_ = options.storage_;
    }

    void Plaintext<Scheme::BFV>::store_in_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                device_locations_ =
                    DeviceVector<Data64>(host_locations_, stream);
                host_locations_.resize(0);
                host_locations_.shrink_to_fit();
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Plaintext<Scheme::BFV>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                host_locations_ = HostVector<Data64>(plain_size_);
                cudaMemcpyAsync(host_locations_.data(),
                                device_locations_.data(),
                                plain_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                device_locations_.resize(0, stream);
                device_locations_.shrink_to_fit(stream);
            }

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    Data64* Plaintext<Scheme::BFV>::data()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_locations_.data();
        }
        else
        {
            return host_locations_.data();
        }
    }

    void Plaintext<Scheme::BFV>::save(std::ostream& os) const
    {
        if (plaintext_generated_)
        {
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &plain_size_, sizeof(plain_size_));

            os.write((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            os.write((char*) &plaintext_generated_,
                     sizeof(plaintext_generated_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            if (storage_type_ == storage_type::DEVICE)
            {
                HostVector<Data64> host_locations_temp(plain_size_);
                cudaMemcpy(host_locations_temp.data(), device_locations_.data(),
                           plain_size_ * sizeof(Data64),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                os.write((char*) &plain_size_, sizeof(plain_size_));
                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * plain_size_);
            }
            else
            {
                os.write((char*) &plain_size_, sizeof(plain_size_));
                os.write((char*) host_locations_.data(),
                         sizeof(Data64) * plain_size_);
            }
        }
        else
        {
            throw std::runtime_error(
                "Plaintext is not generated so can not be serialized!");
        }
    }

    void Plaintext<Scheme::BFV>::load(std::istream& is)
    {
        if ((!plaintext_generated_))
        {
            is.read((char*) &scheme_, sizeof(scheme_));

            if (scheme_ != scheme_type::bfv)
            {
                throw std::runtime_error("Invalid scheme binary!");
            }

            is.read((char*) &plain_size_, sizeof(plain_size_));

            is.read((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            is.read((char*) &plaintext_generated_,
                    sizeof(plaintext_generated_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            storage_type_ = storage_type::DEVICE;
            plaintext_generated_ = true;

            uint32_t plaintext_memory_size;
            is.read((char*) &plaintext_memory_size,
                    sizeof(plaintext_memory_size));

            if (plaintext_memory_size != plain_size_)
            {
                throw std::runtime_error("Invalid plaintext size!");
            }

            HostVector<Data64> host_locations_temp(plaintext_memory_size);
            is.read((char*) host_locations_temp.data(),
                    sizeof(Data64) * plaintext_memory_size);

            device_locations_.resize(plaintext_memory_size);
            cudaMemcpy(device_locations_.data(), host_locations_temp.data(),
                       plaintext_memory_size * sizeof(Data64),
                       cudaMemcpyHostToDevice);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        else
        {
            throw std::runtime_error("Plaintext has been already exist!");
        }
    }

    int Plaintext<Scheme::BFV>::memory_size()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_locations_.size();
        }
        else
        {
            return host_locations_.size();
        }
    }

    void Plaintext<Scheme::BFV>::memory_clear(cudaStream_t stream)
    {
        if (device_locations_.size() > 0)
        {
            device_locations_.resize(0, stream);
            device_locations_.shrink_to_fit(stream);
        }

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void
    Plaintext<Scheme::BFV>::memory_set(DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::DEVICE;
        device_locations_ = std::move(new_device_vector);

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Plaintext<Scheme::BFV>::copy_to_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                device_locations_ =
                    DeviceVector<Data64>(host_locations_, stream);
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Plaintext<Scheme::BFV>::remove_from_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            device_locations_.resize(0, stream);
            device_locations_.shrink_to_fit(stream);

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    void Plaintext<Scheme::BFV>::remove_from_host()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();

            storage_type_ = storage_type::DEVICE;
        }
    }

} // namespace heongpu