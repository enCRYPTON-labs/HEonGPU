// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/publickey.cuh>

namespace heongpu
{
    __host__ Publickey<Scheme::BFV>::Publickey(HEContext<Scheme::BFV> context)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        context_ = context;
        scheme_ = context->scheme_;
        coeff_modulus_count_ = context->Q_prime_size;
        ring_size_ = context->n; // n
        in_ntt_domain_ = false;

        storage_type_ = storage_type::device;
    }

    Data64* Publickey<Scheme::BFV>::data()
    {
        if (storage_type_ == storage_type::device)
        {
            return device_locations_.data();
        }
        else
        {
            return host_locations_.data();
        }
    }

    void Publickey<Scheme::BFV>::store_in_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::device)
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
                // The copies above read host memory until they finish.
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                host_locations_.resize(0);
                host_locations_.shrink_to_fit();
            }

            storage_type_ = storage_type::device;
        }
    }

    void Publickey<Scheme::BFV>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::device)
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                int publickey_memory_size = device_locations_.size();
                host_locations_ = HostVector<Data64>(publickey_memory_size);
                cudaMemcpyAsync(host_locations_.data(),
                                device_locations_.data(),
                                publickey_memory_size * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                device_locations_.resize(0, stream);
                device_locations_.shrink_to_fit(stream);
            }

            storage_type_ = storage_type::host;
        }
        else
        {
            // pass
        }
    }

    void Publickey<Scheme::BFV>::save(std::ostream& os,
                                      serialization_format format) const
    {
        if (public_key_generated_)
        {
            detail::write_serialization_header(
                os, format, detail::serialized_object::public_key, scheme_,
                serialized_context_id());

            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &ring_size_, sizeof(ring_size_));

            os.write((char*) &coeff_modulus_count_,
                     sizeof(coeff_modulus_count_));

            os.write((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            os.write((char*) &public_key_generated_,
                     sizeof(public_key_generated_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            if (storage_type_ == storage_type::device)
            {
                uint32_t publickey_memory_size =
                    2 * coeff_modulus_count_ * ring_size_;
                HostVector<Data64> host_locations_temp(publickey_memory_size);
                cudaMemcpy(host_locations_temp.data(), device_locations_.data(),
                           publickey_memory_size * sizeof(Data64),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                os.write((char*) &publickey_memory_size,
                         sizeof(publickey_memory_size));

                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * publickey_memory_size);
            }
            else
            {
                uint32_t publickey_memory_size = host_locations_.size();
                os.write((char*) &publickey_memory_size,
                         sizeof(publickey_memory_size));
                os.write((char*) host_locations_.data(),
                         sizeof(Data64) * publickey_memory_size);
            }
        }
        else
        {
            throw std::runtime_error(
                "Publickey is not generated so can not be serialized!");
        }
    }

    void Publickey<Scheme::BFV>::load(std::istream& is)
    {
        if ((!public_key_generated_))
        {
            const std::uint64_t loaded_context_id =
                detail::read_serialization_header(
                    is, detail::serialized_object::public_key,
                    scheme_type::bfv);
            detail::validate_loaded_context_identity(
                serialized_context_id(), loaded_context_id, "Publickey");
            if (loaded_context_id != 0)
            {
                context_id_ = loaded_context_id;
            }

            is.read((char*) &scheme_, sizeof(scheme_));

            if (scheme_ != scheme_type::bfv)
            {
                throw SerializationError(
                    error_message("Invalid scheme binary! Expected BFV data, "
                                  "found scheme id ",
                                  static_cast<int>(scheme_), "."));
            }

            is.read((char*) &ring_size_, sizeof(ring_size_));

            is.read((char*) &coeff_modulus_count_,
                    sizeof(coeff_modulus_count_));

            is.read((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            is.read((char*) &public_key_generated_,
                    sizeof(public_key_generated_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            storage_type_ = storage_type::device;
            public_key_generated_ = true;

            uint32_t publickey_memory_size;
            is.read((char*) &publickey_memory_size,
                    sizeof(publickey_memory_size));

            if (publickey_memory_size !=
                (2 * ring_size_ * coeff_modulus_count_))
            {
                throw std::runtime_error("Invalid publickey size!");
            }

            HostVector<Data64> host_locations_temp(publickey_memory_size);
            is.read((char*) host_locations_temp.data(),
                    sizeof(Data64) * publickey_memory_size);

            device_locations_.resize(publickey_memory_size);
            cudaMemcpy(device_locations_.data(), host_locations_temp.data(),
                       publickey_memory_size * sizeof(Data64),
                       cudaMemcpyHostToDevice);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        else
        {
            throw std::runtime_error("Publickey has been already exist!");
        }
    }

    int Publickey<Scheme::BFV>::memory_size()
    {
        if (storage_type_ == storage_type::device)
        {
            return device_locations_.size();
        }
        else
        {
            return host_locations_.size();
        }
    }

    void Publickey<Scheme::BFV>::memory_clear(cudaStream_t stream)
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
    Publickey<Scheme::BFV>::memory_set(DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::device;
        device_locations_ = std::move(new_device_vector);

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Publickey<Scheme::BFV>::copy_to_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::device)
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

            storage_type_ = storage_type::device;
        }
    }

    void Publickey<Scheme::BFV>::remove_from_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::device)
        {
            device_locations_.resize(0, stream);
            device_locations_.shrink_to_fit(stream);

            storage_type_ = storage_type::host;
        }
        else
        {
            // pass
        }
    }

    void Publickey<Scheme::BFV>::remove_from_host()
    {
        if (storage_type_ == storage_type::device)
        {
            // pass
        }
        else
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();

            storage_type_ = storage_type::device;
        }
    }

    __host__ MultipartyPublickey<Scheme::BFV>::MultipartyPublickey(
        HEContext<Scheme::BFV> context, RNGSeed seed)
        : Publickey<Scheme::BFV>(context), seed_(seed)
    {
    }

} // namespace heongpu
