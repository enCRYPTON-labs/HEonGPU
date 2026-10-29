// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/plaintext.cuh>

namespace heongpu
{
    __host__ Plaintext<Scheme::CKKS>::Plaintext(HEContext<Scheme::CKKS> context,
                                                const ExecutionOptions& options)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        scheme_ = context->scheme_;
        context_id_ = context->context_id_;
        plain_size_ = context->n;
        depth_ = 0;
        scale_ = 0;
        encoding_ = encoding::slot;
        storage_type_ = options.storage_;
    }

    void Plaintext<Scheme::CKKS>::store_in_device(cudaStream_t stream)
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

    void Plaintext<Scheme::CKKS>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::device)
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

            storage_type_ = storage_type::host;
        }
        else
        {
            // pass
        }
    }

    Data64* Plaintext<Scheme::CKKS>::data()
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

    void Plaintext<Scheme::CKKS>::save(std::ostream& os,
                                       serialization_format format) const
    {
        if (plaintext_generated_)
        {
            detail::write_serialization_header(
                os, format, detail::serialized_object::plaintext, scheme_,
                context_id_);

            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &plain_size_, sizeof(plain_size_));

            os.write((char*) &depth_, sizeof(depth_));

            os.write((char*) &scale_, sizeof(scale_));

            os.write((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            os.write((char*) &encoding_, sizeof(encoding_));

            os.write((char*) &plaintext_generated_,
                     sizeof(plaintext_generated_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            if (storage_type_ == storage_type::device)
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

    void Plaintext<Scheme::CKKS>::load(std::istream& is)
    {
        if ((!plaintext_generated_))
        {
            const std::uint64_t loaded_context_id =
                detail::read_serialization_header(
                    is, detail::serialized_object::plaintext,
                    scheme_type::ckks);
            detail::validate_loaded_context_identity(
                context_id_, loaded_context_id, "Plaintext");
            if (loaded_context_id != 0)
            {
                context_id_ = loaded_context_id;
            }

            is.read((char*) &scheme_, sizeof(scheme_));

            if (scheme_ != scheme_type::ckks)
            {
                throw SerializationError(
                    error_message("Invalid scheme binary! Expected CKKS data, "
                                  "found scheme id ",
                                  static_cast<int>(scheme_), "."));
            }

            is.read((char*) &plain_size_, sizeof(plain_size_));

            is.read((char*) &depth_, sizeof(depth_));

            is.read((char*) &scale_, sizeof(scale_));

            is.read((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            is.read((char*) &encoding_, sizeof(encoding_));

            is.read((char*) &plaintext_generated_,
                    sizeof(plaintext_generated_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            storage_type_ = storage_type::device;
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

    void Plaintext<Scheme::CKKS>::validate_context(
        const HEContext<Scheme::CKKS>& context) const
    {
        validate_context_identity(context_id_, context, "Plaintext");
    }

    int Plaintext<Scheme::CKKS>::memory_size() const
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

    void Plaintext<Scheme::CKKS>::memory_clear(cudaStream_t stream)
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

    void Plaintext<Scheme::CKKS>::memory_set(
        DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::device;
        device_locations_ = std::move(new_device_vector);

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Plaintext<Scheme::CKKS>::copy_to_device(cudaStream_t stream)
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

    void Plaintext<Scheme::CKKS>::remove_from_device(cudaStream_t stream)
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

    void Plaintext<Scheme::CKKS>::remove_from_host()
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

    std::string Plaintext<Scheme::CKKS>::to_string() const
    {
        if (!plaintext_generated_)
        {
            return "Plaintext<CKKS>(not generated)";
        }
        std::ostringstream stream;
        stream << "Plaintext<CKKS>(size=" << plain_size_ << ", depth=" << depth_
               << ", scale=2^" << std::log2(scale_) << ", "
               << detail::name(encoding_) << ", "
               << (in_ntt_domain_ ? "NTT" : "coefficient") << " domain, "
               << detail::name(storage_type_) << ")";
        return stream.str();
    }

    std::ostream& operator<<(std::ostream& os,
                             const Plaintext<Scheme::CKKS>& object)
    {
        return os << object.to_string();
    }

} // namespace heongpu
