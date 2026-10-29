// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/ciphertext.cuh>

namespace heongpu
{
    __host__
    Ciphertext<Scheme::BFV>::Ciphertext(HEContext<Scheme::BFV> context,
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
        coeff_modulus_count_ = context->Q_size;
        cipher_size_ = 2;
        ring_size_ = context->n;

        int cipher_memory_size =
            cipher_size_ * coeff_modulus_count_ * ring_size_;

        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        relinearization_required_ = false;

        storage_type_ = options.storage_;

        if (storage_type_ == storage_type::device)
        {
            device_locations_ =
                DeviceVector<Data64>(cipher_memory_size, options.stream_);
        }
        else
        {
            host_locations_ = HostVector<Data64>(cipher_memory_size);
        }
    }

    void Ciphertext<Scheme::BFV>::store_in_device(cudaStream_t stream)
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
                host_locations_.resize(0);
                host_locations_.shrink_to_fit();
            }

            storage_type_ = storage_type::device;
        }
    }

    void Ciphertext<Scheme::BFV>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::device)
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                int cipher_memory_size =
                    cipher_size_ * coeff_modulus_count_ * ring_size_;
                host_locations_ = HostVector<Data64>(cipher_memory_size);
                cudaMemcpyAsync(host_locations_.data(),
                                device_locations_.data(),
                                cipher_memory_size * sizeof(Data64),
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

    Data64* Ciphertext<Scheme::BFV>::data()
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

    const Data64* Ciphertext<Scheme::BFV>::data() const
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

    void Ciphertext<Scheme::BFV>::get_data(std::vector<Data64>& cipher,
                                           cudaStream_t stream)
    {
        int cipher_memory_size =
            cipher_size_ * coeff_modulus_count_ * ring_size_;

        if (cipher.size() < cipher_memory_size)
        {
            cipher.resize(cipher_memory_size);
        }

        if (storage_type_ == storage_type::device)
        {
            cudaMemcpyAsync(cipher.data(), device_locations_.data(),
                            cipher_memory_size * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            cipher.resize(host_locations_.size());
            std::memcpy(cipher.data(), host_locations_.data(),
                        host_locations_.size() * sizeof(Data64));
        }
    }

    void Ciphertext<Scheme::BFV>::get_data(HostVector<Data64>& cipher,
                                           cudaStream_t stream)
    {
        int cipher_memory_size =
            cipher_size_ * coeff_modulus_count_ * ring_size_;

        if (cipher.size() < cipher_memory_size)
        {
            cipher.resize(cipher_memory_size);
        }

        if (storage_type_ == storage_type::device)
        {
            cudaMemcpyAsync(cipher.data(), device_locations_.data(),
                            cipher_memory_size * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            cipher.resize(host_locations_.size());
            std::memcpy(cipher.data(), host_locations_.data(),
                        host_locations_.size() * sizeof(Data64));
        }
    }

    void Ciphertext<Scheme::BFV>::save(std::ostream& os,
                                       serialization_format format) const
    {
        if (ciphertext_generated_)
        {
            detail::write_serialization_header(
                os, format, detail::serialized_object::ciphertext, scheme_,
                context_id_);

            os.write((char*) &scheme_, sizeof(scheme_));

            if (scheme_ != scheme_type::bfv)
            {
                throw SerializationError(
                    error_message("Invalid scheme binary! Expected BFV data, "
                                  "found scheme id ",
                                  static_cast<int>(scheme_), "."));
            }

            os.write((char*) &ring_size_, sizeof(ring_size_));

            os.write((char*) &coeff_modulus_count_,
                     sizeof(coeff_modulus_count_));

            os.write((char*) &cipher_size_, sizeof(cipher_size_));

            os.write((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            os.write((char*) &relinearization_required_,
                     sizeof(relinearization_required_));

            os.write((char*) &ciphertext_generated_,
                     sizeof(ciphertext_generated_));

            if (storage_type_ == storage_type::device)
            {
                uint32_t ciphertext_memory_size =
                    cipher_size_ * coeff_modulus_count_ * ring_size_;
                HostVector<Data64> host_locations_temp(ciphertext_memory_size);
                cudaMemcpy(host_locations_temp.data(), device_locations_.data(),
                           ciphertext_memory_size * sizeof(Data64),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                os.write((char*) &ciphertext_memory_size,
                         sizeof(ciphertext_memory_size));
                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * ciphertext_memory_size);
            }
            else
            {
                uint32_t ciphertext_memory_size = host_locations_.size();
                os.write((char*) &ciphertext_memory_size,
                         sizeof(ciphertext_memory_size));
                os.write((char*) host_locations_.data(),
                         sizeof(Data64) * ciphertext_memory_size);
            }
        }
        else
        {
            throw std::runtime_error(
                "Ciphertext is not generated so can not be serialized!");
        }
    }

    void Ciphertext<Scheme::BFV>::load(std::istream& is)
    {
        if ((!ciphertext_generated_))
        {
            const std::uint64_t loaded_context_id =
                detail::read_serialization_header(
                    is, detail::serialized_object::ciphertext,
                    scheme_type::bfv);
            detail::validate_loaded_context_identity(
                context_id_, loaded_context_id, "Ciphertext");
            if (loaded_context_id != 0)
            {
                context_id_ = loaded_context_id;
            }

            is.read((char*) &scheme_, sizeof(scheme_));

            is.read((char*) &ring_size_, sizeof(ring_size_));

            is.read((char*) &coeff_modulus_count_,
                    sizeof(coeff_modulus_count_));

            is.read((char*) &cipher_size_, sizeof(cipher_size_));

            is.read((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            is.read((char*) &relinearization_required_,
                    sizeof(relinearization_required_));

            is.read((char*) &ciphertext_generated_,
                    sizeof(ciphertext_generated_));

            storage_type_ = storage_type::device;
            ciphertext_generated_ = true;

            uint32_t ciphertext_memory_size;
            is.read((char*) &ciphertext_memory_size,
                    sizeof(ciphertext_memory_size));

            if (ciphertext_memory_size !=
                (cipher_size_ * ring_size_ * coeff_modulus_count_))
            {
                throw std::runtime_error("Invalid ciphertext size!");
            }

            HostVector<Data64> host_locations_temp(ciphertext_memory_size);
            is.read((char*) host_locations_temp.data(),
                    sizeof(Data64) * ciphertext_memory_size);

            device_locations_.resize(ciphertext_memory_size);
            cudaMemcpy(device_locations_.data(), host_locations_temp.data(),
                       ciphertext_memory_size * sizeof(Data64),
                       cudaMemcpyHostToDevice);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        else
        {
            throw std::runtime_error("Ciphertext has been already exist!");
        }
    }

    void Ciphertext<Scheme::BFV>::validate_context(
        const HEContext<Scheme::BFV>& context) const
    {
        validate_context_identity(context_id_, context, "Ciphertext");
    }

    int Ciphertext<Scheme::BFV>::memory_size() const
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

    void Ciphertext<Scheme::BFV>::memory_clear(cudaStream_t stream)
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

    void Ciphertext<Scheme::BFV>::memory_set(
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

    void Ciphertext<Scheme::BFV>::copy_to_device(cudaStream_t stream)
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

    void Ciphertext<Scheme::BFV>::remove_from_device(cudaStream_t stream)
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

    void Ciphertext<Scheme::BFV>::remove_from_host()
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

    std::string Ciphertext<Scheme::BFV>::to_string() const
    {
        if (!ciphertext_generated_)
        {
            return "Ciphertext<BFV>(not generated)";
        }
        std::ostringstream stream;
        stream << "Ciphertext<BFV>(N=" << ring_size_
               << ", size=" << cipher_size_
               << ", moduli=" << coeff_modulus_count_ << ", "
               << (in_ntt_domain_ ? "NTT" : "coefficient") << " domain, "
               << detail::name(storage_type_);
        if (relinearization_required_)
        {
            stream << ", relinearization required";
        }
        stream << ")";
        return stream.str();
    }

    std::ostream& operator<<(std::ostream& os,
                             const Ciphertext<Scheme::BFV>& object)
    {
        return os << object.to_string();
    }

} // namespace heongpu