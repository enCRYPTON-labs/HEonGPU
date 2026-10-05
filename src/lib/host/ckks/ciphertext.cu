// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/ciphertext.cuh>
#include <array>
#include <cmath>
#include <limits>

namespace heongpu
{

    __host__
    Ciphertext<Scheme::CKKS>::Ciphertext(HEContext<Scheme::CKKS> context,
                                         const ExecutionOptions& options)
    {
        if (!context || !context->context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context->scheme_;
        coeff_modulus_count_ = context->Q_size;
        cipher_size_ = 2;
        ring_size_ = context->n;
        depth_ = 0;

        int cipher_memory_size =
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;

        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = 0;
        encoding_ = encoding::SLOT;

        storage_type_ = options.storage_;

        if (storage_type_ == storage_type::DEVICE)
        {
            device_locations_ =
                DeviceVector<Data64>(cipher_memory_size, options.stream_);
        }
        else
        {
            host_locations_ = HostVector<Data64>(cipher_memory_size);
        }
    }

    void Ciphertext<Scheme::CKKS>::store_in_device(cudaStream_t stream)
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
                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
                host_locations_.resize(0);
                host_locations_.shrink_to_fit();
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Ciphertext<Scheme::CKKS>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                HEONGPU_CUDA_CHECK(
                    cudaStreamSynchronize(device_locations_.stream()));
                int cipher_memory_size =
                    cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;
                host_locations_ = HostVector<Data64>(cipher_memory_size);
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    host_locations_.data(), device_locations_.data(),
                    cipher_memory_size * sizeof(Data64), cudaMemcpyDeviceToHost,
                    stream));

                HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
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

    Data64* Ciphertext<Scheme::CKKS>::data()
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

    void Ciphertext<Scheme::CKKS>::get_data(std::vector<Data64>& cipher,
                                            cudaStream_t stream)
    {
        int cipher_memory_size =
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;

        if (cipher.size() < cipher_memory_size)
        {
            cipher.resize(cipher_memory_size);
        }

        if (storage_type_ == storage_type::DEVICE)
        {
            HEONGPU_CUDA_CHECK(
                cudaMemcpyAsync(cipher.data(), device_locations_.data(),
                                cipher_memory_size * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream));
        }
        else
        {
            cipher.resize(host_locations_.size());
            std::memcpy(cipher.data(), host_locations_.data(),
                        host_locations_.size() * sizeof(Data64));
        }
    }

    void Ciphertext<Scheme::CKKS>::get_data(HostVector<Data64>& cipher,
                                            cudaStream_t stream)
    {
        int cipher_memory_size =
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;

        if (cipher.size() < cipher_memory_size)
        {
            cipher.resize(cipher_memory_size);
        }

        if (storage_type_ == storage_type::DEVICE)
        {
            HEONGPU_CUDA_CHECK(
                cudaMemcpyAsync(cipher.data(), device_locations_.data(),
                                cipher_memory_size * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream));
        }
        else
        {
            cipher.resize(host_locations_.size());
            std::memcpy(cipher.data(), host_locations_.data(),
                        host_locations_.size() * sizeof(Data64));
        }
    }

    void Ciphertext<Scheme::CKKS>::save(std::ostream& os) const
    {
        if (ciphertext_generated_)
        {
            const size_t expected_size = checked_memory_size();
            if ((storage_type_ == storage_type::DEVICE
                     ? device_locations_.size()
                     : host_locations_.size()) < expected_size)
            {
                throw std::runtime_error("Invalid ciphertext storage size!");
            }
            HEONGPU_CUDA_CHECK(
                cudaStreamSynchronize(device_locations_.stream()));
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &ring_size_, sizeof(ring_size_));

            os.write((char*) &coeff_modulus_count_,
                     sizeof(coeff_modulus_count_));

            os.write((char*) &cipher_size_, sizeof(cipher_size_));

            os.write((char*) &depth_, sizeof(depth_));

            os.write((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            os.write((char*) &scale_, sizeof(scale_));

            os.write((char*) &encoding_, sizeof(encoding_));

            os.write((char*) &rescale_required_, sizeof(rescale_required_));

            os.write((char*) &relinearization_required_,
                     sizeof(relinearization_required_));

            os.write((char*) &ciphertext_generated_,
                     sizeof(ciphertext_generated_));

            if (storage_type_ == storage_type::DEVICE)
            {
                uint32_t ciphertext_memory_size =
                    cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;
                HostVector<Data64> host_locations_temp(ciphertext_memory_size);
                HEONGPU_CUDA_CHECK(cudaMemcpy(
                    host_locations_temp.data(), device_locations_.data(),
                    ciphertext_memory_size * sizeof(Data64),
                    cudaMemcpyDeviceToHost));

                os.write((char*) &ciphertext_memory_size,
                         sizeof(ciphertext_memory_size));
                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * ciphertext_memory_size);
            }
            else
            {
                uint32_t ciphertext_memory_size =
                    static_cast<uint32_t>(expected_size);
                os.write((char*) &ciphertext_memory_size,
                         sizeof(ciphertext_memory_size));
                os.write((char*) host_locations_.data(),
                         sizeof(Data64) * ciphertext_memory_size);
            }
            if (!os)
            {
                throw std::runtime_error("Failed to write ciphertext!");
            }
        }
        else
        {
            throw std::runtime_error(
                "Ciphertext is not generated so can not be serialized!");
        }
    }

    size_t Ciphertext<Scheme::CKKS>::checked_memory_size() const
    {
        if (scheme_ != scheme_type::ckks || ring_size_ < MIN_POLY_DEGREE ||
            ring_size_ > MAX_POLY_DEGREE ||
            (ring_size_ & (ring_size_ - 1)) != 0 || coeff_modulus_count_ <= 0 ||
            depth_ < 0 || depth_ >= coeff_modulus_count_ ||
            (cipher_size_ != 2 && cipher_size_ != 3) ||
            relinearization_required_ != (cipher_size_ == 3) ||
            !std::isfinite(scale_) || scale_ <= 0 ||
            (encoding_ != encoding::SLOT &&
             encoding_ != encoding::COEFFICIENT) ||
            (storage_type_ != storage_type::DEVICE &&
             storage_type_ != storage_type::HOST))
        {
            throw std::runtime_error("Invalid ciphertext parameters!");
        }
        const uint64_t size = uint64_t(cipher_size_) * uint64_t(ring_size_) *
                              uint64_t(coeff_modulus_count_ - depth_);
        // The library's kernel offsets and memory_size() use signed int.
        if (size > uint64_t(std::numeric_limits<int>::max()))
        {
            throw std::runtime_error("Ciphertext size is too large!");
        }
        return static_cast<size_t>(size);
    }

    void Ciphertext<Scheme::CKKS>::validate_context(
        const HEContext<Scheme::CKKS>& context) const
    {
        if (!ciphertext_generated_)
        {
            throw std::logic_error("Ciphertext is not generated!");
        }
        const size_t expected_size = checked_memory_size();
        if (!context || !context->context_generated_ ||
            ring_size_ != context->n || coeff_modulus_count_ != context->Q_size)
        {
            throw std::invalid_argument("Ciphertext context mismatch!");
        }
        const size_t actual_size = storage_type_ == storage_type::DEVICE
                                       ? device_locations_.size()
                                       : host_locations_.size();
        if (actual_size < expected_size)
        {
            throw std::invalid_argument("Invalid ciphertext storage size!");
        }
    }

    void Ciphertext<Scheme::CKKS>::load(std::istream& is)
    {
        if (ciphertext_generated_)
        {
            throw std::runtime_error("Ciphertext already exists!");
        }
        Ciphertext temporary;
        auto read = [&](auto& value)
        {
            is.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!is)
            {
                throw std::runtime_error("Truncated ciphertext binary!");
            }
        };
        auto read_bool = [&](bool& value)
        {
            static_assert(sizeof(bool) == sizeof(uint8_t));
            uint8_t byte;
            read(byte);
            if (byte > 1)
            {
                throw std::runtime_error("Invalid ciphertext boolean!");
            }
            value = byte != 0;
        };
        read(temporary.scheme_);
        read(temporary.ring_size_);
        read(temporary.coeff_modulus_count_);
        read(temporary.cipher_size_);
        read(temporary.depth_);
        read_bool(temporary.in_ntt_domain_);
        read(temporary.storage_type_);
        read(temporary.scale_);
        read(temporary.encoding_);
        read_bool(temporary.rescale_required_);
        read_bool(temporary.relinearization_required_);
        read_bool(temporary.ciphertext_generated_);
        uint32_t serialized_size;
        read(serialized_size);
        const size_t expected_size = temporary.checked_memory_size();
        if (!temporary.ciphertext_generated_ ||
            serialized_size != expected_size)
        {
            throw std::runtime_error(
                "Invalid ciphertext size or generation flag!");
        }
        if ((ring_size_ != 0 && ring_size_ != temporary.ring_size_) ||
            (coeff_modulus_count_ != 0 &&
             coeff_modulus_count_ != temporary.coeff_modulus_count_))
        {
            throw std::invalid_argument("Ciphertext context mismatch!");
        }

        // Do not allocate the claimed payload size before verifying the input.
        std::vector<Data64> payload;
        std::array<Data64, 512> chunk;
        for (size_t remaining = expected_size; remaining != 0;)
        {
            const size_t count = std::min(remaining, chunk.size());
            is.read(reinterpret_cast<char*>(chunk.data()),
                    count * sizeof(Data64));
            if (!is)
            {
                throw std::runtime_error("Truncated ciphertext payload!");
            }
            payload.insert(payload.end(), chunk.begin(), chunk.begin() + count);
            remaining -= count;
        }
        const auto stream = device_locations_.stream();
        DeviceVector<Data64> device(payload.size(), stream);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(device.data(), payload.data(),
                                           payload.size() * sizeof(Data64),
                                           cudaMemcpyHostToDevice, stream));
        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        temporary.memory_set(std::move(device));
        *this = std::move(temporary);
    }

    int Ciphertext<Scheme::CKKS>::memory_size()
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

    void Ciphertext<Scheme::CKKS>::memory_clear(cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::memory_set(
        DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::DEVICE;
        device_locations_ = std::move(new_device_vector);

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Ciphertext<Scheme::CKKS>::copy_to_device(cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::remove_from_device(cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::remove_from_host()
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
