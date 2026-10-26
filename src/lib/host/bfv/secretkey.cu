// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/secretkey.cuh>
#include "../secure_random.h"

namespace heongpu
{
    __host__ Secretkey<Scheme::BFV>::Secretkey(HEContext<Scheme::BFV> context)
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
        n_power_ = context->n_power;

        hamming_weight_ = ring_size_ >> 1; // default
        in_ntt_domain_ = false;

        storage_type_ = storage_type::DEVICE;
    }

    __host__ Secretkey<Scheme::BFV>::Secretkey(HEContext<Scheme::BFV> context,
                                               int hamming_weight)
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
        n_power_ = context->n_power;

        hamming_weight_ = hamming_weight;
        if ((hamming_weight_ <= 0) || (hamming_weight_ > ring_size_))
        {
            throw std::invalid_argument(
                "Hamming weight has to be in range 0 to ring size.");
        }

        in_ntt_domain_ = false;

        storage_type_ = storage_type::DEVICE;
    }

    __host__
    Secretkey<Scheme::BFV>::Secretkey(const std::vector<int>& secret_key,
                                      HEContext<Scheme::BFV> context,
                                      cudaStream_t stream)
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
        n_power_ = context->n_power;

        hamming_weight_ = 0;

        if (!(secret_key.size() == ring_size_))
        {
            throw std::invalid_argument("Secretkey size should be valid!");
        }

        for (size_t i = 0; i < secret_key.size(); ++i)
        {
            if (secret_key[i] < -1 || secret_key[i] > 1)
                throw std::invalid_argument(
                    "Secretkey coefficients must be ternary!");
            hamming_weight_ += secret_key[i] != 0;
        }
        if (hamming_weight_ == 0)
            throw std::invalid_argument("Secretkey cannot be zero!");

        DeviceVector<int> secret_key_device(ring_size_, stream);
        HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
            secret_key_device.data(), secret_key.data(),
            ring_size_ * sizeof(int), cudaMemcpyHostToDevice, stream));

        if (device_locations_.size() < coeff_modulus_count_ * ring_size_)
        {
            device_locations_ =
                DeviceVector<Data64>(coeff_modulus_count_ * ring_size_, stream);
        }

        secretkey_rns_kernel<<<dim3((ring_size_ >> 8), 1, 1), 256, 0, stream>>>(
            secret_key_device.data(), device_locations_.data(),
            context->modulus_->data(), n_power_, coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(device_locations_.data(),
                                context->ntt_table_->data(),
                                context->modulus_->data(), cfg_ntt,
                                coeff_modulus_count_, coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        in_ntt_domain_ = true;
        storage_type_ = storage_type::DEVICE;
        secret_key_generated_ = true;
    }

    __host__
    Secretkey<Scheme::BFV>::Secretkey(const HostVector<int>& secret_key,
                                      HEContext<Scheme::BFV> context,
                                      cudaStream_t stream)
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
        n_power_ = context->n_power;

        hamming_weight_ = 0;

        if (!(secret_key.size() == ring_size_))
        {
            throw std::invalid_argument("Secretkey size should be valid!");
        }

        for (size_t i = 0; i < secret_key.size(); ++i)
        {
            if (secret_key[i] < -1 || secret_key[i] > 1)
                throw std::invalid_argument(
                    "Secretkey coefficients must be ternary!");
            hamming_weight_ += secret_key[i] != 0;
        }
        if (hamming_weight_ == 0)
            throw std::invalid_argument("Secretkey cannot be zero!");

        DeviceVector<int> secret_key_device(secret_key, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (device_locations_.size() < coeff_modulus_count_ * ring_size_)
        {
            device_locations_ =
                DeviceVector<Data64>(coeff_modulus_count_ * ring_size_, stream);
        }

        secretkey_rns_kernel<<<dim3((ring_size_ >> 8), 1, 1), 256, 0, stream>>>(
            secret_key_device.data(), device_locations_.data(),
            context->modulus_->data(), n_power_, coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(device_locations_.data(),
                                context->ntt_table_->data(),
                                context->modulus_->data(), cfg_ntt,
                                coeff_modulus_count_, coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        HEONGPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        in_ntt_domain_ = true;
        storage_type_ = storage_type::DEVICE;
        secret_key_generated_ = true;
    }

    Data64* Secretkey<Scheme::BFV>::data()
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

    void Secretkey<Scheme::BFV>::store_in_device(cudaStream_t stream)
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

    void Secretkey<Scheme::BFV>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            HEONGPU_CUDA_CHECK(
                cudaStreamSynchronize(device_locations_.stream()));
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                int secretkey_memory_size = coeff_modulus_count_ * ring_size_;
                host_locations_ = HostVector<Data64>(secretkey_memory_size);
                HEONGPU_CUDA_CHECK(cudaMemcpyAsync(
                    host_locations_.data(), device_locations_.data(),
                    secretkey_memory_size * sizeof(Data64),
                    cudaMemcpyDeviceToHost, stream));

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

    void Secretkey<Scheme::BFV>::save(std::ostream& os) const
    {
        if (secret_key_generated_)
        {
            HEONGPU_CUDA_CHECK(
                cudaStreamSynchronize(device_locations_.stream()));
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &ring_size_, sizeof(ring_size_));

            os.write((char*) &coeff_modulus_count_,
                     sizeof(coeff_modulus_count_));

            os.write((char*) &n_power_, sizeof(n_power_));

            os.write((char*) &hamming_weight_, sizeof(hamming_weight_));

            os.write((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            os.write((char*) &secret_key_generated_,
                     sizeof(secret_key_generated_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            if (storage_type_ == storage_type::DEVICE)
            {
                uint32_t secretkey_memory_size =
                    coeff_modulus_count_ * ring_size_;
                HostVector<Data64> host_locations_temp(secretkey_memory_size);
                HEONGPU_CUDA_CHECK(cudaMemcpy(
                    host_locations_temp.data(), device_locations_.data(),
                    secretkey_memory_size * sizeof(Data64),
                    cudaMemcpyDeviceToHost));
                HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());

                os.write((char*) &secretkey_memory_size,
                         sizeof(secretkey_memory_size));
                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * secretkey_memory_size);
            }
            else
            {
                uint32_t secretkey_memory_size = host_locations_.size();
                os.write((char*) &secretkey_memory_size,
                         sizeof(secretkey_memory_size));
                os.write((char*) host_locations_.data(),
                         sizeof(Data64) * secretkey_memory_size);
            }
            if (!os)
                throw std::runtime_error("Failed to write Secretkey!");
        }
        else
        {
            throw std::runtime_error(
                "Secretkey is not generated so can not be serialized!");
        }
    }

    void Secretkey<Scheme::BFV>::load(std::istream& is)
    {
        if (secret_key_generated_)
            throw std::runtime_error("Secretkey already exists!");
        // Parse into a temporary object; a failed load must leave this key
        // usable.
        Secretkey temporary;
        temporary.context_ = context_;
        auto read = [&](auto& value)
        {
            is.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!is)
                throw std::runtime_error("Truncated Secretkey binary!");
        };
        read(temporary.scheme_);
        read(temporary.ring_size_);
        read(temporary.coeff_modulus_count_);
        read(temporary.n_power_);
        read(temporary.hamming_weight_);
        // Never read attacker-controlled bytes into a bool object.
        uint8_t ntt, generated;
        static_assert(sizeof(bool) == sizeof(uint8_t));
        read(ntt);
        read(generated);
        storage_type serialized_storage;
        read(serialized_storage);
        uint32_t size;
        read(size);
        const uint64_t expected_size = uint64_t(temporary.ring_size_) *
                                       uint64_t(temporary.coeff_modulus_count_);
        if (temporary.scheme_ != scheme_type::bfv ||
            temporary.ring_size_ < MIN_POLY_DEGREE ||
            temporary.ring_size_ > MAX_POLY_DEGREE || temporary.n_power_ < 0 ||
            temporary.n_power_ >= 31 ||
            (1U << temporary.n_power_) != uint32_t(temporary.ring_size_) ||
            temporary.coeff_modulus_count_ <= 0 ||
            expected_size > uint64_t(std::numeric_limits<int>::max()) ||
            size != expected_size || ntt != 1 || generated != 1 ||
            temporary.hamming_weight_ <= 0 ||
            temporary.hamming_weight_ > temporary.ring_size_ ||
            (serialized_storage != storage_type::DEVICE &&
             serialized_storage != storage_type::HOST))
            throw std::runtime_error("Invalid Secretkey binary parameters!");
        if (context_ &&
            (temporary.ring_size_ != context_->n ||
             temporary.coeff_modulus_count_ != context_->Q_prime_size ||
             temporary.n_power_ != context_->n_power))
            throw ContextMismatchError(error_message(
                "Secretkey context mismatch! The serialized key has ring size ",
                temporary.ring_size_, " and ", temporary.coeff_modulus_count_,
                " key moduli, the context has ring size ", context_->n, " and ",
                context_->Q_prime_size, " key moduli."));

        // Grow only after successful reads, avoiding a claimed-size allocation
        // on truncated input. The unbound deserialize API remains supported.
        detail::PrivateRandomBuffer<Data64> data(0);
        std::array<Data64, 512> chunk;
        for (size_t remaining = size; remaining;)
        {
            const size_t count = std::min(remaining, chunk.size());
            is.read(reinterpret_cast<char*>(chunk.data()),
                    count * sizeof(Data64));
            if (!is)
            {
                OPENSSL_cleanse(chunk.data(), sizeof(chunk));
                throw std::runtime_error("Truncated Secretkey payload!");
            }
            data.values.insert(data.values.end(), chunk.begin(),
                               chunk.begin() + count);
            remaining -= count;
        }
        OPENSSL_cleanse(chunk.data(), sizeof(chunk));
        temporary.device_locations_ = DeviceVector<Data64>(data.values);
        HEONGPU_CUDA_CHECK(
            cudaStreamSynchronize(temporary.device_locations_.stream()));
        temporary.in_ntt_domain_ = true;
        temporary.secret_key_generated_ = true;
        if (context_)
            temporary.validate_key_data(context_);
        *this = std::move(temporary);
    }

    void Secretkey<Scheme::BFV>::validate_context(
        const HEContext<Scheme::BFV>& context)
    {
        if (!context || !context->context_generated_)
            throw StateError("HEContext is not generated! Create the context "
                             "with GenHEContext and call generate() before "
                             "using it.");
        if (ring_size_ != context->n ||
            coeff_modulus_count_ != context->Q_prime_size ||
            n_power_ != context->n_power)
            throw ContextMismatchError(error_message(
                "Secretkey context mismatch! The key has ring size ",
                ring_size_, " and ", coeff_modulus_count_,
                " key moduli, the context has ring size ", context->n, " and ",
                context->Q_prime_size, " key moduli."));
        if (hamming_weight_ <= 0 || hamming_weight_ > ring_size_)
            throw ParameterError(error_message(
                "Invalid Secretkey Hamming weight ", hamming_weight_,
                "! It must be between 1 and the ring size ", ring_size_, "."));
        if (context_)
        {
            const auto actual = context_->get_key_modulus();
            const auto expected = context->get_key_modulus();
            if (actual.size() != expected.size())
                throw ContextMismatchError(error_message(
                    "Secretkey context mismatch! The key was created with ",
                    actual.size(), " key moduli, the context has ",
                    expected.size(), "."));
            for (size_t i = 0; i < actual.size(); ++i)
                if (actual[i].value != expected[i].value)
                    throw ContextMismatchError(error_message(
                        "Secretkey context mismatch! Key modulus ", i, " is ",
                        actual[i].value, " in the key's context and ",
                        expected[i].value, " in this context."));
        }
        else
        {
            if (secret_key_generated_)
                validate_key_data(context);
            context_ = context;
        }
        if (secret_key_generated_ &&
            (!in_ntt_domain_ ||
             memory_size() != ring_size_ * coeff_modulus_count_))
            throw std::invalid_argument("Invalid Secretkey storage!");
        // Keys may have been generated on a nonblocking stream.
        if (storage_type_ == storage_type::DEVICE)
            HEONGPU_CUDA_CHECK(
                cudaStreamSynchronize(device_locations_.stream()));
    }

    void Secretkey<Scheme::BFV>::validate_key_data(
        const HEContext<Scheme::BFV>& context) const
    {
        const size_t size =
            static_cast<size_t>(ring_size_) * coeff_modulus_count_;
        detail::PrivateRandomBuffer<Data64> data(size);
        if (storage_type_ == storage_type::DEVICE)
        {
            HEONGPU_CUDA_CHECK(
                cudaStreamSynchronize(device_locations_.stream()));
            HEONGPU_CUDA_CHECK(
                cudaMemcpy(data.values.data(), device_locations_.data(),
                           size * sizeof(Data64), cudaMemcpyDeviceToHost));
        }
        else
            std::copy(host_locations_.begin(), host_locations_.end(),
                      data.values.begin());
        const auto moduli = context->get_key_modulus();
        const auto roots = generate_primitive_root_of_unity(ring_size_, moduli);
        detail::PrivateRandomBuffer<int> reference(ring_size_);
        for (int m = 0; m < coeff_modulus_count_; ++m)
        {
            const auto modulus = moduli[m];
            detail::PrivateRandomBuffer<Data64> polynomial(ring_size_);
            for (int i = 0; i < ring_size_; ++i)
            {
                const auto value = data.values[m * ring_size_ + i];
                if (value >= modulus.value)
                    throw std::runtime_error("Noncanonical Secretkey residue!");
                polynomial.values[i] = value;
            }
            gpuntt::NTTFactors<Data64> factors(
                modulus, OPERATOR64::mult(roots[m], roots[m], modulus),
                roots[m]);
            gpuntt::NTTCPU<Data64> inverse(gpuntt::NTTParameters<Data64>(
                n_power_, factors, gpuntt::X_N_plus));
            polynomial.values = inverse.intt(polynomial.values);
            int weight = 0;
            for (int i = 0; i < ring_size_; ++i)
            {
                const auto value = polynomial.values[i];
                if (value != 0 && value != 1 && value != modulus.value - 1)
                    throw std::runtime_error(
                        "Invalid Secretkey coefficients or context!");
                const int coefficient =
                    value == modulus.value - 1 ? -1 : int(value);
                if (m == 0)
                    reference.values[i] = coefficient;
                else if (reference.values[i] != coefficient)
                    throw std::runtime_error(
                        "Inconsistent Secretkey RNS coefficients!");
                weight += coefficient != 0;
            }
            if (weight != hamming_weight_)
                throw std::runtime_error("Invalid Secretkey Hamming weight!");
        }
    }

    int Secretkey<Scheme::BFV>::memory_size()
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

    void Secretkey<Scheme::BFV>::memory_clear(cudaStream_t stream)
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
    Secretkey<Scheme::BFV>::memory_set(DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::DEVICE;
        device_locations_ = std::move(new_device_vector);

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Secretkey<Scheme::BFV>::copy_to_device(cudaStream_t stream)
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
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Secretkey<Scheme::BFV>::remove_from_device(cudaStream_t stream)
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

    void Secretkey<Scheme::BFV>::remove_from_host()
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
