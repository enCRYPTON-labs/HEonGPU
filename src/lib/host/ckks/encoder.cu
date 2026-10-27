// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/encoder.cuh>

namespace heongpu
{

    __host__ HEEncoder<Scheme::CKKS>::HEEncoder(HEContext<Scheme::CKKS> context)
    {
        if (!context || !context->context_generated_)
        {
            throw StateError(
                "HEContext is not generated! Create the "
                "context with GenHEContext and call generate() before "
                "using it.");
        }

        context_ = std::move(context);
        const auto* ctx = context_.get();

        slot_count_ = ctx->n >> 1;
        log_slot_count_ = int(log2(slot_count_));
        fft_length = ctx->n * 2;

        two_pow_64 = std::pow(2.0, 64);

        special_root = static_cast<Complex64>(2.0) *
                       static_cast<Complex64>(M_PI) /
                       static_cast<Complex64>(fft_length);
        Complex64 j(0.0, 1.0); // Define the complex unit (imaginary part)
        Complex64 one(1.0); // Define the complex unit (imaginary part)

        // forward fft root table generation
        std::vector<Complex64> special_root_tables;
        for (int i = 0; i < fft_length; i++)
        {
            Complex64 element = complex_arithmetic::exp(
                j * static_cast<Complex64>(i) * special_root);
            special_root_tables.push_back(element);
        }

        // inverse fft root table generation
        std::vector<Complex64> special_inverse_root_tables;
        for (int i = 0; i < fft_length; i++)
        {
            Complex64 element = one / special_root_tables[i];
            special_inverse_root_tables.push_back(element);
        }

        std::vector<int> rot_group;
        rot_group.push_back(1);
        for (int i = 1; i < slot_count_; i++)
        {
            rot_group.push_back((5 * rot_group[i - 1]) % fft_length);
        }

        std::vector<Complex64> new_ordered_root_tables(slot_count_,
                                                       Complex64(0));
        for (int logm = 1; logm <= log_slot_count_; ++logm)
        {
            int idx_mod = 1 << (logm + 2);
            int gap = fft_length / idx_mod;

            int offset = 1 << (logm - 1);
            for (int i = 0; i < (1 << (logm - 1)); ++i)
            {
                int rou_idx = (rot_group[i] % idx_mod) * gap;
                new_ordered_root_tables[offset + i] =
                    special_root_tables[rou_idx];
            }
        }

        std::vector<Complex64> new_ordered_inverse_root_tables(slot_count_,
                                                               Complex64(0));
        for (int logm = log_slot_count_; logm > 0; logm--)
        {
            int idx_mod = 1 << (logm + 2);
            int gap = fft_length / idx_mod;

            int offset = 1 << (logm - 1);
            for (int i = 0; i < (1 << (logm - 1)); ++i)
            {
                int rou_idx = (rot_group[i] % idx_mod) * gap;
                new_ordered_inverse_root_tables[offset + i] =
                    special_inverse_root_tables[rou_idx];
            }
        }

        special_fft_roots_table_ = std::make_shared<DeviceVector<Complex64>>(
            new_ordered_root_tables, sizeof(Complex64));

        special_ifft_roots_table_ = std::make_shared<DeviceVector<Complex64>>(
            new_ordered_inverse_root_tables, sizeof(Complex64));

        std::vector<int> bit_reverse_vec(slot_count_);
        for (int i = 0; i < slot_count_; i++)
        {
            bit_reverse_vec[i] = gpuntt::bitreverse(i, log_slot_count_);
        }

        reverse_order = std::make_shared<DeviceVector<int>>(bit_reverse_vec);
    }

    __host__ void HEEncoder<Scheme::CKKS>::encode_ckks(
        Plaintext<Scheme::CKKS>& plain, const std::vector<double>& message,
        const double scale, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        DeviceVector<double> message_gpu(slot_count_, stream);
        if (message.size() < slot_count_)
        {
            cudaMemsetAsync(message_gpu.data(), 0, slot_count_ * sizeof(double),
                            stream);
        }
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(double), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DeviceVector<Complex64> temp_complex(context_->n, stream);
        double_to_complex_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream>>>(message_gpu.data(),
                                             temp_complex.data());

        double fix = scale / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft{};
        cfg_ifft.n_power = log_slot_count_;
        cfg_ifft.fft_type = gpufft::type::INVERSE;
        cfg_ifft.mod_inverse = Complex64(fix, 0.0);
        cfg_ifft.stream = stream;

        gpufft::GPU_Special_FFT(temp_complex.data(),
                                special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream>>>(
            output_memory.data(), temp_complex.data(),
            context_->modulus_->data(), context_->Q_size, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        plain.scale_ = scale;

        plain.memory_set(std::move(output_memory));
    }

    __host__ void HEEncoder<Scheme::CKKS>::encode_ckks(
        Plaintext<Scheme::CKKS>& plain, const HostVector<double>& message,
        const double scale, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        DeviceVector<double> message_gpu(slot_count_, stream);
        if (message.size() < slot_count_)
        {
            cudaMemsetAsync(message_gpu.data(), 0, slot_count_ * sizeof(double),
                            stream);
        }
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(double), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DeviceVector<Complex64> temp_complex(context_->n, stream);
        double_to_complex_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream>>>(message_gpu.data(),
                                             temp_complex.data());

        double fix = scale / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft{};
        cfg_ifft.n_power = log_slot_count_;
        cfg_ifft.fft_type = gpufft::type::INVERSE;
        cfg_ifft.mod_inverse = Complex64(fix, 0.0);
        cfg_ifft.stream = stream;

        gpufft::GPU_Special_FFT(temp_complex.data(),
                                special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream>>>(
            output_memory.data(), temp_complex.data(),
            context_->modulus_->data(), context_->Q_size, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        plain.scale_ = scale;

        plain.memory_set(std::move(output_memory));
    }

    __host__ void HEEncoder<Scheme::CKKS>::encode_ckks_coeff(
        Plaintext<Scheme::CKKS>& plain, const std::vector<double>& message,
        const double scale, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        DeviceVector<double> message_gpu(context_->n, stream);
        if (message.size() < static_cast<std::size_t>(context_->n))
        {
            cudaMemsetAsync(message_gpu.data(), 0, context_->n * sizeof(double),
                            stream);
        }
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(double), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_coeff_ckks_conversion<<<dim3((context_->n >> 8), 1, 1),
                                              256, 0, stream>>>(
            output_memory.data(), message_gpu.data(),
            context_->modulus_->data(), context_->Q_size, two_pow_64, scale,
            context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        plain.scale_ = scale;
        plain.memory_set(std::move(output_memory));
    }

    __host__ void HEEncoder<Scheme::CKKS>::encode_ckks_coeff(
        Plaintext<Scheme::CKKS>& plain, const HostVector<double>& message,
        const double scale, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        DeviceVector<double> message_gpu(context_->n, stream);
        if (message.size() < static_cast<std::size_t>(context_->n))
        {
            cudaMemsetAsync(message_gpu.data(), 0, context_->n * sizeof(double),
                            stream);
        }
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(double), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        encode_kernel_coeff_ckks_conversion<<<dim3((context_->n >> 8), 1, 1),
                                              256, 0, stream>>>(
            output_memory.data(), message_gpu.data(),
            context_->modulus_->data(), context_->Q_size, two_pow_64, scale,
            context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        plain.scale_ = scale;
        plain.memory_set(std::move(output_memory));
    }

    __host__ void HEEncoder<Scheme::CKKS>::encode_ckks(
        Plaintext<Scheme::CKKS>& plain, const std::vector<Complex64>& message,
        const double scale, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        DeviceVector<Complex64> message_gpu(slot_count_, stream);
        if (message.size() < slot_count_)
        {
            cudaMemsetAsync(message_gpu.data(), 0,
                            slot_count_ * sizeof(Complex64), stream);
        }
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Complex64),
                        cudaMemcpyHostToDevice, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double fix = scale / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft{};
        cfg_ifft.n_power = log_slot_count_;
        cfg_ifft.fft_type = gpufft::type::INVERSE;
        cfg_ifft.mod_inverse = Complex64(fix, 0.0);
        cfg_ifft.stream = stream;

        gpufft::GPU_Special_FFT(message_gpu.data(),
                                special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream>>>(
            output_memory.data(), message_gpu.data(),
            context_->modulus_->data(), context_->Q_size, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        plain.scale_ = scale;

        plain.memory_set(std::move(output_memory));
    }

    __host__ void HEEncoder<Scheme::CKKS>::encode_ckks(
        Plaintext<Scheme::CKKS>& plain, const HostVector<Complex64>& message,
        const double scale, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        DeviceVector<Complex64> message_gpu(slot_count_, stream);
        if (message.size() < slot_count_)
        {
            cudaMemsetAsync(message_gpu.data(), 0,
                            slot_count_ * sizeof(Complex64), stream);
        }
        cudaMemcpyAsync(message_gpu.data(), message.data(),
                        message.size() * sizeof(Complex64),
                        cudaMemcpyHostToDevice, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double fix = scale / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft{};
        cfg_ifft.n_power = log_slot_count_;
        cfg_ifft.fft_type = gpufft::type::INVERSE;
        cfg_ifft.mod_inverse = Complex64(fix, 0.0);
        cfg_ifft.stream = stream;

        gpufft::GPU_Special_FFT(message_gpu.data(),
                                special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1), 256,
                                        0, stream>>>(
            output_memory.data(), message_gpu.data(),
            context_->modulus_->data(), context_->Q_size, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(output_memory.data(),
                                context_->ntt_table_->data(),
                                context_->modulus_->data(), cfg_ntt,
                                context_->Q_size, context_->Q_size);

        plain.scale_ = scale;

        plain.memory_set(std::move(output_memory));
    }

    __host__ void HEEncoder<Scheme::CKKS>::encode_ckks(
        Plaintext<Scheme::CKKS>& plain, const double& message,
        const double scale, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        double value = message * scale;

        encode_kernel_double_ckks_conversion<<<dim3((context_->n >> 8), 1, 1),
                                               256, 0, stream>>>(
            output_memory.data(), value, context_->modulus_->data(),
            context_->Q_size, two_pow_64, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plain.memory_set(std::move(output_memory));
    }

    __host__ void HEEncoder<Scheme::CKKS>::encode_ckks(
        Plaintext<Scheme::CKKS>& plain, const std::int64_t& message,
        const double scale, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(context_->n * context_->Q_size,
                                           stream);

        double value = static_cast<double>(message) * scale;

        encode_kernel_double_ckks_conversion<<<dim3((context_->n >> 8), 1, 1),
                                               256, 0, stream>>>(
            output_memory.data(), value, context_->modulus_->data(),
            context_->Q_size, two_pow_64, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plain.memory_set(std::move(output_memory));
    }

    __host__ void
    HEEncoder<Scheme::CKKS>::decode_ckks(std::vector<double>& message,
                                         Plaintext<Scheme::CKKS>& plain,
                                         const cudaStream_t stream)
    {
        int current_modulus_count = context_->Q_size - plain.depth_;

        DeviceVector<double> message_gpu(slot_count_, stream);

        DeviceVector<Data64> temp_plain(context_->n * current_modulus_count,
                                        stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(plain.data(), temp_plain.data(),
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         current_modulus_count, current_modulus_count);

        int counter = context_->Q_size;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        DeviceVector<Complex64> temp_complex(context_->n, stream);
        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream>>>(
            temp_complex.data(), temp_plain.data(), context_->modulus_->data(),
            context_->Mi_inv_->data() + location1,
            context_->Mi_->data() + location2,
            context_->upper_half_threshold_->data() + location1,
            context_->decryption_modulus_->data() + location1,
            current_modulus_count, plain.scale_, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpufft::fft_configuration<Float64> cfg_fft{};
        cfg_fft.n_power = log_slot_count_;
        cfg_fft.fft_type = gpufft::type::FORWARD;
        cfg_fft.stream = stream;

        gpufft::GPU_Special_FFT(temp_complex.data(),
                                special_fft_roots_table_->data(), cfg_fft, 1);

        complex_to_double_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream>>>(temp_complex.data(),
                                             message_gpu.data());

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEEncoder<Scheme::CKKS>::decode_ckks(HostVector<double>& message,
                                         Plaintext<Scheme::CKKS>& plain,
                                         const cudaStream_t stream)
    {
        int current_modulus_count = context_->Q_size - plain.depth_;

        DeviceVector<double> message_gpu(slot_count_, stream);

        DeviceVector<Data64> temp_plain(context_->n * current_modulus_count,
                                        stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(plain.data(), temp_plain.data(),
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         current_modulus_count, current_modulus_count);

        int counter = context_->Q_size;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        DeviceVector<Complex64> temp_complex(context_->n, stream);

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream>>>(
            temp_complex.data(), temp_plain.data(), context_->modulus_->data(),
            context_->Mi_inv_->data() + location1,
            context_->Mi_->data() + location2,
            context_->upper_half_threshold_->data() + location1,
            context_->decryption_modulus_->data() + location1,
            current_modulus_count, plain.scale_, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpufft::fft_configuration<Float64> cfg_fft{};
        cfg_fft.n_power = log_slot_count_;
        cfg_fft.fft_type = gpufft::type::FORWARD;
        cfg_fft.stream = stream;

        gpufft::GPU_Special_FFT(temp_complex.data(),
                                special_fft_roots_table_->data(), cfg_fft, 1);

        complex_to_double_kernel<<<dim3(((slot_count_) >> 8), 1, 1), 256, 0,
                                   stream>>>(temp_complex.data(),
                                             message_gpu.data());

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEEncoder<Scheme::CKKS>::decode_ckks_coeff(std::vector<double>& message,
                                               Plaintext<Scheme::CKKS>& plain,
                                               const cudaStream_t stream)
    {
        int current_modulus_count = context_->Q_size - plain.depth_;

        DeviceVector<double> message_gpu(context_->n, stream);
        DeviceVector<Data64> temp_plain(context_->n * current_modulus_count,
                                        stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(plain.data(), temp_plain.data(),
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         current_modulus_count, current_modulus_count);

        int counter = context_->Q_size;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        decode_kernel_coeff_ckks_compose<<<dim3((context_->n >> 8), 1, 1), 256,
                                           0, stream>>>(
            message_gpu.data(), temp_plain.data(), context_->modulus_->data(),
            context_->Mi_inv_->data() + location1,
            context_->Mi_->data() + location2,
            context_->upper_half_threshold_->data() + location1,
            context_->decryption_modulus_->data() + location1,
            current_modulus_count, plain.scale_, two_pow_64, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(context_->n);
        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        context_->n * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEEncoder<Scheme::CKKS>::decode_ckks_coeff(HostVector<double>& message,
                                               Plaintext<Scheme::CKKS>& plain,
                                               const cudaStream_t stream)
    {
        int current_modulus_count = context_->Q_size - plain.depth_;

        DeviceVector<double> message_gpu(context_->n, stream);
        DeviceVector<Data64> temp_plain(context_->n * current_modulus_count,
                                        stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(plain.data(), temp_plain.data(),
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         current_modulus_count, current_modulus_count);

        int counter = context_->Q_size;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        decode_kernel_coeff_ckks_compose<<<dim3((context_->n >> 8), 1, 1), 256,
                                           0, stream>>>(
            message_gpu.data(), temp_plain.data(), context_->modulus_->data(),
            context_->Mi_inv_->data() + location1,
            context_->Mi_->data() + location2,
            context_->upper_half_threshold_->data() + location1,
            context_->decryption_modulus_->data() + location1,
            current_modulus_count, plain.scale_, two_pow_64, context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        message.resize(context_->n);
        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        context_->n * sizeof(double), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEEncoder<Scheme::CKKS>::decode_ckks(std::vector<Complex64>& message,
                                         Plaintext<Scheme::CKKS>& plain,
                                         const cudaStream_t stream)
    {
        int current_modulus_count = context_->Q_size - plain.depth_;

        DeviceVector<Complex64> message_gpu(slot_count_, stream);

        DeviceVector<Data64> temp_plain(context_->n * current_modulus_count,
                                        stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(plain.data(), temp_plain.data(),
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         current_modulus_count, current_modulus_count);

        int counter = context_->Q_size;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream>>>(
            message_gpu.data(), temp_plain.data(), context_->modulus_->data(),
            context_->Mi_inv_->data() + location1,
            context_->Mi_->data() + location2,
            context_->upper_half_threshold_->data() + location1,
            context_->decryption_modulus_->data() + location1,
            current_modulus_count, plain.scale_, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpufft::fft_configuration<Float64> cfg_fft{};
        cfg_fft.n_power = log_slot_count_;
        cfg_fft.fft_type = gpufft::type::FORWARD;
        cfg_fft.stream = stream;

        gpufft::GPU_Special_FFT(message_gpu.data(),
                                special_fft_roots_table_->data(), cfg_fft, 1);

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(Complex64), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEEncoder<Scheme::CKKS>::decode_ckks(HostVector<Complex64>& message,
                                         Plaintext<Scheme::CKKS>& plain,
                                         const cudaStream_t stream)
    {
        int current_modulus_count = context_->Q_size - plain.depth_;

        DeviceVector<Complex64> message_gpu(slot_count_, stream);

        DeviceVector<Data64> temp_plain(context_->n * current_modulus_count,
                                        stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = context_->n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = context_->n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT(plain.data(), temp_plain.data(),
                         context_->intt_table_->data(),
                         context_->modulus_->data(), cfg_intt,
                         current_modulus_count, current_modulus_count);

        int counter = context_->Q_size;
        int location1 = 0;
        int location2 = 0;
        for (int i = 0; i < plain.depth_; i++)
        {
            location1 += counter;
            location2 += (counter * counter);
            counter--;
        }

        encode_kernel_compose<<<dim3((slot_count_ >> 8), 1, 1), 256, 0,
                                stream>>>(
            message_gpu.data(), temp_plain.data(), context_->modulus_->data(),
            context_->Mi_inv_->data() + location1,
            context_->Mi_->data() + location2,
            context_->upper_half_threshold_->data() + location1,
            context_->decryption_modulus_->data() + location1,
            current_modulus_count, plain.scale_, two_pow_64,
            reverse_order->data(), context_->n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpufft::fft_configuration<Float64> cfg_fft{};
        cfg_fft.n_power = log_slot_count_;
        cfg_fft.fft_type = gpufft::type::FORWARD;
        cfg_fft.stream = stream;

        gpufft::GPU_Special_FFT(message_gpu.data(),
                                special_fft_roots_table_->data(), cfg_fft, 1);

        message.resize(slot_count_);

        cudaMemcpyAsync(message.data(), message_gpu.data(),
                        slot_count_ * sizeof(Complex64), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu
