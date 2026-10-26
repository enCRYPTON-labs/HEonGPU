// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_CIPHERTEXT_H
#define HEONGPU_CKKS_CIPHERTEXT_H

#include <heongpu/host/ckks/context.cuh>

namespace heongpu
{

    template <> class Ciphertext<Scheme::CKKS>
    {
        template <Scheme S> friend class HEEncryptor;
        template <Scheme S> friend class HEDecryptor;
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;
        template <Scheme S> friend class HEMultiPartyManager;

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
        Ciphertext(HEContext<Scheme::CKKS> context,
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
            if (storage_type_ == storage_type::DEVICE)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /**
         * @brief Returns a pointer to the underlying data of the ciphertext.
         *
         * @return Data64* Pointer to the data.
         */
        Data64* data();

        /**
         * @brief Copies the data from the device to the host.
         *
         * @param cipher Reference to a vector where the device data will be
         * copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void get_data(std::vector<Data64>& cipher,
                      cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Copies the data from the device to the host.
         *
         * @param cipher Reference to a HostVector where the device data will be
         * copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void get_data(HostVector<Data64>& cipher,
                      cudaStream_t stream =
                          cudaStreamDefault); // TODO: add check mechanism

        /**
         * @brief Switches the Ciphertext CUDA stream.
         *
         * @param stream The new CUDA stream to be used.
         */
        void switch_stream(cudaStream_t stream)
        {
            device_locations_.set_stream(stream);
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
            return device_locations_.stream();
        }

        /**
         * @brief Returns the size of the polynomial ring used in ciphertext.
         *
         * @return int Size of the polynomial ring.
         */
        inline int ring_size() const noexcept { return ring_size_; }

        /**
         * @brief Returns the number of coefficient modulus primes used in the
         * ciphertext.
         *
         * @return int Number of coefficient modulus primes.
         */
        inline int coeff_modulus_count() const noexcept
        {
            return coeff_modulus_count_;
        }

        /**
         * @brief Returns the size of the ciphertext.
         *
         * @return int Size of the ciphertext.
         */
        inline int size() const noexcept { return cipher_size_; }

        /**
         * @brief Returns the depth level of the ciphertext.
         *
         * @return int Depth level of the ciphertext.
         */
        inline int depth() const noexcept { return depth_; }

        /**
         * @brief Returns the level of the ciphertext.
         *
         * @return int Level of the ciphertext.
         */
        inline int level() const noexcept
        {
            return coeff_modulus_count_ - (depth_ + 1);
        }

        /**
         * @brief Indicates whether the ciphertext is in the NTT (Number
         * Theoretic Transform) domain.
         *
         * @return bool True if in NTT domain, false otherwise.
         */
        inline bool in_ntt_domain() const noexcept { return in_ntt_domain_; }

        /**
         * @brief Returns the scaling factor used for encoding in CKKS scheme.
         *
         * @return double Scaling factor.
         */
        inline double scale() const noexcept { return scale_; }

        /**
         * @brief Returns the semantic encoding mode carried by ciphertext.
         *
         * @return encoding The encoding mode (`SLOT` or `COEFFICIENT`).
         */
        inline encoding encoding_type() const noexcept { return encoding_; }

        /**
         * @brief Indicates whether rescaling is required for the ciphertext.
         *
         * @return bool True if rescaling is required, false otherwise.
         */
        inline bool rescale_required() const noexcept
        {
            return rescale_required_;
        }

        /**
         * @brief Indicates whether relinearization is required for the
         * ciphertext.
         *
         * @return bool True if relinearization is required, false otherwise.
         */
        inline bool relinearization_required() const noexcept
        {
            return relinearization_required_;
        }

        Ciphertext() = default;

        Ciphertext(const Ciphertext& copy)
            : ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_),
              cipher_size_(copy.cipher_size_), depth_(copy.depth_),
              scheme_(copy.scheme_), in_ntt_domain_(copy.in_ntt_domain_),
              storage_type_(copy.storage_type_), scale_(copy.scale_),
              encoding_(copy.encoding_),
              rescale_required_(copy.rescale_required_),
              relinearization_required_(copy.relinearization_required_),
              ciphertext_generated_(copy.ciphertext_generated_),
              context_id_(copy.context_id_)

        {
            if (copy.storage_type_ == storage_type::DEVICE)
            {
                device_locations_.resize(copy.device_locations_.size(),
                                         copy.device_locations_.stream());
                cudaMemcpyAsync(device_locations_.data(),
                                copy.device_locations_.data(),
                                copy.device_locations_.size() * sizeof(Data64),
                                cudaMemcpyDeviceToDevice,
                                copy.device_locations_
                                    .stream()); // TODO: use cudaStreamPerThread
            }
            else
            {
                host_locations_.resize(copy.host_locations_.size());
                std::memcpy(host_locations_.data(), copy.host_locations_.data(),
                            copy.host_locations_.size() * sizeof(Data64));
            }
        }

        Ciphertext(Ciphertext&& assign) noexcept
            : ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              cipher_size_(std::move(assign.cipher_size_)),
              depth_(std::move(assign.depth_)),
              scheme_(std::move(assign.scheme_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              storage_type_(std::move(assign.storage_type_)),
              scale_(std::move(assign.scale_)),
              encoding_(std::move(assign.encoding_)),
              rescale_required_(std::move(assign.rescale_required_)),
              relinearization_required_(
                  std::move(assign.relinearization_required_)),
              ciphertext_generated_(std::move(assign.ciphertext_generated_)),
              context_id_(assign.context_id_),
              device_locations_(std::move(assign.device_locations_)),
              host_locations_(std::move(assign.host_locations_))
        {
            assign.ciphertext_generated_ = false;
        }

        Ciphertext& operator=(const Ciphertext& copy)
        {
            if (this != &copy)
            {
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;
                cipher_size_ = copy.cipher_size_;
                depth_ = copy.depth_;
                scheme_ = copy.scheme_;
                in_ntt_domain_ = copy.in_ntt_domain_;
                storage_type_ = copy.storage_type_;

                scale_ = copy.scale_;
                encoding_ = copy.encoding_;
                rescale_required_ = copy.rescale_required_;
                relinearization_required_ = copy.relinearization_required_;
                ciphertext_generated_ = copy.ciphertext_generated_;
                context_id_ = copy.context_id_;

                if (copy.storage_type_ == storage_type::DEVICE)
                {
                    device_locations_.resize(copy.device_locations_.size(),
                                             copy.device_locations_.stream());
                    cudaMemcpyAsync(
                        device_locations_.data(), copy.device_locations_.data(),
                        copy.device_locations_.size() * sizeof(Data64),
                        cudaMemcpyDeviceToDevice,
                        copy.device_locations_
                            .stream()); // TODO: use cudaStreamPerThread
                }
                else
                {
                    host_locations_.resize(copy.host_locations_.size());
                    std::memcpy(host_locations_.data(),
                                copy.host_locations_.data(),
                                copy.host_locations_.size() * sizeof(Data64));
                }
            }
            return *this;
        }

        Ciphertext& operator=(Ciphertext&& assign) noexcept
        {
            if (this != &assign)
            {
                ring_size_ = std::move(assign.ring_size_);
                coeff_modulus_count_ = std::move(assign.coeff_modulus_count_);
                cipher_size_ = std::move(assign.cipher_size_);
                depth_ = std::move(assign.depth_);
                scheme_ = std::move(assign.scheme_);
                in_ntt_domain_ = std::move(assign.in_ntt_domain_);
                storage_type_ = std::move(assign.storage_type_);

                scale_ = std::move(assign.scale_);
                encoding_ = std::move(assign.encoding_);
                rescale_required_ = std::move(assign.rescale_required_);
                relinearization_required_ =
                    std::move(assign.relinearization_required_);
                ciphertext_generated_ = std::move(assign.ciphertext_generated_);
                context_id_ = assign.context_id_;

                device_locations_ = std::move(assign.device_locations_);
                host_locations_ = std::move(assign.host_locations_);
                assign.ciphertext_generated_ = false;
            }
            return *this;
        }

        void save(std::ostream& os) const;

        /**
         * @brief Loads a ciphertext after validating its binary representation.
         * Existing ring/modulus dimensions must match; failed loads leave the
         * object unchanged. Default-constructed objects are checked against the
         * execution context before GPU operations.
         */
        void load(std::istream& is);

      private:
        scheme_type scheme_ = scheme_type::ckks;
        int ring_size_ = 0;
        int coeff_modulus_count_ = 0;
        int cipher_size_ = 0;
        int depth_ = 0;

        bool in_ntt_domain_ = true;
        storage_type storage_type_ = storage_type::DEVICE;

        double scale_ = 0;
        encoding encoding_ = encoding::SLOT;
        bool rescale_required_ = false;
        bool relinearization_required_ = false;

        bool ciphertext_generated_ = false;
        std::uint64_t context_id_ = 0;

        DeviceVector<Data64> device_locations_;
        HostVector<Data64> host_locations_;

        size_t checked_memory_size() const;
        void validate_context(const HEContext<Scheme::CKKS>& context) const;
        int memory_size();
        void memory_clear(cudaStream_t stream);
        void memory_set(DeviceVector<Data64>&& new_device_vector);

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_CIPHERTEXT_H
