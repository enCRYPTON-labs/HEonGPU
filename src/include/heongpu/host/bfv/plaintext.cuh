// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_PLAINTEXT_H
#define HEONGPU_BFV_PLAINTEXT_H

#include <heongpu/host/bfv/context.cuh>

namespace heongpu
{

    /**
     * @brief Plaintext represents the unencrypted data used in homomorphic
     * encryption.
     *
     * The Plaintext class is initialized with encryption parameters and
     * optionally a CUDA stream. It provides methods to manage the underlying
     * data, such as resizing, and transferring between host (CPU) and device
     * (GPU). This class is used for preparing data for encoding, encryption,
     * and other homomorphic operations.
     */
    template <> class Plaintext<Scheme::BFV>
    {
        template <Scheme S> friend class HEEncoder;
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
         * @brief Constructs a new Plaintext object with specified parameters
         * and an optional CUDA stream.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        explicit __host__
        Plaintext(HEContext<Scheme::BFV> context,
                  const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Stores the plaintext in the device (GPU) memory.
         */
        void store_in_device(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Stores the plaintext in the host (CPU) memory.
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
         * @brief Returns a pointer to the underlying plaintext data.
         *
         * @return Data64* Pointer to the plaintext data.
         */
        Data64* data();

        /**
         * @brief Returns the size of the plaintext.
         *
         * @return int Size of the plaintext.
         */
        inline int size() const noexcept { return plain_size_; }

        /**
         * @brief Indicates whether the plaintext is in the NTT (Number
         * Theoretic Transform) domain.
         *
         * @return bool True if in NTT domain, false otherwise.
         */
        inline bool in_ntt_domain() const noexcept { return in_ntt_domain_; }

        /**
         * @brief Default constructor for Plaintext.
         *
         * Initializes an empty Plaintext object. All members will have their
         * default values.
         */
        /**
         * @brief Returns a one line description, for example for logs and
         * error reports. Also written by operator<<.
         */
        std::string to_string() const;

        Plaintext() = default;

        Plaintext(const Plaintext& copy)
            : scheme_(copy.scheme_), plain_size_(copy.plain_size_),
              in_ntt_domain_(copy.in_ntt_domain_),
              storage_type_(copy.storage_type_),
              plaintext_generated_(copy.plaintext_generated_),
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

        Plaintext(Plaintext&& assign) noexcept
            : scheme_(std::move(assign.scheme_)),
              plain_size_(std::move(assign.plain_size_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              storage_type_(std::move(assign.storage_type_)),
              plaintext_generated_(std::move(assign.plaintext_generated_)),
              context_id_(assign.context_id_),
              device_locations_(std::move(assign.device_locations_)),
              host_locations_(std::move(assign.host_locations_))
        {
        }

        Plaintext& operator=(const Plaintext& copy)
        {
            if (this != &copy)
            {
                scheme_ = copy.scheme_;
                plain_size_ = copy.plain_size_;
                in_ntt_domain_ = copy.in_ntt_domain_;
                storage_type_ = copy.storage_type_;
                plaintext_generated_ = copy.plaintext_generated_;
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

        Plaintext& operator=(Plaintext&& assign) noexcept
        {
            if (this != &assign)
            {
                scheme_ = std::move(assign.scheme_);
                plain_size_ = std::move(assign.plain_size_);
                in_ntt_domain_ = std::move(assign.in_ntt_domain_);
                storage_type_ = std::move(assign.storage_type_);
                plaintext_generated_ = std::move(assign.plaintext_generated_);
                context_id_ = assign.context_id_;

                device_locations_ = std::move(assign.device_locations_);
                host_locations_ = std::move(assign.host_locations_);
            }
            return *this;
        }

        void save(std::ostream& os,
                  serialization_format format = serialization_format::v2) const;

        void load(std::istream& is);

      private:
        scheme_type scheme_;
        int plain_size_;

        bool in_ntt_domain_ = false;
        storage_type storage_type_;

        bool plaintext_generated_ = false;
        std::uint64_t context_id_ = 0;

        DeviceVector<Data64> device_locations_;
        HostVector<Data64> host_locations_;

        void validate_context(const HEContext<Scheme::BFV>& context) const;
        int memory_size();
        void memory_clear(cudaStream_t stream);
        void memory_set(DeviceVector<Data64>&& new_device_vector);

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

    std::ostream& operator<<(std::ostream& os,
                             const Plaintext<Scheme::BFV>& object);

} // namespace heongpu
#endif // HEONGPU_BFV_PLAINTEXT_H
