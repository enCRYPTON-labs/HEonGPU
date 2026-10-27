// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_CIPHERTEXT_H
#define HEONGPU_BFV_CIPHERTEXT_H

#include <heongpu/host/bfv/context.cuh>

namespace heongpu
{

    template <> class Ciphertext<Scheme::BFV>
    {
        template <Scheme S> friend class HEEncryptor;
        template <Scheme S> friend class HEDecryptor;
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;
        template <Scheme S> friend class HEMultiPartyManager;
        template <Scheme S> friend class HESession;

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
        Ciphertext(HEContext<Scheme::BFV> context,
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
            if (storage_type_ == storage_type::device)
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
         * @brief Indicates whether the ciphertext is in the NTT (Number
         * Theoretic Transform) domain.
         *
         * @return bool True if in NTT domain, false otherwise.
         */
        inline bool in_ntt_domain() const noexcept { return in_ntt_domain_; }

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

        /**
         * @brief Returns a one line description, for example for logs and
         * error reports. Also written by operator<<.
         */
        std::string to_string() const;

        Ciphertext() = default;

        Ciphertext(const Ciphertext& copy)
            : ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_),
              cipher_size_(copy.cipher_size_), scheme_(copy.scheme_),
              in_ntt_domain_(copy.in_ntt_domain_),
              storage_type_(copy.storage_type_),
              relinearization_required_(copy.relinearization_required_),
              ciphertext_generated_(copy.ciphertext_generated_),
              context_id_(copy.context_id_)
        {
            if (copy.storage_type_ == storage_type::device)
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
              scheme_(std::move(assign.scheme_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              storage_type_(std::move(assign.storage_type_)),
              relinearization_required_(
                  std::move(assign.relinearization_required_)),
              ciphertext_generated_(std::move(assign.ciphertext_generated_)),
              context_id_(assign.context_id_),
              device_locations_(std::move(assign.device_locations_)),
              host_locations_(std::move(assign.host_locations_))
        {
        }

        Ciphertext& operator=(const Ciphertext& copy)
        {
            if (this != &copy)
            {
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;
                cipher_size_ = copy.cipher_size_;
                scheme_ = copy.scheme_;
                in_ntt_domain_ = copy.in_ntt_domain_;
                storage_type_ = copy.storage_type_;

                relinearization_required_ = copy.relinearization_required_;
                ciphertext_generated_ = copy.ciphertext_generated_;
                context_id_ = copy.context_id_;

                if (copy.storage_type_ == storage_type::device)
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
                scheme_ = std::move(assign.scheme_);
                in_ntt_domain_ = std::move(assign.in_ntt_domain_);
                storage_type_ = std::move(assign.storage_type_);

                relinearization_required_ =
                    std::move(assign.relinearization_required_);

                ciphertext_generated_ = std::move(assign.ciphertext_generated_);
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
        int ring_size_;
        int coeff_modulus_count_;
        int cipher_size_;

        bool in_ntt_domain_;
        storage_type storage_type_;

        bool relinearization_required_;

        bool ciphertext_generated_ = false;
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
                             const Ciphertext<Scheme::BFV>& object);

} // namespace heongpu
#endif // HEONGPU_BFV_CIPHERTEXT_H