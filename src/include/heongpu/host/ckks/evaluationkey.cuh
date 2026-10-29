// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_EVALUATIONKEY_H
#define HEONGPU_CKKS_EVALUATIONKEY_H

#include <optional>
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/kernel/keygeneration.cuh>

namespace heongpu
{

    /**
     * @brief Relinkey represents a relinearization key used for homomorphic
     * encryption operations.
     *
     * The Relinkey class is initialized with encryption parameters and
     * optionally with key data. It provides methods for storing the
     * relinearization key in GPU or CPU memory, which is essential for
     * performing homomorphic operations that require relinearization, such as
     * reducing the size of ciphertexts after multiplication.
     */
    template <> class Relinkey<Scheme::CKKS>
    {
        template <Scheme S> friend class HEKeyGenerator;
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
        template <typename T, typename F>
        friend void output_storage_manager(T& object, F function,
                                           ExecutionOptions options);

      public:
        /**
         * @brief Constructs a new Relinkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Relinkey(HEContext<Scheme::CKKS> context);

        /**
         * @brief Stores the relinearization key in the device (GPU) memory.
         */
        void store_in_device(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Stores the relinearization key in the host (CPU) memory.
         */
        void store_in_host(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Checks whether the data is stored on the device (GPU) memory.
         */
        bool is_on_device() const
        {
            return (storage_type_ == storage_type::device);
        }

        /**
         * @brief Returns a pointer to the underlying relinearization key data.
         *
         * @return Data64* Pointer to the relinearization key data.
         */
        Data64* data();
        const Data64* data() const;

        /**
         * @brief Returns a pointer to the specific part of the relinearization
         * key data.
         *
         * @param i Index of the key level to access.
         * @return Data64* Pointer to the specified part of the relinearization
         * key data.
         */
        Data64* data(size_t i);
        const Data64* data(size_t i) const;

        /**
         * @brief Copy constructor for creating a new Relinkey object by copying
         * an existing one.
         *
         * This constructor copies data either from GPU or CPU memory, depending
         * on the source. GPU memory operations are performed using
         * `cudaMemcpyAsync`.
         *
         * @param copy The source Relinkey object to copy from.
         */
        Relinkey(const Relinkey& copy)
            : context_(copy.context_), context_id_(copy.context_id_),
              scheme_(copy.scheme_), ring_size(copy.ring_size),
              Q_prime_size_(copy.Q_prime_size_), Q_size_(copy.Q_size_),
              d_(copy.d_), d_tilda_(copy.d_tilda_), r_prime_(copy.r_prime_),
              storage_type_(copy.storage_type_),
              relinkey_size_(copy.relinkey_size_),
              relinkey_size_leveled_(copy.relinkey_size_leveled_),
              relin_key_generated_(copy.relin_key_generated_)
        {
            if (copy.storage_type_ == storage_type::device)
            {
                if (copy.relinkey_size_leveled_.size() == 0)
                {
                    device_location_.resize(copy.device_location_.size(),
                                            copy.device_location_.stream());
                    cudaMemcpyAsync(
                        device_location_.data(), copy.device_location_.data(),
                        copy.device_location_.size() * sizeof(Data64),
                        cudaMemcpyDeviceToDevice,
                        copy.device_location_
                            .stream()); // TODO: use cudaStreamPerThread
                }
                else
                {
                    device_location_leveled_.resize(
                        copy.device_location_leveled_.size());
                    for (int i = 0; i < copy.device_location_leveled_.size();
                         i++)
                    {
                        device_location_leveled_[i].resize(
                            copy.device_location_.size(),
                            copy.device_location_leveled_[i].stream());
                        cudaMemcpyAsync(
                            device_location_leveled_[i].data(),
                            copy.device_location_leveled_[i].data(),
                            copy.device_location_leveled_[i].size() *
                                sizeof(Data64),
                            cudaMemcpyDeviceToDevice,
                            copy.device_location_leveled_[i]
                                .stream()); // TODO: use cudaStreamPerThread
                    }
                }
            }
            else
            {
                if (copy.relinkey_size_leveled_.size() == 0)
                {
                    host_location_ = copy.host_location_;
                }
                else
                {
                    host_location_leveled_.resize(
                        copy.host_location_leveled_.size());
                    for (int i = 0; i < copy.host_location_leveled_.size(); i++)
                    {
                        host_location_leveled_[i] =
                            copy.host_location_leveled_[i];
                    }
                }
            }
        }

        /**
         * @brief Move constructor for transferring ownership of an existing
         * Relinkey object.
         *
         * This constructor transfers resources from the source object to the
         * newly constructed object. GPU and CPU memory resources are moved,
         * leaving the source object in a valid but undefined state.
         *
         * @param assign The source Relinkey object to move from.
         */
        Relinkey(Relinkey&& assign) noexcept
            : context_(std::move(assign.context_)),
              context_id_(assign.context_id_),
              scheme_(std::move(assign.scheme_)),
              ring_size(std::move(assign.ring_size)),
              Q_prime_size_(std::move(assign.Q_prime_size_)),
              Q_size_(std::move(assign.Q_size_)), d_(std::move(assign.d_)),
              d_tilda_(std::move(assign.d_tilda_)),
              r_prime_(std::move(assign.r_prime_)),
              storage_type_(std::move(assign.storage_type_)),
              relinkey_size_(std::move(assign.relinkey_size_)),
              relinkey_size_leveled_(std::move(assign.relinkey_size_leveled_)),
              relin_key_generated_(std::move(assign.relin_key_generated_))
        {
            if (assign.storage_type_ == storage_type::device)
            {
                if (assign.relinkey_size_leveled_.size() == 0)
                {
                    device_location_ = std::move(assign.device_location_);
                }
                else
                {
                    device_location_leveled_.resize(
                        assign.device_location_leveled_.size());
                    for (int i = 0; i < assign.device_location_leveled_.size();
                         i++)
                    {
                        device_location_leveled_[i] =
                            std::move(assign.device_location_leveled_[i]);
                    }
                }
            }
            else
            {
                if (assign.relinkey_size_leveled_.size() == 0)
                {
                    host_location_ = std::move(assign.host_location_);
                }
                else
                {
                    host_location_leveled_.resize(
                        assign.host_location_leveled_.size());
                    for (int i = 0; i < assign.host_location_leveled_.size();
                         i++)
                    {
                        host_location_leveled_[i] =
                            std::move(assign.host_location_leveled_[i]);
                    }
                }
            }
        }

        /**
         * @brief Copy assignment operator for assigning one Relinkey object to
         * another.
         *
         * This operator performs a deep copy of all resources, including GPU
         * and CPU memory, ensuring the target object has its own independent
         * copy of the data.
         *
         * @param copy The source Relinkey object to copy from.
         * @return Reference to the assigned object.
         */
        Relinkey& operator=(const Relinkey& copy)
        {
            if (this != &copy)
            {
                context_ = copy.context_;
                context_id_ = copy.context_id_;
                scheme_ = copy.scheme_;
                ring_size = copy.ring_size;
                Q_prime_size_ = copy.Q_prime_size_;
                Q_size_ = copy.Q_size_;
                d_ = copy.d_;
                d_tilda_ = copy.d_tilda_;
                r_prime_ = copy.r_prime_;
                storage_type_ = copy.storage_type_;
                relinkey_size_ = copy.relinkey_size_;
                relinkey_size_leveled_ = copy.relinkey_size_leveled_;
                relin_key_generated_ = copy.relin_key_generated_;

                if (copy.storage_type_ == storage_type::device)
                {
                    if (copy.relinkey_size_leveled_.size() == 0)
                    {
                        device_location_.resize(copy.device_location_.size(),
                                                copy.device_location_.stream());
                        cudaMemcpyAsync(
                            device_location_.data(),
                            copy.device_location_.data(),
                            copy.device_location_.size() * sizeof(Data64),
                            cudaMemcpyDeviceToDevice,
                            copy.device_location_
                                .stream()); // TODO: use cudaStreamPerThread
                    }
                    else
                    {
                        device_location_leveled_.resize(
                            copy.device_location_leveled_.size());
                        for (int i = 0;
                             i < copy.device_location_leveled_.size(); i++)
                        {
                            device_location_leveled_[i].resize(
                                copy.device_location_.size(),
                                copy.device_location_leveled_[i].stream());
                            cudaMemcpyAsync(
                                device_location_leveled_[i].data(),
                                copy.device_location_leveled_[i].data(),
                                copy.device_location_leveled_[i].size() *
                                    sizeof(Data64),
                                cudaMemcpyDeviceToDevice,
                                copy.device_location_leveled_[i]
                                    .stream()); // TODO: use
                                                // cudaStreamPerThread
                        }
                    }
                }
                else
                {
                    if (copy.relinkey_size_leveled_.size() == 0)
                    {
                        host_location_ = copy.host_location_;
                    }
                    else
                    {
                        host_location_leveled_.resize(
                            copy.host_location_leveled_.size());
                        for (int i = 0; i < copy.host_location_leveled_.size();
                             i++)
                        {
                            host_location_leveled_[i] =
                                copy.host_location_leveled_[i];
                        }
                    }
                }
            }
            return *this;
        }

        /**
         * @brief Move assignment operator for transferring ownership of
         * resources.
         *
         * This operator moves all resources from the source object to the
         * target object. GPU and CPU memory are efficiently transferred,
         * leaving the source object in a valid but undefined state.
         *
         * @param assign The source Relinkey object to move from.
         * @return Reference to the assigned object.
         */
        Relinkey& operator=(Relinkey&& assign) noexcept
        {
            if (this != &assign)
            {
                context_ = std::move(assign.context_);
                context_id_ = assign.context_id_;
                scheme_ = std::move(assign.scheme_);
                ring_size = std::move(assign.ring_size);
                Q_prime_size_ = std::move(assign.Q_prime_size_);
                Q_size_ = std::move(assign.Q_size_);
                d_ = std::move(assign.d_);
                d_tilda_ = std::move(assign.d_tilda_);
                r_prime_ = std::move(assign.r_prime_);
                storage_type_ = std::move(assign.storage_type_);
                relinkey_size_ = std::move(assign.relinkey_size_);
                relinkey_size_leveled_ =
                    std::move(assign.relinkey_size_leveled_);
                relin_key_generated_ = std::move(assign.relin_key_generated_);

                if (assign.storage_type_ == storage_type::device)
                {
                    if (assign.relinkey_size_leveled_.size() == 0)
                    {
                        device_location_ = std::move(assign.device_location_);
                    }
                    else
                    {
                        device_location_leveled_.resize(
                            assign.device_location_leveled_.size());
                        for (int i = 0;
                             i < assign.device_location_leveled_.size(); i++)
                        {
                            device_location_leveled_[i] =
                                std::move(assign.device_location_leveled_[i]);
                        }
                    }
                }
                else
                {
                    if (assign.relinkey_size_leveled_.size() == 0)
                    {
                        host_location_ = std::move(assign.host_location_);
                    }
                    else
                    {
                        host_location_leveled_.resize(
                            assign.host_location_leveled_.size());
                        for (int i = 0;
                             i < assign.host_location_leveled_.size(); i++)
                        {
                            host_location_leveled_[i] =
                                std::move(assign.host_location_leveled_[i]);
                        }
                    }
                }
            }
            return *this;
        }

        /**
         * @brief Default constructor is deleted to ensure proper initialization
         * of relinearization keys.
         */
        Relinkey() = default;

        void save(std::ostream& os,
                  serialization_format format = serialization_format::v2) const;

        void load(std::istream& is);

        void set_context(HEContext<Scheme::CKKS> context)
        {
            context_ = std::move(context);
        }

      private:
        HEContext<Scheme::CKKS> context_;
        // Context identity read by load(); used when context_ is not set.
        std::uint64_t context_id_ = 0;

        std::uint64_t serialized_context_id() const
        {
            return context_ ? context_->context_id() : context_id_;
        }

        void validate_context(const HEContext<Scheme::CKKS>& context) const
        {
            validate_context_identity(serialized_context_id(), context,
                                      "Relinkey");
            if (relin_key_generated_ && context && context->context_generated_)
            {
                validate_key_digit_count(d_, context->d_leveled->operator[](0),
                                         "Relinkey");
            }
        }

        scheme_type scheme_;

        int ring_size;
        int Q_prime_size_;
        int Q_size_;

        int d_;
        int d_tilda_;
        int r_prime_;

        storage_type storage_type_;
        Data64 relinkey_size_;
        std::vector<size_t> relinkey_size_leveled_;

        bool relin_key_generated_ = false;

        // store key in device (GPU)
        DeviceVector<Data64> device_location_;
        std::vector<DeviceVector<Data64>> device_location_leveled_;

        // store key in host (CPU)
        HostVector<Data64> host_location_;
        std::vector<HostVector<Data64>> host_location_leveled_;

        int memory_size() const;
        void memory_clear(cudaStream_t stream);
        void memory_set(DeviceVector<Data64>&& new_device_vector);
        void memory_set(DeviceVector<Data64>&& new_device_vector, int i);

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

    /**
     * @brief MultipartyRelinkey is a specialized class for managing
     * relinearization keys in multiparty computation (MPC) settings.
     *
     * This class extends the `Relinkey` class to include additional
     * functionality specific to MPC, such as seed management for reproducible
     * key generation across participants. It integrates with `HEKeyGenerator`
     * for collaborative key generation processes.
     */
    template <>
    class MultipartyRelinkey<Scheme::CKKS> : public Relinkey<Scheme::CKKS>
    {
        template <Scheme S> friend class HEKeyGenerator;
        template <typename T, typename F>
        friend void input_storage_manager(T& object, F function,
                                          ExecutionOptions options,
                                          bool check_initial_condition);
        template <typename T, typename F>
        friend void input_vector_storage_manager(std::vector<T>& objects,
                                                 F function,
                                                 ExecutionOptions options,
                                                 bool is_input_output_same);
        template <typename T, typename F>
        friend void output_storage_manager(T& object, F function,
                                           ExecutionOptions options);

      public:
        __host__ MultipartyRelinkey(HEContext<Scheme::CKKS> context,
                                    const RNGSeed seed);

        /**
         * @brief Retrieves the seed value used for key generation.
         *
         * @return RNGSeed The seed value.
         */
        inline RNGSeed seed() const noexcept { return seed_; }

      private:
        RNGSeed seed_;
    };

    /**
     * @brief Galoiskey represents a Galois key used for performing homomorphic
     * operations such as rotations on encrypted data.
     *
     * The Galoiskey class is initialized with encryption parameters and
     * provides options to allow certain types of rotations. The Galois key can
     * be used to rotate encrypted data by specific shifts or elements, which is
     * useful in operations like ciphertext multiplication or encoding
     * manipulations. The class also offers flexibility to store the key in
     * either GPU or CPU memory.
     */
    template <> class Galoiskey<Scheme::CKKS>
    {
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;
        template <Scheme S> friend class HEMultiPartyManager;
        template <Scheme S> friend class HESession;

      public:
        /**
         * @brief Constructs a new Galoiskey object with default settings that
         * allow rotations in the range (-255, 255).
         *
         * The range is set by MAX_SHIFT in kernel/defines.h. Use
         * Galoiskey(context, max_shift) to choose another range without
         * rebuilding the library.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Galoiskey(HEContext<Scheme::CKKS> context);

        /**
         * @brief Constructs a new Galoiskey object with in the range
         * (-2^max_shift, 2^max_shift).
         *
         * This Galois key object supports all homomorphic rotations, though at
         * some performance cost. Rotation speed improves as max_shift
         * approaches half of the slot count.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Galoiskey(HEContext<Scheme::CKKS> context, int max_shift);

        /**
         * @brief Constructs a new Galoiskey object allowing specific rotations
         * based on the given vector of shifts.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param shift_vec Vector of integers representing the allowed shifts
         * for rotations.
         */
        __host__ Galoiskey(HEContext<Scheme::CKKS> context,
                           std::vector<int>& shift_vec);

        /**
         * @brief Constructs a new Galoiskey object allowing certain Galois
         * elements given in a vector.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param galois_elts Vector of Galois elements (represented as unsigned
         * 32-bit integers) specifying the rotations allowed.
         */
        __host__ Galoiskey(HEContext<Scheme::CKKS> context,
                           std::vector<uint32_t>& galois_elts);

        /**
         * @brief Stores the galois key in the device (GPU) memory.
         */
        void store_in_device(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Stores the galois key in the host (CPU) memory.
         */
        void store_in_host(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Checks whether the data is stored on the device (GPU) memory.
         */
        bool is_on_device() const
        {
            return (storage_type_ == storage_type::device);
        }

        /**
         * @brief Returns a pointer to the specified part of the Galois key
         * data.
         *
         * @param i Index of the key elvel to access.
         * @return Data64* Pointer to the specified part of the Galois key data.
         */
        Data64* data(size_t i);
        const Data64* data(size_t i) const;

        /**
         * @brief Returns a pointer to the Galois key data for conjugation.(for
         * CKKS)
         *
         * @return Data64* Pointer to the Galois key data for column rotation.
         */
        Data64* c_data();
        const Data64* c_data() const;

        /**
         * @brief Copy constructor for creating a new Galoiskey object by
         * copying an existing one.
         *
         * This constructor copies data from an existing Galoiskey object,
         * handling both GPU and CPU memory efficiently. GPU memory operations
         * use `cudaMemcpyAsync` to ensure asynchronous data transfer.
         *
         * @param copy The source Galoiskey object to copy from.
         */
        Galoiskey(const Galoiskey& copy)
            : context_(copy.context_), context_id_(copy.context_id_),
              scheme_(copy.scheme_), ring_size(copy.ring_size),
              Q_prime_size_(copy.Q_prime_size_), Q_size_(copy.Q_size_),
              d_(copy.d_), customized(copy.customized),
              group_order_(copy.group_order_),
              storage_type_(copy.storage_type_),
              galoiskey_size_(copy.galoiskey_size_),
              custom_galois_elt(copy.custom_galois_elt),
              max_shift_(copy.max_shift_), max_log_slot_(copy.max_log_slot_),
              galois_elt(copy.galois_elt),
              galois_elt_zero(copy.galois_elt_zero),
              galois_key_generated_(copy.galois_key_generated_)
        {
            if (copy.storage_type_ == storage_type::device)
            {
                for (const auto& [key, value] : copy.device_location_)
                {
                    device_location_[key].resize(value.size(), value.stream());
                    cudaMemcpyAsync(
                        device_location_[key].data(), value.data(),
                        value.size() * sizeof(Data64), cudaMemcpyDeviceToDevice,
                        value.stream()); // TODO: use cudaStreamPerThread
                }

                zero_device_location_.resize(
                    copy.zero_device_location_.size(),
                    copy.zero_device_location_.stream());

                cudaMemcpyAsync(zero_device_location_.data(),
                                copy.zero_device_location_.data(),
                                copy.zero_device_location_.size() *
                                    sizeof(Data64),
                                cudaMemcpyDeviceToDevice,
                                copy.zero_device_location_
                                    .stream()); // TODO: use cudaStreamPerThread
            }
            else
            {
                for (const auto& [key, value] : copy.host_location_)
                {
                    host_location_[key] = value;
                }

                zero_host_location_ = copy.zero_host_location_;
            }
        }

        /**
         * @brief Move constructor for transferring ownership of an existing
         * Galoiskey object.
         *
         * Transfers all resources, including GPU and CPU memory, from the
         * source object to the newly created object. The source object is left
         * in a valid but undefined state.
         *
         * @param assign The source Galoiskey object to move from.
         */
        Galoiskey(Galoiskey&& assign) noexcept
            : context_(std::move(assign.context_)),
              context_id_(assign.context_id_),
              scheme_(std::move(assign.scheme_)),
              ring_size(std::move(assign.ring_size)),
              Q_prime_size_(std::move(assign.Q_prime_size_)),
              Q_size_(std::move(assign.Q_size_)), d_(std::move(assign.d_)),
              customized(std::move(assign.customized)),
              group_order_(std::move(assign.group_order_)),
              storage_type_(std::move(assign.storage_type_)),
              galoiskey_size_(std::move(assign.galoiskey_size_)),
              custom_galois_elt(std::move(assign.custom_galois_elt)),
              max_shift_(assign.max_shift_),
              max_log_slot_(assign.max_log_slot_),
              galois_elt(std::move(assign.galois_elt)),
              galois_elt_zero(std::move(assign.galois_elt_zero)),
              galois_key_generated_(std::move(assign.galois_key_generated_))
        {
            if (assign.storage_type_ == storage_type::device)
            {
                for (const auto& [key, value] : assign.device_location_)
                {
                    device_location_[key] = std::move(value);
                }

                assign.device_location_.clear();

                zero_device_location_ = std::move(assign.zero_device_location_);
            }
            else
            {
                for (const auto& [key, value] : assign.host_location_)
                {
                    host_location_[key] = std::move(value);
                }

                assign.host_location_.clear();

                zero_host_location_ = std::move(assign.zero_host_location_);
            }
        }

        /**
         * @brief Copy assignment operator for assigning one Galoiskey object to
         * another.
         *
         * Performs a deep copy of all resources, ensuring that the target
         * object gets its own independent copy of the data, whether stored in
         * GPU or CPU memory.
         *
         * @param copy The source Galoiskey object to copy from.
         * @return Reference to the assigned object.
         */
        Galoiskey& operator=(const Galoiskey& copy)
        {
            if (this != &copy)
            {
                context_ = copy.context_;
                context_id_ = copy.context_id_;
                scheme_ = copy.scheme_;
                ring_size = copy.ring_size;
                Q_prime_size_ = copy.Q_prime_size_;
                Q_size_ = copy.Q_size_;
                d_ = copy.d_;
                customized = copy.customized;
                group_order_ = copy.group_order_;
                storage_type_ = copy.storage_type_;
                galoiskey_size_ = copy.galoiskey_size_;
                custom_galois_elt = copy.custom_galois_elt;
                max_shift_ = copy.max_shift_;
                max_log_slot_ = copy.max_log_slot_;
                galois_elt = copy.galois_elt;
                galois_elt_zero = copy.galois_elt_zero;
                galois_key_generated_ = copy.galois_key_generated_;

                if (copy.storage_type_ == storage_type::device)
                {
                    for (const auto& [key, value] : copy.device_location_)
                    {
                        device_location_[key].resize(value.size(),
                                                     value.stream());
                        cudaMemcpyAsync(
                            device_location_[key].data(), value.data(),
                            value.size() * sizeof(Data64),
                            cudaMemcpyDeviceToDevice,
                            value.stream()); // TODO: use cudaStreamPerThread
                    }

                    zero_device_location_.resize(
                        copy.zero_device_location_.size(),
                        copy.zero_device_location_.stream());

                    cudaMemcpyAsync(
                        zero_device_location_.data(),
                        copy.zero_device_location_.data(),
                        copy.zero_device_location_.size() * sizeof(Data64),
                        cudaMemcpyDeviceToDevice,
                        copy.zero_device_location_
                            .stream()); // TODO: use cudaStreamPerThread
                }
                else
                {
                    for (const auto& [key, value] : copy.host_location_)
                    {
                        host_location_[key] = value;
                    }

                    zero_host_location_ = copy.zero_host_location_;
                }
            }
            return *this;
        }

        /**
         * @brief Move assignment operator for transferring ownership of
         * resources.
         *
         * Transfers all resources, including GPU and CPU memory, from the
         * source object to the target object. The source object is left in a
         * valid but undefined state.
         *
         * @param assign The source Galoiskey object to move from.
         * @return Reference to the assigned object.
         */
        Galoiskey& operator=(Galoiskey&& assign) noexcept
        {
            if (this != &assign)
            {
                context_ = std::move(assign.context_);
                context_id_ = assign.context_id_;
                scheme_ = std::move(assign.scheme_);
                ring_size = std::move(assign.ring_size);
                Q_prime_size_ = std::move(assign.Q_prime_size_);
                Q_size_ = std::move(assign.Q_size_);
                d_ = std::move(assign.d_);
                customized = std::move(assign.customized);
                group_order_ = std::move(assign.group_order_);
                storage_type_ = std::move(assign.storage_type_);
                galoiskey_size_ = std::move(assign.galoiskey_size_);
                custom_galois_elt = std::move(assign.custom_galois_elt);
                max_shift_ = assign.max_shift_;
                max_log_slot_ = assign.max_log_slot_;
                galois_elt = std::move(assign.galois_elt);
                galois_elt_zero = std::move(assign.galois_elt_zero);
                galois_key_generated_ = std::move(assign.galois_key_generated_);

                if (assign.storage_type_ == storage_type::device)
                {
                    for (const auto& [key, value] : assign.device_location_)
                    {
                        device_location_[key] = std::move(value);
                    }

                    assign.device_location_.clear();

                    zero_device_location_ =
                        std::move(assign.zero_device_location_);
                }
                else
                {
                    for (const auto& [key, value] : assign.host_location_)
                    {
                        host_location_[key] = std::move(value);
                    }

                    assign.host_location_.clear();

                    zero_host_location_ = std::move(assign.zero_host_location_);
                }
            }
            return *this;
        }

        /**
         * @brief Default constructor for Galoiskey.
         *
         * Initializes an empty Galoiskey object. All members will have their
         * default values.
         */
        Galoiskey() = default;

        void save(std::ostream& os,
                  serialization_format format = serialization_format::v2) const;

        void load(std::istream& is);

        void set_context(HEContext<Scheme::CKKS> context)
        {
            context_ = std::move(context);
        }

      private:
        HEContext<Scheme::CKKS> context_;
        // Context identity read by load(); used when context_ is not set.
        std::uint64_t context_id_ = 0;

        std::uint64_t serialized_context_id() const
        {
            return context_ ? context_->context_id() : context_id_;
        }

        void validate_context(const HEContext<Scheme::CKKS>& context) const
        {
            validate_context_identity(serialized_context_id(), context,
                                      "Galoiskey");
            if (galois_key_generated_ && context && context->context_generated_)
            {
                validate_key_digit_count(d_, context->d_leveled->operator[](0),
                                         "Galoiskey");
            }
        }

        void set_rotation_limits();

        scheme_type scheme_;

        int ring_size;

        int Q_prime_size_;
        int Q_size_;

        int d_;

        bool customized;
        int group_order_;

        storage_type storage_type_;
        Data64 galoiskey_size_;
        std::vector<u_int32_t> custom_galois_elt;
        // Rotations without their own key are built from the keys for the
        // steps 1, 2, 4, ..., 2^max_shift_ (see set_rotation_limits()).
        int max_shift_ = -1;
        int max_log_slot_ = 0;

        bool galois_key_generated_ = false;

        // for rotate
        std::unordered_map<int, int> galois_elt;
        std::unordered_map<int, DeviceVector<Data64>> device_location_;
        std::unordered_map<int, HostVector<Data64>> host_location_;

        // for rotate_columns
        int galois_elt_zero;
        DeviceVector<Data64> zero_device_location_;
        HostVector<Data64> zero_host_location_;
    };

    /**
     * @brief MultipartyGaloiskey is a specialized class for managing Galois
     * keys in multiparty computation (MPC) settings.
     *
     * This class extends the `Galoiskey` class to include functionality
     * specific to MPC scenarios, such as managing custom Galois elements and
     * seeds for deterministic key generation. It integrates with
     * `HEKeyGenerator` for collaborative key generation processes.
     */
    template <>
    class MultipartyGaloiskey<Scheme::CKKS> : public Galoiskey<Scheme::CKKS>
    {
        template <Scheme S> friend class HEKeyGenerator;

      public:
        __host__ MultipartyGaloiskey(HEContext<Scheme::CKKS> context,
                                     const RNGSeed seed);

        __host__ MultipartyGaloiskey(HEContext<Scheme::CKKS> context,
                                     std::vector<int>& shift_vec,
                                     const RNGSeed seed);

        __host__ MultipartyGaloiskey(HEContext<Scheme::CKKS> context,
                                     std::vector<uint32_t>& galois_elts,
                                     const RNGSeed seed);

        /**
         * @brief Retrieves the seed value used for key generation.
         *
         * @return int The seed value.
         */
        inline RNGSeed seed() const noexcept { return seed_; }

      private:
        RNGSeed seed_;
    };

    /**
     * @brief Switchkey represents a key used for switching between different
     * secret keys in homomorphic encryption.
     *
     * The Switchkey class is initialized with encryption parameters and
     * provides the functionality to store the switching key either in GPU or
     * CPU memory. Key switching is an essential operation in homomorphic
     * encryption schemes, allowing ciphertexts encrypted under one key to be
     * transformed for use with a different secret key, which is particularly
     * useful for multi-party computation and other advanced encryption
     * scenarios.
     */
    template <> class Switchkey<Scheme::CKKS>
    {
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;

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
         * @brief Constructs a new Switchkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Switchkey(HEContext<Scheme::CKKS> context);

        /**
         * @brief Stores the switch key in the device (GPU) memory.
         */
        void store_in_device(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Stores the switch key in the host (CPU) memory.
         */
        void store_in_host(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Checks whether the data is stored on the device (GPU) memory.
         */
        bool is_on_device() const
        {
            return (storage_type_ == storage_type::device);
        }

        /**
         * @brief Returns a pointer to the underlying switch key data.
         *
         * @return Data64* Pointer to the switch key data.
         */
        Data64* data();
        const Data64* data() const;

        Switchkey() = default;
        Switchkey(const Switchkey& copy) = default;
        Switchkey(Switchkey&& source) = default;
        Switchkey& operator=(const Switchkey& assign) = default;
        Switchkey& operator=(Switchkey&& assign) = default;

        void save(std::ostream& os,
                  serialization_format format = serialization_format::v2) const;

        void load(std::istream& is);

        void set_context(HEContext<Scheme::CKKS> context)
        {
            context_ = std::move(context);
        }

      private:
        HEContext<Scheme::CKKS> context_;
        // Context identity read by load(); used when context_ is not set.
        std::uint64_t context_id_ = 0;

        std::uint64_t serialized_context_id() const
        {
            return context_ ? context_->context_id() : context_id_;
        }

        void validate_context(const HEContext<Scheme::CKKS>& context) const
        {
            validate_context_identity(serialized_context_id(), context,
                                      "Switchkey");
            if (switch_key_generated_ && context && context->context_generated_)
            {
                validate_key_digit_count(d_, context->d_leveled->operator[](0),
                                         "Switchkey");
            }
        }

        scheme_type scheme_;

        int ring_size;

        int Q_prime_size_;
        int Q_size_;

        int d_;

        storage_type storage_type_;
        Data64 switchkey_size_;

        bool switch_key_generated_ = false;

        DeviceVector<Data64> device_location_;
        HostVector<Data64> host_location_;

        int memory_size() const;
        void memory_clear(cudaStream_t stream);
        void memory_set(DeviceVector<Data64>&& new_device_vector);
        void memory_set(DeviceVector<Data64>&& new_device_vector, int i);

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };

    /**
     * @brief The keys CKKS bootstrapping uses, as HEKeyGenerator::
     * bootstrapping_keys generates them. The switch keys between the secret
     * key and a sparse secret key are present when bootstrapping uses one.
     */
    struct CKKSBootstrappingKeys
    {
        Relinkey<Scheme::CKKS> relin_key;
        Galoiskey<Scheme::CKKS> galois_key;
        std::optional<Switchkey<Scheme::CKKS>> dense_to_sparse;
        std::optional<Switchkey<Scheme::CKKS>> sparse_to_dense;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_EVALUATIONKEY_H
