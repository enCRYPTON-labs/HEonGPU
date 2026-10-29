// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_STORAGE_MANAGER_H
#define HEONGPU_STORAGE_MANAGER_H

#include "gpuntt/common/common.cuh"
#include "gpuntt/common/nttparameters.cuh"
#include <stdexcept>
#include <vector>
#include <heongpu/util/error.h>

namespace heongpu
{
    /**
     * @enum storage_type
     * @brief Specifies the storage type for objects.
     *
     * - `host`: Indicates that the object is stored in the host memory.
     * - `device`: Indicates that the object is stored in the device memory.
     */
    enum class storage_type : std::uint8_t
    {
        host = 0x1, // Host memory storage
        device = 0x2 // Device memory storage
    };

    /**
     * @struct ExecutionOptions
     * @brief Manages execution options, including CUDA stream configuration,
     * storage type selection, and the behavior regarding initial data location.
     */
    struct ExecutionOptions
    {
        cudaStream_t stream_ =
            cudaStreamDefault; // CUDA stream to be used for execution. Defaults
                               // to cudaStreamDefault.
        storage_type storage_ =
            storage_type::device; // Default storage type for the operation.
                                  // Defaults to DEVICE.
        bool keep_initial_condition_ =
            true; // Determines whether to maintain the initial data location.
                  // Defaults to true.

        /**
         * @brief Sets the CUDA stream to be used for execution.
         *
         * This method allows you to specify the CUDA stream that will be used
         * during the execution of operations. By default, the stream is set to
         * `cudaStreamDefault`.
         *
         * @param stream The CUDA stream to set.
         * @return ExecutionOptions& A reference to the updated ExecutionOptions
         * object, allowing method chaining.
         */
        ExecutionOptions& set_stream(cudaStream_t stream)
        {
            stream_ = stream;
            return *this;
        }

        /**
         * @brief Configures the storage type for execution.
         *
         * Sets the storage type where the operation's data will be stored. This
         * can typically be either `storage_type::device` or
         * `storage_type::host`.
         *
         * @param storage The storage type to set (storage_type::device or
         * storage_type::host).
         * @return ExecutionOptions& A reference to the updated ExecutionOptions
         * object, allowing method chaining.
         */
        ExecutionOptions& set_storage_type(storage_type storage)
        {
            storage_ = storage;
            return *this;
        }

        /**
         * @brief Specifies whether the object an *_inplace operation changes
         * returns to its initial location after the operation.
         *
         * The inputs of an operation never move: an input in host memory is
         * read from a temporary copy on the device. Only the object changed
         * in place is affected; if set to `true`, an object that was in host
         * memory is brought back there unless the storage type is device.
         *
         * @param keep_initial_condition Boolean flag to determine whether to
         * keep the initial location.
         * @return ExecutionOptions& A reference to the updated ExecutionOptions
         * object, allowing method chaining.
         */
        ExecutionOptions& set_initial_location(bool keep_initial_condition)
        {
            keep_initial_condition_ = keep_initial_condition;
            return *this;
        }
    };

    /**
     * @brief Manages the input storage and conditionally transfers data to the
     * appropriate location (e.g., device or host) before executing a function
     * on the object.
     *
     * @tparam T The type of the object to manage.
     * @tparam F The callable type of the function to execute on the object.
     * @param object The object whose storage is being managed.
     * @param function The callable function that will operate on the object.
     * @param options A set of execution options defining storage behavior and
     * stream configuration.
     * @param is_input_output_same A flag indicating whether the input and
     * output storage should remain consistent.
     */
    template <typename T, typename F>
    void input_storage_manager(T& object, F function, ExecutionOptions options,
                               bool is_input_output_same)
    {
        storage_type initial_condition = object.storage_type_;

        if (!object.is_on_device())
        {
            if (options.keep_initial_condition_)
            {
                object.copy_to_device(options.stream_);
            }
            else
            {
                object.store_in_device(options.stream_);
            }
        }

        function(object);

        if (is_input_output_same)
        {
            if (options.keep_initial_condition_ &&
                (initial_condition == storage_type::host) &&
                (options.storage_ == storage_type::device))
            {
                object.remove_from_host();
            }
        }
        else
        {
            if (options.keep_initial_condition_)
            {
                if (initial_condition == storage_type::host)
                {
                    object.remove_from_device(options.stream_);
                }
            }
            else
            {
                if (options.storage_ == storage_type::device)
                {
                    object.store_in_device(options.stream_);
                }
                else if (options.storage_ == storage_type::host)
                {
                    object.store_in_host(options.stream_);
                }
                else
                {
                    throw ParameterError(
                        error_message("Invalid storage type ",
                                      static_cast<int>(options.storage_),
                                      "! Expected storage_type::device or "
                                      "storage_type::host."));
                }
            }
        }
    }

    /**
     * @brief Recursively manages the input storage for a vector of objects and
     * conditionally transfers data to the appropriate location (e.g., device or
     * host) before executing a function on each object.
     *
     * @tparam T The type of the objects in the vector to manage.
     * @tparam F The callable type of the function to execute on each object.
     * @param objects A vector of objects whose storage is being managed.
     * @param function The callable function that will operate on each object in
     * the vector.
     * @param options A set of execution options defining storage behavior and
     * stream configuration.
     * @param is_input_output_same A flag indicating whether the input and
     * output storage should remain consistent for each object.
     */
    inline void synchronize_input_stream(cudaStream_t stream)
    {
        cudaError_t error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess)
        {
            throw SystemError(
                error_message("CUDA error: ", cudaGetErrorString(error)));
        }
    }

    /**
     * @brief Runs function on a const input of an operation, on the device.
     *
     * An input on the device is passed as it is. For an input in host memory,
     * function gets a device copy, so that the object itself never changes:
     * neither its data nor where it is stored, whatever options says. The
     * operations only read their inputs, although their implementation takes
     * non-const references, so function must not change the object.
     *
     * @param object The input of the operation.
     * @param function The callable function that reads the input.
     * @param options Execution options; the stream of the device copy.
     * @param is_input_output_same Unused: an input that is also the output
     * of the operation is read before the output is written, from the copy
     * when it is in host memory.
     */
    template <typename T, typename F>
    void input_storage_manager(const T& object, F function,
                               ExecutionOptions options,
                               bool is_input_output_same)
    {
        (void) is_input_output_same;
        if (object.is_on_device())
        {
            function(const_cast<T&>(object));
            return;
        }
        // Earlier work on the stream, such as an encoding stored in host
        // memory, may still be writing the host data.
        synchronize_input_stream(options.stream_);
        T device_copy(object);
        device_copy.store_in_device(options.stream_);
        function(device_copy);
    }

    template <typename T, typename F>
    void input_vector_storage_manager(std::vector<T>& objects, F function,
                                      ExecutionOptions options,
                                      bool is_input_output_same)
    {
        std::vector<storage_type> initial_conditions(objects.size());

        for (int i = 0; i < objects.size(); i++)
        {
            if (!objects[i].is_on_device())
            {
                if (options.keep_initial_condition_)
                {
                    objects[i].copy_to_device(options.stream_);
                }
                else
                {
                    objects[i].store_in_device(options.stream_);
                }
            }
        }

        function(objects);

        for (int i = 0; i < objects.size(); i++)
        {
            if (is_input_output_same)
            {
                if (options.keep_initial_condition_ &&
                    (initial_conditions[i] == storage_type::host) &&
                    (options.storage_ == storage_type::device))
                {
                    objects[i].remove_from_host();
                }
            }
            else
            {
                if (options.keep_initial_condition_)
                {
                    if (initial_conditions[i] == storage_type::host)
                    {
                        objects[i].remove_from_device(options.stream_);
                    }
                }
                else
                {
                    if (options.storage_ == storage_type::device)
                    {
                        objects[i].store_in_device(options.stream_);
                    }
                    else if (options.storage_ == storage_type::host)
                    {
                        objects[i].store_in_host(options.stream_);
                    }
                    else
                    {
                        throw ParameterError(
                            error_message("Invalid storage type ",
                                          static_cast<int>(options.storage_),
                                          "! Expected storage_type::device or "
                                          "storage_type::host."));
                    }
                }
            }
        }
    }

    /**
     * @brief Manages the output storage of an object after executing a
     * function.
     *
     * @tparam T The type of the object to manage.
     * @tparam F The callable type of the function to execute on the object.
     * @param object The object whose output storage is being managed.
     * @param function The callable function that will operate on the object.
     * @param options A set of execution options defining storage behavior and
     * stream configuration.
     */
    /**
     * @brief Runs function on const inputs of an operation, on the device; see
     * input_storage_manager for a const input.
     */
    template <typename T, typename F>
    void input_vector_storage_manager(const std::vector<T>& objects, F function,
                                      ExecutionOptions options,
                                      bool is_input_output_same)
    {
        (void) is_input_output_same;
        bool on_device = true;
        for (const T& object : objects)
        {
            on_device = on_device && object.is_on_device();
        }
        if (on_device)
        {
            function(const_cast<std::vector<T>&>(objects));
            return;
        }
        synchronize_input_stream(options.stream_);
        std::vector<T> device_copies(objects);
        for (T& copy : device_copies)
        {
            if (!copy.is_on_device())
            {
                copy.store_in_device(options.stream_);
            }
        }
        function(device_copies);
    }

    template <typename T, typename F>
    void output_storage_manager(T& object, F function, ExecutionOptions options)
    {
        function(object);

        if (options.storage_ == storage_type::device)
        {
            object.store_in_device(options.stream_);
        }
        else if (options.storage_ == storage_type::host)
        {
            object.store_in_host(options.stream_);
        }
        else
        {
            throw ParameterError(error_message(
                "Invalid storage type ", static_cast<int>(options.storage_),
                "! Expected storage_type::device or storage_type::host."));
        }
    }

} // namespace heongpu
#endif // HEONGPU_STORAGE_MANAGER_H
