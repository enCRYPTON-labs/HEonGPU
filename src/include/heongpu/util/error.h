// Copyright 2024-2026 Alişah Özcan
// Copyright 2026 enCRYPTON-Labs
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_ERROR_H
#define HEONGPU_ERROR_H

#include <sstream>
#include <stdexcept>
#include <string>

namespace heongpu
{
    /**
     * @brief Common base of the exceptions thrown by HEonGPU.
     *
     * Every HEonGPU exception also derives from a standard exception type, so
     * existing `catch (const std::invalid_argument&)`,
     * `catch (const std::logic_error&)` and
     * `catch (const std::runtime_error&)` blocks keep working. Catch
     * `heongpu::Error` to handle every HEonGPU exception in one place, or one
     * of the derived types to react to a specific problem.
     */
    class Error
    {
      public:
        virtual ~Error() = default;

        /**
         * @brief Returns the error message.
         */
        virtual const char* what() const noexcept = 0;
    };

    /**
     * @brief Thrown when a parameter or an argument has an invalid value, for
     * example a scale out of bounds or a ciphertext with the wrong size.
     */
    class ParameterError : public std::invalid_argument, public Error
    {
      public:
        explicit ParameterError(const std::string& message)
            : std::invalid_argument(message)
        {
        }

        const char* what() const noexcept override
        {
            return std::invalid_argument::what();
        }
    };

    /**
     * @brief Thrown when an object is used before it is ready or after it can
     * no longer change, for example a key that is not generated yet or a key
     * that is generated twice.
     */
    class StateError : public std::invalid_argument, public Error
    {
      public:
        explicit StateError(const std::string& message)
            : std::invalid_argument(message)
        {
        }

        const char* what() const noexcept override
        {
            return std::invalid_argument::what();
        }
    };

    /**
     * @brief Thrown when an object created with one HEContext is used with a
     * different HEContext.
     */
    class ContextMismatchError : public std::invalid_argument, public Error
    {
      public:
        explicit ContextMismatchError(const std::string& message)
            : std::invalid_argument(message)
        {
        }

        const char* what() const noexcept override
        {
            return std::invalid_argument::what();
        }
    };

    /**
     * @brief Thrown when the operands of an operation are not at the levels it
     * requires, for example two ciphertexts at different levels.
     */
    class LevelMismatchError : public std::invalid_argument, public Error
    {
      public:
        explicit LevelMismatchError(const std::string& message)
            : std::invalid_argument(message)
        {
        }

        const char* what() const noexcept override
        {
            return std::invalid_argument::what();
        }
    };

    /**
     * @brief Thrown when a key needed by an operation is missing, for example
     * a Galois key for a rotation step that was not generated.
     */
    class KeyNotFoundError : public std::invalid_argument, public Error
    {
      public:
        explicit KeyNotFoundError(const std::string& message)
            : std::invalid_argument(message)
        {
        }

        const char* what() const noexcept override
        {
            return std::invalid_argument::what();
        }
    };

    /**
     * @brief Thrown when serialized data is malformed, truncated or written
     * for a different scheme.
     */
    class SerializationError : public std::runtime_error, public Error
    {
      public:
        explicit SerializationError(const std::string& message)
            : std::runtime_error(message)
        {
        }

        const char* what() const noexcept override
        {
            return std::runtime_error::what();
        }
    };

    /**
     * @brief Thrown when a system service HEonGPU depends on fails, for
     * example the operating system random number generator.
     */
    class SystemError : public std::runtime_error, public Error
    {
      public:
        explicit SystemError(const std::string& message)
            : std::runtime_error(message)
        {
        }

        const char* what() const noexcept override
        {
            return std::runtime_error::what();
        }
    };

    /**
     * @brief Builds an error message by streaming all arguments in order.
     */
    template <typename... Args>
    inline std::string error_message(const Args&... args)
    {
        std::ostringstream stream;
        (stream << ... << args);
        return stream.str();
    }

} // namespace heongpu
#endif // HEONGPU_ERROR_H
