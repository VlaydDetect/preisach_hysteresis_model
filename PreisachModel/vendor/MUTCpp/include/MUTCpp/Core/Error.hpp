// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
//
// License
// Copyright (c) 2024 Vlayd Detect.
// All rights reserved.

#pragma once

#include <iostream>
#include <string>
#include <stdexcept>

#include "Core/Types.hpp"

#define THROW_INVALID_ARGUMENT_ERROR(msg) mc::error::throwError<std::invalid_argument>(__FILE__, __func__, __LINE__, msg)
#define THROW_RUNTIME_ERROR(msg) mc::error::throwError<std::runtime_error>(__FILE__, __func__, __LINE__, msg)
#define THROW_OVERFLOW_ERROR(msg) mc::error::throwError<std::overflow_error>(__FILE__, __func__, __LINE__, msg)
#define THROW_UNDERFLOW_ERROR(msg) mc::error::throwError<std::underflow_error>(__FILE__, __func__, __LINE__, msg)
#define THROW_OUT_OF_RANGE_ERROR(msg) mc::error::throwError<std::out_of_range>(__FILE__, __func__, __LINE__, msg)

namespace mc {
    namespace error {
        /// Makes the full error message string
        ///
        /// @param file: the file
        /// @param function: the function
        /// @param line: the line of the file
        /// @param msg: the message to throw (default "")
        ///
        template<typename ErrorType>
        void throwError(const std::string &file, const std::string &function, const uint32 line,
                        const std::string &msg = "") {
            std::string errMsg =
                    "File: " + file + "\n\tFunction: " + function + "\n\tLine: " + std::to_string(line) + "\n\tError: "
                    + msg;
            std::cerr << errMsg;
            throw ErrorType(errMsg);
        }
    }
}
