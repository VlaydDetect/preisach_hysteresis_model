// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Matrix.hpp"

namespace mc
{
    /// Write an array to a file as binary.
    /// The data produced by this method can be recovered
    /// using the function fromfile().
    template <typename dtype>
    void tofile(const Matrix<dtype> &inArray, const std::string &inFilename)
    {
        return inArray.tofile(inFilename);
    }

    /// Write an array to a file as text.
    /// The data produced by this method can be recovered
    /// using the function fromfile().
    /// @param inArray
    /// @param inFilename
    /// @param inSep: Separator between array items for text output.
    template <typename dtype>
    void tofile(const Matrix<dtype> &inArray, const std::string &inFilename, const char inSep)
    {
        return inArray.tofile(inFilename, inSep);
    }
}
