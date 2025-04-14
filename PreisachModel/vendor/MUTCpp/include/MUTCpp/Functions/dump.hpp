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
    /// Dump a binary file of the array to the specified file.
    /// The array can be read back with or mc::load.
    template<typename dtype>
    void dump(const Matrix<dtype>& inArray, const std::string& inFilename)
    {
        inArray.dump(inFilename);
    }
}
