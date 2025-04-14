// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Types.hpp"
#include "Functions/fromstring.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Construct an array from data in a binary file.
    template<typename dtype>
    Matrix<dtype> fromfile(const std::string& inFilename)
    {
        if (!std::filesystem::exists(inFilename))
        {
            THROW_INVALID_ARGUMENT_ERROR("input filename does not exist.\n\t" + inFilename);
        }

        // read in as binary file
        std::ifstream file(inFilename.c_str(), std::ios::in | std::ios::binary);
        if (!file.is_open())
        {
            THROW_INVALID_ARGUMENT_ERROR("unable to open file\n\t" + inFilename);
        }

        file.seekg(0, std::ifstream::end);
        const uint32 fileSize = static_cast<uint32>(file.tellg());
        file.seekg(0, std::ifstream::beg);

        std::vector<char> fileBuffer;
        fileBuffer.reserve(fileSize);
        file.read(fileBuffer.data(), fileSize);

        if (file.bad() || file.fail())
        {
            THROW_INVALID_ARGUMENT_ERROR("error occured while reading the file");
        }

        file.close();

        Matrix<dtype> returnArray(reinterpret_cast<dtype*>(fileBuffer.data()), fileSize / sizeof(dtype));

        return returnArray;
    }

    /// Construct an array from data in a text file.
    template<typename dtype>
    Matrix<dtype> fromfile(const std::string& inFilename, const char inSep)
    {
        std::ifstream file(inFilename.c_str());
        if (!file.is_open())
        {
            THROW_INVALID_ARGUMENT_ERROR("unable to open file\n\t" + inFilename);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        return fromstring<dtype>(buffer.str(), inSep);
    }
}
