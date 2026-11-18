// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/fromstring.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace mc
{
    template <typename T>
    requires std::is_trivially_copyable_v<T>
    Eigen::RowVectorX<T> fromfile(const std::filesystem::path &path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Unable to open " + path.string());
        const auto bytes = static_cast<std::streamoff>(file.tellg());
        if (bytes < 0 || bytes % sizeof(T) != 0)
        {
            throw std::runtime_error("Invalid binary file size for " + path.string());
        }
        Eigen::RowVectorX<T> result(static_cast<Eigen::Index>(bytes / sizeof(T)));
        file.seekg(0);
        if (bytes != 0) file.read(reinterpret_cast<char *>(result.data()), bytes);
        if (!file) throw std::runtime_error("Unable to read " + path.string());
        return result;
    }

    template <typename T>
    Eigen::RowVectorX<T> fromfile(const std::filesystem::path &path, char separator)
    {
        std::ifstream file(path);
        if (!file) throw std::runtime_error("Unable to open " + path.string());
        std::ostringstream text;
        text << file.rdbuf();
        if (file.bad()) throw std::runtime_error("Unable to read " + path.string());
        return fromstring<T>(text.str(), separator);
    }
}
