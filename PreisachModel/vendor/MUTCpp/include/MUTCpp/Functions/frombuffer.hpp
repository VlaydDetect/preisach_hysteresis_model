// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace mc
{
    template <typename T>
    requires std::is_trivially_copyable_v<T>
    Eigen::RowVectorX<T> frombuffer(const char *data, std::size_t bytes)
    {
        if (bytes % sizeof(T) != 0 || (bytes != 0 && !data))
        {
            throw std::invalid_argument("Invalid byte buffer for requested scalar type");
        }
        Eigen::RowVectorX<T> result(static_cast<Eigen::Index>(bytes / sizeof(T)));
        if (bytes != 0) std::memcpy(result.data(), data, bytes);
        return result;
    }
}
