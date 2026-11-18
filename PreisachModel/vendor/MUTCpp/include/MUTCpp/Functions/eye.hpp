// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>

#include "Core/Types.hpp"

namespace mc
{
    template <typename T>
    Eigen::MatrixX<T> eye(uint32 rows, uint32 columns, int32 offset = 0)
    {
        Eigen::MatrixX<T> result = Eigen::MatrixX<T>::Zero(rows, columns);
        if (static_cast<Eigen::Index>(offset) > -static_cast<Eigen::Index>(rows) &&
            static_cast<Eigen::Index>(offset) < static_cast<Eigen::Index>(columns) && rows != 0 && columns != 0)
        {
            result.diagonal(offset).setOnes();
        }
        return result;
    }

    template <typename T>
    Eigen::MatrixX<T> eye(uint32 size, int32 offset = 0)
    {
        return eye<T>(size, size, offset);
    }
}
