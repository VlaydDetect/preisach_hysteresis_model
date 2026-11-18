// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include <limits>

namespace mc
{
    inline Eigen::MatrixXd nans(Eigen::Index rows, Eigen::Index columns)
    {
        return Eigen::MatrixXd::Constant(rows, columns, std::numeric_limits<double>::quiet_NaN());
    }

    inline Eigen::MatrixXd nans(Eigen::Index size)
    {
        return nans(size, size);
    }
}
