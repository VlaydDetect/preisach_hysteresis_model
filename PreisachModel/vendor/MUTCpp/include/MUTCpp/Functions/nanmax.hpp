// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Types.hpp"
#include "Functions/max.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return the maximum of an array or maximum along an axis ignoring NaNs.
    template <typename dtype>
    Matrix<dtype> nanmax(const Matrix<dtype> &inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_FLOAT(dtype);

        Matrix<dtype> arrayCopy(inArray);
        algo::for_each(arrayCopy.begin(),
                       arrayCopy.end(),
                       [](dtype &value) noexcept -> void
                       {
                           if (std::isnan(value))
                           {
                               value = DtypeInfo<dtype>::min();
                           };
                       });

        return max(arrayCopy, inAxis);
    }
}
