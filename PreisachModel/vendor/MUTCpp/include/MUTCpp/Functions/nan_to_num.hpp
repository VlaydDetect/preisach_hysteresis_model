// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <utility>

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Functions/isinf.hpp"
#include "Functions/isnan.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Replace NaN with zero and infinity with large finite numbers (default behavior)
    /// or with the numbers defined by the user using the nan, posinf and/or neginf keywords.
    /// @param inArray
    /// @param nan: value to be used to fill NaN values, default 0
    /// @param posInf: value to be used to fill positive infinity values, default a very large number
    /// @param negInf: value to be used to fill negative infinity values, default a very large negative number
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> nan_to_num(Matrix<dtype> inArray,
                             dtype nan = static_cast<dtype>(0.),
                             dtype posInf = DtypeInfo<dtype>::max(),
                             dtype negInf = DtypeInfo<dtype>::min())
    {
        STATIC_ASSERT_FLOAT(dtype);

        algo::for_each(inArray.begin(),
                       inArray.end(),
                       [nan, posInf, negInf](dtype &value)
                       {
                           if (isnan(value))
                           {
                               value = nan;
                           }
                           else if (isinf(value))
                           {
                               if (value > static_cast<dtype>(0.))
                               {
                                   value = posInf;
                               }
                               else
                               {
                                   value = negInf;
                               }
                           }
                       });

        return inArray;
    }
}
