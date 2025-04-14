// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Returns True if input arrays are shape consistent and all elements equal.
    ///
    /// Shape consistent means they are either the same shape, or one input array
    /// can be broadcasted to create the same shape as the other one.
    template<typename dtype>
    bool array_equiv(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2) noexcept
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        if (inArray1.size() != inArray2.size())
        {
            return false;
        }

        if (DtypeInfo<dtype>::isInteger())
        {
            return algo::equal(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin());
        }

        const auto comparitor = [](dtype value1, dtype value2) noexcept -> bool
        { return utils::essentiallyEqual(value1, value2); };

        return algo::equal(inArray1.cbegin(), inArray1.cend(), inArray2.cbegin(), comparitor);
    }
}
