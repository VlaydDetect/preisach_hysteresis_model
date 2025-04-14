// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Computes the number of 1-bits in an integer
    template <typename dtype>
    constexpr int bit_count(dtype inValue) noexcept
    {
        STATIC_ASSERT_UNSIGNED_INTEGER(dtype);

        int count = 0;
        for (int bit = 0; bit < DtypeInfo<dtype>::bits(); ++bit)
        {
            count += static_cast<int>((inValue & (dtype{1} << bit)) >> bit);
        }

        return count;
    }

    /// Computes the number of 1-bits in an integer
    template <typename dtype>
    Matrix<int> bit_count(const Matrix<dtype> &inArray)
    {
        Matrix<int> returnArray(inArray.shape());

        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [](dtype inValue) noexcept -> int { return bit_count(inValue); });

        return returnArray;
    }
}
