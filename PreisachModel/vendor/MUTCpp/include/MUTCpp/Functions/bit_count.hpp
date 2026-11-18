// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"
#include <Eigen/Core>

namespace mc
{
    /// Computes the number of 1-bits in an integer
    template <typename dtype>
        requires (std::is_arithmetic_v<dtype> || mc::is_complex_v<dtype>)
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
    template <typename Derived>
    auto bit_count(const Eigen::DenseBase<Derived> &inArray)
    {
        return inArray.derived().unaryExpr([](const auto &inValue)
        {
            return mc::bit_count(inValue);
        }).eval();
    }
}
