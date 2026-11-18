// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    template <typename Derived>
    auto none(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        using Scalar = typename Derived::Scalar;
        return detail::reduce(input, axis, [](const auto &values)
        {
            const auto isNonzero = [](const Scalar &value)
            {
                return !utils::essentiallyEqual(value, Scalar{});
            };
            return !values.unaryExpr(isNonzero).any();
        });
    }
}
