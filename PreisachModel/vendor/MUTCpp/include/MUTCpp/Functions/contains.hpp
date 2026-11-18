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
    auto contains(const Eigen::DenseBase<Derived> &input, typename Derived::Scalar target, Axis axis = Axis::NONE)
    {
        return detail::reduce(input, axis, [target](const auto &values)
        {
            return std::any_of(values.begin(), values.end(), [target](const auto &value)
            {
                return utils::essentiallyEqual(value, target);
            });
        });
    }
}
