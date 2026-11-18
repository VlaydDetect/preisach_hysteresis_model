// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include "Utils/essentiallyEqual.hpp"
#include <string_view>

namespace mc
{
    template <typename Derived>
    Eigen::RowVectorX<typename Derived::Scalar> trim_zeros(const Eigen::DenseBase<Derived> &input, std::string_view trim = "fb")
    {
        if (trim != "f" && trim != "b" && trim != "fb")
        {
            throw std::invalid_argument("Trim options are f, b, and fb");
        }
        using T = typename Derived::Scalar;
        const auto values = detail::rowMajor(input);
        Eigen::Index first = 0;
        Eigen::Index last = values.size();
        while (trim != "b" && first < last && utils::essentiallyEqual(values(first), T{}))
        {
            ++first;
        }
        while (trim != "f" && first < last && utils::essentiallyEqual(values(last - 1), T{}))
        {
            --last;
        }
        return values.segment(first, last - first).transpose();
    }
}
