// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <complex>
#include "Core/TypeTraits.hpp"

namespace mc
{
    template <typename Derived>
    auto gradient(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::ROW)
    {
        return detail::mapAxis(input, axis, 0, [&](const auto &slice)
        {
            using T = std::conditional_t<mc::is_complex_v<typename Derived::Scalar>, std::complex<double>, double>;
            Eigen::RowVectorX<T> result = Eigen::RowVectorX<T>::Zero(slice.size());
            if (slice.size() < 2) return result;
            result(0) = static_cast<T>(slice(1)) - static_cast<T>(slice(0));
            for (Eigen::Index i = 1; i + 1 < slice.size(); ++i)
            {
                result(i) = (static_cast<T>(slice(i + 1)) - static_cast<T>(slice(i - 1))) / 2.0;
            }
            result(slice.size() - 1) = static_cast<T>(slice(slice.size() - 1)) - static_cast<T>(slice(slice.size() - 2));
            return result;
        });
    }
}
