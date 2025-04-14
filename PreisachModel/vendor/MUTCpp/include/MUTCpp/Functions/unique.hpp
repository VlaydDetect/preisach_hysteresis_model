// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <complex>
#include <vector>

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Functions/sort.hpp"

namespace mc
{
    /// Find the unique elements of an array.
    ///
    /// Returns the sorted unique elements of an array.
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> unique(const Matrix<dtype> &inArray)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto comp = [](const dtype lhs, const dtype rhs) noexcept -> bool
        {
            return utils::essentiallyEqual(lhs, rhs);
        };

        const auto sorted = sort(inArray);

        std::vector<dtype> res(sorted.size());
        const auto last = algo::unique_copy(sorted.begin(), sorted.end(), res.begin(), comp);

        return Matrix<dtype>(res.begin(), last);
    }
}
