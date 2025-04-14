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
#include "Functions/unique.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Find the set difference of two arrays.
    /// Return the sorted, unique values in ar1 that are not in ar2.
    template <typename dtype>
    Matrix<dtype> setdiff1d(const Matrix<dtype> &inArray1, const Matrix<dtype> &inArray2)
    {
        STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

        const auto comp = [](const dtype lhs, const dtype rhs) noexcept -> bool { return lhs < rhs; };

        const auto set1 = unique(inArray1);
        const auto set2 = unique(inArray2);

        std::vector<dtype> res(set1.size());
        const auto last = algo::set_difference(set1.begin(), set1.end(), set2.begin(),
                                               set2.end(), res.begin(), comp);

        return Matrix<dtype>(res.begin(), last);
    }
}
