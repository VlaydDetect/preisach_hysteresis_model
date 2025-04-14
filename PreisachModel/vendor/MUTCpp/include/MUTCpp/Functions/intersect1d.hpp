// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <set>
#include <vector>

#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Find the intersection of two arrays.
    ///
    /// Return the sorted, unique values that are in both of the input arrays.
    template<typename dtype>
    Matrix<dtype> intersect1d(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        std::vector<dtype>    res(inArray1.size() + inArray2.size());
        const std::set<dtype> in1(inArray1.cbegin(), inArray1.cend());
        const std::set<dtype> in2(inArray2.cbegin(), inArray2.cend());

        const auto iter = algo::set_intersection(in1.begin(), in1.end(), in2.begin(), in2.end(), res.begin());
        res.resize(iter - res.begin());
        return Matrix<dtype>(res);
    }
}
