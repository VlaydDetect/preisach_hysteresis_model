// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Matrix.hpp"

namespace mc
{
    //============================================================================
    // Method Description:
    /// Given the "legs" of a right triangle, return its hypotenuse.
    ///
    /// Equivalent to sqrt(x1**2 + x2**2), element - wise.
    template<typename dtype>
    double hypot(dtype inValue1, dtype inValue2) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        return std::hypot(static_cast<double>(inValue1), static_cast<double>(inValue2));
    }

    //============================================================================
    // Method Description:
    /// Given the "legs" of a right triangle, return its hypotenuse.
    ///
    /// Equivalent to sqrt(x1**2 + x2**2 + x3**2), element - wise.
    template<typename dtype>
    double hypot(dtype inValue1, dtype inValue2, dtype inValue3) noexcept
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

#ifdef __cpp_lib_hypot
        return std::hypot(static_cast<double>(inValue1), static_cast<double>(inValue2), static_cast<double>(inValue3));
#else
        return std::sqrt(utils::sqr(static_cast<double>(inValue1)) + utils::sqr(static_cast<double>(inValue2)) +
                         utils::sqr(static_cast<double>(inValue3)));
#endif
    }

    //============================================================================
    // Method Description:
    /// Given the "legs" of a right triangle, return its hypotenuse.
    ///
    /// Equivalent to sqrt(x1**2 + x2**2), element - wise.
    template<typename dtype>
    Matrix<double> hypot(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2)
    {
        return broadcast::broadcaster<double>(inArray1,
                                              inArray2,
                                              [](dtype inValue1, dtype inValue2) noexcept -> double
                                              { return hypot(inValue1, inValue2); });
    }

    //============================================================================
    // Method Description:
    /// Given the "legs" of a right triangle, return its hypotenuse.
    ///
    /// Equivalent to sqrt(x1**2 + x2**2), element - wise.
    template<typename dtype>
    Matrix<double>
        hypot(const Matrix<dtype>& inArray1, const Matrix<dtype>& inArray2, const Matrix<dtype>& inArray3)
    {
        if (inArray1.size() != inArray2.size() || inArray1.size() != inArray3.size())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array sizes are not consistant.");
        }

        Matrix<double> returnArray(inArray1.shape());
        for (typename Matrix<dtype>::size_type i = 0; i < inArray1.size(); ++i)
        {
            returnArray[i] = hypot(inArray1[i], inArray2[i], inArray3[i]);
        }

        return returnArray;
    }
}
