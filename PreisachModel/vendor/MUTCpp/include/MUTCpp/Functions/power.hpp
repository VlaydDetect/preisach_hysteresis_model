// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"
#include "Utils/mathOperations.hpp"

namespace mc
{
    /// Raises the elements of the array to the input integer power
    template <typename dtype>
    constexpr dtype power(dtype inValue, uint8 inExponent) noexcept
    {
        return utils::power(inValue, inExponent);
    }

    /// Raises the elements of the array to the input integer power
    template <typename dtype>
    Matrix<dtype> power(const Matrix<dtype> &inArray, uint8 inExponent)
    {
        Matrix<dtype> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        returnArray.begin(),
                        [inExponent](dtype inValue) noexcept -> dtype { return mc::power(inValue, inExponent); });

        return returnArray;
    }

    /// Raises the elements of the array to the input integer powers
    template <typename dtype>
    Matrix<dtype> power(const Matrix<dtype> &inArray, const Matrix<uint8> &inExponents)
    {
        if (inArray.shape() != inExponents.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input array shapes are not consistant.");
        }

        Matrix<dtype> returnArray(inArray.shape());
        algo::transform(inArray.cbegin(),
                        inArray.cend(),
                        inExponents.cbegin(),
                        returnArray.begin(),
                        [](dtype inValue, uint8 inExponent) -> dtype { return mc::power(inValue, inExponent); });

        return returnArray;
    }
}
