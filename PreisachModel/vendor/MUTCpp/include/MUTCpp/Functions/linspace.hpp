// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Enums.hpp"
#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return evenly spaced numbers over a specified interval.
    /// Returns num evenly spaced samples, calculated over the
    /// interval[start, stop].
    /// The endpoint of the interval can optionally be excluded.
    /// Mostly only useful if called with a floating point type
    /// for the template argument.
    /// @param inStart
    /// @param inStop
    /// @param inNum: number of points (default = 50)
    /// @param endPoint: include endPoint (default = true)
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> linspace(dtype inStart, dtype inStop, uint32 inNum = 50, EndPoint endPoint = EndPoint::YES)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inNum == 0)
        {
            return Matrix<dtype>(0);
        }

        if (inNum == 1)
        {
            Matrix<dtype> returnArray = {inStart};
            return returnArray;
        }

        if (inStop <= inStart)
        {
            THROW_INVALID_ARGUMENT_ERROR("stop value must be greater than the start value.");
        }

        if (endPoint == EndPoint::YES)
        {
            if (inNum == 2)
            {
                Matrix<dtype> returnArray = {inStart, inStop};
                return returnArray;
            }

            Matrix<dtype> returnArray(1, inNum);
            returnArray.front() = inStart;
            returnArray.back() = inStop;

            dtype step = (inStop - inStart) / static_cast<dtype>(inNum - 1);

            for (uint32 i = 1; i < inNum - 1; ++i)
            {
                returnArray[i] = inStart + static_cast<dtype>(i) * step;
            }

            return returnArray;
        }

        if (inNum == 2)
        {
            dtype step = (inStop - inStart) / (inNum);
            Matrix<dtype> returnArray = {inStart, inStart + step};
            return returnArray;
        }

        Matrix<dtype> returnArray(1, inNum);
        returnArray.front() = inStart;

        dtype step = (inStop - inStart) / static_cast<dtype>(inNum);

        for (uint32 i = 1; i < inNum; ++i)
        {
            returnArray[i] = inStart + static_cast<dtype>(i) * step;
        }

        return returnArray;
    }
}
