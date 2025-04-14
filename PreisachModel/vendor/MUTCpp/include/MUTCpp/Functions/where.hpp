// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <string>

#include "Core/Error.hpp"
#include "Core/Shape.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Return elements, either from x or y, depending on the input mask.
    /// The output array contains elements of x where mask is True, and
    /// elements from y elsewhere.
    template <typename dtype>
    Matrix<dtype> where(const Matrix<bool> &inMask, const Matrix<dtype> &inA, const Matrix<dtype> &inB)
    {
        const auto shapeMask = inMask.shape();
        const auto shapeA = inA.shape();
        if (shapeA != inB.shape())
        {
            THROW_INVALID_ARGUMENT_ERROR("input inA and inB must be the same shapes.");
        }

        if (shapeMask != shapeA)
        {
            THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
        }

        auto outArray = Matrix<dtype>(shapeMask);

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA[idx];
            }
            else
            {
                outArray[idx] = inB[idx];
            }
            ++idx;
        }

        return outArray;
    }

    /// Return elements, either from x or y, depending on the input mask.
    /// The output array contains elements of x where mask is True, and
    /// elements from y elsewhere.
    template <typename dtype>
    Matrix<dtype> where(const Matrix<bool> &inMask, const Matrix<dtype> &inA, dtype inB)
    {
        const auto shapeMask = inMask.shape();
        const auto shapeA = inA.shape();
        if (shapeMask != shapeA)
        {
            THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
        }

        auto outArray = Matrix<dtype>(shapeMask);

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA[idx];
            }
            else
            {
                outArray[idx] = inB;
            }
            ++idx;
        }

        return outArray;
    }

    /// Return elements, either from x or y, depending on the input mask.
    /// The output array contains elements of x where mask is True, and
    /// elements from y elsewhere.
    template <typename dtype>
    Matrix<dtype> where(const Matrix<bool> &inMask, dtype inA, const Matrix<dtype> &inB)
    {
        const auto shapeMask = inMask.shape();
        const auto shapeB = inB.shape();
        if (shapeMask != shapeB)
        {
            THROW_INVALID_ARGUMENT_ERROR("input inMask must be the same shape as the input arrays.");
        }

        auto outArray = Matrix<dtype>(shapeMask);

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA;
            }
            else
            {
                outArray[idx] = inB[idx];
            }
            ++idx;
        }

        return outArray;
    }

    /// Return elements, either from x or y, depending on the input mask.
    /// The output array contains elements of x where mask is True, and
    /// elements from y elsewhere.
    template <typename dtype>
    Matrix<dtype> where(const Matrix<bool> &inMask, dtype inA, dtype inB)
    {
        auto outArray = Matrix<dtype>(inMask.shape());

        uint32 idx = 0;
        for (auto maskValue : inMask)
        {
            if (maskValue)
            {
                outArray[idx] = inA;
            }
            else
            {
                outArray[idx] = inB;
            }
            ++idx;
        }

        return outArray;
    }
}
