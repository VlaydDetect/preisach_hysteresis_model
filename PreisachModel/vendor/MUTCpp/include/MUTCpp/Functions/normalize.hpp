// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>

#include "Core/StaticAssert.hpp"
#include "Core/Types.hpp"
#include "Functions/norm.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Matrix or vector norm.
    template<typename dtype>
    Matrix<double> normalize(const Matrix<dtype>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                return inArray / norm(inArray, Axis::NONE);
            }
            case Axis::COL:
            {
                Matrix<double> returnArray(inArray.shape());
                const auto      cSlice = returnArray.cSlice();
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    returnArray.put(row, cSlice, normalize(inArray.row(row), Axis::NONE));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return normalize(inArray.transpose(), Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }

    //============================================================================
    // Method Description:
    /// Matrix or vector norm.
    template<typename dtype>
    Matrix<std::complex<double>> normalize(const Matrix<std::complex<dtype>>& inArray, Axis inAxis = Axis::NONE)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        switch (inAxis)
        {
            case Axis::NONE:
            {
                return inArray / norm(inArray, Axis::NONE);
            }
            case Axis::COL:
            {
                Matrix<std::complex<double>> returnArray(inArray.shape());
                const auto                    cSlice = returnArray.cSlice();
                for (uint32 row = 0; row < inArray.numRows(); ++row)
                {
                    returnArray.put(row, cSlice, normalize(inArray.row(row), Axis::NONE));
                }

                return returnArray;
            }
            case Axis::ROW:
            {
                return normalize(inArray.transpose(), Axis::COL).transpose();
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
                return {}; // get rid of compiler warning
            }
        }
    }
}
