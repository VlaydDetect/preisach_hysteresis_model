// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>

#include "Core/Enums.hpp"
#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/argmin.hpp"
#include "Functions/clip.hpp"
#include "Matrix.hpp"
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    /// Compute the qth percentile of the data along the specified axis.
    /// @param inArray
    /// @param inPercentile: percentile must be in the range [0, 100]
    /// @param inAxis (Optional, default NONE)
    /// @param inInterpMethod (Optional) interpolation method
    /// linear: i + (j - i) * fraction, where a fraction is the fractional part of the index surrounded by i and j.
    /// lower: i.
    /// higher: j.
    /// nearest: i or j, whichever is nearest.
    /// midpoint: (i + j) / 2.
    /// @return Matrix
    template <typename dtype>
    Matrix<double> percentile(const Matrix<dtype> &inArray,
                              double inPercentile,
                              Axis inAxis = Axis::NONE,
                              InterpolationMethod inInterpMethod = InterpolationMethod::LINEAR)
    {
        STATIC_ASSERT_ARITHMETIC(dtype);

        if (inPercentile < 0. || inPercentile > 100.)
        {
            THROW_INVALID_ARGUMENT_ERROR("input percentile value must be of the range [0, 100].");
        }

        if (inArray.isempty())
        {
            return {};
        }
        else if (inArray.isscalar())
        {
            Matrix<double> returnArray = {static_cast<double>(inArray.front())};
            return returnArray;
        }

        switch (inAxis)
        {
        case Axis::NONE:
        {
            Matrix<double> arrayCopy = inArray.template astype<double>();
            algo::sort(arrayCopy.begin(), arrayCopy.end());

            if (utils::essentiallyEqual(inPercentile, 0.))
            {
                Matrix<double> returnArray = {arrayCopy.front()};
                return returnArray;
            }
            if (utils::essentiallyEqual(inPercentile, 100.))
            {
                Matrix<double> returnArray = {arrayCopy.back()};
                return returnArray;
            }

            const auto i =
                static_cast<uint32>(std::floor(static_cast<double>(inArray.size() - 1) * inPercentile / 100.));
            const auto indexLower = clip<uint32>(i, 0, inArray.size() - 2);

            switch (inInterpMethod)
            {
            case InterpolationMethod::LINEAR:
            {
                const double percentI =
                    static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                const double fraction =
                    (inPercentile / 100. - percentI) /
                    (static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1) - percentI);

                Matrix<double> returnArray = {
                    arrayCopy[indexLower] + (arrayCopy[indexLower + 1] - arrayCopy[indexLower]) * fraction
                };
                return returnArray;
            }
            case InterpolationMethod::LOWER:
            {
                Matrix<double> returnArray = {arrayCopy[indexLower]};
                return returnArray;
            }
            case InterpolationMethod::HIGHER:
            {
                Matrix<double> returnArray = {arrayCopy[indexLower + 1]};
                return returnArray;
            }
            case InterpolationMethod::NEAREST:
            {
                const double percent = inPercentile / 100.;
                const double percent1 =
                    static_cast<double>(indexLower) / static_cast<double>(inArray.size() - 1);
                const double percent2 =
                    static_cast<double>(indexLower + 1) / static_cast<double>(inArray.size() - 1);
                const double diff1 = percent - percent1;
                const double diff2 = percent2 - percent;

                switch (argmin<double>({diff1, diff2}).item())
                {
                case 0:
                {
                    Matrix<double> returnArray = {arrayCopy[indexLower]};
                    return returnArray;
                }
                case 1:
                {
                    Matrix<double> returnArray = {arrayCopy[indexLower + 1]};
                    return returnArray;
                }
                }

                return {}; // get rid of compiler warning
            }
            case InterpolationMethod::MIDPOINT:
            {
                Matrix<double> returnArray = {(arrayCopy[indexLower] + arrayCopy[indexLower + 1]) / 2.};
                return returnArray;
            }
            default:
            {
                THROW_INVALID_ARGUMENT_ERROR("Unimplemented Interpolation method.");
                return {}; // get rid of compiler warning
            }
            }
        }
        case Axis::COL:
        {
            const Shape inShape = inArray.shape();

            Matrix<double> returnArray(1, inShape.m_Rows);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                returnArray[row] =
                    percentile(Matrix<dtype>(const_cast<typename Matrix<dtype>::pointer>(&inArray.front(row)),
                                             inShape.m_Cols,
                                             PointerPolicy::SHELL),
                               inPercentile,
                               Axis::NONE,
                               inInterpMethod)
                    .item();
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return percentile(inArray.transpose(), inPercentile, Axis::COL, inInterpMethod);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }

        return {}; // get rid of compiler warning
    }
}
