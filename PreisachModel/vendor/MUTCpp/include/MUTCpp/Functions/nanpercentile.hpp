// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "Core/Enums.hpp"
#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/argmin.hpp"
#include "Functions/clip.hpp"
#include "Functions/isnan.hpp"
#include "Functions/percentile.hpp"
#include "Matrix.hpp"
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    /// Compute the qth percentile of the data along the specified axis, while ignoring nan values.
    /// @param inArray
    /// @param inPercentile
    /// @param inAxis (Optional, default NONE)
    /// @param inInterpMethod (default linear) choices = ['linear','lower','higher','nearest','midpoint']
    /// @return Matrix
    template <typename dtype>
    Matrix<double> nanpercentile(const Matrix<dtype> &inArray,
                                 double inPercentile,
                                 Axis inAxis = Axis::NONE,
                                 InterpolationMethod inInterpMethod = InterpolationMethod::LINEAR)
    {
        STATIC_ASSERT_FLOAT(dtype);

        switch (inAxis)
        {
        case Axis::NONE:
        {
            std::vector<double> arrayCopy;
            arrayCopy.reserve(inArray.size());
            for (auto value : inArray)
            {
                if (!isnan(value))
                {
                    arrayCopy.push_back(static_cast<double>(value));
                }
            }

            if (arrayCopy.empty())
            {
                Matrix<double> returnArray = {consts::nan};
                return returnArray;
            }

            return percentile(Matrix<double>(arrayCopy.data(),
                                             static_cast<typename Matrix<dtype>::size_type>(arrayCopy.size()),
                                             PointerPolicy::SHELL),
                              inPercentile,
                              Axis::NONE,
                              inInterpMethod);
        }
        case Axis::COL:
        {
            const Shape inShape = inArray.shape();

            Matrix<double> returnArray(1, inShape.m_Rows);
            for (uint32 row = 0; row < inShape.m_Rows; ++row)
            {
                Matrix<double> outValue = nanpercentile(Matrix<dtype>(&inArray.front(row), inShape.m_Cols),
                                                        inPercentile,
                                                        Axis::NONE,
                                                        inInterpMethod);

                if (outValue.isscalar())
                {
                    returnArray[row] = outValue.item();
                }
                else
                {
                    returnArray[row] = consts::nan;
                }
            }

            return returnArray;
        }
        case Axis::ROW:
        {
            return nanpercentile(inArray.transpose(), inPercentile, Axis::COL, inInterpMethod);
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
