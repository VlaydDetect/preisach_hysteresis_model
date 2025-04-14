// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Functions/zeros.hpp"
#include "Linalg/det.hpp"
#include "Matrix.hpp"
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    namespace linalg
    {
        /// matrix inverse
        template <typename dtype>
        Matrix<double> inv(const Matrix<dtype> &inArray)
        {
            STATIC_ASSERT_ARITHMETIC_OR_COMPLEX(dtype);

            const Shape inShape = inArray.shape();
            if (inShape.m_Rows != inShape.m_Cols)
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be square.");
            }

            Matrix<double> inArrayDouble = inArray.template astype<double>();
            Matrix<int> incidence = mc::zeros<int>(inShape);

            for (uint32 k = 0; k < inShape.m_Rows - 1; ++k)
            {
                if (utils::essentiallyEqual(inArrayDouble(k, k), 0.))
                {
                    uint32 l = k;
                    while (l < inShape.m_Cols && utils::essentiallyEqual(inArrayDouble(k, l), 0.))
                    {
                        ++l;
                    }

                    inArrayDouble.swapRows(k, l);
                    incidence(k, k) = 1;
                    incidence(k, l) = 1;
                }
            }

            Matrix<double> result(inShape);

            for (uint32 k = 0; k < inShape.m_Rows; ++k)
            {
                result(k, k) = -1. / inArrayDouble(k, k);
                for (uint32 i = 0; i < inShape.m_Rows; ++i)
                {
                    for (uint32 j = 0; j < inShape.m_Cols; ++j)
                    {
                        if ((i - k) && (j - k))
                        {
                            result(i, j) = inArrayDouble(i, j) + inArrayDouble(k, j) * inArrayDouble(i, k) * result(k, k);
                        }
                        else if ((i - k) && !(j - k))
                        {
                            result(i, k) = inArrayDouble(i, k) * result(k, k);
                        }
                        else if (!(i - k) && (j - k))
                        {
                            result(k, j) = inArrayDouble(k, j) * result(k, k);
                        }
                    }
                }

                inArrayDouble = result;
            }

            result *= -1.;

            for (int i = static_cast<int>(inShape.m_Rows) - 1; i >= 0; --i)
            {
                if (incidence(i, i) != 1)
                {
                    continue;
                }

                int k = 0;
                for (; k < static_cast<int>(inShape.m_Cols); ++k)
                {
                    if ((k - i) && incidence(i, k) != 0)
                    {
                        result.swapCols(i, k);
                        break;
                    }
                }
            }

            return result;
        }
    }
}
