// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/TypeTraits.hpp"
#include "Core/Shape.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    namespace linalg
    {
        namespace detail
        {
            /// matrix determinant.
            template <typename dtype>
            auto det(const Matrix<dtype> &inArray, uint32 order)
                -> std::conditional_t<std::is_integral_v<dtype>, int64, double>
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                using ReturnType = std::conditional_t<std::is_integral_v<dtype>, int64, double>;

                if (order == 1)
                {
                    return static_cast<ReturnType>(inArray.front());
                }

                if (order == 2)
                {
                    return static_cast<ReturnType>(inArray(0, 0)) * static_cast<ReturnType>(inArray(1, 1)) -
                        static_cast<ReturnType>(inArray(0, 1)) * static_cast<ReturnType>(inArray(1, 0));
                }

                if (order == 3)
                {
                    const auto aei = static_cast<ReturnType>(inArray(0, 0)) * static_cast<ReturnType>(inArray(1, 1)) *
                        static_cast<ReturnType>(inArray(2, 2));
                    const auto bfg = static_cast<ReturnType>(inArray(0, 1)) * static_cast<ReturnType>(inArray(1, 2)) *
                        static_cast<ReturnType>(inArray(2, 0));
                    const auto cdh = static_cast<ReturnType>(inArray(0, 2)) * static_cast<ReturnType>(inArray(1, 0)) *
                        static_cast<ReturnType>(inArray(2, 1));
                    const auto ceg = static_cast<ReturnType>(inArray(0, 2)) * static_cast<ReturnType>(inArray(1, 1)) *
                        static_cast<ReturnType>(inArray(2, 0));
                    const auto bdi = static_cast<ReturnType>(inArray(0, 1)) * static_cast<ReturnType>(inArray(1, 0)) *
                        static_cast<ReturnType>(inArray(2, 2));
                    const auto afh = static_cast<ReturnType>(inArray(0, 0)) * static_cast<ReturnType>(inArray(1, 2)) *
                        static_cast<ReturnType>(inArray(2, 1));

                    return aei + bfg + cdh - ceg - bdi - afh;
                }

                ReturnType determinant = 0;
                ReturnType sign = 1;
                Matrix<dtype> submat(order - 1);

                for (uint32 c = 0; c < order; ++c)
                {
                    uint32 subi = 0;
                    for (uint32 i = 1; i < order; ++i)
                    {
                        uint32 subj = 0;
                        for (uint32 j = 0; j < order; ++j)
                        {
                            if (j == c)
                            {
                                continue;
                            }

                            submat(subi, subj++) = inArray(i, j);
                        }
                        ++subi;
                    }

                    determinant += (sign * static_cast<ReturnType>(inArray(0, c)) * det(submat, order - 1));
                    sign *= -1;
                }

                return determinant;
            }
        }

        /// matrix determinant.
        /// NOTE: can get verrrrry slow for large matrices (order > 10)
        template <typename dtype>
        auto det(const Matrix<dtype> &inArray)
        {
            STATIC_ASSERT_ARITHMETIC(dtype);

            const Shape inShape = inArray.shape();
            if (!inShape.issquare())
            {
                THROW_INVALID_ARGUMENT_ERROR("input array must be square.");
            }

            return detail::det(inArray, inShape.m_Rows);
        }
    }
}
