// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <random>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Random/generator.hpp"

namespace mc
{
    namespace random
    {
        namespace detail
        {
            /// Return random integer from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype randInt(GeneratorType &generator, dtype inLow, dtype inHigh = 0)
            {
                STATIC_ASSERT_INTEGER(dtype);

                if (inLow == inHigh)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
                }
                else if (inLow > inHigh)
                {
                    std::swap(inLow, inHigh);
                }

                std::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);
                return dist(generator);
            }

            /// Return random integers from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            template <typename dtype, typename GeneratorType = std::mt19937>
            Eigen::MatrixX<dtype> randInt(GeneratorType &generator, Eigen::Index rows, Eigen::Index columns, dtype inLow, dtype inHigh = 0)
            {
                STATIC_ASSERT_INTEGER(dtype);

                if (inLow == inHigh)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
                }
                else if (inLow > inHigh - 1)
                {
                    std::swap(inLow, inHigh);
                }

                Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> returnArray(rows, columns);

                std::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);

                std::for_each(returnArray.data(),
                              returnArray.data() + returnArray.size(),
                              [&dist, &generator](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Return random integer from low (inclusive) to high (exclusive),
        /// with the given shape. If no high value is input then the range will
        /// go from [0, low).
        template <typename dtype>
        dtype randInt(dtype inLow, dtype inHigh = 0)
        {
            return detail::randInt(generator_, inLow, inHigh);
        }

        /// Return random integers from low (inclusive) to high (exclusive),
        /// with the given shape. If no high value is input then the range will
        /// go from [0, low).
        template <typename dtype>
        Eigen::MatrixX<dtype> randInt(Eigen::Index rows, Eigen::Index columns, dtype inLow, dtype inHigh = 0)
        {
            return detail::randInt(generator_, rows, columns, inLow, inHigh);
        }
    }
}
