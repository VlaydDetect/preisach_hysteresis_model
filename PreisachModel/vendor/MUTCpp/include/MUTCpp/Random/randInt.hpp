// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <random>
#include <string>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Matrix.hpp"
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
            Matrix<dtype> randInt(GeneratorType &generator, const Shape &inShape, dtype inLow, dtype inHigh = 0)
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

                Matrix<dtype> returnArray(inShape);

                std::uniform_int_distribution<dtype> dist(inLow, inHigh - 1);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
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
        Matrix<dtype> randInt(const Shape &inShape, dtype inLow, dtype inHigh = 0)
        {
            return detail::randInt(generator_, inShape, inLow, inHigh);
        }
    }
}
