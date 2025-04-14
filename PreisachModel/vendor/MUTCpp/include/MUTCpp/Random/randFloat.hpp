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
#include "Utils/essentiallyEqual.hpp"

namespace mc
{
    namespace random
    {
        namespace detail
        {
            /// Return a single random float from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype randFloat(GeneratorType &generator, dtype inLow, dtype inHigh = 0.)
            {
                STATIC_ASSERT_FLOAT(dtype);

                if (utils::essentiallyEqual(inLow, inHigh))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
                }
                else if (inLow > inHigh)
                {
                    std::swap(inLow, inHigh);
                }

                std::uniform_real_distribution<dtype> dist(inLow, inHigh - DtypeInfo<dtype>::epsilon());
                return dist(generator);
            }

            /// Return random floats from low (inclusive) to high (exclusive),
            /// with the given shape. If no high value is input then the range will
            /// go from [0, low).
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> randFloat(GeneratorType &generator, const Shape &inShape, dtype inLow, dtype inHigh = 0.)
            {
                STATIC_ASSERT_FLOAT(dtype);

                if (utils::essentiallyEqual(inLow, inHigh))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input low value must be less than the input high value.");
                }
                else if (inLow > inHigh)
                {
                    std::swap(inLow, inHigh);
                }

                Matrix<dtype> returnArray(inShape);

                std::uniform_real_distribution<dtype> dist(inLow, inHigh - DtypeInfo<dtype>::epsilon());

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Return a single random float from low (inclusive) to high (exclusive),
        /// with the given shape. If no high value is input then the range will
        /// go from [0, low).
        template <typename dtype>
        dtype randFloat(dtype inLow, dtype inHigh = 0.)
        {
            return detail::randFloat(generator_, inLow, inHigh);
        }

        /// Return random floats from low (inclusive) to high (exclusive),
        /// with the given shape. If no high value is input then the range will
        /// go from [0, low).
        template <typename dtype>
        Matrix<dtype> randFloat(const Shape &inShape, dtype inLow, dtype inHigh = 0.)
        {
            return detail::randFloat(generator_, inShape, inLow, inHigh);
        }
    }
}
