// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/DtypeInfo.hpp"
#include "Core/StaticAssert.hpp"
#include "Random/randFloat.hpp"

namespace mc
{
    namespace random
    {
        namespace detail
        {
            /// Draw sample from a uniform distribution.
            ///
            /// Samples are uniformly distributed over the half -
            /// open interval[low, high) (includes low, but excludes high)
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype uniform(GeneratorType &generator, dtype inLow, dtype inHigh)
            {
                STATIC_ASSERT_FLOAT(dtype);

                return detail::randFloat(generator, inLow, inHigh);
            }

            /// Draw samples from a uniform distribution.
            ///
            /// Samples are uniformly distributed over the half -
            /// open interval[low, high) (includes low, but excludes high)
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> uniform(GeneratorType &generator, const Shape &inShape, dtype inLow, dtype inHigh)
            {
                STATIC_ASSERT_FLOAT(dtype);

                return detail::randFloat(generator, inShape, inLow, inHigh);
            }
        }

        /// Draw sample from a uniform distribution.
        ///
        /// Samples are uniformly distributed over the half -
        /// open interval[low, high) (includes low, but excludes high)
        template <typename dtype>
        dtype uniform(dtype inLow, dtype inHigh)
        {
            return detail::uniform(generator_, inLow, inHigh);
        }

        /// Draw samples from a uniform distribution.
        ///
        /// Samples are uniformly distributed over the half -
        /// open interval[low, high) (includes low, but excludes high)
        template <typename dtype>
        Matrix<dtype> uniform(const Shape &inShape, dtype inLow, dtype inHigh)
        {
            return detail::uniform(generator_, inShape, inLow, inHigh);
        }
    }
}
