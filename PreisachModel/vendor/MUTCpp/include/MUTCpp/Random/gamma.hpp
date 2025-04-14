// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <random>

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
            /// Single random value sampled from the "gamma" distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype gamma(GeneratorType &generator, dtype inGammaShape, dtype inScaleValue = 1)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inGammaShape <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input gamma shape should be greater than zero.");
                }

                if (inScaleValue <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input scale should be greater than zero.");
                }

                std::gamma_distribution<dtype> dist(inGammaShape, inScaleValue);
                return dist(generator);
            }


            /// Create an array of the given shape and populate it with
            /// random samples from a "gamma" distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> gamma(GeneratorType &generator, const Shape &inShape, dtype inGammaShape,
                                dtype inScaleValue = 1)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inGammaShape <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input gamma shape should be greater than zero.");
                }

                if (inScaleValue <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input scale should be greater than zero.");
                }

                Matrix<dtype> returnArray(inShape);

                std::gamma_distribution<dtype> dist(inGammaShape, inScaleValue);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the "gamma" distribution.
        template <typename dtype>
        dtype gamma(dtype inGammaShape, dtype inScaleValue = 1)
        {
            return detail::gamma(generator_, inGammaShape, inScaleValue);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from a "gamma" distribution.
        template <typename dtype>
        Matrix<dtype> gamma(const Shape &inShape, dtype inGammaShape, dtype inScaleValue = 1)
        {
            return detail::gamma(generator_, inShape, inGammaShape, inScaleValue);
        }
    }
}
