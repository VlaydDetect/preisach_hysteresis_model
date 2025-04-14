// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <random>

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
            /// Returns a single random value sampled from the "standard normal" distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype randN(GeneratorType &generator)
            {
                STATIC_ASSERT_FLOAT(dtype);

                std::normal_distribution<dtype> dist;
                return dist(generator);
            }

            /// Create an array of the given shape and populate it with
            /// random samples from the "standard normal" distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> randN(GeneratorType &generator, const Shape &inShape)
            {
                STATIC_ASSERT_FLOAT(dtype);

                Matrix<dtype> returnArray(inShape);

                std::normal_distribution<dtype> dist;

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Returns a single random value sampled from the "standard normal" distribution.
        template <typename dtype>
        dtype randN()
        {
            return detail::randN<dtype>(generator_);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from the "standard normal" distribution.
        template <typename dtype>
        Matrix<dtype> randN(const Shape &inShape)
        {
            return detail::randN<dtype>(generator_, inShape);
        }
    }
}
