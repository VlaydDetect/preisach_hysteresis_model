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
            /// Single random value sampled from the
            /// "discrete" distribution.  It produces integers in the
            /// range [0, n) with the probability of producing each value
            /// is specified by the parameters of the distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype discrete(GeneratorType &generator, const Matrix<double> &inWeights)
            {
                STATIC_ASSERT_INTEGER(dtype);

                std::discrete_distribution<dtype> dist(inWeights.cbegin(), inWeights.cend());
                return dist(generator);
            }

            /// Create an array of the given shape and populate it with
            /// random samples from a "discrete" distribution.  It produces
            /// integers in the range [0, n) with the probability of
            /// producing each value is specified by the parameters
            /// of the distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> discrete(GeneratorType &generator, const Shape &inShape, const Matrix<double> &inWeights)
            {
                STATIC_ASSERT_INTEGER(dtype);

                Matrix<dtype> returnArray(inShape);

                std::discrete_distribution<dtype> dist(inWeights.cbegin(), inWeights.cend());

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the
        /// "discrete" distribution.  It produces integers in the
        /// range [0, n) with the probability of producing each value
        /// is specified by the parameters of the distribution.
        template <typename dtype>
        dtype discrete(const Matrix<double> &inWeights)
        {
            return detail::discrete<dtype>(generator_, inWeights);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from a "discrete" distribution.  It produces
        /// integers in the range [0, n) with the probability of
        /// producing each value is specified by the parameters
        /// of the distribution.
        template <typename dtype>
        Matrix<dtype> discrete(const Shape &inShape, const Matrix<double> &inWeights)
        {
            return detail::discrete<dtype>(generator_, inShape, inWeights);
        }
    }
}
