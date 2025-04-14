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
            /// Single random value sampled from the "binomial" distribution.
            /// @param generator: instance of a random number generator
            /// @param inN (number of trials)
            /// @param inP (probability of success [0, 1])
            /// @return Matrix
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype binomial(GeneratorType &generator, dtype inN, double inP = 0.5)
            {
                STATIC_ASSERT_INTEGER(dtype);

                if (inN < 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input number of trials must be greater than or equal to zero.");
                }

                if (inP < 0 || inP > 1)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input probability of sucess must be of the range [0, 1].");
                }

                std::binomial_distribution<dtype> dist(inN, inP);
                return dist(generator);
            }

            /// Create an array of the given shape and populate it with
            /// random samples from the "binomial" distribution.
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @param inN (number of trials)
            /// @param inP (probablity of success [0, 1])
            /// @return Matrix
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> binomial(GeneratorType &generator, const Shape &inShape, dtype inN, double inP = 0.5)
            {
                STATIC_ASSERT_INTEGER(dtype);

                if (inN < 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input number of trials must be greater than or equal to zero.");
                }

                if (inP < 0 || inP > 1)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input probability of sucess must be of the range [0, 1].");
                }

                Matrix<dtype> returnArray(inShape);

                std::binomial_distribution<dtype> dist(inN, inP);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the "binomial" distribution.
        /// @param inN (number of trials)
        /// @param inP (probability of success [0, 1])
        /// @return Matrix
        template <typename dtype>
        dtype binomial(dtype inN, double inP = 0.5)
        {
            return detail::binomial(generator_, inN, inP);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from the "binomial" distribution.
        /// @param inShape
        /// @param inN (number of trials)
        /// @param inP (probability of success [0, 1])
        /// @return Matrix
        template <typename dtype>
        Matrix<dtype> binomial(const Shape &inShape, dtype inN, double inP = 0.5)
        {
            return detail::binomial(generator_, inShape, inN, inP);
        }
    }
}
