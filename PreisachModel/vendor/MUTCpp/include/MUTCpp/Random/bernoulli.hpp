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
            /// Single random value sampled from the "bernoulli" distribution.
            /// @param generator: instance of a random number generator
            /// @param inP (probability of success [0, 1]). Default 0.5
            /// @return Matrix
            template <typename GeneratorType = std::mt19937>
            bool bernoulli(GeneratorType &generator, double inP = 0.5)
            {
                if (inP < 0 || inP > 1)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input probability of success must be of the range [0, 1].");
                }

                std::bernoulli_distribution dist(inP);
                return dist(generator);
            }

            /// Create an array of the given shape and populate it with
            /// random samples from the "bernoulli" distribution.
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @param inP (probability of success [0, 1]). Default 0.5
            /// @return Matrix
            template <typename GeneratorType = std::mt19937>
            Matrix<bool> bernoulli(GeneratorType &generator, const Shape &inShape, double inP = 0.5)
            {
                if (inP < 0 || inP > 1)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input probability of success must be of the range [0, 1].");
                }

                Matrix<bool> returnArray(inShape);

                std::bernoulli_distribution dist(inP);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](bool &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the "bernoulli" distribution.
        /// @param inP (probability of success [0, 1]). Default 0.5
        /// @return Matrix
        inline bool bernoulli(double inP = 0.5)
        {
            return detail::bernoulli(generator_, inP);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from the "bernoulli" distribution.
        /// @param inShape
        /// @param inP (probability of success [0, 1]). Default 0.5
        /// @return Matrix
        inline Matrix<bool> bernoulli(const Shape &inShape, double inP = 0.5)
        {
            return detail::bernoulli(generator_, inShape, inP);
        }
    }
}
