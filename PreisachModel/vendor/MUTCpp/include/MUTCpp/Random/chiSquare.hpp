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
            /// Single random value sampled from the "chi square" distribution.
            /// @param generator: instance of a random number generator
            /// @param inDof (independent random variables)
            /// @return value
            template<typename dtype, typename GeneratorType = std::mt19937>
            dtype chiSquare(GeneratorType& generator, dtype inDof)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inDof <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("numerator degrees of freedom must be greater than zero.");
                }

                std::chi_squared_distribution<dtype> dist(inDof);
                return dist(generator);
            }

            /// Create an array of the given shape and populate it with
            /// random samples from the "chi square" distribution.
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @param inDof (independent random variables)
            /// @return Matrix
            template<typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> chiSquare(GeneratorType& generator, const Shape& inShape, dtype inDof)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inDof <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("numerator degrees of freedom must be greater than zero.");
                }

                Matrix<dtype> returnArray(inShape);

                std::chi_squared_distribution<dtype> dist(inDof);

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype& value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the "chi square" distribution.
        /// @param inDof (independent random variables)
        /// @return value
        template<typename dtype>
        dtype chiSquare(dtype inDof)
        {
            return detail::chiSquare(generator_, inDof);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from the "chi square" distribution.
        /// @param inShape
        /// @param inDof (independent random variables)
        /// @return Matrix
        template<typename dtype>
        Matrix<dtype> chiSquare(const Shape& inShape, dtype inDof)
        {
            return detail::chiSquare(generator_, inShape, inDof);
        }
    }
}
