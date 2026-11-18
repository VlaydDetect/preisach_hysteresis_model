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
            /// @param rows, columns
            /// @param inDof (independent random variables)
            /// @return Matrix
            template<typename dtype, typename GeneratorType = std::mt19937>
            Eigen::MatrixX<dtype> chiSquare(GeneratorType& generator, Eigen::Index rows, Eigen::Index columns, dtype inDof)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inDof <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("numerator degrees of freedom must be greater than zero.");
                }

                Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> returnArray(rows, columns);

                std::chi_squared_distribution<dtype> dist(inDof);

                std::for_each(returnArray.data(),
                              returnArray.data() + returnArray.size(),
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
        /// @param rows, columns
        /// @param inDof (independent random variables)
        /// @return Matrix
        template<typename dtype>
        Eigen::MatrixX<dtype> chiSquare(Eigen::Index rows, Eigen::Index columns, dtype inDof)
        {
            return detail::chiSquare(generator_, rows, columns, inDof);
        }
    }
}
