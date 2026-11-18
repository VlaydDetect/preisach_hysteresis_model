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
            /// @param rows, columns
            /// @param inP (probability of success [0, 1]). Default 0.5
            /// @return Matrix
            template <typename GeneratorType = std::mt19937>
            Eigen::MatrixX<bool> bernoulli(GeneratorType &generator, Eigen::Index rows, Eigen::Index columns, double inP = 0.5)
            {
                if (inP < 0 || inP > 1)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input probability of success must be of the range [0, 1].");
                }

                Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> returnArray(rows, columns);

                std::bernoulli_distribution dist(inP);

                std::for_each(returnArray.data(),
                              returnArray.data() + returnArray.size(),
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
        /// @param rows, columns
        /// @param inP (probability of success [0, 1]). Default 0.5
        /// @return Matrix
        inline Eigen::MatrixX<bool> bernoulli(Eigen::Index rows, Eigen::Index columns, double inP = 0.5)
        {
            return detail::bernoulli(generator_, rows, columns, inP);
        }
    }
}
