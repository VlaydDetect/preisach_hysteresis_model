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
            /// Single random value sampled from the "normal" distribution.
            /// @param generator: instance of a random number generator
            /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
            /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
            ///     Default is 1.
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype normal(GeneratorType &generator, dtype inMean = 0, dtype inSigma = 1)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inSigma <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input sigma must be greater than zero.");
                }

                std::normal_distribution<dtype> dist(inMean, inSigma);
                return dist(generator);
            }

            /// Create an array of the given shape and populate it with
            /// random samples from a "normal" distribution.
            /// @param generator: instance of a random number generator
            /// @param rows, columns
            /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
            /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
            ///     Default is 1.
            template <typename dtype, typename GeneratorType = std::mt19937>
            Eigen::MatrixX<dtype> normal(GeneratorType &generator, Eigen::Index rows, Eigen::Index columns, dtype inMean = 0, dtype inSigma = 1)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inSigma <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input sigma must be greater than zero.");
                }

                Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> returnArray(rows, columns);

                std::normal_distribution<dtype> dist(inMean, inSigma);

                std::for_each(returnArray.data(),
                              returnArray.data() + returnArray.size(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the "normal" distribution.
        /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
        /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
        ///     Default is 1.
        template <typename dtype>
        dtype normal(dtype inMean = 0, dtype inSigma = 1)
        {
            return detail::normal(generator_, inMean, inSigma);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from a "normal" distribution.
        /// @param rows, columns
        /// @param inMean: Mean value of the underlying normal distribution. Default is 0.
        /// @param inSigma: Standard deviation of the underlying normal distribution. Should be greater than zero.
        ///     Default is 1.
        template <typename dtype>
        Eigen::MatrixX<dtype> normal(Eigen::Index rows, Eigen::Index columns, dtype inMean = 0, dtype inSigma = 1)
        {
            return detail::normal(generator_, rows, columns, inMean, inSigma);
        }
    }
}
