// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <random>

#include "Utils/DenseOperations.hpp"
#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
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
            dtype discrete(GeneratorType &generator, const Eigen::MatrixX<double> &inWeights)
            {
                STATIC_ASSERT_INTEGER(dtype);

                const auto weights = mc::detail::rowMajor(inWeights);
                std::discrete_distribution<dtype> dist(weights.begin(), weights.end());
                return dist(generator);
            }

            /// Create an array of the given shape and populate it with
            /// random samples from a "discrete" distribution.  It produces
            /// integers in the range [0, n) with the probability of
            /// producing each value is specified by the parameters
            /// of the distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            Eigen::MatrixX<dtype> discrete(GeneratorType &generator, Eigen::Index rows, Eigen::Index columns, const Eigen::MatrixX<double> &inWeights)
            {
                STATIC_ASSERT_INTEGER(dtype);

                Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> returnArray(rows, columns);

                const auto weights = mc::detail::rowMajor(inWeights);
                std::discrete_distribution<dtype> dist(weights.begin(), weights.end());

                std::for_each(returnArray.data(),
                              returnArray.data() + returnArray.size(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the
        /// "discrete" distribution.  It produces integers in the
        /// range [0, n) with the probability of producing each value
        /// is specified by the parameters of the distribution.
        template <typename dtype>
        dtype discrete(const Eigen::MatrixX<double> &inWeights)
        {
            return detail::discrete<dtype>(generator_, inWeights);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from a "discrete" distribution.  It produces
        /// integers in the range [0, n) with the probability of
        /// producing each value is specified by the parameters
        /// of the distribution.
        template <typename dtype>
        Eigen::MatrixX<dtype> discrete(Eigen::Index rows, Eigen::Index columns, const Eigen::MatrixX<double> &inWeights)
        {
            return detail::discrete<dtype>(generator_, rows, columns, inWeights);
        }
    }
}
