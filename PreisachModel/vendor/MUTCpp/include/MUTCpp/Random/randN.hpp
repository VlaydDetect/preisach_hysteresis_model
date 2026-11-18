// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <random>

#include "Core/StaticAssert.hpp"
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
            Eigen::MatrixX<dtype> randN(GeneratorType &generator, Eigen::Index rows, Eigen::Index columns)
            {
                STATIC_ASSERT_FLOAT(dtype);

                Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> returnArray(rows, columns);

                std::normal_distribution<dtype> dist;

                std::for_each(returnArray.data(),
                              returnArray.data() + returnArray.size(),
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
        Eigen::MatrixX<dtype> randN(Eigen::Index rows, Eigen::Index columns)
        {
            return detail::randN<dtype>(generator_, rows, columns);
        }
    }
}
