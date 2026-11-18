// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <random>

#include "Core/Error.hpp"
#include "Core/StaticAssert.hpp"
#include "Random/generator.hpp"

namespace mc
{
    namespace random
    {
        namespace detail
        {
            /// Single random value sampled from the "gamma" distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype gamma(GeneratorType &generator, dtype inGammaShape, dtype inScaleValue = 1)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inGammaShape <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input gamma shape should be greater than zero.");
                }

                if (inScaleValue <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input scale should be greater than zero.");
                }

                std::gamma_distribution<dtype> dist(inGammaShape, inScaleValue);
                return dist(generator);
            }


            /// Create an array of the given shape and populate it with
            /// random samples from a "gamma" distribution.
            template <typename dtype, typename GeneratorType = std::mt19937>
            Eigen::MatrixX<dtype> gamma(GeneratorType &generator, Eigen::Index rows, Eigen::Index columns, dtype inGammaShape,
                                dtype inScaleValue = 1)
            {
                STATIC_ASSERT_ARITHMETIC(dtype);

                if (inGammaShape <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input gamma shape should be greater than zero.");
                }

                if (inScaleValue <= 0)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input scale should be greater than zero.");
                }

                Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> returnArray(rows, columns);

                std::gamma_distribution<dtype> dist(inGammaShape, inScaleValue);

                std::for_each(returnArray.data(),
                              returnArray.data() + returnArray.size(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the "gamma" distribution.
        template <typename dtype>
        dtype gamma(dtype inGammaShape, dtype inScaleValue = 1)
        {
            return detail::gamma(generator_, inGammaShape, inScaleValue);
        }

        /// Create an array of the given shape and populate it with
        /// random samples from a "gamma" distribution.
        template <typename dtype>
        Eigen::MatrixX<dtype> gamma(Eigen::Index rows, Eigen::Index columns, dtype inGammaShape, dtype inScaleValue = 1)
        {
            return detail::gamma(generator_, rows, columns, inGammaShape, inScaleValue);
        }
    }
}
