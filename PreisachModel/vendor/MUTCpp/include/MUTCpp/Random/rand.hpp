// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>
#include <random>

#include "Core/StaticAssert.hpp"
#include "Core/Shape.hpp"
#include "Matrix.hpp"
#include "generator.hpp"

namespace mc
{
    namespace random
    {
        namespace detail
        {
            /// Single random value sampled from the uniform distribution over [0, 1).
            /// @param generator: instance of a random number generator
            /// @return Matrix
            template <typename dtype, typename GeneratorType = std::mt19937>
            dtype rand(GeneratorType &generator)
            {
                STATIC_ASSERT_FLOAT(dtype);

                std::uniform_real_distribution<dtype> dist(static_cast<dtype>(0.),
                                                           static_cast<dtype>(1.) - DtypeInfo<dtype>::epsilon());
                return dist(generator);
            }

            //============================================================================
            // Method Description:
            /// Create an array of the given shape and populate it with
            /// random samples from a uniform distribution over [0, 1).
            /// @param generator: instance of a random number generator
            /// @param inShape
            /// @return Matrix
            template <typename dtype, typename GeneratorType = std::mt19937>
            Matrix<dtype> rand(GeneratorType &generator, const Shape &inShape)
            {
                STATIC_ASSERT_FLOAT(dtype);

                Matrix<dtype> returnArray(inShape);

                std::uniform_real_distribution<dtype> dist(static_cast<dtype>(0.),
                                                           static_cast<dtype>(1.) - DtypeInfo<dtype>::epsilon());

                std::for_each(returnArray.begin(),
                              returnArray.end(),
                              [&generator, &dist](dtype &value) -> void { value = dist(generator); });

                return returnArray;
            }
        }

        /// Single random value sampled from the uniform distribution over [0, 1).
        template <typename dtype>
        dtype rand()
        {
            return detail::rand<dtype>(generator_);
        }
        
        /// Create an array of the given shape and populate it with
        /// random samples from a uniform distribution over [0, 1).
        template <typename dtype>
        Matrix<dtype> rand(const Shape &inShape)
        {
            return detail::rand<dtype>(generator_, inShape);
        }
    }
}
