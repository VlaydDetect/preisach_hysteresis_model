// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <algorithm>

#include "Matrix.hpp"
#include "Random/generator.hpp"

namespace mc
{
    namespace random
    {
        namespace detail
        {
            /// Modify a sequence in-place by shuffling its contents.
            template <typename dtype, typename GeneratorType = std::mt19937>
            void shuffle(GeneratorType &generator, Matrix<dtype> &inArray)
            {
                std::shuffle(inArray.begin(), inArray.end(), generator);
            }
        }

        /// Modify a sequence in-place by shuffling its contents.
        template <typename dtype>
        void shuffle(Matrix<dtype> &inArray)
        {
            return detail::shuffle(generator_, inArray);
        }
    }
}
