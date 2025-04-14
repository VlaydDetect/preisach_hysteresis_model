// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <random>

namespace mc
{
    namespace random
    {
        /// generator function
        inline std::mt19937_64 generator_;

        /// Seeds the random number generator
        inline void seed(int inSeed)
        {
            generator_.seed(inSeed);
        }
    }
}
