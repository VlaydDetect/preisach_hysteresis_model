// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <functional>

#include "Core/Error.hpp"
#include "Core/Types.hpp"

namespace mc
{
    namespace roots
    {
        /// ABC for iteration classes to derive from
        class Iteration
        {
        public:
            explicit Iteration(double epsilon) noexcept :
                epsilon_(epsilon)
            {
            }
            
            /// @param epsilon: the epsilon value
            /// @param maxNumIterations: the maximum number of iterations to perform
            Iteration(double epsilon, uint32 maxNumIterations) noexcept :
                epsilon_(epsilon),
                maxNumIterations_(maxNumIterations)
            {
            }
            
            virtual ~Iteration() noexcept = default;

            
            /// Returns the number of iterations
            /// @return: number of iterations
            [[nodiscard]] uint32 numIterations() const noexcept
            {
                return numIterations_;
            }

        protected:
            
            /// Resets the number of iterations
            void resetNumberOfIterations() noexcept
            {
                numIterations_ = 0;
            }

            
            /// Increments the number of iterations
            void incrementNumberOfIterations()
            {
                ++numIterations_;
                if (numIterations_ > maxNumIterations_)
                {
                    THROW_RUNTIME_ERROR(
                        "Maximum number of iterations has been reached; no root has been found within epsilon.");
                }
            }

            const double epsilon_;
            uint32 maxNumIterations_{1000};
            uint32 numIterations_{0};
        };
    }
}
