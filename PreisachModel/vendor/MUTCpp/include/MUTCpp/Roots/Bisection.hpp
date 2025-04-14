// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <functional>
#include <utility>

#include "Core/Types.hpp"
#include "Roots/Iteration.hpp"

namespace mc
{
    namespace roots
    {
        /// Bisection root finding method
        class Bisection : public Iteration
        {
        public:
            Bisection(const double epsilon, std::function<double(double)> f) noexcept :
                Iteration(epsilon),
                f_(std::move(f))
            {
            }

            /// Constructor
            /// @param epsilon: the epsilon value
            /// @param maxNumIterations: the maximum number of iterations to perform
            /// @param f: the function
            Bisection(const double epsilon, const uint32 maxNumIterations, std::function<double(double)> f) noexcept :
                Iteration(epsilon, maxNumIterations),
                f_(std::move(f))
            {
            }

            virtual ~Bisection() override = default;

            /// Solves for the root in the range [a, b]
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @return root between the bound
            double solve(double a, double b)
            {
                resetNumberOfIterations();
                checkAndFixAlgorithmCriteria(a, b);

                double x = 0.5 * (a + b);
                double fx = f_(x);

                while (std::fabs(fx) >= epsilon_)
                {
                    x = calculateX(x, a, b, fx);
                    fx = f_(x);

                    incrementNumberOfIterations();
                }

                return x;
            }

        private:
            const std::function<double(double)> f_;

            /// Checks the bounds criteria
            /// @param a: the lower bound
            /// @param b: the upper bound
            void checkAndFixAlgorithmCriteria(double &a, double &b) const noexcept
            {
                // The Algorithm works in range [a,b] if criteria f(a)*f(b) < 0 and f(a) > f(b) is fulfilled
                if (f_(a) < f_(b))
                {
                    std::swap(a, b);
                }
            }

            /// Calculates the bisection point
            /// @param x: the evaluation point
            /// @param a: the lower bound
            /// @param b: the upper bound
            /// @param fx: the function evaluated at x
            /// @return x
            static double calculateX(double x, double &a, double &b, double fx) noexcept
            {
                if (fx < 0)
                {
                    b = x;
                }
                else
                {
                    a = x;
                }

                return 0.5 * (a + b);
            }
        };
    }
}
