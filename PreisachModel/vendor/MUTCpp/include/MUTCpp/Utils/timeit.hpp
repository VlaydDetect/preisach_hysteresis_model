// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <chrono>
#include <iostream>
#include <type_traits>

#include "Core/Enums.hpp"
#include "Core/Timer.hpp"
#include "Core/Types.hpp"

namespace mc
{
    namespace utils
    {
        namespace timeit_detail
        {
            /// @brief  Result statistics of a timeit run
            template <typename TimeUnit>
            struct Result
            {
                TimeUnit min{};
                TimeUnit max{};
                TimeUnit mean{};
            };

            template <typename TimeUnit>
            std::ostream &operator<<(std::ostream &os, const Result<TimeUnit> result)
            {
                std::string unit{};
                if constexpr (std::is_same_v<TimeUnit, std::chrono::hours>)
                {
                    unit = " hours";
                }
                else if constexpr (std::is_same_v<TimeUnit, std::chrono::minutes>)
                {
                    unit = " minutes";
                }
                else if constexpr (std::is_same_v<TimeUnit, std::chrono::seconds>)
                {
                    unit = " seconds";
                }
                else if constexpr (std::is_same_v<TimeUnit, std::chrono::milliseconds>)
                {
                    unit = " milliseconds";
                }
                else if constexpr (std::is_same_v<TimeUnit, std::chrono::microseconds>)
                {
                    unit = " microseconds";
                }
                else if constexpr (std::is_same_v<TimeUnit, std::chrono::nanoseconds>)
                {
                    unit = " nanoseconds";
                }
                else
                {
                    unit = " time units of some sort";
                }

                os << "Timeit results:\n";
                os << "\tmin: " << result.min.count() << unit << "\n";
                os << "\tmax: " << result.max.count() << unit << "\n";
                os << "\tmean: " << result.mean.count() << unit << "\n";

                return os;
            }
        }

        /// Timing of a function
        ///
        /// @param numIterations: number of iterations for the timing statistics
        /// @param printResults: true to print the results
        /// @param function: the function to time
        /// @param args: the arguements that are forwarded to the function input
        ///
        /// @return timing statistics
        template <typename TimeUnit, typename Function, typename... Args>
        timeit_detail::Result<TimeUnit>
        timeit(uint32 numIterations, PrintResults printResults, Function function, Args &&... args) noexcept
        {
            auto result = timeit_detail::Result<TimeUnit>{};
            auto timer = Timer<TimeUnit>{};

            for (uint32 i = 0; i < numIterations; ++i)
            {
                if (i == 0)
                {
                    result.min = TimeUnit::max();
                }

                timer.tic();

                using ResultType = std::invoke_result_t<Function, Args...>;
                if constexpr (std::is_same_v<ResultType, void>)
                {
                    function(std::forward<Args>(args)...);
                }
                else
                {
                    // cppcheck-suppress redundantAssignment
                    [[maybe_unused]] const ResultType functionResult = function(std::forward<Args &>(args)...);
                }

                const auto elapsedTime = timer.toc(false);

                result.mean = result.mean + elapsedTime;
                result.min = std::min(result.min, elapsedTime);
                result.max = std::max(result.max, elapsedTime);
            }

            result.mean = result.mean / numIterations;

            if (printResults == PrintResults::YES)
            {
                std::cout << result;
            }

            return result;
        }
    }
}
