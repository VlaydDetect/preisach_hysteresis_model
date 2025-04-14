// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
//
// License
// Copyright (c) 2024 Vlayd Detect.
// All rights reserved.

#pragma once

#include <chrono>
#include <string>
#include <thread>
#include <print>

#include "Core/Types.hpp"

namespace mc
{
    /// A timer class for timing code execution
    template <typename TimeUnit = std::chrono::milliseconds>
    class Timer
    {
    public:
        using ChronoClock = std::chrono::steady_clock;
        using TimePoint = std::chrono::time_point<ChronoClock>;

        Timer() :
            m_Start(ChronoClock::now())
        {
            setUnits();
        }

        explicit Timer(const std::string &inName) :
            m_Name(inName + " "),
            m_Start(ChronoClock::now())
        {
            setUnits();
        }

        /// Sets/changes the timer name
        /// @param inName
        void setName(const std::string& inName)
        {
            m_Name = inName + " ";
        }

        /// Sleeps the current thread
        /// @param length: the length of time to sleep
        void sleep(uint32 length)
        {
            std::this_thread::sleep_for(TimeUnit(length));
        }

        /// Starts the timer
        ///
        void tic() noexcept
        {
            m_Start = ChronoClock::now();
        }

        /// Stops the timer
        ///
        /// @param printElapsedTime: should the elapsed time be output for the console?
        /// @return elapsed time in specified time units
        ///
        TimeUnit toc(bool printElapsedTime = true)
        {
            const auto duration = std::chrono::duration_cast<TimeUnit>(ChronoClock::now() - m_Start);

            if (printElapsedTime)
            {
                std::print("{}: Elapsed Time = {}{}\n", m_Name, duration.count(), m_Unit);
            }

            return duration;
        }

    private:
        void setUnits()
        {
            if constexpr (std::is_same_v<TimeUnit, std::chrono::hours>)
            {
                m_Unit = " hours";
            }
            else if constexpr (std::is_same_v<TimeUnit, std::chrono::minutes>)
            {
                m_Unit = " minutes";
            }
            else if constexpr (std::is_same_v<TimeUnit, std::chrono::seconds>)
            {
                m_Unit = " seconds";
            }
            else if constexpr (std::is_same_v<TimeUnit, std::chrono::milliseconds>)
            {
                m_Unit = " milliseconds";
            }
            else if constexpr (std::is_same_v<TimeUnit, std::chrono::microseconds>)
            {
                m_Unit = " microseconds";
            }
            else if constexpr (std::is_same_v<TimeUnit, std::chrono::nanoseconds>)
            {
                m_Unit = " nanoseconds";
            }
            else
            {
                m_Unit = " time units of some sort";
            }
        }

    private:
        std::string m_Name{""};
        std::string m_Unit{""};
        TimePoint m_Start{};
    };
}
