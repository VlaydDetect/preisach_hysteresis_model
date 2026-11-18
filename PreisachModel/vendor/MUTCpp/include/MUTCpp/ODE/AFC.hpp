// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DynamicSystem.hpp"
#include "Functions/find_extrema.hpp"

#include <Eigen/Dense>
#include <utility>
#include "MUTCpp/Utils/Eigen.hpp"

namespace mc
{
    namespace ode
    {
        struct AFCSweep
        {
            Eigen::VectorXd frequencies;
            Eigen::VectorXd amplitudes;
        };

        struct AFCResult
        {
            double time;
            double dt;
            double excitation_amplitude;
            AFCSweep forward;
            AFCSweep backward;
        };

        /// Builds an amplitude-frequency characteristic for a DynamicSystem
        /// At least the function f(x) must depend on the frequency.
        /// @param system: DynamicSystem.
        /// @param time: the time during which the system will be solved
        /// @param freqs:sequence of frequencies
        /// @param eps: Epsilon value for amplitude condition
        /// @param set_frequency Typed update of the selected frequency in the context.
        /// @param excitation_amplitude Amplitude reported with the sweep.
        template <int N, typename Context, typename SetFrequency>
        AFCResult AFC(DynamicalSystem<N, Context> &system, double time, const Eigen::VectorXd &freqs,
                      SetFrequency set_frequency, double excitation_amplitude, double eps = 0.05)
        {
            AL_PROFILE_FUNC("AFC");
            
            system.ResetArgs();
            system.Reset();

            Eigen::Vector<double, N> x0 = Eigen::Vector<double, N>::Zero();
            const int steps = static_cast<int>(time / system.GetDeltaTime());

            const Eigen::VectorXd& freqs_inc = freqs;
            Eigen::VectorXd freqs_dec = freqs.reverse().eval();

            Eigen::VectorXd amplitudes_inc(freqs_inc.size());
            Eigen::VectorXd amplitudes_dec(freqs_dec.size());

            auto freqLoop = [&](const Eigen::VectorXd &f_sweep, Eigen::VectorXd &amplitudes)
            {
                AL_PROFILE_FUNC("AFC::freqLoop");

                for (Eigen::Index k = 0; k < f_sweep.size(); ++k)
                {
                    const double w = f_sweep[k];

                    system.ResetTo(x0);
                    auto args = system.GetArgs();
                    std::invoke(set_frequency, args, w);
                    system.SetArgs(args);

                    const auto traj = system.Forward(steps);
                    
                    const auto xi = traj.col(0); 

                    const auto extrema = find_extrema_1d(xi.array());
                    const auto& peaks_max = extrema.maxima;
                    const auto& peaks_min = extrema.minima;

                    double amplitude = std::numeric_limits<double>::quiet_NaN();
                    const size_t num_max = peaks_max.size();
                    const size_t num_min = peaks_min.size();
                    if (num_max >= 2 && num_min >= 2)
                    {
                        // Установившийся режим (Steady State) проверяется только по последним пикам.
                        // Проверяем разницу между двумя последними пиками симуляции
                        const double last_max = xi[peaks_max[num_max - 1]];
                        const double prev_max = xi[peaks_max[num_max - 2]];
                        const double last_min = xi[peaks_min[num_min - 1]];
                        const double prev_min = xi[peaks_min[num_min - 2]];
                        
                        // Установившийся режим требует стабилизации как верхней, так и нижней границы аттрактора
                        bool max_stable = std::abs(last_max - prev_max) < eps;
                        bool min_stable = std::abs(last_min - prev_min) < eps;
                        
                        if (max_stable && min_stable)
                        {
                            // Peak-to-Peak амплитуда: устойчива к дрейфу и асимметрии
                            amplitude = (last_max - last_min) / 2.0;
                        }
                    }
                    
                    amplitudes[k] = amplitude;
                    // Использование bottomRows<1>() избегает промежуточных объектов при транспонировании
                    x0 = traj.template bottomRows<1>().transpose();
                }
            };

            freqLoop(freqs_inc, amplitudes_inc);
            freqLoop(freqs_dec, amplitudes_dec);

            const double excitation_amplitude = system.GetArgs().at("A").toDouble();
            
            system.ResetArgs();
            system.Reset();
            
            
            return {time, system.GetDeltaTime(), excitation_amplitude,
                    {freqs, std::move(amplitudes_inc)},
                    {std::move(freqs_dec), std::move(amplitudes_dec)}};
        }
    }
}
