// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Matrix.hpp"
#include "Functions/byDelta.hpp"
#include "Functions/reverse.hpp"
#include "Functions/zeros.hpp"
#include "DynamicSystem.hpp"
#include "Json.hpp"
#include "Functions/abs.hpp"
#include "Functions/find_peaks.hpp"
#include "Functions/range.hpp"

namespace mc
{
    namespace ode
    {
        /// Builds an amplitude-frequency characteristic for a DynamicSystem
        /// At least the function f(x) must depend on the frequency.
        /// Default symbol for frequency in functions - 'w'.
        /// @param system: DynamicSystem.
        /// @param time: the time during which the system will be solved
        /// @param freq_max: maximum frequency value (included to sequence of frequencies)
        /// @param freq_delta: delta step for frequency
        /// @param eps: Epsilon value for amplitude condition
        /// @param freqArgName: name of frequency argument in system arguments map (default is 'w')
        inline json::JsonDocument AFC(DynamicalSystem *system, double time, double freq_max, double freq_delta,
                                      double eps = 0.1, std::string freqArgName = "w")
        {
            AL_PROFILE_FUNC("AFC");
            Matrix<double> x0 = zeros<double>(1, system->GetDimension());

            auto steps = static_cast<int>(time / system->GetDeltaTime());

            Matrix<double> freqs_inc = byDelta(0.0, freq_max + freq_delta, freq_delta);
            Matrix<double> freqs_dec = reverse(freqs_inc);

            std::vector<Matrix<double>> trajs_inc = {};
            std::vector<Matrix<double>> trajs_dec = {};

            Matrix<double> amplitudes_inc = {};
            Matrix<double> amplitudes_dec = {};

            auto freqLoop = [steps, freqArgName, eps](DynamicalSystem *system, const Matrix<double> &freqs, std::vector<Matrix<double>> &trajs,
                                                              Matrix<double> &amplitudes, Matrix<double> &x0)
            {
                AL_PROFILE_FUNC("AFC::freqLoop");
                double amplitude = consts::nan;

                for (const auto w : freqs)
                {
                    std::print("{}\n", w);
                    system->Reset(x0);
                    system->AddAndSetArg(freqArgName, Vote(w));
                    
                    auto traj = system->Forward(steps);
                    trajs.push_back(traj);
                    
                    auto xi = traj(traj.rSlice(), 0);
                    auto [peaks, props] = find_peaks(xi);
                    for (const auto i : range(peaks.size() - 1))
                    {
                        if (abs(xi[peaks[i]] - xi[peaks[i - 1]]) < eps)
                        {
                            amplitude = xi[peaks[-1]];
                        }
                    }
                    amplitudes = append(amplitudes, {amplitude});
                    // x0 = traj(-1, traj.cSlice());
                    x0 = {xi[-1], traj(-1, 1), x0[-1]};
                }
            };

            freqLoop(system, freqs_inc, trajs_inc, amplitudes_inc, x0);
            freqLoop(system, freqs_dec, trajs_dec, amplitudes_dec, x0);

            json::JsonDocument doc({"name", "time", "dt", "A", "freqs", "forward", "backward"});
            doc.AddField("name", "AFC");
            doc.AddField("time", time);
            doc.AddField("dt", system->GetDeltaTime());
            doc.AddField("A", system->GetArgs().at("A").toDouble());
            doc.AddField("freqs", freqs_inc);
            doc.AddSubField({"forward", "amplitudes"}, amplitudes_inc);
            doc.AddSubField({"forward", "frequencies"}, freqs_inc);
            doc.AddSubField({"forward", "trajectories"}, trajs_inc);
            doc.AddSubField({"backward", "amplitudes"}, amplitudes_dec);
            doc.AddSubField({"backward", "frequencies"}, freqs_dec);
            doc.AddSubField({"backward", "trajectories"}, trajs_dec);

            return doc;
        }
    }
}
