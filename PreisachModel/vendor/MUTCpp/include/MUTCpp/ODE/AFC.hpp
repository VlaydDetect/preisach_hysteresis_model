// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DynamicSystem.hpp"
#include "Json.hpp"
#include "Functions/find_peaks.hpp"
#include "Functions/range.hpp"

#include <Eigen/Dense>
#include "Eigen/utils.hpp"

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
        inline json::JsonDocument AFC(Ref<DynamicalSystem> system, double time, double freq_max, double freq_delta,
                                      double eps = 0.1, std::string freqArgName = "w")
        {
            AL_PROFILE_FUNC("AFC");

            const int dim = system->GetDimension();
            Eigen::VectorXd x0 = Eigen::VectorXd::Zero(dim);
            auto steps = static_cast<int>(time / dim);

            Eigen::VectorXd freqs_inc = Eigen::arange(0.0, freq_max + freq_delta, freq_delta);
            Eigen::VectorXd freqs_dec = freqs_inc.reverse().eval();

            std::vector<Eigen::VectorXd> trajs_inc = {};
            trajs_inc.reserve(freqs_inc.size());
            std::vector<Eigen::VectorXd> trajs_dec;
            trajs_dec.reserve(freqs_dec.size());

            Eigen::VectorXd amplitudes_inc(freqs_inc.size());
            Eigen::VectorXd amplitudes_dec(freqs_dec.size());

            auto freqLoop = [&](const Eigen::VectorXd &freqs, std::vector<Eigen::VectorXd> &trajs,
                                Eigen::VectorXd &amplitudes)
            {
                AL_PROFILE_FUNC("AFC::freqLoop");

                for (Eigen::Index k = 0; k < freqs.size(); ++k)
                {
                    const double w = freqs[k];
                    std::print("{}\n", w);

                    system->ResetTo(x0);
                    system->AddAndSetArg(freqArgName, Vote(w));

                    Eigen::MatrixXd traj = system->Forward(steps);
                    trajs.push_back(std::move(traj));

                    const Eigen::VectorXd xi = trajs.back().col(0);
                    auto [peaks, props] = find_peaks(xi.array());

                    double amplitude = consts::nan;
                    if (peaks.size() > 1)
                    {
                        for (const auto i : range(peaks.size() - 1))
                        {
                            if (abs(xi[peaks[i]] - xi[peaks[i - 1]]) < eps)
                            {
                                amplitude = xi[peaks[peaks.size() - 1]];
                            }
                        }
                    }

                    amplitudes[k] = amplitude;
                    x0 = trajs.back().row(trajs.back().rows() - 1).transpose();
                }
            };

            freqLoop(freqs_inc, trajs_inc, amplitudes_inc);
            freqLoop(freqs_dec, trajs_dec, amplitudes_dec);

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
