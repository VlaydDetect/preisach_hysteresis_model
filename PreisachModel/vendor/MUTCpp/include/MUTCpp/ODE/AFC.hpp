// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DynamicSystem.hpp"
#include "Json.hpp"
#include "Functions/find_extrema.hpp"

#include <Eigen/Dense>
#include "MUTCpp/Utils/Eigen.hpp"

namespace mc
{
    namespace ode
    {
        /// Builds an amplitude-frequency characteristic for a DynamicSystem
        /// At least the function f(x) must depend on the frequency.
        /// Default symbol for frequency in functions - 'w'.
        /// @param system: DynamicSystem.
        /// @param time: the time during which the system will be solved
        /// @param freqs:sequence of frequencies
        /// @param eps: Epsilon value for amplitude condition
        /// @param freqArgName: name of frequency argument in system arguments map (default is 'w')
        template <int N>
        json::JsonDocument AFC(DynamicalSystem<N> &system, double time, const Eigen::VectorXd &freqs,
                                      double eps = 0.05, const std::string &freqArgName = "w")
        {
            AL_PROFILE_FUNC("AFC");
            
            system.ResetArgs();
            system.Reset();

            Eigen::Vector<double, N> x0 = Eigen::Vector<double, N>::Zero();
            const int steps = static_cast<int>(time / system.GetDeltaTime());

            const Eigen::VectorXd& freqs_inc = freqs;
            const Eigen::VectorXd freqs_dec = freqs.reverse().eval();

            Eigen::VectorXd amplitudes_inc(freqs_inc.size());
            Eigen::VectorXd amplitudes_dec(freqs_dec.size());

            auto freqLoop = [&](const Eigen::VectorXd &f_sweep, Eigen::VectorXd &amplitudes)
            {
                AL_PROFILE_FUNC("AFC::freqLoop");

                for (Eigen::Index k = 0; k < f_sweep.size(); ++k)
                {
                    const double w = f_sweep[k];

                    system.ResetTo(x0);
                    system.SetArg(freqArgName, mc::ode::Vote(w));

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

            json::JsonDocument doc({"name", "time", "dt", "A", "freqs", "forward", "backward"});
            doc.AddField("name", "AFC");
            doc.AddField("time", time);
            doc.AddField("dt", system.GetDeltaTime());
            doc.AddField("A", system.GetArgs().at("A").toDouble());
            doc.AddField("freqs", freqs_inc);
            doc.AddSubField({"forward", "amplitudes"}, amplitudes_inc);
            doc.AddSubField({"forward", "frequencies"}, freqs_inc);
            doc.AddSubField({"backward", "amplitudes"}, amplitudes_dec);
            doc.AddSubField({"backward", "frequencies"}, freqs_dec);
            
            system.ResetArgs();
            system.Reset();
            
            
            // system.Reset();
            //
            // const int dim = system.GetDimension();
            // Eigen::VectorXd x0 = Eigen::VectorXd::Zero(dim);
            // auto steps = static_cast<int>(time / dim);
            //
            // Eigen::VectorXd freqs_inc = freqs;
            // Eigen::VectorXd freqs_dec = freqs_inc.reverse().eval();
            //
            // std::vector<Eigen::VectorXd> trajs_inc = {};
            // trajs_inc.reserve(freqs_inc.size());
            // std::vector<Eigen::VectorXd> trajs_dec;
            // trajs_dec.reserve(freqs_dec.size());
            //
            // Eigen::VectorXd amplitudes_inc(freqs_inc.size());
            // Eigen::VectorXd amplitudes_dec(freqs_dec.size());
            //
            // auto freqLoop = [&](const Eigen::VectorXd &freqs, std::vector<Eigen::VectorXd> &trajs,
            //                     Eigen::VectorXd &amplitudes)
            // {
            //     AL_PROFILE_FUNC("AFC::freqLoop");
            //
            //     for (Eigen::Index k = 0; k < freqs.size(); ++k)
            //     {
            //         const double w = freqs[k];
            //         std::print("{}\n", w);
            //
            //         system.ResetTo(x0);
            //         system.AddAndSetArg(freqArgName, Vote(w));
            //
            //         const auto traj = system.Forward(steps);
            //         trajs.push_back(std::move(traj));
            //
            //         const Eigen::VectorXd xi = trajs.back().col(0);
            //         auto [peaks, props] = find_peaks(xi.array());
            //
            //         double amplitude = consts::nan;
            //         if (peaks.size() > 1)
            //         {
            //             for (const auto i : range(peaks.size() - 1))
            //             {
            //                 if (abs(xi[peaks[i]] - xi[peaks[i - 1]]) < eps)
            //                 {
            //                     amplitude = xi[peaks[peaks.size() - 1]];
            //                 }
            //             }
            //         }
            //
            //         amplitudes[k] = amplitude;
            //         x0 = trajs.back().row(trajs.back().rows() - 1).transpose();
            //     }
            // };
            //
            // freqLoop(freqs_inc, trajs_inc, amplitudes_inc);
            // freqLoop(freqs_dec, trajs_dec, amplitudes_dec);
            //
            // json::JsonDocument doc({"name", "time", "dt", "A", "freqs", "forward", "backward"});
            // doc.AddField("name", "AFC");
            // doc.AddField("time", time);
            // doc.AddField("dt", system.GetDeltaTime());
            // doc.AddField("A", system.GetArgs().at("A").toDouble());
            // doc.AddField("freqs", freqs_inc);
            // doc.AddSubField({"forward", "amplitudes"}, amplitudes_inc);
            // doc.AddSubField({"forward", "frequencies"}, freqs_inc);
            // doc.AddSubField({"forward", "trajectories"}, trajs_inc);
            // doc.AddSubField({"backward", "amplitudes"}, amplitudes_dec);
            // doc.AddSubField({"backward", "frequencies"}, freqs_dec);
            // doc.AddSubField({"backward", "trajectories"}, trajs_dec);

            return doc;
        }
    }
}
