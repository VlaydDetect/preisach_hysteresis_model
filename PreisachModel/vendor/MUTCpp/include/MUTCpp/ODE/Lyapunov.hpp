// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DynamicSystem.hpp"
#include "Matrix.hpp"
#include "Functions/log.hpp"
#include "Functions/norm.hpp"
#include "Functions/normalize.hpp"
#include "Functions/eye.hpp"
#include "Linalg/QRDecomposition.hpp"
#include "Random/rand.hpp"
#include "Random/randN.hpp"
#include "Utils/doubleToString.hpp"
#include "Utils/nestedLoop.hpp"

namespace mc
{
    namespace ode
    {
        namespace detail
        {
            inline std::array<double, 2> divergenceDegree(Ref<DynamicalSystem> system, double timeForward, double e,
                                                          Eigen::VectorXd v, double T, int M,
                                                          std::vector<Eigen::MatrixXd> &traj1,
                                                          std::vector<Eigen::MatrixXd> &traj2,
                                                          bool linearize = false)
            {
                AL_PROFILE_FUNC("divergenceDegree");

                system->Reset();

                system->Forward(timeForward);
                traj1.push_back(system->GetX());

                v.normalize();
                v *= e;
                v = system->GetX() + v;
                traj2.push_back(v);

                double n = 0.0;

                for (int i = 1; i < M + 1; ++i)
                {
                    auto count = static_cast<uint32>(T / system->GetDeltaTime());
                    while (count--)
                    {
                        v = linearize ? system->NextLTM(v) : system->NextTM(v);
                        traj2.push_back(v);

                        auto traj = system->Forward(1);
                        traj1.push_back(system->GetX());
                    }
                    v -= system->GetX();

                    auto norm_v = v.norm() / e;
                    n += mc::log(norm_v);

                    v.normalize();
                    v *= e;
                    v = system->GetX() + v;
                    traj2[traj2.size() - 1] = v;
                }

                return {n / M, mc::log(T)};
            }
        }

        /**
         * Compute maximal 1-LCE.
         * @param system Dynamical system for which we want to compute the mLCE.
         * @param numForward Number of steps before starting the mLCE computation.
         * @param numCompute The number of steps to compute the mLCE, can be adjusted using keep_evolution.
         * @return Pair of Maximum 1-LCE and Evolution history of mLCE during the computation.
         */
        inline std::pair<double, Eigen::VectorXd> mLCE(Ref<DynamicalSystem> system, int numForward, int numCompute)
        {
            AL_PROFILE_FUNC("mLCE");
            system->Forward(numForward);

            double mLCE = 0.0;
            Eigen::VectorXd w = Eigen::Rand(1, numCompute);
            w.normalize();
            Eigen::VectorXd history = Eigen::VectorXd::Zero(numCompute);

            for (int i = 1; i < numCompute + 1; i++)
            {
                w = system->NextLTM(w);
                system->Forward(1);
                mLCE += Eigen::logonorm(w);
                history[i - 1] = mLCE / (i * system->GetDeltaTime());
                w.normalize();
            }
            mLCE /= numCompute * system->GetDeltaTime();

            return {mLCE, history};
        }

        /**
         * Compute LCEs.
         * @param system Dynamical system for which we want to compute the mLCE.
         * @param p Number of LCE to compute.
         * @param numForward Number of steps before starting the mLCE computation.
         * @param numCompute Number of steps to compute the mLCE, can be adjusted using keep_evolution.
         * @return Pair of Lyapunov Characteristic Exponents array and Evolution history of mLCE during the computation.
         */
        inline std::pair<Eigen::VectorXd, Eigen::MatrixXd> LCE(Ref<DynamicalSystem> system, int p, int numForward,
                                                               int numCompute)
        {
            // Forward the system before the computation of mLCE
            system->Forward(numForward);

            int dim = system->GetDimension();
            if (p > dim)
            {
                printf(
                    "WARNING: The number p (= %d) of calculated Lyapunov exponents cannot exceed the system dimension (= %d)!\n",
                    p, system->GetDimension());
                printf("INFO: The number of calculated exponents is set equal to the system dimension (= %d)!\n",
                       system->GetDimension());
                p = dim;
            }

            Eigen::MatrixXd w = Eigen::MatrixXd::Identity(dim, p);
            Eigen::VectorXd LCE = Eigen::VectorXd::Zero(p);
            Eigen::MatrixXd history = Eigen::MatrixXd::Zero(numCompute, p);

            for (int i = 1; i < numCompute + 1; i++)
            {
                w = system->NextLTM(w);
                system->Forward(1);
                auto [Q, R] = Eigen::GramSchimidtQR(w);

                for (int j = 0; j < p; j++)
                {
                    double val = abs(R(i, j));
                    if (val > 0.0)
                    {
                        LCE[j] += log(val);
                    }
                    history(i - 1, j) = LCE[j] / (i * system->GetDeltaTime());
                }

                LCE /= numCompute * system->GetDeltaTime();
            }

            return {LCE, history};
        }

        /// Implements Benettin's mLCE computation algorithm
        inline std::pair<double, Eigen::VectorXd> Benettin_mLCE(Ref<DynamicalSystem> system, double timeForward,
                                                                double e,
                                                                double T = 8.0,
                                                                int M = 30, bool linearize = true)
        {
            AL_PROFILE_FUNC("Benettin_mLCE");
            system->Forward(timeForward);

            double mLCE = 0.0;
            Eigen::VectorXd history = Eigen::VectorXd::Zero(1, M);

            Eigen::VectorXd v = Eigen::Rand(1, system->GetDimension());

            for (int i = 1; i < M + 1; ++i)
            {
                v = system->GetX() + v.normalized() * e;
                auto count = static_cast<uint32>(T / system->GetDeltaTime());
                while (count--)
                {
                    v = linearize ? system->NextLTM(v) : system->NextTM(v);
                    system->Forward(1);
                }
                v -= system->GetX();

                mLCE += log(v.norm() / e);
                history[i - 1] = mLCE / (i * T);
            }

            mLCE /= M * T;
            return {mLCE, history};
        }

        inline json::JsonDocument TrajsCircle(Ref<DynamicalSystem> system, const std::vector<double> &es, double time,
                                              double timeForward = consts::nan, double areaCoeff = consts::nan)
        {
            std::vector<std::string> esStr(es.size());
            std::ranges::transform(es, esStr.begin(), [](const double &elem) { return doubleToString(elem, 2); });

            json::JsonDocument message({"name", "es", "coords", "time", "timeForward", "trajs"});
            message.AddField("name", "TrajsCircle");
            message.AddField("es", esStr);
            message.AddField("time", time);
            message.AddField("timeForward", timeForward);

            std::vector<Eigen::Vector3d> coords = {
                {1., 0., 0.},
                {sqrt(2.) / 2, sqrt(2.) / 2, 0.},
                {0., 1., 0.},
                {-sqrt(2.) / 2, sqrt(2.) / 2, 0.},
                {-1., 0., 0.},
                {-sqrt(2.) / 2, -sqrt(2.) / 2, 0.},
                {0., -1., 0.},
                {sqrt(2.) / 2, -sqrt(2.) / 2, 0.},
            };
            std::vector<std::string> coordsStr(coords.size());
            std::ranges::transform(coords, coordsStr.begin(), [](const Eigen::VectorXd &elem)
            {
                return Eigen::DoubleVectorToString(elem, 4);
            });
            message.AddField("coords", coordsStr);

            if (!isnan(timeForward))
            {
                system->Forward(timeForward);
            }

            std::unordered_map<std::string,
                               std::unordered_map<std::string,
                                                  std::array<std::vector<std::vector<double>>, 2>>> allTrajs = {};

            std::unordered_map<std::string,
                               std::unordered_map<std::string, std::array<std::vector<double>, 4>>> allLoops = {};

            system->SetResetFn([areaCoeff](DSArgs &args, DSArgs &nextArgs)
            {
                constexpr double L = 1.0;
                auto model1 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);

                auto model2 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
                if (!isnan(areaCoeff))
                {
                    model2->P(L, -2);
                    model2->P(areaCoeff * L, -1);
                }

                args.insert_or_assign("model", Vote(model1));
                nextArgs.insert_or_assign("model", Vote(model2));
            });

            for (const auto &e : es)
            {
                std::unordered_map<std::string,
                                   std::array<std::vector<std::vector<double>>, 2>> trajs = {};

                std::unordered_map<std::string, std::array<std::vector<double>, 4>> loops = {};

                for (const auto v : coords)
                {
                    system->Reset();

                    std::vector<std::vector<double>> mainTraj = {};
                    mainTraj.emplace_back(Eigen::to_std(system->GetX()));

                    Eigen::VectorXd w = system->GetX() + v.normalized() * e;

                    std::vector<std::vector<double>> secondTraj = {};
                    secondTraj.emplace_back(Eigen::to_std(w));

                    auto count = static_cast<uint32>(time / system->GetDeltaTime());
                    while (count--)
                    {
                        auto traj = system->Forward(1);
                        mainTraj.emplace_back(Eigen::to_std(system->GetX()));

                        w = system->NextTM(w);
                        secondTraj.emplace_back(Eigen::to_std(w));
                    }

                    // TODO:
                    // auto [in1, out1] = system->GetArgs().at("model").toPreisachModel()->HysteresisLoop();
                    // auto [in2, out2] = system->GetNextArgs().at("model").toPreisachModel()->HysteresisLoop();
                    auto [in1, out1] = system->GetArgs().at("model").toPreisachModel()->HysteresisLoop();
                    auto [in2, out2] = system->GetNextArgs().at("model").toPreisachModel()->HysteresisLoop();

                    trajs.insert({Eigen::DoubleVectorToString(v, 4), {mainTraj, secondTraj}});
                    loops.insert({Eigen::DoubleVectorToString(v, 4), {in1, out1, in2, out2}});
                }
                allTrajs.insert({doubleToString(e, 2), trajs});
                allLoops.insert({doubleToString(e, 2), loops});
            }

            auto &doc = message.GetDoc();

            doc["trajs"] = allTrajs;
            doc["loops"] = allLoops;

            return message;
        }

        //#region ------------------------------------------ Find Best Params ------------------------------------------

        /**
         * A result of metric function.
         * m_MetricValue -- value of metric.
         * m_Start -- the start of the segment where the metric is calculated (std::nullopt if the metric is considered for the entire solution)
         * m_End -- the end of the segment where the metric is calculated (std::nullopt if the metric is considered for the entire solution)
         */
        struct MetricResult
        {
            MetricResult() = default;

            explicit MetricResult(double value) :
                m_MetricValue(value)
            {
            }

            MetricResult(double value, uint32_t start, uint32_t end) :
                m_MetricValue(value), m_Start(start), m_End(end)
            {
            }

            double m_MetricValue = consts::nan;
            std::optional<uint32_t> m_Start = std::nullopt;
            std::optional<uint32_t> m_End = std::nullopt;
        };

        using SolutionFn = std::function<Eigen::VectorXd(Ref<DynamicalSystem> system,
                                                         const std::unordered_map<DSArgs::key_type, Vote> &args)>;
        using SolutionsMetricFn = std::function<MetricResult(const Eigen::VectorXd &)>;
        using MetricsCompareFn = std::function<bool(double, double)>;

        /**
         * A metric to find a growth segment of time series
         * @param ts: time series
         * @return slope (i.e., growth divided by growth interval length) or consts::nan if no growth segment is found
         */
        static MetricResult GrowthMetric(const Eigen::VectorXd &ts)
        {
            const uint32_t n = ts.size();
            if (n < 2)
                return MetricResult();

            auto rng = std::views::iota(0u, n - 1u);
            auto it_start = std::ranges::find_if(rng, [&](auto i) { return ts[i + 1] > ts[i]; });
            if (it_start == rng.end())
            {
                return MetricResult();
            }

            uint32_t start = *it_start;
            uint32_t end = start + 1;

            while (end < n && ts[end] > ts[end - 1])
            {
                ++end;
            }
            --end;

            if (end == start)
            {
                return MetricResult();
            }

            double growth = ts[end] - ts[start];
            return MetricResult(growth / (end - start), start, end);
        }

        static bool MaxCompare(double new_metric, double best_metric)
        {
            return new_metric > best_metric;
        }

        /**
         * Params for candidate.
         */
        struct CandidateParams
        {
            using CandidateFn = std::function<std::vector<Vote>()>;

            CandidateParams(std::vector<Vote> coarse_range, std::optional<double> fine_step)
            {
                this->coarse_range = coarse_range;
                this->fine_step = fine_step;
            }

            CandidateParams(CandidateFn coarse_range, std::optional<double> fine_step)
            {
                this->coarse_range = coarse_range;
                this->fine_step = fine_step;
            }

            std::variant<std::vector<Vote>, CandidateFn> coarse_range;
            std::optional<double> fine_step;
        };

        struct ParamsSearchResult
        {
            ParamsSearchResult(const std::unordered_map<DSArgs::key_type, Vote> &p, const MetricResult &m,
                               const Eigen::VectorXd &s) :
                params(p), metric(m), solution(s)
            {
            }

            std::unordered_map<DSArgs::key_type, Vote> params;
            MetricResult metric;
            Eigen::VectorXd solution;
        };

        /**
         * Find the best params with specific metric for system solutions.
         * @param system: dynamical system
         * @param solve: function that gets a system solution
         * @param candidates: a candidate map, the keys of which are the names of the candidate params, and the values are the steps for these candidates.
         * @param metric: a solution metric.
         * @param comparator: compares results of applying metric for solutions
         * @returns vector of CoarseSearchResult
         */
        inline std::vector<ParamsSearchResult> FindBestParams(Ref<DynamicalSystem> system, const SolutionFn &solve,
                                                              const std::unordered_map<
                                                                  DSArgs::key_type, CandidateParams> &candidates,
                                                              const SolutionsMetricFn &metric = GrowthMetric,
                                                              const MetricsCompareFn &comparator = MaxCompare)
        {
            auto keys = candidates | std::views::keys;
            std::vector<std::string> params_names{keys.begin(), keys.end()};
            std::ranges::sort(params_names);

            std::unordered_map<DSArgs::key_type, std::vector<Vote>> coarse_grid;
            std::ranges::transform(params_names, std::inserter(coarse_grid, coarse_grid.begin()),
                                   [candidates](const auto &name)
                                   {
                                       std::vector<Vote> value;
                                       if (const auto &range = candidates.at(name).coarse_range; std::holds_alternative<
                                           std::vector<Vote>>(range))
                                       {
                                           value = std::get<std::vector<Vote>>(range);
                                       }
                                       else
                                       {
                                           value = std::get<CandidateParams::CandidateFn>(range)();
                                       }

                                       return std::make_pair(name, value);
                                   });

            std::vector<ParamsSearchResult> coarse_results = {};

            for (const auto &candidate : mc::utils::getNestedCombinations(coarse_grid))
            {
                system->Reset();

                for (const auto &[param_name, param_value] : candidate)
                {
                    system->SetArg(param_name, param_value);
                }

                Eigen::VectorXd solution = solve(system, candidate);
                auto metric_value = metric(solution);
                coarse_results.emplace_back(candidate, metric_value, solution);
            }

            std::vector<ParamsSearchResult> coarse_bests;
            {
                std::vector<ParamsSearchResult> best_elements;
                std::optional<MetricResult> best_metric;

                for (const auto &coarse_result : coarse_results)
                {
                    if (!isnan(coarse_result.metric.m_MetricValue))
                    {
                        if (!best_metric.has_value() || comparator(coarse_result.metric.m_MetricValue,
                                                                   best_metric.value().m_MetricValue))
                        {
                            best_metric = coarse_result.metric;
                            best_elements = {coarse_result};
                        }
                        else if (mc::utils::essentiallyEqual(best_metric.value().m_MetricValue,
                                                             coarse_result.metric.m_MetricValue))
                        {
                            best_elements.push_back(coarse_result);
                        }
                    }
                }

                coarse_bests = best_elements;
            }

            // if (coarse_bests.size() > 1)
            // {
            //     std::println("Coarse search revealed more than one best solution. Fine search will be ignored.");
            //     return coarse_bests;
            // }

            return coarse_bests;
        }

        //#endregion ------------------------------------------ Find Best Params ------------------------------------------

        inline double FindDivergenceDegree(const Ref<DynamicalSystem> &system, double timeForward, double e,
                                           const Eigen::VectorXd &v,
                                           double T, int M, std::vector<Eigen::MatrixXd> &traj1,
                                           std::vector<Eigen::MatrixXd> &traj2, bool linearize = false)
        {
            AL_PROFILE_FUNC("FindDivergenceDegree");

            const auto [lnN, lnT] = detail::divergenceDegree(system, timeForward, e, v, T, M, traj1, traj2,
                                                             linearize);
            return lnN / lnT;
        }

        inline mc::json::JsonDocument DivergenceDegreeRegressionData(const Ref<DynamicalSystem> &system,
                                                                     double timeForward,
                                                                     double e, const Eigen::VectorXd &v,
                                                                     const Eigen::VectorXd &Ts, int M,
                                                                     std::unordered_map<
                                                                         std::string, std::vector<Eigen::MatrixXd>> &
                                                                     trajs1,
                                                                     std::unordered_map<
                                                                         std::string, std::vector<Eigen::MatrixXd>> &
                                                                     trajs2,
                                                                     std::vector<double> &ns)
        {
            AL_PROFILE_FUNC("FindDivergenceDegree");

            for (const auto &T : Ts)
            {
                std::vector<Eigen::MatrixXd> traj1, traj2;
                const auto [n, t] = detail::divergenceDegree(system, timeForward, e, v, T, M, traj1, traj2, false);

                const auto strT = mc::doubleToString(T, 2);

                trajs1.insert({strT, traj1});
                trajs2.insert({strT, traj2});
                std::println("n: {}", n);
                ns.push_back(n);
            }

            Eigen::VectorXd TsLog = Ts.array().log();
            mc::json::JsonDocument message({"name", "method", "dt", "e", "M", "Ts", "ns", "trajs1", "trajs2"});
            message.AddField("name", "RodosLCEs");
            message.AddField("method", "plot");
            message.AddField("dt", system->GetDeltaTime());
            message.AddField("e", e);
            message.AddField("M", M);
            message.AddField("Ts", TsLog);
            message.AddField("ns", ns);
            message.AddField("trajs1", trajs1);
            message.AddField("trajs2", trajs2);

            return message;
        }
    }
}
