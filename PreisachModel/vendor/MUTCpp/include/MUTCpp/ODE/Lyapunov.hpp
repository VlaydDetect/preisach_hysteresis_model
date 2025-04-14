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
            inline std::array<double, 2> divergenceDegree(Ref<DynamicalSystem> system, double timeForward, double e, Matrix<double> v,
                                                          double T, int M, std::vector<Matrix<double>> &traj1,
                                                          std::vector<Matrix<double>> &traj2,
                                                          bool linearize = false)
            {
                AL_PROFILE_FUNC("divergenceDegree");

                system->Reset();

                system->Forward(timeForward);
                traj1.push_back(system->GetX());
                
                double n = 0.0;

                std::vector<double> logs = {};

                for (int i = 1; i < M + 1; ++i)
                {
                    v = normalize(v) * e;
                    v = system->GetX() + v;
                    traj2.push_back(v);

                    auto count = static_cast<uint32>(T / system->GetDeltaTime());
                    while (count--)
                    {
                        v = linearize ? system->NextLTM(v) : system->NextTM(v);
                        traj2.push_back(v);

                        auto traj = system->Forward(1);
                        traj1.push_back(system->GetX());
                    }
                    v -= system->GetX();

                    auto norm_v = norm(v) / e;
                    n += mc::log(norm_v)[0];
                }

                return {n / M, mc::log(T)};
            }
        }

        /**
         * Compute maximal 1-LCE.
         * @param system Dynamical system for which we want to compute the mLCE.
         * @param numForward Number of steps before starting the mLCE computation.
         * @param numCompute Number of steps to compute the mLCE, can be adjusted using keep_evolution.
         * @return Pair of Maximum 1-LCE and Evolution history of mLCE during the computation.
         */
        inline std::pair<double, Matrix<double>> mLCE(DynamicalSystem *system, int numForward, int numCompute)
        {
            AL_PROFILE_FUNC("mLCE");
            system->Forward(numForward);

            double mLCE = 0.0;
            auto w = mc::random::rand<double>(static_cast<Shape>(system->GetDimension()));
            w = normalize(w);
            Matrix<double> history = mc::zeros<double>(1, numCompute);

            for (int i = 1; i < numCompute + 1; i++)
            {
                w = system->NextLTM(w);
                system->Forward(1);
                mLCE += logonorm(w)[0];
                history[i - 1] = mLCE / (i * system->GetDeltaTime());
                w = normalize(w);
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
        inline std::pair<Matrix<double>, Matrix<double>> LCE(DynamicalSystem *system, int p, int numForward,
                                                             int numCompute)
        {
            // Forward the system before the computation of mLCE
            system->Forward(numForward);

            auto eye = mc::eye<double>(system->GetDimension());
            if (p > system->GetDimension())
            {
                printf(
                    "WARNING: The number p (= %d) of calculated Lyapunov exponents cannot exceed the system dimension (= %d)!\n",
                    p, system->GetDimension());
                printf("INFO: The number of calculated exponents is set equal to the system dimension (= %d)!\n",
                       system->GetDimension());
            }
            auto w = eye(eye.rSlice(), Slice(std::min(system->GetDimension(), p)));
            auto LCE = zeros<double>(p);

            Matrix<double> history = zeros<double>(numCompute, p);
            for (int i = 1; i < numCompute + 1; i++)
            {
                w = system->NextLTM(w);
                system->Forward(1);
                auto [W, R] = linalg::qr_decomposition(w);
                for (int j = 0; j < p; j++)
                {
                    LCE[j] += log(abs(R(i, j)));
                    history(i - 1, j) = LCE[j] / (i * system->GetDeltaTime());
                }
                LCE /= numCompute * system->GetDeltaTime();
            }

            return {LCE, history};
        }

        /// Implements Benettin's mLCE computation algorithm
        inline std::pair<double, Matrix<double>> Benettin_mLCE(DynamicalSystem *system, double timeForward, double e,
                                                               double T = 8.0,
                                                               int M = 30, bool linearize = true)
        {
            AL_PROFILE_FUNC("Benettin_mLCE");
            system->Forward(timeForward);

            double mLCE = 0.0;
            Matrix<double> history = zeros<double>(1, M);

            auto v = mc::random::rand<double>(Shape(1, system->GetDimension()));

            for (int i = 1; i < M + 1; ++i)
            {
                v = normalize(v) * e;
                v = system->m_X + v;

                auto count = static_cast<uint32>(T / system->GetDeltaTime());
                while (count--)
                {
                    v = linearize ? system->NextLTM(v) : system->NextTM(v);
                    system->Forward(1);
                }
                v -= system->m_X;

                mLCE += log(norm(v) / e)[0];
                history[i - 1] = mLCE / (i * T);
            }

            mLCE /= M * T;
            return {mLCE, history};
        }

        inline std::vector<std::pair<double, int>> mLCE_DivergenceDegreeTest(
            DynamicalSystem *system, double timeForward, double e,
            double T = 8.0, int M = 30,
            uint8_t n = 7, bool linearize = false,
            bool useMult = false)
        {
            AL_PROFILE_FUNC("mLCE_DivergenceDegreeTest");

            system->Forward(timeForward);

            const auto x0 = system->m_X;
            auto v = mc::random::rand<double>(Shape(system->GetDimension(), 1));
            v = normalize(v) * e;
            v = x0 + v;

            double mLCE = useMult ? 1. : 0.;
            Matrix<double> history = zeros<double>(1, M);

            for (int i = 1; i < M + 1; ++i)
            {
                auto count = static_cast<uint32>(T / system->GetDeltaTime());
                while (count--)
                {
                    v = linearize ? system->NextLTM(v) : system->NextTM(v);
                    system->Forward(1);
                }

                if (useMult)
                {
                    mLCE *= mc::norm(v)[0] / e;
                }
                else
                {
                    mLCE += mc::norm(v)[0] / e;
                }
                history[i - 1] = mLCE / (i * T);

                v = normalize(v);
                v *= e;
            }

            std::vector<std::pair<double, int>> res = {};
            for (uint8_t i = 1; i < n + 1; i++)
            {
                auto a = (useMult ? std::pow(mLCE, 1 / M) : mLCE) / (M * mc::power(T, i));
                res.emplace_back(a, i);
            }

            return res;
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

            std::vector<mc::Matrix<double>> coords = {
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
            std::ranges::transform(coords, coordsStr.begin(), [](const Matrix<double> &elem)
            {
                return doubleToString(elem, 4);
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
                    mainTraj.push_back(system->GetX().toFlattenVector());

                    auto w = system->GetX() + normalize(v) * e;

                    std::vector<std::vector<double>> secondTraj = {};
                    secondTraj.push_back(w.toFlattenVector());

                    auto count = static_cast<uint32>(time / system->GetDeltaTime());
                    while (count--)
                    {
                        auto traj = system->Forward(1);
                        mainTraj.push_back(system->GetX().toFlattenVector());

                        w = system->NextTM(w);
                        secondTraj.push_back(w.toFlattenVector());
                    }

                    auto [in1, out1] = system->GetArgs().at("model").toPreisachModel()->HysteresisLoop();
                    auto [in2, out2] = system->GetNextArgs().at("model").toPreisachModel()->HysteresisLoop();

                    trajs.insert({doubleToString(v, 4), {mainTraj, secondTraj}});
                    loops.insert({doubleToString(v, 4), {in1, out1, in2, out2}});
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
        using SolutionFn = std::function<Matrix<double>(Ref<DynamicalSystem> system, const std::unordered_map<DSArgs::key_type, Vote>& args)>;
        using SolutionsMetricFn = std::function<double(const Matrix<double> &)>;
        using MetricsCompareFn = std::function<bool(double, double)>;

        /**
         * A metric to find a growth segment of time series
         * @param ts: time series
         * @return slope (i.e., growth divided by growth interval length) or consts::nan if no growth segment is found
         */
        static double GrowthMetric(const Matrix<double> &ts)
        {
            if (!ts.isflat())
            {
                THROW_INVALID_ARGUMENT_ERROR("Matrix must be 1-dim");
            }

            uint32_t n = ts.size();
            if (n < 2)
                return consts::nan;

            int32_t start = -1;
            for (uint32_t i = 0; i < n - 1; i++)
            {
                if (ts[i + 1] > ts[i])
                {
                    start = static_cast<int32_t>(i);
                    break;
                }
            }
            if (start == -1)
                return consts::nan;

            uint32_t end = static_cast<uint32_t>(start) + 1;
            while (end < n && ts[end] > ts[end - 1])
            {
                end++;
            }
            end--;

            if (end == start)
                return consts::nan;

            double growth = ts[end] - ts[start];
            uint32_t length = end - start;
            return growth / length;
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
            
            CandidateParams(std::vector<Vote> coarse_range, std::optional<double> fine_step/*, std::optional<double> threshold*/)
            {
                this->coarse_range = coarse_range;
                this->fine_step = fine_step;
                // this->threshold_ratio = threshold;
            }

            CandidateParams(CandidateFn coarse_range, std::optional<double> fine_step/*, std::optional<double> threshold*/)
            {
                this->coarse_range = coarse_range;
                this->fine_step = fine_step;
                // this->threshold_ratio = threshold;
            }
            
            std::variant<std::vector<Vote>, CandidateFn> coarse_range;
            std::optional<double> fine_step;
            // std::optional<double> threshold_ratio;
        };

        struct ParamsSearchResult
        {
            ParamsSearchResult(const std::unordered_map<DSArgs::key_type, Vote> &params, double metric_value,
                               const Matrix<double> &solution)
            {
                this->params = params;
                this->metric_value = metric_value;
                this->solution = solution;
            }

            std::unordered_map<DSArgs::key_type, Vote> params;
            double metric_value;
            Matrix<double> solution;
        };

        /**
         * Find best params with specific metric for system solutions.
         * @param system: dynamical system
         * @param solve: function that gets system solution
         * @param candidates: a candidate map, the keys of which are the names of the candidate params, and the values are the steps for these candidates.
         * @param metric: a solution metric.
         * @param comparator: compares results of applying metric for solutions
         * @returns vector of CoarseSearchResult
         */
        inline std::vector<ParamsSearchResult> FindBestParams(Ref<DynamicalSystem> system, const SolutionFn &solve,
                                   const std::unordered_map<DSArgs::key_type, CandidateParams> &candidates,
                                   const SolutionsMetricFn &metric = GrowthMetric,
                                   const MetricsCompareFn &comparator = MaxCompare)
        {
            auto keys = candidates | std::views::keys;
            std::vector<std::string> params_names{ keys.begin(), keys.end() };
            std::ranges::sort(params_names);

            std::unordered_map<DSArgs::key_type, std::vector<Vote>> coarse_grid;
            std::ranges::transform(params_names, std::inserter(coarse_grid, coarse_grid.begin()),
                                   [candidates](const auto &name)
                                   {
                                       std::vector<Vote> value;
                                       if (auto range = candidates.at(name).coarse_range; std::holds_alternative<
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

            for (const auto &candidate : utils::getNestedCombinations(coarse_grid))
            {
                system->Reset();
                
                for (const auto &[param_name, param_value] : candidate)
                {
                    system->SetArg(param_name, param_value);
                }

                Matrix<double> solution = solve(system, candidate);
                auto metric_value = metric(solution);
                // if (isnan(metric_value))
                //     metric_value = -consts::inf;
                coarse_results.emplace_back(candidate, metric_value, solution);
            }

            std::vector<ParamsSearchResult> coarse_bests;
            {
                std::vector<ParamsSearchResult> best_elements;
                std::optional<double> best_metric;

                for (const auto &coarse_result : coarse_results)
                {
                    if (!isnan(coarse_result.metric_value))
                    {
                        if (!best_metric.has_value() || comparator(coarse_result.metric_value, best_metric.value()))
                        {
                            best_metric = coarse_result.metric_value;
                            best_elements = {coarse_result};
                        }
                        else if (utils::essentiallyEqual(best_metric.value(), coarse_result.metric_value))
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

        inline double FindDivergenceDegree(DynamicalSystem *system, double timeForward, double e, Matrix<double> v,
                                           double T, int M, std::vector<Matrix<double>> &traj1,
                                           std::vector<Matrix<double>> &traj2, bool linearize = false)
        {
            AL_PROFILE_FUNC("FindDivergenceDegree");

            const auto [lnN, lnT] = detail::divergenceDegree(system, timeForward, e, v, T, M, traj1, traj2,
                                                             linearize);
            return lnN / lnT;
        }

        inline std::array<double, 2> DivergenceDegreeRegressionData(const Ref<DynamicalSystem> &system, double timeForward,
                                                                    double e, Matrix<double> v,
                                                                    double T, int M, std::vector<Matrix<double>> &traj1,
                                                                    std::vector<Matrix<double>> &traj2,
                                                                    bool linearize = false)
        {
            AL_PROFILE_FUNC("FindDivergenceDegree");

            return detail::divergenceDegree(system, timeForward, e, v, T, M, traj1, traj2, linearize);
        }
    }
}
