// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DynamicSystem.hpp"
#include "RadonsSystem.hpp"
#include <cmath>
#include <format>
#include <print>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <variant>

#include "Hysteresis/ArealModel.hpp"
#include "Core/Error.hpp"
#include "Utils/Eigen.hpp"
#include "Utils/doubleToString.hpp"
#include "Utils/nestedLoop.hpp"

namespace mc
{
    namespace ode
    {
        template <int N>
        using DivergenceTrajectories = std::unordered_map<std::string, std::vector<Eigen::Vector<double, N>>>;

        namespace detail
        {
            template <int N, typename Context>
            std::array<double, 2> divergenceDegree(DynamicalSystem<N, Context> &system, double timeForward, double e,
                                                          typename DynamicalSystem<N, Context>::State v, double T, int M,
                                                  std::vector<typename DynamicalSystem<N, Context>::State> &traj1,
                                                  std::vector<typename DynamicalSystem<N, Context>::State> &traj2)
            {
                AL_PROFILE_FUNC("divergenceDegree");

                system.Reset();

                system.Forward(timeForward);
                traj1.push_back(system.GetX());

                v.normalize();
                v *= e;
                v = system.GetX() + v;
                traj2.push_back(v);

                double n = 0.0;

                for (int i = 1; i < M + 1; ++i)
                {
                    auto count = static_cast<uint32>(T / system.GetDeltaTime());
                    while (count--)
                    {
                        v = system.NextTM(v);
                        traj2.push_back(v);

                        auto traj = system.Forward(1);
                        traj1.push_back(system.GetX());
                    }
                    v -= system.GetX();

                    n += std::log(v.norm() / e);

                    v.normalize();
                    v *= e;
                    v = system.GetX() + v;
                    traj2[traj2.size() - 1] = v;
                }

                return {n / M, std::log(T)};
            }

            /**
             * Детектор сингулярных точек производной оператора Прейзаха.
             * В данном случае - это смена монотонности входа x, или смена знака производной v.
             * @param curr_x Текущее состояние системы.
             * @param next_x Следующее состояние системы.
             * @return bool  True, если следующая точка сингулярная.
             */
            template <int N>
            bool PreisachInputSingularityDetector(const Eigen::Vector<double, N> &curr_x,
                                                   const Eigen::Vector<double, N> &next_x)
            {
                if constexpr (N < 2)
                {
                    return false;
                }
                else
                {
                    constexpr double eps = 0.001;
                    return !(std::abs(curr_x[1]) < eps) && (curr_x[1] * next_x[1] < 0.0 || std::abs(next_x[1]) < eps);
                }
            }
        }
        
        namespace power_law
        {
            /**
             * @brief Вспомогательная функция вычисления усредненного логарифма расстояния для одного значения T.
             */
            inline double ComputeAvgLogDistForT(
                 RadonsSystem &system,
                const Eigen::Vector2d &initial_dir, double eps, int T_steps, int M,
                std::vector<Eigen::Vector2d> &traj1, std::vector<Eigen::Vector2d> &traj2)
            {
                AL_PROFILE_FUNC("mc::ode::ComputePowerLawExponent::ComputeAvgLogDistForT");

                Eigen::Vector2d dir = initial_dir.normalized();
                 const double w02 = system.GetArgs().params.w0;

                const auto calc_logonormal_dist = [eps, dir, w02](
                    const Eigen::Vector2d &main_x, Eigen::Vector2d &pert_x,
                    double &sum_log_dist)
                {
                    Eigen::Vector2d delta = pert_x - main_x;
                    // Используем энергетическую метрику
                    // p(x1, x2) = std::sqrt(w02 / 2 * (dx)^2 + (dv)^2 / 2)
                    double dist = std::sqrt(w02 * (delta(0) * delta(0)) + (delta(1) * delta(1))) / std::sqrt(2.);

                    if (dist > 1e-16)
                    {
                        sum_log_dist += std::log(dist);
                        // Нормализация вектора возмущения "в том же направлении" 
                        // (в направлении разошедшихся траекторий) до длины eps
                        pert_x = main_x + delta * (eps / dist);
                    }
                    else
                    {
                        // Защита от схлопывания (деления на ноль)
                        sum_log_dist += std::log(1e-16);
                        pert_x = main_x + eps * dir; // Сброс в исходное направление
                    }
                };

                Eigen::Vector2d main_x = system.GetX();
                // Возмущенная траектория в заданном направлении
                Eigen::Vector2d pert_x = main_x + eps * dir;

                traj1.push_back(main_x);
                traj1.push_back(pert_x);

                double sum_log_dist = 0.0;
                
                for (int m = 0; m < M; ++m)
                {
                    for (int k = 0; k < T_steps; ++k)
                    {
                        pert_x = system.NextTM(pert_x);
                        system.Next();
                        main_x = system.GetX();

                        traj1.push_back(main_x);
                        traj2.push_back(pert_x);
                    }
                    
                    calc_logonormal_dist(main_x, pert_x, sum_log_dist);

                    traj2[traj2.size() - 1] = pert_x;
                }

                // Усреднение за M периодов
                return sum_log_dist / static_cast<double>(M);
            }
            
            struct PowerLawResult
            {
                double epsilon;
                Eigen::ArrayXd log_times;
                Eigen::ArrayXd mean_log_distances;
                double exponent;
                double intercept;
                double r_squared;
                DivergenceTrajectories<2> primary;
                DivergenceTrajectories<2> perturbed;
            };

            inline PowerLawResult ComputePowerLawExponent(
                 RadonsSystem &system,
                const Eigen::Vector2d &initial_dir,
                double eps,
                double T_step, double additional_time = 0.0)
            {
                AL_PROFILE_FUNC("mc::ode::ComputePowerLawExponent");
                 system.GetNextArgs();
                
                const double dt = system.GetDeltaTime();
                 const double L = system.GetArgs().model->GetL() + 0.1;
                 const double w02 = system.GetArgs().params.w0;

                 system.SetResetFn([](RadonsContext &args, std::optional<RadonsContext> &nextArgs, uint32_t)
                {
                     args.model->ResetState();
                     if (nextArgs) nextArgs->model->ResetState();
                });
                
                system.ResetArgs();
                system.ResetNextArgs();
                system.Reset();
                
                // Считаем максимально доступное количество шагов в активной зоне
                uint32_t max_steps = 0;
                Eigen::Vector2d pert_x = system.GetX() + eps * initial_dir;

                double prev_dist = -mc::consts::inf;
                while (true)
                {
                    pert_x = system.NextTM(pert_x);
                    system.Next();
                    max_steps++;
                    
                    // Защита от зависания, если траектория застряла в аттракторе
                    if (max_steps > static_cast<uint32_t>(200. / dt)) break; 
                    
                    bool in_bounds = std::abs(system.GetX()(0)) <= L;
                    
                    auto delta = system.GetX() - pert_x;
                    double dist = std::sqrt(w02 * (delta(0) * delta(0)) + (delta(1) * delta(1))) / std::sqrt(2.);
                    bool is_diverging = (dist - prev_dist) > 1e-5;
                    prev_dist = dist;

                    if (!in_bounds && !is_diverging) break;
                }
                
                max_steps += static_cast<uint32_t>(additional_time / dt);
                const double max_T = static_cast<double>(max_steps) * dt;
                
                const Eigen::VectorXd Ts = Eigen::arange(max_T * T_step, max_T / 2., T_step);
                
                Eigen::ArrayXd X; // ln(T)
                Eigen::ArrayXd Y; // ln||delta x||

                X.resize(Ts.size());
                Y.resize(Ts.size());

                std::unordered_map<std::string, std::vector<Eigen::Vector2d>> trajs1 = {}, trajs2 = {};

                for (int i = 0; i < Ts.size(); ++i)
                {
                    const double T = Ts(i);
                    const uint32_t T_steps = static_cast<uint32_t>(T / dt);
                    const int M = max_steps / T_steps;
                    
                    // Если T слишком велико и не помещается даже 1 раз в окно - пропускаем
                    if (M < 1) continue;
                    
                    system.ResetArgs();
                    system.ResetNextArgs();
                    system.Reset();

                    std::vector<Eigen::Vector2d> traj1, traj2;
                    double avg_log_d = ComputeAvgLogDistForT(system, initial_dir, eps, T_steps, M, traj1, traj2);

                    const auto strT = mc::doubleToString(T, 2);
                    trajs1.insert({strT, traj1});
                    trajs2.insert({strT, traj2});

                    X(i) = T;
                    Y(i) = avg_log_d;
                }
                X = X.array().log();
                
                const auto reg = Eigen::Regression(X, Y);

                system.ResetArgs();
                system.ResetNextArgs();
                system.Reset();

                return {eps, std::move(X), std::move(Y), reg.slope, reg.intercept, reg.r * reg.r,
                        std::move(trajs1), std::move(trajs2)};
            }
        }

        /// Implements Benettin's mLCE computation algorithm
        template <int N, typename Context>
        std::pair<double, Eigen::VectorXd> Benettin_mLCE(DynamicalSystem<N, Context> &system, double timeForward,
                                                                double e,
                                                                double T = 8.0,
                                                                int M = 30, bool linearize = true)
        {
            AL_PROFILE_FUNC("Benettin_mLCE");
            system.Reset();
            system.Forward(timeForward);

            double mLCE = 0.0;
            Eigen::VectorXd history = Eigen::VectorXd::Zero(M);

            Eigen::Vector<double, N> v = Eigen::Rand(1, N).transpose();

            for (int i = 1; i < M + 1; ++i)
            {
                v = system.GetX() + v.normalized() * e;
                auto count = static_cast<uint32>(T / system.GetDeltaTime());
                while (count--)
                {
                    v = linearize ? system.NextLTM(v) : system.NextTM(v);
                    system.Forward(1);
                }
                v -= system.GetX();

                mLCE += std::log(v.norm() / e);
                history[i - 1] = mLCE / (i * T);
            }

            mLCE /= M * T;
            return {mLCE, history};
        }

        struct TrajsCircleResult
        {
            using DirectionTrajectories = std::unordered_map<std::string,
                std::array<std::vector<Eigen::Vector3d>, 2>>;
            using DirectionLoops = std::unordered_map<std::string, std::array<std::vector<double>, 4>>;

            std::vector<double> perturbations;
            std::vector<Eigen::Vector3d> directions;
            double time;
            double time_forward;
            std::unordered_map<std::string, DirectionTrajectories> trajectories;
            std::unordered_map<std::string, DirectionLoops> loops;
        };

         inline TrajsCircleResult TrajsCircle(DynamicalSystem<3, RadonsContext> &system, const std::vector<double> &es, double time,
                                             double timeForward = consts::nan, double areaCoeff = consts::nan)
        {
             system.GetNextArgs();
            std::vector<Eigen::Vector3d> coords = {
                {1., 0., 0.},
                {std::sqrt(2.) / 2, std::sqrt(2.) / 2, 0.},
                {0., 1., 0.},
                {-std::sqrt(2.) / 2, std::sqrt(2.) / 2, 0.},
                {-1., 0., 0.},
                {-std::sqrt(2.) / 2, -std::sqrt(2.) / 2, 0.},
                {0., -1., 0.},
                {std::sqrt(2.) / 2, -std::sqrt(2.) / 2, 0.},
            };
            if (!std::isnan(timeForward))
            {
                system.Forward(timeForward);
            }

            TrajsCircleResult result{es, std::move(coords), time, timeForward, {}, {}};

             system.SetResetFn([areaCoeff](RadonsContext &args, std::optional<RadonsContext> &nextArgs, uint32_t)
            {
                constexpr double L = 1.0;
                auto model1 = std::make_shared<mc::ArealPreisachModel>(L, false, false);

                auto model2 = std::make_shared<mc::ArealPreisachModel>(L, false, false);
                if (!std::isnan(areaCoeff))
                {
                    model2->P(L, 0.0, -2);
                    model2->P(areaCoeff * L, 0.0, -1);
                }

                 args.model = std::move(model1);
                 nextArgs.value().model = std::move(model2);
            });

            for (const auto &e : es)
            {
                TrajsCircleResult::DirectionTrajectories trajs;
                TrajsCircleResult::DirectionLoops loops;

                for (const auto &v : result.directions)
                {
                    system.Reset();

                    std::vector<Eigen::Vector3d> mainTraj{system.GetX()};

                    Eigen::Vector3d w = system.GetX() + v.normalized() * e;

                    std::vector<Eigen::Vector3d> secondTraj{w};

                    auto count = static_cast<uint32>(time / system.GetDeltaTime());
                    while (count--)
                    {
                        auto traj = system.Forward(1);
                        mainTraj.emplace_back(system.GetX());

                        w = system.NextTM(w);
                        secondTraj.emplace_back(w);
                    }

                    // TODO:
                    // auto [in1, out1] = system.GetArgs().at("model").toPreisachModel()->HysteresisLoop();
                    // auto [in2, out2] = system.GetNextArgs().at("model").toPreisachModel()->HysteresisLoop();
                    auto [in1, out1] = system.GetArgs().model->HysteresisLoop();
                    auto [in2, out2] = system.GetNextArgs().model->HysteresisLoop();

                    trajs.insert({mc::doubleToString(v, 4), {std::move(mainTraj), std::move(secondTraj)}});
                    loops.insert({mc::doubleToString(v, 4),
                                  {std::move(in1), std::move(out1), std::move(in2), std::move(out2)}});
                }
                result.trajectories.emplace(doubleToString(e, 2), std::move(trajs));
                result.loops.emplace(doubleToString(e, 2), std::move(loops));
            }

            return result;
        }


        struct LCEResult
        {
            Eigen::VectorXd spectrum; // Итоговые показатели Ляпунова
            Eigen::MatrixXd history; // История: строки - шаги (время), столбцы - показатели
        };

        /**
         * @brief Вычисление спектра показателей Ляпунова (LCEs) модифицированным алгоритмом Бенеттина.
         * Адаптировано для негладких систем.
         * @param system             Система.
         * @param timeForward        Пропускаемое время (время выхода на аттрактор).
         * @param total_time         Общее время интегрирования.
         * @param ortho_steps        Период (в шагах) классической регулярной ортогонализации (например, 10-100).
         * @param singularity_check  Предикат-детектор пересечения поверхности сингулярности 
         * между текущим состоянием и состоянием на следующем шаге.
         * @param num_lce             Количество показателей Ляпунова для вычисления 
         * (если не задано, используется размерность системы) 
         * @return [Eigen::VectorXd, Eigen::MatrixXd]   Спектр из N показателей Ляпунова (где N - размерность системы)
         * и вектор истории каждого из них (построчно).
         */
        template <int N, typename Context>
        LCEResult ComputeLCEs(
            DynamicalSystem<N, Context> &system,
            double timeForward,
            double total_time,
            uint32_t ortho_steps,
            std::optional<int> num_lce = std::nullopt,
            const std::type_identity_t<std::function<bool(const Eigen::Vector<double, N> &,
                                                        const Eigen::Vector<double, N> &)>> &singularity_check =
                detail::PreisachInputSingularityDetector<N>)
        {
            const int n_lce = num_lce.value_or(N);
            if (n_lce < 1 || n_lce > N)
            {
                THROW_INVALID_ARGUMENT_ERROR("Number of exponents must be between one and the system dimension");
            }

            AL_PROFILE_FUNC("mc::ode::ComputeLCEs");

            system.ResetArgs();
            system.Reset();
            system.Forward(timeForward);

            const double dt = system.GetDeltaTime();
            const int num_iterations = static_cast<int>(total_time / dt);

            LCEResult result;
            result.spectrum = Eigen::VectorXd::Zero(n_lce);
            result.history = Eigen::MatrixXd::Zero(ortho_steps, n_lce);

            // Инициализация матрицы возмущений единичной матрицей
            using PerturbationBasis = Eigen::Matrix<double, N, Eigen::Dynamic>;
            PerturbationBasis W = PerturbationBasis::Identity(N, n_lce);

            Eigen::VectorXd lce_sums = Eigen::VectorXd::Zero(n_lce);
            Eigen::VectorXd current_norms = Eigen::VectorXd::Zero(n_lce);

            const uint32_t orto_step = num_iterations / ortho_steps;
            int32_t hist_idx = -1;
            bool is_next_singularity = false;
            bool is_singularity_now = false;
            for (int step = 1; step <= num_iterations; ++step)
            {
                const auto current_x = system.GetX();
                // memoize = true, потому что мы предсказываем следующую точку
                const auto next_x = system.ShiftTrajNext(current_x, system.GetT(), true);

                if (is_next_singularity)
                {
                    is_singularity_now = true;
                }

                is_next_singularity = singularity_check(current_x, next_x);

                // Детекция возможного пересечения сингулярности в пределах [current_t, current_t + dt]
                // "Заглядываем" на шаг вперёд (без изменения состояния системы), 
                // чтобы понять, пересечем ли мы поверхность потери гладкости.
                // Интегрирование касательного пространства (уравнений в вариациях)
                PerturbationBasis W_next(N, n_lce);
                for (int i = 0; i < n_lce; ++i)
                {
                    W_next.col(i) = system.NextLTM(W.col(i));
                }

                // Делаем физический шаг системы (обновляет внутренние m_X и m_T)
                system.Next();

                // Если попали на точку сингулярности ИЛИ подошло время плановой ортогонализации
                if (is_next_singularity || (!is_singularity_now && (step > 0 && step % orto_step == 0)))
                {
                    ++hist_idx;
                    // Вызов QR-ортогонализации
                    Eigen::OrthogonalizeQR(W_next, current_norms, n_lce);

                    for (int i = 0; i < n_lce; ++i)
                    {
                        if (current_norms(i) > 1e-15)
                        {
                            lce_sums(i) += std::log(current_norms(i));
                        }
                    }
                    W = W_next;
                }
                else
                {
                    W = W_next;
                    is_singularity_now = false;
                    continue;
                }

                if (hist_idx >= result.history.rows())
                {
                    result.history.conservativeResize(static_cast<Eigen::Index>(1.5 * result.history.rows()),
                                                      Eigen::NoChange);
                }

                // Запись в историю. LCE = (Сумма логарифмов) / Время
                // Для первого шага предотвращаем деление на ноль
                double time_divisor = (system.GetT() > 0.0) ? system.GetT() : dt;
                for (int i = 0; i < n_lce; ++i)
                {
                    result.history(hist_idx, i) = lce_sums(i) / time_divisor;
                }
            }

            result.history.conservativeResize(hist_idx, Eigen::NoChange);

            // Вычисляем итоговые показатели как предел усредненной суммы
            result.spectrum = lce_sums / system.GetT();

            system.ResetArgs();
            system.Reset();

            return result;
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

        using CandidateValue = std::variant<bool, int, double, Eigen::MatrixXd>;
        using CandidateValues = std::unordered_map<std::string, CandidateValue>;

        template <int N>
        struct DivergenceCandidate
        {
            double epsilon;
            Eigen::Vector<double, N> direction;
        };

        template <int N>
        DivergenceCandidate<N> DecodeDivergenceCandidate(const CandidateValues &values)
        {
            for (const auto &[name, value] : values)
            {
                if (name != "e" && name != "v")
                {
                    throw std::invalid_argument("Unknown divergence candidate parameter: " + name);
                }
            }
            const auto e = values.find("e");
            const auto *epsilon = e == values.end() ? nullptr : std::get_if<double>(&e->second);
            if (!epsilon || !std::isfinite(*epsilon))
            {
                throw std::invalid_argument("Divergence candidate e must be a finite double");
            }
            const auto v = values.find("v");
            const auto *direction = v == values.end() ? nullptr : std::get_if<Eigen::MatrixXd>(&v->second);
            if (!direction || direction->rows() != N || direction->cols() != 1 ||
                !direction->allFinite() || direction->isZero(0.0))
            {
                throw std::invalid_argument(std::format("Divergence candidate v must be a finite nonzero {} x 1 vector", N));
            }
            return {*epsilon, *direction};
        }

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
            using CandidateFn = std::function<std::vector<CandidateValue>()>;

            CandidateParams(std::vector<CandidateValue> coarse_range, std::optional<double> fine_step)
            {
                this->coarse_range = coarse_range;
                this->fine_step = fine_step;
            }

            CandidateParams(CandidateFn coarse_range, std::optional<double> fine_step)
            {
                this->coarse_range = coarse_range;
                this->fine_step = fine_step;
            }

            std::variant<std::vector<CandidateValue>, CandidateFn> coarse_range;
            std::optional<double> fine_step;
        };

        struct ParamsSearchResult
        {
            ParamsSearchResult(const CandidateValues &p, const MetricResult &m,
                               const Eigen::VectorXd &s) :
                params(p), metric(m), solution(s)
            {
            }

            CandidateValues params;
            MetricResult metric;
            Eigen::VectorXd solution;
        };

        /**
         * Find the best params with specific metric for system solutions.
         * @param system: dynamical system
         * @param validate: converts a complete candidate before resetting or advancing the system
         * @param solve: evaluates a validated, typed candidate
         * @param candidates: a candidate map, the keys of which are the names of the candidate params, and the values are the steps for these candidates.
         * @param metric: a solution metric.
         * @param comparator: compares results of applying metric for solutions
         * @returns vector of CoarseSearchResult
         */
        template <int N, typename Context, typename Validate, typename Solve>
        std::vector<ParamsSearchResult> FindBestParams(DynamicalSystem<N, Context> &system,
                                                      Validate validate, Solve solve,
                                                              const std::unordered_map<std::string, CandidateParams> &candidates,
                                                              const SolutionsMetricFn &metric = GrowthMetric,
                                                              const MetricsCompareFn &comparator = MaxCompare)
        {
            auto keys = candidates | std::views::keys;
            std::vector<std::string> params_names{keys.begin(), keys.end()};
            std::ranges::sort(params_names);

            std::unordered_map<std::string, std::vector<CandidateValue>> coarse_grid;
            std::ranges::transform(params_names, std::inserter(coarse_grid, coarse_grid.begin()),
                                   [candidates](const auto &name)
                                   {
                                       std::vector<CandidateValue> value;
                                       if (const auto &range = candidates.at(name).coarse_range; std::holds_alternative<
                                           std::vector<CandidateValue>>(range))
                                       {
                                           value = std::get<std::vector<CandidateValue>>(range);
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
                const auto checked = std::invoke(validate, candidate);
                system.Reset();

                Eigen::VectorXd solution = std::invoke(solve, system, checked);
                auto metric_value = metric(solution);
                coarse_results.emplace_back(candidate, metric_value, solution);
            }

            std::vector<ParamsSearchResult> coarse_bests;
            {
                std::vector<ParamsSearchResult> best_elements;
                std::optional<MetricResult> best_metric;

                for (const auto &coarse_result : coarse_results)
                {
                    if (!std::isnan(coarse_result.metric.m_MetricValue))
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

        template <int N>
        struct DivergenceRegressionResult
        {
            double epsilon;
            int repetitions;
            Eigen::VectorXd log_times;
            std::vector<double> mean_log_distances;
            DivergenceTrajectories<N> primary;
            DivergenceTrajectories<N> perturbed;
        };

        template <int N, typename Context, typename Derived>
        DivergenceRegressionResult<N> DivergenceDegreeRegressionData(DynamicalSystem<N, Context> &system,
                                                                      double timeForward,
                                                                      double e, const Eigen::MatrixBase<Derived> &v,
                                                                      const Eigen::VectorXd &Ts, int M)
        {
            AL_PROFILE_FUNC("FindDivergenceDegree");

            using State = typename DynamicalSystem<N, Context>::State;
            const State direction = DynamicalSystem<N, Context>::CheckedState(v);
            system.Reset();

            std::unordered_map<std::string, std::vector<State>>
                trajs1 = {}, trajs2 = {};
            std::vector<double> ns;

            trajs1.reserve(Ts.size());
            trajs2.reserve(Ts.size());
            ns.reserve(Ts.size());


            for (const auto &T : Ts)
            {
                std::vector<State> traj1, traj2;
                const auto [n, t] = detail::divergenceDegree(system, timeForward, e, direction, T, M, traj1, traj2);

                const auto strT = mc::doubleToString(T, 2);

                trajs1.insert({strT, traj1});
                trajs2.insert({strT, traj2});
                std::println("n: {}", n);
                ns.push_back(n);
            }

            return {e, M, Ts.array().log().matrix(), std::move(ns), std::move(trajs1), std::move(trajs2)};
        }
    }
}
