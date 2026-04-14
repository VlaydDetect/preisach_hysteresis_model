// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DynamicSystem.hpp"
#include "Functions/log.hpp"
#include "Random/randN.hpp"
#include "Utils/doubleToString.hpp"
#include "Utils/nestedLoop.hpp"

namespace mc
{
    namespace ode
    {
        namespace detail
        {
            /**
             * @brief Ортогонализация базиса возмущений через QR-разложение.
             * @param W Матрица векторов возмущений (in/out).
             * @param norms Вектор для записи норм роста (out).
             * @param n_lce Количество вычисляемых показателей.
             */
            inline void OrthogonalizeQR(Eigen::MatrixXd &W, Eigen::VectorXd &norms, int n_lce)
            {
                // Выполняем QR-разложение для первых n_lce столбцов
                Eigen::HouseholderQR<Eigen::MatrixXd> qr(W.leftCols(n_lce));

                // Q матрица содержит ортонормированный базис
                Eigen::MatrixXd Q = qr.householderQ();

                // R матрица (верхнетреугольная). Ее диагональные элементы равны длинам векторов
                // возмущений после проекции на предыдущие орты.
                Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

                for (int i = 0; i < n_lce; ++i)
                {
                    // Длина базисного вектора не может быть отрицательной
                    norms(i) = std::abs(R(i, i));

                    // Обновляем матрицу возмущений ортонормированными векторами
                    W.col(i) = Q.col(i);
                }
            }

            inline std::array<double, 2> divergenceDegree(Ref<DynamicalSystem> system, double timeForward, double e,
                                                          Eigen::VectorXd v, double T, int M,
                                                          std::vector<Eigen::MatrixXd> &traj1,
                                                          std::vector<Eigen::MatrixXd> &traj2)
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
                        v = system->NextTM(v);
                        traj2.push_back(v);

                        auto traj = system->Forward(1);
                        traj1.push_back(system->GetX());
                    }
                    v -= system->GetX();

                    n += mc::log(v.norm() / e);

                    v.normalize();
                    v *= e;
                    v = system->GetX() + v;
                    traj2[traj2.size() - 1] = v;
                }

                return {n / M, mc::log(T)};
            }
        }

        namespace power_law
        {
            /**
             * @brief Вспомогательная функция вычисления усредненного логарифма расстояния для одного значения T.
             */
            inline double ComputeAvgLogDistForT(
                Ref<ode::ContinuousDS> system,
                const Vec &initial_dir, double eps, int T_steps, int M,
                std::vector<Eigen::MatrixXd> traj1, std::vector<Eigen::MatrixXd> traj2)
            {
                AL_PROFILE_FUNC("mc::ode::ComputePowerLawExponent::ComputeAvgLogDistForT");
                
                Vec main_x = system->GetX();

                // Возмущенная траектория в заданном направлении
                Vec dir = initial_dir.normalized();
                Vec pert_x = main_x + eps * dir;
                
                traj1.push_back(main_x);
                traj1.push_back(pert_x);

                double sum_log_dist = 0.0;

                for (int m = 0; m < M; ++m)
                {
                    for (int k = 0; k < T_steps; ++k)
                    {
                        // Важно: вызываем NextTM для pert_x до сдвига самой системы,
                        // чтобы внутреннее время m_T системы и аргументы совпадали.
                        pert_x = system->NextTM(pert_x);
                        system->Next();
                        
                        traj1.push_back(system->GetX());
                        traj2.push_back(pert_x);
                    }

                    main_x = system->GetX();
                    Vec delta = pert_x - main_x;
                    double dist = delta.norm();

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
                    
                    traj2[traj2.size() - 1] = pert_x;
                }

                // Усреднение за M периодов
                return sum_log_dist / static_cast<double>(M);
            }

            /**
             * @brief Вычисляет степенной показатель расхождения траекторий.
             * Оценивает зависимость \ln||\delta x(T)|| = \nu \ln T + C через МНК.
             * * @param system           Указатель на инстанс ContinuousDS.
             * @param initial_dir      Вектор направления начального возмущения.
             * @param eps              Длина начального возмущения \epsilon.
             * @param Ts               Массив различных периодов времени T.
             * @param M                Количество повторений нормализации для каждого T.
             * @return PowerLawResult  Структура с коэффициентами \nu, C и историей вычислений.
             */
            inline mc::json::JsonDocument ComputePowerLawExponent(
                Ref<ode::ContinuousDS> system,
                double timeForward,
                const Vec &initial_dir,
                double eps,
                const Eigen::VectorXd &Ts,
                int M)
            {
                AL_PROFILE_FUNC("mc::ode::ComputePowerLawExponent");

                const double dt = system->GetDeltaTime();
                const int num_points = static_cast<int>(Ts.size());

                Eigen::VectorXd X; // ln(T)
                Eigen::VectorXd Y; // ln||delta x||

                X.resize(num_points);
                Y.resize(num_points);
                
                std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> trajs1 = {}, trajs2 = {};

                // 1. Сбор данных для каждого T
                for (int i = 0; i < Ts.size(); ++i)
                {
                    const double T = Ts(i);
                    // Сброс системы в начальное положение m_X0 для каждого измерения,
                    // чтобы сравнивать расхождение на одной и той же части аттрактора.
                    // (При необходимости нужно также сбрасывать состояние модели Прейзаха через ResetFn).
                    system->Reset();
                    system->Forward(timeForward);

                    std::vector<Eigen::MatrixXd> traj1, traj2;
                    double avg_log_d = ComputeAvgLogDistForT(system, initial_dir, eps, T / dt, M, traj1, traj2);
                    
                    const auto strT = mc::doubleToString(T, 2);
                    trajs1.insert({strT, traj1});
                    trajs2.insert({strT, traj2});
                    
                    X(i) = T;
                    Y(i) = avg_log_d;
                }
                X = X.array().log();

                // 2. Линейная регрессия (МНК)
                double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
                for (int i = 0; i < num_points; ++i)
                {
                    sum_x += X[i];
                    sum_y += Y[i];
                    sum_xx += X[i] * X[i];
                    sum_xy += X[i] * Y[i];
                }

                double nu;
                double C;
                
                double denominator = num_points * sum_xx - sum_x * sum_x;
                if (std::abs(denominator) > 1e-12)
                {
                    nu = (num_points * sum_xy - sum_x * sum_y) / denominator;
                    C = (sum_xx * sum_y - sum_x * sum_xy) / denominator; // Эквивалент (sum_y - nu * sum_x) / N
                }
                else
                {
                    // Защита от вырожденного случая (например, если передано только одно значение T)
                    nu = 0.0;
                    C = 0.0;
                }
                
                mc::json::JsonDocument message({"name", "e", "M", "Ts", "ns", "nu", "C", "trajs1", "trajs2"});
                message.AddField("name", "DivergenceDegreeRegressionData");
                message.AddField("e", eps);
                message.AddField("M", M);
                message.AddField("Ts", X);
                message.AddField("ns", Y);
                message.AddField("nu", nu);
                message.AddField("C", C);
                message.AddField("trajs1", trajs1);
                message.AddField("trajs2", trajs2);

                return message;
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
            system->Reset();
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

            system->SetResetFn([areaCoeff](DSArgs &args, DSArgs &nextArgs, uint32_t)
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

        /**
         * Детектор сингулярных точек производной оператора Прейзаха.
         * В данном случае - это смена монотонности входа x, или смена знака производной v.
         * @param curr_x Текущее состояние системы.
         * @param next_x Следующее состояние системы.
         * @return bool  True, если следующая точка сингулярная.
         */
        inline bool PreisachInputSingularityDetector(const Vec &curr_x, const Vec &next_x)
        {
            constexpr double eps = 0.001;
            return !(std::abs(curr_x[1]) < eps) && (curr_x[1] * next_x[1] < 0.0 || std::abs(next_x[1]) < eps);
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
        inline LCEResult ComputeLCEs(
            Ref<ode::DynamicalSystem> system,
            double timeForward,
            double total_time,
            uint32_t ortho_steps,
            std::optional<int> num_lce = std::nullopt,
            const std::function<bool(const Vec &curr_x, const Vec &next_x)> &singularity_check =
                PreisachInputSingularityDetector)
        {
            if (num_lce.has_value())
            {
                assert(*num_lce <= system->GetDimension());
            }

            AL_PROFILE_FUNC("mc::ode::ComputeLCEs");

            system->Reset();
            system->Forward(timeForward);

            const int dim = system->GetDimension();
            const int n_lce = num_lce.value_or(dim);
            const double dt = system->GetDeltaTime();
            const int num_iterations = static_cast<int>(total_time / dt);

            LCEResult result;
            result.spectrum = Eigen::VectorXd::Zero(n_lce);
            result.history = Eigen::MatrixXd::Zero(ortho_steps, n_lce);

            // Инициализация матрицы возмущений единичной матрицей
            Eigen::MatrixXd W = Eigen::MatrixXd::Identity(dim, dim);

            Eigen::VectorXd lce_sums = Eigen::VectorXd::Zero(n_lce);
            Eigen::VectorXd current_norms = Eigen::VectorXd::Zero(n_lce);

            const uint32_t orto_step = num_iterations / ortho_steps;
            int32_t hist_idx = -1;
            bool crossed_singularity = false;
            for (int step = 1; step <= num_iterations; ++step)
            {
                Vec current_x = system->GetX();
                Vec next_x = system->ShiftTrajNext(current_x, system->GetT());
                
                if (crossed_singularity)
                {
                    crossed_singularity = singularity_check(current_x, next_x);
                    continue;
                }
                
                crossed_singularity = singularity_check(current_x, next_x);

                // Детекция возможного пересечения сингулярности в пределах [current_t, current_t + dt]
                // "Заглядываем" на шаг вперёд (без изменения состояния системы), 
                // чтобы понять, пересечем ли мы поверхность потери гладкости.
                // Интегрирование касательного пространства (уравнений в вариациях)
                Eigen::MatrixXd W_next(dim, dim);
                for (int i = 0; i < n_lce; ++i)
                {
                    W_next.col(i) = system->NextLTM(W.col(i));
                }

                // Делаем физический шаг системы (обновляет внутренние m_X и m_T)
                system->Next();

                // Если попали на точку сингулярности ИЛИ подошло время плановой ортогонализации
                if (crossed_singularity || (step > 0 && step % orto_step == 0))
                {
                    ++hist_idx;
                    // Вызов QR-ортогонализации
                    detail::OrthogonalizeQR(W_next, current_norms, n_lce);

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
                    continue;
                }
                
                if (hist_idx >= result.history.rows())
                {
                    result.history.conservativeResize(static_cast<Eigen::Index>(1.5 * result.history.rows()), Eigen::NoChange);
                }

                // Запись в историю. LCE = (Сумма логарифмов) / Время
                // Для первого шага предотвращаем деление на ноль
                double time_divisor = (system->GetT() > 0.0) ? system->GetT() : dt;
                for (int i = 0; i < n_lce; ++i)
                {
                    result.history(hist_idx, i) = lce_sums(i) / time_divisor;
                }
            }
            
            result.history.conservativeResize(hist_idx, Eigen::NoChange);

            // Вычисляем итоговые показатели как предел усредненной суммы
            result.spectrum = lce_sums / system->GetT();
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
                                           std::vector<Eigen::MatrixXd> &traj2)
        {
            AL_PROFILE_FUNC("FindDivergenceDegree");

            const auto [lnN, lnT] = detail::divergenceDegree(system, timeForward, e, v, T, M, traj1, traj2);
            return lnN / lnT;
        }

        inline mc::json::JsonDocument DivergenceDegreeRegressionData(Ref<DynamicalSystem> system,
                                                                     double timeForward,
                                                                     double e, const Eigen::VectorXd &v,
                                                                     const Eigen::VectorXd &Ts, int M)
        {
            AL_PROFILE_FUNC("FindDivergenceDegree");

            system->Reset();

            std::unordered_map<std::string, std::vector<Eigen::MatrixXd>>
                trajs1 = {}, trajs2 = {};
            std::vector<double> ns;

            trajs1.reserve(Ts.size());
            trajs2.reserve(Ts.size());
            ns.reserve(Ts.size());


            for (const auto &T : Ts)
            {
                std::vector<Eigen::MatrixXd> traj1, traj2;
                const auto [n, t] = detail::divergenceDegree(system, timeForward, e, v, T, M, traj1, traj2);

                const auto strT = mc::doubleToString(T, 2);

                trajs1.insert({strT, traj1});
                trajs2.insert({strT, traj2});
                std::println("n: {}", n);
                ns.push_back(n);
            }

            Eigen::VectorXd TsLog = Ts.array().log();
            mc::json::JsonDocument message({"name", "e", "M", "Ts", "ns", "trajs1", "trajs2"});
            message.AddField("name", "DivergenceDegreeRegressionData");
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
