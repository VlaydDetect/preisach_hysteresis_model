// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "DynamicSystem.hpp"

namespace mc
{
    namespace ode
    {
        template <int N>
        class PoincareMapGenerator
        {
        public:
            using State = typename DynamicalSystem<N>::State;
            using Trajectory = typename DynamicalSystem<N>::Trajectory;

            // Структура для определения секущей плоскости
            struct PoincareSection
            {
                State Normal;
                State Point;

                PoincareSection(const State &normal, const State &point) :
                    Normal(normal.normalized()), Point(point)
                {
                }
            };

            // Основная функция генерации отображения
            static Trajectory ComputeMap(
                const Trajectory &traj,
                const PoincareSection &section)
            {
                const Eigen::Index numSteps = traj.rows();

                if (numSteps < 2)
                    return Trajectory(0, N);

                std::vector<State> intersections;
                intersections.reserve(1024);

                // 2. Векторизованное вычисление расстояний до плоскости для всей траектории.
                // Математически: d = (X - X_0) * n
                // Использование rowwise() позволяет Eigen задействовать SIMD инструкции
                Eigen::VectorXd distances = (traj.rowwise() - section.Point.transpose()) * section.Normal;

                // 3. Поиск пересечений и интерполяция
                for (Eigen::Index i = 0; i < numSteps - 1; ++i)
                {
                    double d1 = distances(i);
                    double d2 = distances(i + 1);

                    // Фиксация пересечения строго в одном направлении (снизу вверх)
                    if (d1 < 0.0 && d2 >= 0.0)
                    {
                        // Линейная интерполяция для нахождения точной точки на плоскости
                        // t - доля шага, на которой происходит пересечение
                        double t = d1 / (d1 - d2);

                        State exactIntersection =
                            traj.row(i).transpose() + t * (traj.row(i + 1) - traj.row(i)).transpose();

                        intersections.push_back(exactIntersection);
                    }
                }

                // 4. Формирование результирующей матрицы
                Trajectory poincareMap(intersections.size(), N);
                for (size_t i = 0; i < intersections.size(); ++i)
                {
                    poincareMap.row(i) = intersections[i].transpose();
                }

                return poincareMap;
            }

            static Trajectory ComputeStroboscopicMap(const Trajectory &traj, double period, double dt)
            {
                const uint32_t stepsPerPeriod = period / dt;
                
                Eigen::Index totalPoints = traj.rows();
                Eigen::Index numIntersections = totalPoints / stepsPerPeriod;
    
                Trajectory poincareMap(numIntersections, N);
    
                for (Eigen::Index i = 0; i < numIntersections; ++i)
                {
                    // Выборка строк с шагом, равным периоду
                    poincareMap.row(i) = traj.row(i * stepsPerPeriod);
                }
    
                return poincareMap;
            }
        };

        template <int N, typename Normal, typename Point>
        typename DynamicalSystem<N>::Trajectory PoincareMapping(
            DynamicalSystem<N> &system, double time, const Eigen::MatrixBase<Normal> &normal,
            const Eigen::MatrixBase<Point> &point)
        {
            const typename PoincareMapGenerator<N>::PoincareSection section(
                DynamicalSystem<N>::CheckedState(normal), DynamicalSystem<N>::CheckedState(point));
            system.ResetArgs();
            system.Reset();
            const auto traj = system.Forward(time);
            system.ResetArgs();
            system.Reset();
            return PoincareMapGenerator<N>::ComputeMap(traj, section);
        }

        template <int N>
        typename DynamicalSystem<N>::Trajectory PoincareStroboscopicMapping(
            DynamicalSystem<N> &system, double periodMultiplier, double period, double dt)
        {
            system.ResetArgs();
            system.Reset();
            const auto traj = system.Forward(periodMultiplier * period);
            system.ResetArgs();
            system.Reset();
            return PoincareMapGenerator<N>::ComputeStroboscopicMap(traj, period, dt);
        }
    }
}
