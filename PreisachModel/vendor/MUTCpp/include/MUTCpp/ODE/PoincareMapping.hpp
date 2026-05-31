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
        class PoincareMapGenerator
        {
        public:
            // Структура для определения секущей плоскости
            struct PoincareSection
            {
                Eigen::VectorXd Normal;
                Eigen::VectorXd Point;

                PoincareSection(const Eigen::VectorXd &normal, const Eigen::VectorXd &point) :
                    Normal(normal.normalized()), Point(point)
                {
                }
            };

            // Основная функция генерации отображения
            static Eigen::MatrixXd ComputeMap(
                Eigen::MatrixXd traj,
                uint32_t dim,
                const PoincareSection &section)
            {
                const Eigen::Index numSteps = traj.rows();

                if (numSteps < 2)
                    return Eigen::MatrixXd(0, dim);

                std::vector<Eigen::VectorXd> intersections;
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

                        Eigen::VectorXd exactIntersection =
                            traj.row(i).transpose() + t * (traj.row(i + 1) - traj.row(i)).transpose();

                        intersections.push_back(exactIntersection);
                    }
                }

                // 4. Формирование результирующей матрицы
                Eigen::MatrixXd poincareMap(intersections.size(), dim);
                for (size_t i = 0; i < intersections.size(); ++i)
                {
                    poincareMap.row(i) = intersections[i].transpose();
                }

                return poincareMap;
            }

            static Eigen::MatrixXd ComputeStroboscopicMap(Eigen::MatrixXd traj, uint32_t dim, double period, double dt)
            {
                const uint32_t stepsPerPeriod = period / dt;
                
                Eigen::Index totalPoints = traj.rows();
                Eigen::Index numIntersections = totalPoints / stepsPerPeriod;
    
                Eigen::MatrixXd poincareMap(numIntersections, dim); 
    
                for (Eigen::Index i = 0; i < numIntersections; ++i)
                {
                    // Выборка строк с шагом, равным периоду
                    poincareMap.row(i) = traj.row(i * stepsPerPeriod).head(dim); 
                }
    
                return poincareMap;
            }
        };

        inline Eigen::MatrixXd PoincareMapping(mc::Ref<mc::ode::DynamicalSystem> system, double time, uint32_t dim,
                                               const Eigen::VectorXd &PlaneNormal, const Eigen::VectorXd &PlanePoint)
        {
            system->ResetArgs();
            system.Reset();
            
            const auto traj = system->Forward(time);
            
            system->ResetArgs();
            system.Reset();
            
            return PoincareMapGenerator::ComputeMap(traj, dim, PoincareMapGenerator::PoincareSection(PlaneNormal, PlanePoint));
        }
        
        inline Eigen::MatrixXd PoincareStroboscopicMapping(mc::Ref<mc::ode::DynamicalSystem> system, double periodMultiplier, uint32_t dim, double period, double dt)
        {
            system->ResetArgs();
            system.Reset();
            
            const auto traj = system->Forward(periodMultiplier * period);
            
            system->ResetArgs();
            system.Reset();
            
            return PoincareMapGenerator::ComputeStroboscopicMap(traj, dim, period, dt);
        }
    }
}
