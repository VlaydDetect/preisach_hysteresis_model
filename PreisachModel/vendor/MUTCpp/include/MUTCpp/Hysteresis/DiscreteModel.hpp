// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "ModelBase.hpp"
#include "Functions/nans.hpp"
#include "Utils/conversions.hpp"

#include <Eigen/Dense>

namespace mc
{
    class DiscretePreisachModel : public SinglePreisachModel
    {
    public:
        DiscretePreisachModel(const double L, const double h, bool keepDerivative = false, bool keepAnimation = false) :
            SinglePreisachModel(L, keepDerivative, keepAnimation), m_H(h), m_Count(static_cast<int>(2.0 * L / h))
        {
            Init();
        }

    private:
        virtual double P_Impl(const double u) override
        {
            double p = 0.0;
            for (const auto &[i, j] : m_Indices)
            {
                if (utils::essentiallyEqual(m_R(i, j), m_Down) && u >= -m_L + m_H * j)
                {
                    m_R(i, j) = m_Up;
                }

                if (utils::essentiallyEqual(m_R(i, j), m_Up) && u <= -m_L + m_H * i)
                {
                    m_R(i, j) = m_Down;
                }

                p += m_R(i, j);
            }

            p /= GetSum();

            // auto c = std::ranges::count_if(m_R, [&](const double& elem) { return utils::essentiallyEqual(elem, m_Down); });

            // std::print("Input: {:.3f};   Output: {:.5f};    DownCount: {}/{}\n", u, p, c, m_Count * m_Count);

            return 2. * p;
        }

        void Init()
        {
            m_R = Eigen::Matrix2d::Constant(m_Count, m_Count, std::numeric_limits<double>::quiet_NaN());

            m_Indices.reserve(static_cast<std::size_t>(m_Count) * static_cast<std::size_t>(m_Count + 1) / 2);

            for (int i = 0; i < m_Count; i++)
            {
                for (int j = 0; j < m_Count; j++)
                {
                    if (i <= j)
                    {
                        m_Indices.emplace_back(i, j);
                        m_R(i, j) = m_Down;
                    }
                }
            }
        }

        double GetSum() const
        {
            return m_Count * (m_Count + 1) / 2.0;
        }

    private:
        Eigen::Matrix2d m_R;
        std::vector<std::pair<int, int>> m_Indices;
        double m_H;
        int m_Count;
        double m_Up = 1.0, m_Down = -1.0;
    };
}
