// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "ModelBase.hpp"
#include "MUTCpp/Integrate.hpp"
#include "Utils/gauss_area.hpp"
#include "hep/mc.hpp"

namespace mc
{
    class ArealPreisachModel : public SinglePreisachModel
    {
    public:
        ArealPreisachModel(const double L, bool keepDerivative = false, bool keepAnimation = false,
                           bool isDownZero = false) :
            SinglePreisachModel(L, keepDerivative, keepAnimation), m_IsDownZero(isDownZero)
        {
        }

        ArealPreisachModel(const ArealPreisachModel &) = delete;
        ArealPreisachModel &operator=(const ArealPreisachModel &) = delete;

        void SetEverettFunction(std::function<double(double, double)> func)
        {
            m_EverettFunction = func;
        }

        virtual double GetMaxArea() const override
        {
            if (m_EverettFunction.has_value())
            {
                return integrate::monteCarloTriangle(m_EverettFunction.value(), {-1., -1.}, {-1., 1.}, {1., 1.});
            }
            return 2 * utils::power(m_L, 2);
        }

    private:
        virtual double P_Impl(double u) override
        {
            AL_PROFILE_FUNC("ArealPreisachModel::P_Impl");

            auto points = MakePoints();

            if (!m_EverettFunction.has_value())
            {
                const auto O = Eigen::Vector2d(m_Bounds[0], m_Bounds[0]);
                auto res = gauss_area(points, O);
                const double ret = CalcS(res.total_area);
                m_PreviousArea = res.total_area;
                return ret;
            }

            return IntegrateEverett();
        }

        double IntegrateEverett() const
        {
            double positive = 0.0;
            {
                auto points = MakePoints();
                const auto pivot = Eigen::Vector2d(m_Bounds[0], m_Bounds[0]);

                auto res = gauss_area(points, pivot);

                algo::for_each(res.areas_n_triangles.begin(), res.areas_n_triangles.end(),
                               [&positive, this](
                               std::tuple<double, std::array<Eigen::Vector2d, 3>> elem)
                               {
                                   const auto &[area, triangle] = elem;

                                   positive += integrate::monteCarloTriangle(
                                       m_EverettFunction.value(), triangle);
                               });
            }

            double negative = 0.0;
            {
                auto points = MakePoints();
                points.insert(points.begin(), Eigen::Vector2d(m_Bounds[0], m_Bounds[1]));

                const auto pivot = Eigen::Vector2d(m_Bounds[1], m_Bounds[1]);

                auto res = gauss_area(points, pivot);

                algo::for_each(res.areas_n_triangles.begin(), res.areas_n_triangles.end(),
                               [&negative, this](
                               std::tuple<double, std::array<Eigen::Vector2d, 3>> elem)
                               {
                                   const auto &[area, triangle] = elem;
                                   negative += integrate::monteCarloTriangle(
                                       m_EverettFunction.value(), triangle);
                               });
            }

            return positive - negative;
        }

        double CalcS(double p) const
        {
            if (m_IsDownZero)
            {
                return p - m_PreviousArea;
            }
            return 2 * p - GetMaxArea();
        }

        std::vector<Eigen::Vector2d> MakePoints() const
        {
            AL_PROFILE_FUNC("ArealPreisachModel::MakePoints");
            std::vector<Eigen::Vector2d> res;
            const bool first = m_FirstElemType == ElementType::Max;
            const bool last = m_LastElemType == ElementType::Max;

            const uint32 m1 = m_InterfaceMax.size();
            const uint32 m2 = m_InterfaceMin.size();
            const uint32 n = std::min(m1, m2);
            uint32 k = 2 * n + 1;

            if (m1 > 0 && m2 > 0 && m_FirstElemType != m_LastElemType)
            {
                k--;
            }

            // const auto O = Point(-m_L, -m_L);
            // const auto O = Point(m_Bounds[0], m_Bounds[0]);
            const auto Mf = first
                ? Eigen::Vector2d(m_Bounds[0], m_InterfaceMax[0])
                : Eigen::Vector2d(m_InterfaceMin[0], m_Bounds[1]);
            // res.push_back(O);
            res.push_back(Mf);

            for (uint32 m = 1; m < k; m++)
            {
                if (m % 2 == 0)
                {
                    const uint32 i = m / 2;
                    const uint32 j = i - 1;
                    const auto Mm = first
                        ? Eigen::Vector2d(m_InterfaceMin[j], m_InterfaceMax[i])
                        : Eigen::Vector2d(m_InterfaceMin[i], m_InterfaceMax[j]);
                    res.push_back(Mm);
                }
                else
                {
                    const uint32 i = std::floor(m / 2);
                    const auto Mm = Eigen::Vector2d(m_InterfaceMin[i], m_InterfaceMax[i]);
                    res.push_back(Mm);
                }
            }

            const auto Ml = last
                // ? Point(m_InterfaceMax[-1], m_InterfaceMax[-1])
                // : Point(m_InterfaceMin[-1], m_InterfaceMin[-1]);
                ? Eigen::Vector2d(m_InterfaceMax.back(), m_InterfaceMax.back())
                : Eigen::Vector2d(m_InterfaceMin.back(), m_InterfaceMin.back());
            res.push_back(Ml);

            return res;
        }

    private:
        double m_PreviousArea = 0.0;
        bool m_IsDownZero = false;
        Matrix<double> m_D = {0.0, 0.0};

        Matrix<double> m_V1 = {-1., -1.}, m_V2 = {1., 1.}, m_V3 = {-1., 1.};

        std::optional<std::function<double(double, double)>> m_EverettFunction = std::nullopt;
    };

    class DoubleArealPreisachModel : public DoublePreisachModel
    {
    public:
        DoubleArealPreisachModel(double L, double d) :
            DoublePreisachModel(L, d)
        {
            m_UpperModel = Ref<ArealPreisachModel>::Create(m_L);
            m_UpperModel->SetD(m_D[0]);

            m_LowerModel = Ref<ArealPreisachModel>::Create(m_L);
            m_LowerModel->SetD(m_D[1]);
        }

        DoubleArealPreisachModel(double L, const std::array<double, 2> &d) :
            DoublePreisachModel(L, d)
        {
            m_UpperModel = Ref<ArealPreisachModel>::Create(m_L);
            m_UpperModel->SetD(m_D[0]);

            m_LowerModel = Ref<ArealPreisachModel>::Create(m_L);
            m_LowerModel->SetD(m_D[1]);
        }
    };
}
