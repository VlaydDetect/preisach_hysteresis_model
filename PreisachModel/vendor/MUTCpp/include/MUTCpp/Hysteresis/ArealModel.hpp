// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "ModelBase.hpp"
#include "MUTCpp/Integrate.hpp"
#include "hep/mc.hpp"

namespace mc
{
    class ArealPreisachModel : public PreisachModelBase
    {
    public:
        ArealPreisachModel(const double L, bool keepDerivative = false, bool keepAnimation = false,
                           bool isDownZero = false) :
            PreisachModelBase(L, keepDerivative, keepAnimation), m_IsDownZero(isDownZero)
        {
        }

        ArealPreisachModel(const ArealPreisachModel &) = delete;
        ArealPreisachModel &operator=(const ArealPreisachModel &) = delete;

        void SetEverettFunction(std::function<double(double, double)> func)
        {
            m_EverettFunction = func;
        }

    private:
        virtual double P_Impl(double u) override
        {
            AL_PROFILE_FUNC("ArealPreisachModel::P_Impl");

            auto points = MakePoints();

            if (!m_EverettFunction.has_value())
            {
                uint32 n = points.size();
                double s = 0.0;
                for (uint32 i = 0; i < n; i++)
                {
                    auto point1 = points[i];
                    auto point2 = points[(i + 1) % n];
                    s += point1.x * point2.y - point2.x * point1.y;
                }

                const double p = abs(s) / 2;

                const double ret = CalcS(p);

                m_PreviousArea = p;
                return ret;
            }

            return IntegrateEverett(points);
        }

        struct Point
        {
            double x = 0.0, y = 0.0;
        };

        double IntegrateEverett(const std::vector<Point> &points) const
        {
            AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett");
            auto check_bounds = [](const Point &point, const std::vector<Point> &bounds)
            {
                AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett::check_bounds");
                for (uint32 i = 0; i < bounds.size() - 1; i++)
                {
                    const auto &[x1, y1] = bounds[i];
                    const auto &[x2, y2] = bounds[i + 1];
            
                    if (x1 == x2)
                    {
                        if (point.x == x1 && min(y1, y2) <= point.y <= max(y1, y2))
                            return 0;
                    }
                    else if (y1 == y2)
                    {
                        if (point.y == y1 && min(x1, x2) <= point.x <= max(x1, x2))
                            return 0;
                    }
            
                    if (x1 <= point.x <= x2 && point.y > y1)
                        return -1;
                }
            
                return 1;
            };
            //
            // auto func = [points, check_bounds, this](double x, double y)
            // {
            //     const auto modifier = check_bounds(Point{x, y}, points);
            //     return m_EverettFunction.value()(x, y) * modifier;
            // };
            //
            // const Matrix v1 = {-1., -1.};
            // const Matrix v2 = {1., 1.};
            // const Matrix v3 = {-1., 1.};
            //
            // return integrate::gaussianQuadratureTriangle(func, v1, v2, v3);
            // // return integrate::monteCarloTriangle(func, v1, v2, v3, 1000, .001, 1);


            // double sum = 0.0;
            // for (uint32 i = 1; i < points.size(); i++)
            // {
            //     double Mk = points[i].y;
            //     double mk = points[i].x;
            //     double mkOld = points[i - 1].x;
            //     sum += m_EverettFunction.value()(mkOld, Mk) - m_EverettFunction.value()(mk, Mk);
            // }
            // return -m_EverettFunction.value()(-m_L, m_L) + 2. * sum;

            auto fn = [points, check_bounds, this](const hep::mc_point<double> &point, hep::projector<double>& projector)
            {
                AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett::EverettFunction");
                const double range = 2 * m_L;
                const double x = range * point.point()[0] - m_L;
                const double y = range * point.point()[1] - m_L;
                const auto modifier = check_bounds(Point{x, y}, points);
                const double value = m_EverettFunction.value()(x, y) * modifier;
                projector.add(0, x, y, value);
                return value;
            };

            double result;
            
            {
                AL_PROFILE_SCOPE("ArealPreisachModel::IntegrateEverett::integration");
                auto integrand = hep::make_integrand<double>(
                   fn, 2, hep::distribution_parameters<double>{50, 50, -m_L, m_L, -m_L, m_L, ""});
                const auto chkpt = hep::plain(integrand, std::vector<std::size_t>(1, 200));
                result = chkpt.results().back().value();
            }
            return result;
        }

        [[nodiscard]] double GetMaxArea() const
        {
            return 2 * std::pow(m_L, 2);
        }

        double CalcS(double p) const
        {
            if (m_IsDownZero)
            {
                return p - m_PreviousArea;
            }
            return (2 * p - GetMaxArea()) / GetMaxArea();
            // return 2 * p - GetMaxArea();
        }

        std::vector<Point> MakePoints() const
        {
            AL_PROFILE_FUNC("ArealPreisachModel::MakePoints");
            std::vector<Point> res;
            const bool first = m_FirstElemType == ElementType::Max;
            const bool last = m_LastElemType == ElementType::Max;

            const uint32 m1 = m_InterfaceMax.size();
            const uint32 m2 = m_InterfaceMin.size();
            const uint32 n = min(m1, m2);
            uint32 k = 2 * n + 1;

            if (m1 > 0 && m2 > 0 && m_FirstElemType != m_LastElemType)
            {
                k--;
            }

            const auto O = Point(-m_L, -m_L);
            const auto Mf = first ? Point(-m_L, m_InterfaceMax[0]) : Point(m_InterfaceMin[0], m_L);
            res.push_back(O);
            res.push_back(Mf);

            for (uint32 m = 1; m < k; m++)
            {
                if (m % 2 == 0)
                {
                    const uint32 i = m / 2;
                    const uint32 j = i - 1;
                    const auto Mm = first
                        ? Point(m_InterfaceMin[j], m_InterfaceMax[i])
                        : Point(m_InterfaceMin[i], m_InterfaceMax[j]);
                    res.push_back(Mm);
                }
                else
                {
                    const uint32 i = std::floor(m / 2);
                    const auto Mm = Point(m_InterfaceMin[i], m_InterfaceMax[i]);
                    res.push_back(Mm);
                }
            }

            const auto Ml = last
                // ? Point(m_InterfaceMax[-1], m_InterfaceMax[-1])
                // : Point(m_InterfaceMin[-1], m_InterfaceMin[-1]);
                ? Point(m_InterfaceMax.back(), m_InterfaceMax.back())
                : Point(m_InterfaceMin.back(), m_InterfaceMin.back());
            res.push_back(Ml);

            // algo::for_each(res.begin(), res.end(), [this](Point& p)
            // {
            //     p.x += m_D[0];
            //     p.y += m_D[1];
            // });

            return res;
        }

    private:
        double m_PreviousArea = 0.0;
        bool m_IsDownZero = false;
        Matrix<double> m_D = {0.0, 0.0};

        Matrix<double> m_V1 = {-1., -1.}, m_V2 = {1., 1.}, m_V3 = {-1., 1.};

        std::optional<std::function<double(double, double)>> m_EverettFunction = std::nullopt;
    };
}
