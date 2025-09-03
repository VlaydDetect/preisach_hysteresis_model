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
    inline Vec2 interpolate_triangle(const Vec2 &A, const Vec2 &B, const Vec2 &C, double u, double v)
    {
        const double w = 1 - u - v;
        return Vec2(u * A.x + v * B.x + w * C.x, u * A.y + v * B.y + w * C.y);
    };

    constexpr double w = 1. / 3.;
    constexpr std::array<std::pair<double, double>, 3> nodes = {{
        {0.5, 0.0},
        {0.5, 0.5},
        {0.0, 0.5}
    }};

    inline std::vector<std::size_t> iterations(1, 10);

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
                const std::vector points = {Vec2(-1., 1.), Vec2(1., 1.)};
                constexpr Vec2 pivot = Vec2(-1., -1.);

                return integrate::monteCarloTriangle(m_EverettFunction.value(), {-1., -1.}, {-1., 1.}, {1., 1.},
                                                     iterations);

                double sum = 0.0;
                for (const auto &[u, v] : nodes)
                {
                    Vec2 p = interpolate_triangle(pivot, points[0], points[1], u, v);
                    sum += m_EverettFunction.value()(p.x, p.y) * w;
                }
                return sum * gauss_area(points, pivot).total_area;
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
                // uint32 n = points.size();
                // double s = 0.0;
                // for (uint32 i = 0; i < n; i++)
                // {
                //     auto point1 = points[i];
                //     auto point2 = points[(i + 1) % n];
                //     s += point1.x * point2.y - point2.x * point1.y;
                // }
                //
                // const double p = abs(s) / 2;

                const auto O = Vec2(m_Bounds[0], m_Bounds[0]);
                // const auto O = Vec2(0., 0.);
                // points.push_back(Vec2(m_Bounds[0], m_Bounds[0]));
                auto res = gauss_area(points, O);

                const double ret = CalcS(res.total_area);

                m_PreviousArea = res.total_area;
                return ret;
            }

            // return IntegrateEverett(points);
            return IntegrateEverett();
        }

        // struct Point
        // {
        //     double x = 0.0, y = 0.0;
        // };

        double IntegrateEverett() const
        {
            auto fn = [this](
                const Vec2 &point)
                // const hep::mc_point<double> &point)
            {
                AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett::EverettFunction");
                // const double range = 2 * m_L;
                // const double x = range * point.point()[0] - m_L;
                // const double y = range * point.point()[1] - m_L;

                const double value = m_EverettFunction.value()(point.x, point.y);
                return value;

                // if (x > y)
                //     return 0.;
                //
                // const double value = m_EverettFunction.value()(x, y);
                // return value;
            };

            double positive = 0.0;
            {
                auto points = MakePoints();
                const auto pivot = Vec2(m_Bounds[0], m_Bounds[0]);

                auto res = gauss_area(points, pivot);

                algo::for_each(res.areas_n_triangles.begin(), res.areas_n_triangles.end(),
                               [fn, &positive, this](
                               std::tuple<double, std::array<Vec2, 3>> elem)
                               {
                                   const auto &[area, triangle] = elem;

                                   // double sum = 0.0;
                                   // for (const auto &[u, v] : nodes)
                                   // {
                                   //     Vec2 p = interpolate_triangle(triangle[0], triangle[1], triangle[2], u, v);
                                   //     sum += fn(p) * w;
                                   // }
                                   // positive += sum * area;

                                   positive += integrate::monteCarloTriangle(
                                       m_EverettFunction.value(), triangle[0].toMatrix(),
                                       triangle[1].toMatrix(), triangle[2].toMatrix(),
                                       iterations);
                               });


                // const uint32 n = points.size();
                // for (uint32 i = 0; i < n; i++)
                // {
                //     auto point1 = points[i];
                //     auto point2 = points[(i + 1) % n];
                //     const double s = (point1.x * point2.y - point2.x * point1.y) / 2.;
                //
                //     double sum = 0.0;
                //     // for (const auto& [u, v] : nodes)
                //     // {
                //     //     Point p = interpolate_triangle(point1, point2, u, v);
                //     //     sum += fn(p) * w;
                //     // }
                //
                //     std::vector<std::size_t> iterations(1, 10000);
                //     
                //     auto results = hep::vegas(
                //         hep::make_integrand<double>(fn, 2),
                //         iterations
                //     ).results();
                //     
                //     sum = results.back().value();
                //     
                //     sum *= s;
                //     positive += sum;
                // }
            }

            double negative = 0.0;
            {
                auto points = MakePoints();
                points.insert(points.begin(), Vec2(m_Bounds[0], m_Bounds[1]));

                const auto pivot = Vec2(m_Bounds[1], m_Bounds[1]);

                auto res = gauss_area(points, pivot);

                algo::for_each(res.areas_n_triangles.begin(), res.areas_n_triangles.end(),
                               [fn, &negative, this](
                               std::tuple<double, std::array<Vec2, 3>> elem)
                               {
                                   const auto &[area, triangle] = elem;

                                   // double sum = 0.0;
                                   // for (const auto &[u, v] : nodes)
                                   // {
                                   //     Vec2 p = interpolate_triangle(triangle[0], triangle[1], triangle[2], u, v);
                                   //     sum += fn(p) * w;
                                   // }
                                   // negative += sum * area;

                                   negative += integrate::monteCarloTriangle(
                                       m_EverettFunction.value(), triangle[0].toMatrix(),
                                       triangle[1].toMatrix(), triangle[2].toMatrix(),
                                       iterations);
                               });

                // auto points = MakePoints();
                // points.erase(points.begin());
                // points.push_back(Vec2(m_Bounds[1], m_Bounds[1]));
                // points.push_back(Vec2(m_Bounds[0], m_Bounds[1]));
                // const uint32 n = points.size();
                // for (uint32 i = 0; i < n; i++)
                // {
                //     auto point1 = points[i];
                //     auto point2 = points[(i + 1) % n];
                //     const double s = -(point1.x * point2.y - point2.x * point1.y) / 2.;
                //
                //     double sum = 0.0;
                //     // for (const auto& [u, v] : nodes)
                //     // {
                //     //     Point p = interpolate_triangle(point1, point2, u, v);
                //     //     sum += fn(p) * w;
                //     // }
                //
                //     std::vector<std::size_t> iterations(1, 10000);
                //
                //     auto results = hep::vegas(
                //         hep::make_integrand<double>(fn, 2),
                //         iterations
                //         ).results();
                //
                //     sum = results.back().value();
                //
                //     sum *= s;
                //     negative += sum;
                // }
            }

            return positive - negative;
        }

        // double IntegrateEverett(const std::vector<Point> &points) const
        // {
        //     AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett");
        //     // auto check_bounds = [](const Point &point, const std::vector<Point> &bounds)
        //     // {
        //     //     AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett::check_bounds");
        //     //     for (uint32 i = 1; i < bounds.size() - 1; i++)
        //     //     {
        //     //         const auto &[x1, y1] = bounds[i];
        //     //         const auto &[x2, y2] = bounds[i + 1];
        //     //
        //     //         // if (x1 == x2)
        //     //         // {
        //     //         //     if (point.x == x1 && min(y1, y2) <= point.y <= max(y1, y2))
        //     //         //         return 0;
        //     //         // }
        //     //         // else if (y1 == y2)
        //     //         // {
        //     //         //     if (point.y == y1 && min(x1, x2) <= point.x <= max(x1, x2))
        //     //         //         return 0;
        //     //         // }
        //     //         //
        //     //         // if (x1 <= point.x <= x2 && point.y > y1)
        //     //         //     return -1;
        //     //
        //     //         if (utils::essentiallyEqual(x1, x2))
        //     //         {
        //     //             if (min(y1, y2) <= point.y <= max(y1, y2) && utils::essentiallyEqual(point.x, x1))
        //     //                 return 0;
        //     //             if (point.y > y1)
        //     //                 return -1;
        //     //             if (point.y < y1)
        //     //                 return 1;
        //     //         }
        //     //         if (utils::essentiallyEqual(y1, y2))
        //     //         {
        //     //             if (min(x1, x2) <= point.x <= max(x1, x2) && utils::essentiallyEqual(point.y, y1))
        //     //                 return 0;
        //     //             if (point.x > x1)
        //     //                 return -1;
        //     //             if (point.x < x1)
        //     //                 return 1;
        //     //         }
        //     //     }
        //     //
        //     //     return 1;
        //     // };
        //
        //     auto is_upper_that_state = [](const Point &point, const std::vector<Point> &bounds) -> int
        //     {
        //         AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett::check_bounds");
        //         for (uint32 i = 1; i < bounds.size() - 1; i++)
        //         {
        //             const auto &[x1, y1] = bounds[i];
        //             const auto &[x2, y2] = bounds[i + 1];
        //
        //             if (utils::essentiallyEqual(x1, x2))
        //             {
        //                 if (min(y1, y2) <= point.y <= max(y1, y2) && utils::essentiallyEqual(point.x, x1))
        //                     return 0;
        //                 if (point.y > y1)
        //                     return 1;
        //                 if (point.y < y1)
        //                     return 0;
        //             }
        //             if (utils::essentiallyEqual(y1, y2))
        //             {
        //                 if (min(x1, x2) <= point.x <= max(x1, x2) && utils::essentiallyEqual(point.y, y1))
        //                     return 0;
        //                 if (point.x > x1)
        //                     return 1;
        //                 if (point.x < x1)
        //                     return 0;
        //             }
        //         }
        //
        //         return 0;
        //     };
        //
        //     // auto fn = [points, check_bounds, this](
        //     //     const hep::mc_point<double> &point/*, hep::projector<double> &projector*/)
        //     // {
        //     //     AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett::EverettFunction");
        //     //     const double range = 2 * m_L;
        //     //     const double x = range * point.point()[0] - m_L;
        //     //     const double y = range * point.point()[1] - m_L;
        //     //
        //     //     if (x > y)
        //     //         return 0.;
        //     //
        //     //     const auto modifier = check_bounds(Point{x, y}, points);
        //     //     const double value = m_EverettFunction.value()(x, y) * modifier;
        //     //     // projector.add(0, x, y, value);
        //     //     return value;
        //     // };
        //
        //     auto fn1 = [points, is_upper_that_state, this](
        //         const hep::mc_point<double> &point/*, hep::projector<double> &projector*/)
        //     {
        //         AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett::EverettFunction");
        //         const double range = 2 * m_L;
        //         const double x = range * point.point()[0] - m_L;
        //         const double y = range * point.point()[1] - m_L;
        //
        //         if (x > y)
        //             return 0.;
        //
        //         const auto modifier = 1 - is_upper_that_state(Point{x, y}, points);
        //         const double value = m_EverettFunction.value()(x, y) * static_cast<double>(modifier);
        //         // projector.add(0, x, y, value);
        //         return value;
        //     };
        //
        //     auto fn2 = [points, is_upper_that_state, this](
        //         const hep::mc_point<double> &point/*, hep::projector<double> &projector*/)
        //     {
        //         AL_PROFILE_FUNC("ArealPreisachModel::IntegrateEverett::EverettFunction");
        //         const double range = 2 * m_L;
        //         const double x = range * point.point()[0] - m_L;
        //         const double y = range * point.point()[1] - m_L;
        //
        //         if (x > y)
        //             return 0.;
        //
        //         const auto modifier = is_upper_that_state(Point{x, y}, points);
        //         const double value = m_EverettFunction.value()(x, y) * static_cast<double>(modifier);
        //         // projector.add(0, x, y, value);
        //         return value;
        //     };
        //
        //     double result1;
        //     double result2;
        //
        //     {
        //         AL_PROFILE_SCOPE("ArealPreisachModel::IntegrateEverett::integration");
        //         // auto integrand = hep::make_integrand<double>(
        //         // fn, 2, hep::distribution_parameters{50, 50, -m_L, m_L, -m_L, m_L, "PreisachPlain"});
        //         // const auto chkpt = hep::plain(integrand, std::vector<std::size_t>(1, 200));
        //         // result = chkpt.results().back().value();
        //
        //         std::vector<std::size_t> iterations(1, 10000);
        //
        //         auto results = hep::vegas(
        //             hep::make_integrand<double>(fn1, 2),
        //             iterations
        //         ).results();
        //
        //         if (iterations.size() == 1)
        //         {
        //             result1 = results.back().value();
        //         }
        //         else
        //         {
        //             result1 = hep::accumulate<hep::weighted_with_variance>(results.begin() + 1, results.end()).value();
        //         }
        //     }
        //
        //     {
        //         AL_PROFILE_SCOPE("ArealPreisachModel::IntegrateEverett::integration");
        //         // auto integrand = hep::make_integrand<double>(
        //         // fn, 2, hep::distribution_parameters{50, 50, -m_L, m_L, -m_L, m_L, "PreisachPlain"});
        //         // const auto chkpt = hep::plain(integrand, std::vector<std::size_t>(1, 200));
        //         // result = chkpt.results().back().value();
        //
        //         std::vector<std::size_t> iterations(1, 10000);
        //
        //         auto results = hep::vegas(
        //             hep::make_integrand<double>(fn2, 2),
        //             iterations
        //         ).results();
        //
        //         if (iterations.size() == 1)
        //         {
        //             result2 = results.back().value();
        //         }
        //         else
        //         {
        //             result2 = hep::accumulate<hep::weighted_with_variance>(results.begin() + 1, results.end()).value();
        //         }
        //     }
        //     
        //     // return result * 2 * m_L * m_L;
        //     return (result1 - result2) * 2 * m_L * m_L;
        // }

        double CalcS(double p) const
        {
            if (m_IsDownZero)
            {
                return p - m_PreviousArea;
            }
            // return (2 * p - GetMaxArea()) / GetMaxArea();
            return 2 * p - GetMaxArea();
        }

        std::vector<Vec2> MakePoints() const
        {
            AL_PROFILE_FUNC("ArealPreisachModel::MakePoints");
            std::vector<Vec2> res;
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
            const auto Mf = first ? Vec2(m_Bounds[0], m_InterfaceMax[0]) : Vec2(m_InterfaceMin[0], m_Bounds[1]);
            // res.push_back(O);
            res.push_back(Mf);

            for (uint32 m = 1; m < k; m++)
            {
                if (m % 2 == 0)
                {
                    const uint32 i = m / 2;
                    const uint32 j = i - 1;
                    const auto Mm = first
                        ? Vec2(m_InterfaceMin[j], m_InterfaceMax[i])
                        : Vec2(m_InterfaceMin[i], m_InterfaceMax[j]);
                    res.push_back(Mm);
                }
                else
                {
                    const uint32 i = std::floor(m / 2);
                    const auto Mm = Vec2(m_InterfaceMin[i], m_InterfaceMax[i]);
                    res.push_back(Mm);
                }
            }

            const auto Ml = last
                // ? Point(m_InterfaceMax[-1], m_InterfaceMax[-1])
                // : Point(m_InterfaceMin[-1], m_InterfaceMin[-1]);
                ? Vec2(m_InterfaceMax.back(), m_InterfaceMax.back())
                : Vec2(m_InterfaceMin.back(), m_InterfaceMin.back());
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
