// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <functional>
#include <optional>

#include "ModelBase.hpp"
#include "Utils/gauss_area.hpp"

namespace mc
{
    class ArealPreisachModel : public SinglePreisachModel
    {
    public:
        using EverettFunction = std::function<double(double, double)>;

    public:
        ArealPreisachModel(const double L, bool keepDerivative = false, bool keepAnimation = false,
                           bool isDownZero = false) :
            SinglePreisachModel(L, keepDerivative, keepAnimation), m_IsDownZero(isDownZero)
        {
        }

        ArealPreisachModel(const ArealPreisachModel &) = delete;
        ArealPreisachModel &operator=(const ArealPreisachModel &) = delete;

        void SetEverettFunction(EverettFunction func)
        {
            m_EverettFunction = std::move(func);
        }

        virtual double GetMaxArea() const override
        {
            const double lo = m_Bounds[0];
            const double hi = m_Bounds[1];

            if (m_EverettFunction.has_value())
            {
                // Total weighted mass of the Preisach support.
                return m_EverettFunction.value()(lo, hi);
            }

            // Geometric area of beta <= alpha,
            // beta,alpha in [lo,hi].
            const double width = hi - lo;
            return 0.5 * width * width;
        }

    protected:
        virtual double P_Impl(double u) override
        {
            AL_PROFILE_FUNC("ArealPreisachModel::P_Impl");

            if (m_EverettFunction.has_value())
            {
                const double positive = EverettPositiveMass();
                
                /*
                 * For elementary relays {-1,+1}:
                 *
                 * output =
                 *     positiveWeight - negativeWeight
                 *
                 * and
                 *
                 * negativeWeight =
                 *     totalWeight - positiveWeight.
                 *
                 * Therefore
                 *
                 * output = 2*positiveWeight - totalWeight.
                 */
                
                const double result = 2.0 * positive - GetMaxArea();

                m_PreviousArea = positive;
                return result;
            }
            
            auto points = MakePoints();
            const auto O = Eigen::Vector2d(m_Bounds[0], m_Bounds[0]);
            auto res = gauss_area(points, O);
            const double ret = CalcS(res.total_area);
            m_PreviousArea = res.total_area;
            return ret;
        }

        virtual void ResetImpl() override
        {
            SinglePreisachModel::ResetImpl();
            m_PreviousArea = 0.0;
        }

    private:
        double EverettPositiveMass() const
        {
            assert(m_EverettFunction.has_value());

            const auto &E = m_EverettFunction.value();
            const double lo = m_Bounds[0];

            auto eval = [&E](double x, double y) -> ddouble
            {
                // Degenerate Everett triangle.
                if (!(y > x))
                    return 0.0L;

                const double value = E(x, y);

                assert(std::isfinite(value));

                // E must be non-negative.
                // Allow microscopic negative roundoff from a user function.
                return std::max(0.0, value);
            };

            ddouble positive = 0.0L;

            if (m_FirstElemType == ElementType::Max)
            {
                /*
                 * State:
                 *
                 * + E(lo, M0)
                 * - E(m0, M0)
                 * + E(m0, M1)
                 * - E(m1, M1)
                 * + ...
                 */

                double left = lo;

                for (std::size_t i = 0; i < m_InterfaceMax.size(); ++i)
                {
                    const double upper = m_InterfaceMax[i];

                    positive += eval(left, upper);

                    if (i < m_InterfaceMin.size())
                    {
                        const double lower = m_InterfaceMin[i];

                        positive -= eval(lower, upper);
                        left = lower;
                    }
                }
            }
            else
            {
                /*
                 * After lower saturation the first minimum
                 * is the lower-bound sentinel:
                 *
                 * m_InterfaceMin[0] == lo.
                 *
                 * Then:
                 *
                 * + E(lo, M0)
                 * - E(m1, M0)
                 * + E(m1, M1)
                 * ...
                 */

                if (m_InterfaceMin.empty())
                    return 0.0;

                double left = m_InterfaceMin.front();

                std::size_t minIndex = 1;

                for (std::size_t i = 0; i < m_InterfaceMax.size(); ++i)
                {
                    const double upper = m_InterfaceMax[i];

                    positive += eval(left, upper);

                    if (minIndex < m_InterfaceMin.size())
                    {
                        const double lower = m_InterfaceMin[minIndex++];

                        positive -= eval(lower, upper);
                        left = lower;
                    }
                }
            }

            /*
             * Alternating Everett sums may accumulate a tiny
             * negative roundoff near saturation.
             */
            if (positive < 0.0L)
            {
                const ddouble tolerance = 128.0L * std::numeric_limits<double>::epsilon() * std::max(1.0L, std::abs(positive));

                if (positive >= -tolerance)
                    positive = 0.0L;
            }

            assert(positive >= 0.0L);

            return static_cast<double>(positive);
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

            const auto Mf = first
                ? Eigen::Vector2d(m_Bounds[0], m_InterfaceMax[0])
                : Eigen::Vector2d(m_InterfaceMin[0], m_Bounds[1]);
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
                ? Eigen::Vector2d(m_InterfaceMax.back(), m_InterfaceMax.back())
                : Eigen::Vector2d(m_InterfaceMin.back(), m_InterfaceMin.back());
            res.push_back(Ml);

            return res;
        }

    private:
        double m_PreviousArea = 0.0;
        bool m_IsDownZero = false;

        std::optional<EverettFunction> m_EverettFunction = std::nullopt;
    };

    class DoubleArealPreisachModel : public DoublePreisachModel
    {
    public:
        DoubleArealPreisachModel(double L, double d, double k, bool keepDerivative = false) :
            DoublePreisachModel(L, d, k, keepDerivative)
        {
            m_UpperModel = std::make_unique<ArealPreisachModel>(m_L, keepDerivative);
            m_UpperModel->SetBounds({d, 2. * L + d});

            m_LowerModel = std::make_unique<ArealPreisachModel>(m_L, keepDerivative);
            m_LowerModel->SetBounds({-2. * L - d, -d});
        }

        virtual double GetMaxArea() const override
        {
            return m_UpperModel->GetMaxArea() + m_LowerModel->GetMaxArea();
        }
    };
}
