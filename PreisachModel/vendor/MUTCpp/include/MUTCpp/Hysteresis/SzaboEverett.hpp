// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

#include "Core/Constants.hpp"
#include "Core/Error.hpp"
#include "Functions/expm1.hpp"
#include "Utils/essentiallyEqual.hpp"
#include "Core/Types.hpp"
#include "Json/JsonDocument.hpp"

namespace mc
{
    // See: https://www.researchgate.net/publication/279530104_Preisach_Type_Hysteresis_Models_with_Everett_Function_in_Closed_Form
    class SzaboEverett
    {
    private:
        struct Component
        {
            ddouble a = 0.0L;
            ddouble b = 0.0L;
            ddouble c = 0.0L;

            ddouble inv_c = 1.0L;
            ddouble a2c2 = 0.0L;

            // r = b / c
            ddouble r = 0.0L;

            // k = exp(-2 |r|)
            ddouble k = 1.0L;

            // s = 1 - k, computed accurately even for r -> 0
            ddouble s = 0.0L;
        };

        struct EverettSurfaceData
        {
            Eigen::VectorXd x; // coordinates along x axis
            Eigen::VectorXd y; // coordinates along y axis

            // z(row, col) = E(x[col], y[row])
            //
            // Region x > y is outside the Preisach triangle
            // and is filled with NaN.
            Eigen::MatrixXd z;

            mc::json::JsonDocument toJSON()
            {
                // mc::json::JsonDocument doc({"x", "y", "E"});
                //  doc.AddField("x", x);
                //  doc.AddField("y", y);
                //  doc.AddField("E", z);
                //  return doc;

                if (this->z.rows() != this->y.size() ||
                    this->z.cols() != this->x.size())
                {
                    throw std::invalid_argument(
                        "EverettSurfaceData: z dimensions do not match x/y");
                }

                nlohmann::json x = nlohmann::json::array();
                nlohmann::json y = nlohmann::json::array();
                nlohmann::json z = nlohmann::json::array();

                // x
                for (Eigen::Index i = 0; i < this->x.size(); ++i)
                {
                    const double value = this->x(i);

                    if (!std::isfinite(value))
                    {
                        throw std::invalid_argument(
                            "EverettSurfaceData: x contains non-finite value");
                    }

                    x.push_back(value);
                }

                // y
                for (Eigen::Index i = 0; i < this->y.size(); ++i)
                {
                    const double value = this->y(i);

                    if (!std::isfinite(value))
                    {
                        throw std::invalid_argument(
                            "EverettSurfaceData: y contains non-finite value");
                    }

                    y.push_back(value);
                }

                /*
                 * Flatten z in row-major logical order:
                 *
                 * z[row * cols + col] = this->z(row, col)
                 *
                 * Note:
                 * Eigen::MatrixXd is column-major internally by default,
                 * but that does not matter because we index explicitly.
                 */
                for (Eigen::Index row = 0; row < this->z.rows(); ++row)
                {
                    for (Eigen::Index col = 0; col < this->z.cols(); ++col)
                    {
                        const double value = this->z(row, col);

                        if (std::isfinite(value))
                        {
                            z.push_back(value);
                        }
                        else
                        {
                            // x > y region -> JSON null
                            z.push_back(nullptr);
                        }
                    }
                }

                nlohmann::json result = nlohmann::json::object();

                result["type"] = "everett_surface";

                result["rows"] =
                    static_cast<std::int64_t>(this->z.rows());

                result["cols"] =
                    static_cast<std::int64_t>(this->z.cols());

                result["x"] = std::move(x);
                result["y"] = std::move(y);
                result["z"] = std::move(z);

                return result;
            }
        };

    public:
        template <typename DerivedA, typename DerivedB, typename DerivedC>
        SzaboEverett(const Eigen::DenseBase<DerivedA> &a,
                     const Eigen::DenseBase<DerivedB> &b,
                     const Eigen::DenseBase<DerivedC> &c)
        {
            if (a.size() != b.size() || a.size() != c.size())
            {
                THROW_INVALID_ARGUMENT_ERROR("SzaboEverett: a, b and c must have the same size");
            }

            m_Components.reserve(static_cast<std::size_t>(a.size()));

            for (Eigen::Index i = 0; i < a.size(); ++i)
            {
                const ddouble ai = static_cast<ddouble>(a.derived().coeff(i));
                const ddouble bi = static_cast<ddouble>(b.derived().coeff(i));
                ddouble ci = static_cast<ddouble>(c.derived().coeff(i));

                if (std::isinf(ai) || std::isinf(bi) || std::isinf(ci))
                {
                    THROW_INVALID_ARGUMENT_ERROR("SzaboEverett: parameters must be finite");
                }

                if (mc::utils::essentiallyEqual(ci, 0.0L))
                {
                    THROW_INVALID_ARGUMENT_ERROR("SzaboEverett: c must be non-zero");
                }

                // phi written through cosh is insensitive to sign(c).
                ci = std::abs(ci);

                Component p;
                p.a = ai;
                p.b = bi;
                p.c = ci;
                p.inv_c = 1.0L / ci;
                p.a2c2 = (ai * ci) * (ai * ci);

                p.r = bi / ci;

                const ddouble abs_r = std::abs(p.r);

                p.k = std::exp(-2.0L * abs_r);

                // 1 - exp(-2 |r|), without loss near r = 0.
                p.s = -mc::expm1(-2.0L * abs_r);

                m_Components.push_back(p);
            }
        }

        [[nodiscard]] double operator()(double x, double y) const noexcept
        {
            // Everett triangle is defined for x <= y.
            assert(x <= y);

            if (!(y > x))
                return 0.0;

            ddouble result = 0.0L;

            for (const Component &p : m_Components)
            {
                result += EvaluateComponent(static_cast<ddouble>(x), static_cast<ddouble>(y), p);
            }

            // Individual components are non-negative.
            // Remove only tiny negative round-off.
            if (result < 0.0L)
                result = 0.0L;

            return static_cast<double>(result);
        }

        [[nodiscard]] EverettSurfaceData GetoOutputSurfaceData(double low, double high,
                                                                std::size_t samples = 101) const
        {
            if (samples < 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("GetSurfaceData: samples must be >= 2");
            }

            if (!(high > low))
            {
                THROW_INVALID_ARGUMENT_ERROR("GetSurfaceData: invalid Preisach bounds");
            }

            EverettSurfaceData result;

            result.x.resize(static_cast<Eigen::Index>(samples));

            result.y.resize(static_cast<Eigen::Index>(samples));

            result.z.resize(static_cast<Eigen::Index>(samples), static_cast<Eigen::Index>(samples));

            result.z.setConstant(mc::consts::nan);

            /*
             * Use multiplication rather than repeatedly adding dx.
             * This guarantees that the last point is exactly hi.
             */
            for (std::size_t i = 0; i < samples; ++i)
            {
                const double t = static_cast<double>(i) / static_cast<double>(samples - 1);

                const double value = std::lerp(low, high, t);

                result.x[static_cast<Eigen::Index>(i)] = value;
                result.y[static_cast<Eigen::Index>(i)] = value;
            }

            /*
             * Only the Preisach triangle x <= y has to be evaluated.
             *
             * Therefore:
             *
             *      y
             *      ^
             *      | ********
             *      |  *******
             *      |   ******
             *      |    *****
             *      |     ****
             *      +----------> x
             *
             * Number of evaluations:
             *
             *      N(N+1)/2
             *
             * rather than N^2.
             */
            for (std::size_t row = 0; row < samples; ++row)
            {
                const double y = result.y[static_cast<Eigen::Index>(row)];

                for (std::size_t col = 0; col <= row; ++col)
                {
                    const double x = result.x[static_cast<Eigen::Index>(col)];

                    double value = this->operator()(x, y);

                    if (std::isinf(value))
                    {
                        THROW_RUNTIME_ERROR("GetSurfaceData: Everett function returned a non-finite value");
                    }

                    /*
                     * Exact Everett properties:
                     *
                     * E(x,x) = 0
                     * E(x,y) >= 0
                     *
                     * Avoid displaying microscopic negative values
                     * produced only by roundoff.
                     */
                    if (row == col)
                    {
                        value = 0.0;
                    }
                    else if (value < 0.0)
                    {
                        const double scale = std::max(1.0, std::abs(value));

                        const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() * scale;

                        if (value >= -tolerance)
                        {
                            value = 0.0;
                        }
                    }

                    result.z(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = value;
                }
            }

            return result;
        }

        [[nodiscard]] EverettSurfaceData GetoMeasureSurfaceData(double minBound, double maxBound, Eigen::Index resolution = 101,
                                                                double densityScale = 1.0) const
        {
            if (resolution < 2 || !(minBound < maxBound))
                throw std::invalid_argument("Invalid surface parameters");

            if (!std::isfinite(densityScale) || densityScale < 0.0)
                throw std::invalid_argument("Invalid density scale");

            EverettSurfaceData data;

            // X = beta (нижний порог)
            // Y = alpha (верхний порог)
            data.x = Eigen::VectorXd::LinSpaced(
                resolution, minBound, maxBound);

            data.y = data.x;

            data.z = Eigen::MatrixXd::Zero(
                resolution, resolution);

            /*
             * mu(beta, alpha) =
             *     sum_i phi_i(beta) * phi_i(-alpha).
             *
             * Используем сепарабельность плотности:
             * каждая компонента добавляется как внешнее
             * произведение двух векторов.
             *
             * Это существенно быстрее вычисления exp()
             * отдельно для каждой точки сетки.
             */
            Eigen::VectorXd phiBeta(resolution);
            Eigen::VectorXd phiMinusAlpha(resolution);

            for (const auto &component : m_Components)
            {
                for (Eigen::Index i = 0; i < resolution; ++i)
                {
                    const double h = data.x[i];

                    phiBeta[i] =
                        EvaluatePhi(h, component);

                    phiMinusAlpha[i] =
                        EvaluatePhi(-h, component);
                }

                data.z.noalias() +=
                    phiMinusAlpha * phiBeta.transpose();
            }

            // Если оператор использует нормированную E,
            // для визуализации применяем тот же множитель.
            data.z *= densityScale;

            /*
             * Отсекаем область вне носителя Прейзаха.
             *
             * Matrix(row, col) = mu(x[col], y[row]).
             *
             * Допустимая область: y >= x.
             *
             * На диагонали y == x оставляем настоящие
             * значения плотности, а не нули.
             */
            const double nan =
                std::numeric_limits<double>::quiet_NaN();

            for (Eigen::Index row = 0; row < resolution; ++row)
            {
                for (Eigen::Index col = row + 1;
                     col < resolution; ++col)
                {
                    data.z(row, col) = nan;
                }
            }

            return data;
        }

    private:
        // Численно устойчивая версия
        // phi(t) = (a / 2) / (1 + cosh((t-b)/c)).
        static double EvaluatePhi(
            double t,
            const Component &p) noexcept
        {
            const long double z =
                std::abs((static_cast<long double>(t) - p.b) / p.c);

            const long double e = std::exp(-z);
            const long double denominator = 1.0L + e;

            return static_cast<double>(
                p.a * e / (denominator * denominator)
            );
        }

        [[nodiscard]]
        double EvaluateDensity(double beta, double alpha) const noexcept
        {
            // Носитель: beta <= alpha.
            if (beta > alpha)
                return 0.0;

            long double result = 0.0L;

            for (const auto &component : m_Components)
            {
                result +=
                    static_cast<long double>(
                        EvaluatePhi(beta, component)) *
                    EvaluatePhi(-alpha, component);
            }

            return static_cast<double>(result);
        }

        static ddouble Sigmoid(ddouble x) noexcept
        {
            // Never evaluates exp(+large_number).
            if (x >= 0.0L)
            {
                const ddouble e = std::exp(-x);
                return 1.0L / (1.0L + e);
            }

            const ddouble e = std::exp(x);
            return e / (1.0L + e);
        }

        // sigma(v) - sigma(u), v >= u.
        // More accurate than direct subtraction when u ~= v
        // or both values lie in a saturated tail.
        static ddouble SigmoidDifference(ddouble u, ddouble v) noexcept
        {
            if (!(v > u))
                return 0.0L;

            const ddouble delta = v - u;

            if (delta < 0.5L)
            {
                if (u >= 0.0L)
                {
                    const ddouble eu = std::exp(-u);
                    const ddouble ev = eu * std::exp(-delta);

                    const ddouble numerator = eu * (-mc::expm1(-delta));

                    return numerator / ((1.0L + eu) * (1.0L + ev));
                }

                if (v <= 0.0L)
                {
                    const ddouble eu = std::exp(u);
                    const ddouble ev = eu * std::exp(delta);

                    const ddouble numerator = eu * mc::expm1(delta);

                    return numerator / ((1.0L + eu) * (1.0L + ev));
                }
            }

            return Sigmoid(v) - Sigmoid(u);
        }

        // R(z) = [ z - log(1 + z) ] / z^2
        //
        // Direct evaluation loses precision for z -> 0.
        // R(0) = 1/2.
        static ddouble Log1pRemainderRatio(ddouble z) noexcept
        {
            if (std::abs(z) < 1.0e-4L)
            {
                // z - log(1+z)
                // = z^2/2 - z^3/3 + z^4/4 - ...
                //
                // divide by z^2:
                // R = 1/2 - z/3 + z^2/4 - ...

                ddouble result = 0.0L;
                ddouble power = 1.0L;

                for (int n = 2; n <= 14; ++n)
                {
                    const ddouble sign = (n % 2 == 0) ? 1.0L : -1.0L;

                    result += sign * power / static_cast<ddouble>(n);

                    power *= z;
                }

                return result;
            }

            return (z - std::log1p(z)) / (z * z);
        }

        static ddouble EvaluateComponent(ddouble x, ddouble y, const Component &p) noexcept
        {
            if (p.a2c2 == 0.0L)
                return 0.0L;

            const ddouble ux = (x + p.b) * p.inv_c;

            const ddouble uy = (y + p.b) * p.inv_c;

            const ddouble qx = Sigmoid(ux);
            const ddouble qy = Sigmoid(uy);

            // Compute complements directly rather than 1-q:
            // important when q ~= 1.
            const ddouble qcx = Sigmoid(-ux);
            const ddouble qcy = Sigmoid(-uy);

            const ddouble dq = SigmoidDifference(ux, uy);

            if (!(dq > 0.0L))
                return 0.0L;

            ddouble Dx;
            ddouble Dy;
            ddouble d;

            if (p.r >= 0.0L)
            {
                /*
                 * p(q) = k q / [1 - q + k q]
                 *
                 * D(q) = 1 - q + k q
                 * D'(q) = k - 1 = -s
                 */
                Dx = qcx + p.k * qx;
                Dy = qcy + p.k * qy;
                d = -p.s;
            }
            else
            {
                /*
                 * Equivalent reciprocal representation:
                 *
                 * p(q) = q / [q + k(1-q)]
                 *
                 * D(q) = q + k(1-q)
                 * D'(q) = 1-k = s
                 *
                 * This avoids exp(-2r) overflow for r < 0.
                 */
                Dx = qx + p.k * qcx;
                Dy = qy + p.k * qcy;
                d = p.s;
            }

            if (!(Dx > 0.0L) || !(Dy > 0.0L))
                return 0.0L;

            /*
             * Exact identity:
             *
             * E_i / (a_i^2 c_i^2)
             *
             *   = k/s^2 * [ z - log(1+z) ],
             *
             * z = d * (qy-qx) / Dx.
             *
             * For s -> 0 this has the finite limit
             * 0.5 * (qy-qx)^2.
             */

            if (p.s == 0.0L)
            {
                return p.a2c2 * 0.5L * dq * dq;
            }

            const ddouble z = d * dq / Dx;

            ddouble normalized;

            if (std::abs(z) < 0.125L)
            {
                // Rewrite so there is no s^2 cancellation:
                //
                // k * (dq/Dx)^2 *
                // [z-log1p(z)]/z^2

                const ddouble w = dq / Dx;

                normalized = p.k * w * w * Log1pRemainderRatio(z);
            }
            else
            {
                // Here cancellation is not dangerous.
                // log(Dy/Dx) is equal to log(1+z).
                const ddouble logRatio = std::log(Dy) - std::log(Dx);

                normalized = (p.k / (p.s * p.s)) * (z - logRatio);
            }

            if (normalized < 0.0L)
                normalized = 0.0L;

            return p.a2c2 * normalized;
        }

    private:
        std::vector<Component> m_Components;
    };
}
