#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

#include "Json/Eigen.hpp"
#include "Json/JsonDocument.hpp"

namespace mc
{
    struct EverettSurfaceData
    {
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        Eigen::MatrixXd z;

        json::JsonDocument toJSON() const
        {
            if (z.rows() != y.size() || z.cols() != x.size())
            {
                throw std::invalid_argument("EverettSurfaceData: z dimensions do not match x/y");
            }
            return nlohmann::json{
                {"type", "everett_surface"},
                {"rows", z.rows()},
                {"cols", z.cols()},
                {"x", json::EncodeVector(x)},
                {"y", json::EncodeVector(y)},
                {"z", json::EncodeVector(z.reshaped<Eigen::RowMajor>(), json::NonFinite::AsNull)}
            };
        }
    };

    class PreisachMeasure
    {
    public:
        virtual ~PreisachMeasure() = default;
        virtual double Density(double lower, double upper) const = 0;
        virtual double TriangleMass(double lower, double upper) const = 0;
        virtual std::array<double, 2> BoundaryDerivatives(double lower, double upper) const = 0;

        virtual EverettSurfaceData GetoMeasureSurfaceData(double lo, double hi, Eigen::Index resolution = 101,
                                                           double scale = 1.0) const
        {
            ValidateSurfaceParameters(lo, hi, resolution, scale);
            EverettSurfaceData data;
            data.x = Eigen::VectorXd::LinSpaced(resolution, lo, hi);
            data.y = data.x;
            data.z.resize(resolution, resolution);
            const double nan = std::numeric_limits<double>::quiet_NaN();
            for (Eigen::Index row = 0; row < resolution; ++row)
            {
                for (Eigen::Index col = 0; col < resolution; ++col)
                {
                    data.z(row, col) = col <= row ? scale * Density(data.x(col), data.y(row)) : nan;
                }
            }
            return data;
        }

    protected:
        static void ValidateSurfaceParameters(double lo, double hi, Eigen::Index resolution, double scale)
        {
            if (resolution < 2 || !std::isfinite(lo) || !std::isfinite(hi) || !(lo < hi) ||
                !std::isfinite(scale) || scale < 0.0)
            {
                throw std::invalid_argument("Invalid Preisach surface parameters");
            }
        }
    };

    class UniformPreisachMeasure final : public PreisachMeasure
    {
    public:
        double Density(double lower, double upper) const override
        {
            return lower <= upper ? 1.0 : 0.0;
        }

        double TriangleMass(double lower, double upper) const override
        {
            const double width = upper > lower ? upper - lower : 0.0;
            return 0.5 * width * width;
        }

        std::array<double, 2> BoundaryDerivatives(double lower, double upper) const override
        {
            const double width = upper > lower ? upper - lower : 0.0;
            return {-width, width};
        }
    };
}
