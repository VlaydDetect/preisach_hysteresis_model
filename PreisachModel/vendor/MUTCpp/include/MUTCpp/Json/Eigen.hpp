#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <Eigen/Core>
#include "vendor/json/nlohmann_json.hpp"

namespace mc::json
{
    enum class NonFinite
    {
        Reject,
        AsNull
    };

    template <typename Scalar>
        requires std::is_arithmetic_v<Scalar>
    nlohmann::json EncodeNumber(Scalar value, NonFinite non_finite = NonFinite::Reject)
    {
        if constexpr (std::is_floating_point_v<Scalar>)
        {
            if (!std::isfinite(value))
            {
                if (non_finite == NonFinite::AsNull)
                {
                    return nullptr;
                }
                throw std::invalid_argument("Non-finite number in JSON");
            }
        }
        return value;
    }

    template <typename Derived>
    nlohmann::json EncodeVector(const Eigen::DenseBase<Derived> &vector,
                                NonFinite non_finite = NonFinite::Reject)
    {
        if (vector.rows() != 1 && vector.cols() != 1)
        {
            throw std::invalid_argument("Expected a row or column vector");
        }
        const auto &values = vector.eval();
        auto result = nlohmann::json::array();
        const bool row_vector = values.rows() == 1;
        for (Eigen::Index i = 0; i < values.size(); ++i)
        {
            result.push_back(EncodeNumber(values.coeff(row_vector ? 0 : i, row_vector ? i : 0), non_finite));
        }
        return result;
    }

    template <typename Derived>
    nlohmann::json EncodeRows(const Eigen::DenseBase<Derived> &matrix,
                              NonFinite non_finite = NonFinite::Reject)
    {
        const auto &values = matrix.eval();
        auto result = nlohmann::json::array();
        for (Eigen::Index row = 0; row < values.rows(); ++row)
        {
            auto encoded_row = nlohmann::json::array();
            for (Eigen::Index col = 0; col < values.cols(); ++col)
            {
                encoded_row.push_back(EncodeNumber(values.coeff(row, col), non_finite));
            }
            result.push_back(std::move(encoded_row));
        }
        return result;
    }

    template <typename Derived>
    nlohmann::json EncodeMatrix(const Eigen::DenseBase<Derived> &matrix,
                                NonFinite non_finite = NonFinite::Reject)
    {
        return {{"shape", {matrix.rows(), matrix.cols()}}, {"values", EncodeRows(matrix, non_finite)}};
    }

    template <int Rows, int Cols>
    Eigen::Matrix<double, Rows, Cols> DecodeRows(const nlohmann::json &data)
    {
        static_assert(Rows > 0 && Cols > 0);
        if (!data.is_array() || data.size() != Rows)
        {
            throw std::invalid_argument("Unexpected matrix row count in JSON");
        }
        Eigen::Matrix<double, Rows, Cols> result;
        for (int row = 0; row < Rows; ++row)
        {
            const auto &values = data.at(row);
            if (!values.is_array() || values.size() != Cols)
            {
                throw std::invalid_argument("Unexpected matrix column count in JSON");
            }
            for (int col = 0; col < Cols; ++col)
            {
                const double value = values.at(col).get<double>();
                if (!std::isfinite(value))
                {
                    throw std::invalid_argument("Non-finite matrix coefficient in JSON");
                }
                result(row, col) = value;
            }
        }
        return result;
    }
}
