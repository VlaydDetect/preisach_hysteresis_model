// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <numeric>
#include <vector>
#include <span>
#include <Eigen/Dense>

#include "Core/Algorithm.hpp"
#include "Core/Enums.hpp"
#include "Core/Error.hpp"
#include "Utils/doubleToString.hpp"

namespace Eigen
{
    template <typename Scalar>
    Eigen::VectorX<Scalar> from_std(const std::vector<Scalar> &src)
    {
        return Eigen::VectorX<Scalar>::Map(src.data(), src.size());
    }

    template <typename Scalar, int R, int C>
    std::vector<Scalar> to_std(const Eigen::Matrix<Scalar, R, C> &v)
    {
        if (v.rows() > 1 && v.cols() > 1)
        {
            THROW_INVALID_ARGUMENT_ERROR("In Eigen::to_std: excepted vector, but matrix provided");
        }
        std::vector<double> vec(v.data(), v.data() + v.size());
        return vec;
    }

    template <typename Scalar>
    std::span<Scalar> to_span(const Eigen::VectorX<Scalar> &v)
    {
        return std::span(v.data(), v.size());
    }

    template <typename Scalar>
    Eigen::VectorX<Scalar> concat(const Eigen::VectorX<Scalar> &a, const Eigen::VectorX<Scalar> &b)
    {
        Eigen::VectorX<Scalar> out(a.size() + b.size());
        out.head(a.size()) = a;
        out.tail(b.size()) = b;
        return out;
    }

    template <typename Scalar>
    void appendScalar(Eigen::VectorX<Scalar> &v, double x)
    {
        const Eigen::Index n = v.size();
        v.conservativeResize(n + 1);
        v[n] = x;
    }

    template <typename Scalar>
    Eigen::VectorX<Scalar> selectByMask(const Eigen::VectorX<Scalar> &arr, const std::vector<char> &mask)
    {
        assert(static_cast<size_t>(arr.size()) == mask.size());
        size_t cnt = 0;
        for (const char c : mask)
        {
            if (c)
                ++cnt;
        }
        Eigen::VectorXd out(cnt);
        size_t k = 0;
        for (Eigen::Index i = 0; i < arr.size(); ++i)
        {
            if (mask[static_cast<size_t>(i)])
            {
                out[static_cast<Eigen::Index>(k++)] = arr[i];
            }
        }
        return out;
    }

    template <typename Scalar>
    Eigen::VectorX<Scalar> arange(Scalar start, Scalar stop, Scalar step = Scalar(1))
    {
        if (step == Scalar(0))
        {
            THROW_INVALID_ARGUMENT_ERROR("arange: step must be non-zero");
        }

        // Количество элементов (округляем вниз)
        const auto n = static_cast<Eigen::Index>(
            std::floor((stop - start) / step + Scalar(1e-12))
        );

        if (n <= 0)
        {
            return Eigen::VectorX<Scalar>(0);
        }

        Eigen::VectorX<Scalar> result(n);

        // Векторизованная генерация: seq * step + start
        result = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::LinSpaced(n, Scalar(0), Scalar(n - 1));
        result = result.array() * step + start;

        return result;
    }

    template <typename Derived>
    std::vector<std::vector<typename Derived::Scalar>> toVector2D(const Eigen::MatrixBase<Derived> &mat,
                                                                  bool rowMajor = true)
    {
        using Scalar = typename Derived::Scalar;
        std::vector<std::vector<Scalar>> res;

        if (rowMajor)
        {
            res.resize(mat.rows());
            for (Eigen::Index i = 0; i < mat.rows(); ++i)
            {
                res[i].reserve(mat.cols());
                for (Eigen::Index j = 0; j < mat.cols(); ++j)
                {
                    res[i].push_back(mat(i, j));
                }
            }
        }
        else
        {
            res.resize(mat.cols());
            for (Eigen::Index j = 0; j < mat.cols(); ++j)
            {
                res[j].reserve(mat.rows());
                for (Eigen::Index i = 0; i < mat.rows(); ++i)
                {
                    res[j].push_back(mat(i, j));
                }
            }
        }
        return res;
    }

    template <typename Derived>
    bool isFlat(const Eigen::MatrixBase<Derived> &mat)
    {
        return mat.rows() == 1 || mat.cols() == 1;
    }

    /// Returns the indices that would sort a matrix.
    /// @param axis: Row Major - 0, Col Major - 1
    template <typename Scalar, int R, int C>
    Eigen::Matrix<Eigen::Index, R, C> argsort(const Eigen::Matrix<Scalar, R, C> &mat, uint8_t axis = 0)
    {
        if (axis == 0)
        {
            const Eigen::Index rows = mat.rows();
            const Eigen::Index cols = mat.cols();
            Eigen::Matrix<Eigen::Index, R, C> result(rows, cols);
            std::vector<Eigen::Index> idx(cols);

            for (Eigen::Index r = 0; r < rows; ++r)
            {
                std::iota(idx.begin(), idx.end(), 0);
                mc::algo::stable_sort(idx.begin(), idx.end(), [&mat, r](Eigen::Index c1, Eigen::Index c2) noexcept
                {
                    return mat(r, c1) < mat(r, c2);
                });

                result.row(r) = Eigen::Map<Eigen::Matrix<Eigen::Index, 1, Eigen::Dynamic>>(idx.data(), cols);
            }

            return result;
        }
        if (axis == 1)
        {
            Eigen::Matrix<Index, R, C> tmp = argsort(mat.transpose(), 0);
            return tmp.transpose();
        }
        THROW_INVALID_ARGUMENT_ERROR(
            "Unimplemented axis type. If you If you need to sort a vector, choose the overloaded function version for vector.");
        return {}; // get rid of compiler warning
    }

    /// Returns the indices that would sort a vector.
    template <typename Scalar, int R>
    Eigen::Vector<Eigen::Index, R> argsort(const Eigen::Vector<Scalar, R> &v)
    {
        const Eigen::Index n = v.size();
        std::vector<Eigen::Index> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        mc::algo::stable_sort(idx.begin(), idx.end(), [&v](Eigen::Index i1, Eigen::Index i2) noexcept
        {
            return v[i1] < v[i2];
        });
        Eigen::Vector<Eigen::Index, R> result = Eigen::Map<Eigen::Vector<Eigen::Index, R>>(idx.data(), n);
        return result;
    }

    /// Returns the indices that would sort an array.
    template <typename Scalar, int R, int C>
    Eigen::Array<Eigen::Index, R, C> argsort(const Eigen::Array<Scalar, R, C> &v)
    {
        const Eigen::Index n = v.size();
        std::vector<Eigen::Index> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        mc::algo::stable_sort(idx.begin(), idx.end(), [&v](Eigen::Index i1, Eigen::Index i2) noexcept
        {
            return v[i1] < v[i2];
        });
        Eigen::Array<Eigen::Index, R, C> result = Eigen::Map<Eigen::Array<Eigen::Index, R, C>>(idx.data(), n);
        return result;
    }

    template <typename Scalar, int R, int C>
    Eigen::Array<Scalar, R, C> mask(const Eigen::Array<Scalar, R, C> &data, const Eigen::Array<bool, R, C> &mask)
    {
        if (data.rows() != mask.rows())
        {
            THROW_INVALID_ARGUMENT_ERROR("mask must be the same size as data.");
        }

        std::vector<Scalar> selected;
        selected.reserve(data.size());

        for (Eigen::Index i = 0; i < data.rows(); ++i)
        {
            if (mask[i])
            {
                selected.push_back(data[i]);
            }
        }

        return selected.empty()
            ? Eigen::ArrayX<Scalar>()
            : Eigen::Map<Eigen::ArrayX<Scalar>>(selected.data(), selected.size());
    }

    template <typename Derived>
    double logonorm(const Eigen::MatrixBase<Derived> &mat)
    {
        return static_cast<typename Derived::Scalar>(std::log(mat.norm()));
    }

    inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GramSchimidtQR(const Eigen::MatrixXd &A)
    {
        const int n = A.cols();
        if (A.rows() != n)
        {
            THROW_RUNTIME_ERROR("Input matrix should be square.");
        }

        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, n);
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

        for (int i = 0; i < n; ++i)
        {
            Eigen::VectorXd v = A.col(i);

            if (i > 0)
            {
                // R(0..i-1, i) = Q(:,0..i-1)^T * a_i
                R.block(0, i, i, 1).noalias() = Q.leftCols(i).transpose() * v;

                // v = a_i - Q(:,0..i-1) * R(0..i-1, i)
                v.noalias() -= Q.leftCols(i) * R.block(0, i, i, 1);
            }

            R(i, i) = v.norm();
            Q.col(i) = v / R(i, i);
        }

        return {Q, R};
    }

    /// Numbers are uniformly spread through their whole definition range for integer types, and in the [0, 1) range for floating point scalar types.
    inline Eigen::MatrixXd Rand(uint32_t rows, uint32_t cols)
    {
        return (Eigen::MatrixXd::Random(rows, cols).array() + 1.) / 2.;
    }

    inline std::string DoubleVectorToString(const Eigen::VectorXd &value, int precision)
    {
        std::string ret;
        ret += "{";
        for (int i = 0; i < value.size(); i++)
        {
            ret += mc::doubleToString(value[i], precision);

            if (i != value.size() - 1)
            {
                ret += ", ";
            }
        }
        ret += "}";

        return ret;
    }
}
