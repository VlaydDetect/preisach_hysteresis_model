// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Core>
#include "Core/Enums.hpp"
#include <stdexcept>

namespace mc
{
    template <typename Derived>
    Eigen::RowVectorXd centerOfMass(const Eigen::DenseBase<Derived> &input, Axis axis = Axis::NONE)
    {
        const auto values = input.derived().matrix().template cast<double>();
        const auto rows = Eigen::VectorXd::LinSpaced(input.rows(), 0.0, static_cast<double>(input.rows() - 1));
        const auto cols = Eigen::VectorXd::LinSpaced(input.cols(), 0.0, static_cast<double>(input.cols() - 1));
        if (axis == Axis::ROW)
        {
            return (rows.transpose() * values).array() / values.colwise().sum().array();
        }
        if (axis == Axis::COL)
        {
            return (values * cols).transpose().array() / values.rowwise().sum().transpose().array();
        }
        if (axis != Axis::NONE) throw std::invalid_argument("Unknown axis");
        const double mass = values.sum();
        return Eigen::RowVector2d(rows.dot(values.rowwise().sum()) / mass, cols.dot(values.colwise().sum().transpose()) / mass);
    }
}
