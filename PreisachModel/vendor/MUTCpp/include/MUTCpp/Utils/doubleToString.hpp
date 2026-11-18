// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

#include <Eigen/Core>

namespace mc
{
    inline std::string doubleToString(double value, int precision)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(precision) << value;
        return ss.str();
    }

    template <typename Derived>
        requires std::is_same_v<typename Derived::Scalar, double>
    std::string doubleToString(const Eigen::DenseBase<Derived> &value, int precision)
    {
        const auto flat = value.derived().template reshaped<Eigen::RowMajor>();
        std::string ret = "{";
        for (Eigen::Index i = 0; i < flat.size(); ++i)
        {
            if (i != 0)
            {
                ret += ", ";
            }
            ret += doubleToString(flat[i], precision);
        }
        ret += "}";

        return ret;
    }
}
