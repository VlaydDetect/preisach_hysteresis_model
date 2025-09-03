// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <Eigen/Dense>

namespace Eigen
{
    inline Eigen::VectorXd make_vector(std::initializer_list<double> init)
    {
        Eigen::VectorXd v(static_cast<Eigen::Index>(init.size()));
        Eigen::Index i = 0;
        for (const double x : init)
            v[i++] = x;
        return v;
    }

    inline Eigen::VectorXd from_std(const std::vector<double> &src)
    {
        Eigen::VectorXd v(static_cast<Eigen::Index>(src.size()));
        for (Eigen::Index i = 0; i < v.size(); ++i)
            v[i] = src[static_cast<size_t>(i)];
        return v;
    }

    inline std::vector<double> to_std(const Eigen::VectorXd &v)
    {
        std::vector<double> out(static_cast<size_t>(v.size()));
        for (Eigen::Index i = 0; i < v.size(); ++i)
            out[static_cast<size_t>(i)] = v[i];
        return out;
    }

    inline Eigen::VectorXd concat(const Eigen::VectorXd &a, const Eigen::VectorXd &b)
    {
        Eigen::VectorXd out(a.size() + b.size());
        out.head(a.size()) = a;
        out.tail(b.size()) = b;
        return out;
    }
}
