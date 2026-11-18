// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/dump.hpp"

namespace mc
{
    template <typename Derived>
    void tofile(const Eigen::DenseBase<Derived> &input, const std::filesystem::path &path)
    {
        dump(input, path);
    }

    template <typename Derived>
    void tofile(const Eigen::DenseBase<Derived> &input, std::filesystem::path path, char separator)
    {
        if (!path.has_extension()) path.replace_extension("txt");
        std::ofstream file(path);
        if (!file) throw std::runtime_error("Unable to open " + path.string());
        const auto values = detail::rowMajor(input);
        for (Eigen::Index i = 0; i < values.size(); ++i)
        {
            if (i != 0) file << separator;
            file << values(i);
        }
        file.close();
        if (!file) throw std::runtime_error("Unable to write " + path.string());
    }
}
