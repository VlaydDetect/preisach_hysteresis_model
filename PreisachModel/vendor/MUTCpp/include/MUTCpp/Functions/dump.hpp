// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <filesystem>
#include <fstream>

namespace mc
{
    template <typename Derived>
    requires std::is_trivially_copyable_v<typename Derived::Scalar>
    void dump(const Eigen::DenseBase<Derived> &input, std::filesystem::path path)
    {
        using T = typename Derived::Scalar;
        if (!path.has_extension()) path.replace_extension("bin");
        std::ofstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Unable to open " + path.string());
        const Eigen::RowVectorX<T> values = detail::rowMajor(input).transpose();
        if (values.size() != 0)
        {
            file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(T));
        }
        file.close();
        if (!file) throw std::runtime_error("Unable to write " + path.string());
    }
}
