// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Functions/searchsorted.hpp"
#include "Functions/unique.hpp"
#include "Core/Types.hpp"

namespace mc
{
    template <typename Left, typename Right>
    Eigen::RowVectorX<uint32> digitize(const Eigen::DenseBase<Left> &left, const Eigen::DenseBase<Right> &right)
    {
        return searchsorted(unique(right), left, Side::RIGHT).template cast<uint32>();
    }
}
