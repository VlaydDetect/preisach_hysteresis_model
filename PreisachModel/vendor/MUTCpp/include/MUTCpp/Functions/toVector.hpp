// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <vector>

#include "split.hpp"
#include "range.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Write an array to an STL vector of vectors
    /// @return std::vector
    template <typename dtype>
    [[nodiscard]] std::vector<std::vector<dtype>> toVector(Matrix<dtype> inArray, Axis inAxis = Axis::ROW)
    {
        if (inArray.isempty())
        {
            return {};
        }
        
        switch (inAxis)
        {
        case Axis::ROW:
        {
            std::vector<Matrix<dtype>> splitted = split(inArray, range(1,inArray.numRows()), Axis::ROW);
            std::vector<std::vector<dtype>> res(inArray.numRows());
            algo::transform(splitted.begin(), splitted.end(), res.begin(),
                            [](auto &elem) { return elem.toFlattenVector(); });
            return res;
        }

        case Axis::COL:
        {
            std::vector<Matrix<dtype>> splitted = split(inArray, range(1, inArray.numCols()), Axis::COL);
            std::vector<std::vector<dtype>> res(inArray.numCols());
            algo::transform(splitted.begin(), splitted.end(), res.begin(),
                            [](auto &elem) { return elem.toFlattenVector(); });
            return res;
        }

        case Axis::NONE:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
