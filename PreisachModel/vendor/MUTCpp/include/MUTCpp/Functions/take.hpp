// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Error.hpp"
#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Evenly round to the given number of decimals.
    template <typename dtype, typename Indices, type_traits::matrix_int_concept<Indices>  = 0>
    Matrix<dtype> take(const Matrix<dtype> &inArray, const Indices &inIndices, Axis inAxis = Axis::NONE)
    {
        switch (inAxis)
        {
        case Axis::NONE:
        {
            return inArray[inIndices];
        }
        case Axis::ROW:
        {
            return inArray(inIndices, inArray.cSlice());
        }
        case Axis::COL:
        {
            return inArray(inArray.rSlice(), inIndices);
        }
        default:
        {
            THROW_INVALID_ARGUMENT_ERROR("Unimplemented axis type.");
            return {}; // get rid of compiler warning
        }
        }
    }
}
