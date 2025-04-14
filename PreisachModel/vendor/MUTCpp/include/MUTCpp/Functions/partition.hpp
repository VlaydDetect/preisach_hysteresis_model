// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/Types.hpp"
#include "Matrix.hpp"

namespace mc
{
    /// Rearranges the elements in the array in such a way that
    /// the value of the element in kth position is in the position it
    /// would be in a sorted array. All elements smaller than the kth
    /// element are moved before this element and all equal or greater
    /// are moved behind it. The ordering of the elements in the two
    /// partitions is undefined.
    /// @param inArray
    /// @param inKth: kth element
    /// @param inAxis (Optional, default NONE)
    /// @return Matrix
    template <typename dtype>
    Matrix<dtype> partition(const Matrix<dtype> &inArray, uint32 inKth, Axis inAxis = Axis::NONE)
    {
        Matrix<dtype> returnArray(inArray);
        returnArray.partition(inKth, inAxis);
        return returnArray;
    }
}
