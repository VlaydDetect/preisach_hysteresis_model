// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <utility>

#include "Core/Error.hpp"
#include "Functions/dot.hpp"
#include "Functions/append.hpp"
#include "Matrix.hpp"
#include "Functions/zeros.hpp"
#include "MUTCpp/Functions/norm.hpp"

namespace mc
{
    namespace linalg
    {
        inline std::pair<Matrix<double>, Matrix<double>> qr_decomposition(const Matrix<double> &inMatrix)
        {
            const uint32_t n = inMatrix.numCols();
            if (!inMatrix.issquare())
            {
                THROW_RUNTIME_ERROR("Input matrix should be square.");
            }

            Matrix<double> Q;
            auto R = zeros<double>(n, n);

            for (uint32_t i = 0; i < n; ++i)
            {
                auto v = inMatrix.column(i);

                for (uint32_t j = 0; j < i; ++j)
                {
                    R(j, i) = dot(Q.column(j), inMatrix.column(i))[0];
                    v -= R(j, i) * Q.column(j);
                }

                R(i, i) = mc::norm(v)[0];
                Q = append(Q, v / R(i, i), Axis::COL);
            }

            return {Q, R};
        }
    }
}
