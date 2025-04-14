// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Core/StaticAssert.hpp"
#include "Matrix.hpp"

namespace mc
{
    template<typename U, typename F>
    std::pair<Matrix<double>, Matrix<double>> ContinueFunction(const Matrix<F>& t0, const Matrix<U>& v0, double dt)
    {
        STATIC_ASSERT_ARITHMETIC(F);
        STATIC_ASSERT_ARITHMETIC(U);

        if (t0.size() != v0.size())
        {
            THROW_RUNTIME_ERROR("Matrices sizes must be equal");
        }

        Matrix<double> t;
        Matrix<double> v;

        for (int i = 1; i < v0.size(); i++)
        {
            double ta = t0[i - 1];
            double tb = t0[i];
            double va = v0[i - 1];
            double vb = v0[i];

            double h = std::abs(vb - va) * dt / std::abs(tb - ta);
        
            for (uint32_t j = 0; j < static_cast<uint32_t>(std::abs((tb - ta) / dt)); ++j)
            {
                t = append(t, {ta + j * dt});
                // t.append({ta + j * dt});
            }

            for (uint32_t j = 0; j < static_cast<uint32_t>(std::abs((vb - va) / h)); ++j)
            {
                const int a = vb > va ? 1 : -1;
                v = append(v, {va + j * h * a});
                // v.append({va + j * h * a});
            }
        }
        return {t, v};
    }
}
