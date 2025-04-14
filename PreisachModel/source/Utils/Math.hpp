#pragma once

#include <cmath>
#include <vector>

#include "Array.hpp"
#include "MUTCpp.hpp"

#define NAN mc::constants::nan

inline bool eqf(double a, double b, double epsilon = 1.0e-6)
{
    // return std::abs(a - b) < std::numeric_limits<double>::epsilon();
    return std::abs(a - b) < epsilon;
}

inline std::complex<double> operator*(const int& a, const std::complex<double>& b)
{
    if (a == 0) return {0, 0};

    return {b.real() * a, b.imag() * a};
}

template<typename U, typename F>
std::pair<std::vector<double>, std::vector<double>> ContinueFunction(double dt, const std::vector<F>& t0, const std::vector<U>& v0)
{
    if (t0.size() != v0.size())
    {
        THROW_RUNTIME_ERROR("Vector sizes of t0 and v0 must be equal");
    }
    
    std::vector<double> t = {};
    std::vector<double> v = {};
    for (int i = 1; i < v0.size(); i++)
    {
        double ta = t0[i - 1];
        double tb = t0[i];
        double va = v0[i - 1];
        double vb = v0[i];

        double h = std::abs(vb - va) * dt / std::abs(tb - ta);
        
        for (uint32_t j = 0; j < static_cast<uint32_t>(std::abs((tb - ta) / dt)); ++j)
        {
            t.push_back(ta + j * dt);
        }

        for (uint32_t j = 0; j < static_cast<uint32_t>(std::abs((vb - va) / h)); ++j)
        {
            const int a = vb > va ? 1 : -1;
            v.push_back(va + j * h * a);
        }
    }
    return {t, v};
};

// std::vector<double> DiscreteFourierTransform(std::vector<double>& x)
// {
//     mc::Matrix<double> xn = mc::asarray<double>({x});
//     auto N = static_cast<double>(xn.shape().rows);
//     auto n = mc::arange(N);
//     auto k = n.reshape(N, 1);
//     auto M = mc::exp(-2 * mc::constants::j * k * n / N );
//     return mc::dot(M, x);
// }
