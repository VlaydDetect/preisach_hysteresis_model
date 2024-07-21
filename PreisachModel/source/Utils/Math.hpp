#pragma once
#include <limits>
#include <cmath>
#include <vector>
#include "NumCpp.hpp"

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

// std::vector<double> DiscreteFourierTransform(std::vector<double>& x)
// {
//     nc::NdArray<double> xn = nc::asarray<double>({x});
//     auto N = static_cast<double>(xn.shape().rows);
//     auto n = nc::arange(N);
//     auto k = n.reshape(N, 1);
//     auto M = nc::exp(-2 * nc::constants::j * k * n / N );
//     return nc::dot(M, x);
// }