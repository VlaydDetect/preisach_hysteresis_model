// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <iomanip>
#include <sstream>
#include <string>

#include "Matrix/MatrixCore.hpp"

namespace mc
{
    inline std::string doubleToString(double value, int precision)
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(precision) << value;
        return ss.str();
    }

    inline std::string doubleToString(const Matrix<double>& value, int precision)
    {
        std::string ret;
        ret += "{";
        for (int i = 0; i < value.size(); i++)
        {
            ret += doubleToString(value[i], precision);

            if (i != value.size() - 1)
            {
                ret += ", ";
            }
        }
        ret += "}";

        return ret;
    }
}
