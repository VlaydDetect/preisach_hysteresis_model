// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>

#include "Core/Error.hpp"
#include "Matrix.hpp"
#include "Utils/essentiallyEqual.hpp"
#include "Utils/interp.hpp"

//====================================================================================

namespace mc
{
    /// Holds a 2D vector
    class Vec2
    {
    public:
        double x{0.};
        double y{0.};

        constexpr Vec2() = default;

        constexpr Vec2(double inX, double inY) noexcept :
            x(inX),
            y(inY)
        {
        }

        Vec2(const std::initializer_list<double> &inList)
        {
            if (inList.size() != 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input initializer list must have a size = 2");
            }

            x = *inList.begin();
            y = *(inList.begin() + 1);
        }

        Vec2(const Matrix<double> &ndArray)
        {
            if (ndArray.size() != 2)
            {
                THROW_INVALID_ARGUMENT_ERROR("input Matrix must have a size = 2");
            }

            x = ndArray[0];
            y = ndArray[1];
        }

        std::vector<double> toVector()
        {
            return {x, y};
        }

        std::array<double, 2> toArray()
        {
            return {x, y};
        }

        /// Returns the angle between the two vectors
        /// @return the angle in radians
        [[nodiscard]] double angle(const Vec2 &otherVec) const noexcept
        {
            double dotProduct = dot(otherVec);
            dotProduct /= norm();
            dotProduct /= otherVec.norm();

            // clamp the value to the acos range just to be safe
            dotProduct = std::max(std::min(dotProduct, 1.), -1.);

            return std::acos(dotProduct);
        }

        /// Returns a copy of the vector with its magnitude clamped
        /// to maxLength
        /// @param maxLength
        /// @return Vec2
        [[nodiscard]] Vec2 clampMagnitude(double maxLength) const noexcept
        {
            const double magnitude = norm();
            if (magnitude <= maxLength)
            {
                return *this;
            }

            Vec2 returnVec = Vec2(*this).normalize();
            returnVec *= maxLength;
            return returnVec;
        }

        /// Returns the distance between the two vectors
        /// @return the distance (equivalent to (a - b).norm()
        [[nodiscard]] double distance(const Vec2 &otherVec) const noexcept
        {
            return (Vec2(*this) -= otherVec).norm();
        }

        /// Returns the dot product of the two vectors
        /// @return the dot product
        [[nodiscard]] double dot(const Vec2 &otherVec) const noexcept
        {
            return x * otherVec.x + y * otherVec.y;
        }

        /// Linearly interpolates between two vectors
        /// @param otherVec
        /// @param t the amount to interpolate by (clamped from [0, 1]);
        /// @return Vec2
        [[nodiscard]] Vec2 lerp(const Vec2 &otherVec, double t) const noexcept
        {
            t = std::max(std::min(t, 1.), 0.);

            Vec2 trajectory = otherVec;
            trajectory -= *this;
            const double xInterp = utils::interp(0., trajectory.x, t);
            const double yInterp = utils::interp(0., trajectory.y, t);

            return Vec2(*this) += Vec2(xInterp, yInterp);
        }

        /// Returns the norm of the vector
        /// @return norm of the vector
        [[nodiscard]] double norm() const noexcept
        {
            return std::hypot(x, y);
        }

        /// Returns a new normalized Vec2
        /// @return Vec2
        [[nodiscard]] Vec2 normalize() const noexcept
        {
            return Vec2(*this) /= norm();
        }

        /// Projects the vector onto the input vector
        /// @return Vec2
        [[nodiscard]] Vec2 project(const Vec2 &otherVec) const noexcept
        {
            const double projectedMagnitude = norm() * std::cos(angle(otherVec));
            return otherVec.normalize() *= projectedMagnitude;
        }

        /// Returns the unit vector [0, 1]
        /// @return Vec2
        static constexpr Vec2 up() noexcept
        {
            return Vec2(0., 1.); // NOLINT(modernize-return-braced-init-list)
        }

        /// Returns the unit vector [0, -1]
        /// @return Vec2
        static constexpr Vec2 down() noexcept
        {
            return Vec2(0., -1.); // NOLINT(modernize-return-braced-init-list)
        }

        /// Returns the unit vector [1, 0]
        /// @return Vec2
        static constexpr Vec2 right() noexcept
        {
            return Vec2(1., 0.); // NOLINT(modernize-return-braced-init-list)
        }

        /// Returns the unit vector [-1, 0]
        /// @return Vec2
        static constexpr Vec2 left() noexcept
        {
            return Vec2(-1., 0.); // NOLINT(modernize-return-braced-init-list)
        }

        /// Returns the Vec2 as a string
        /// @return std::string
        [[nodiscard]] std::string toString() const
        {
            std::stringstream stream;
            stream << "Vec2[" << x << ", " << y << "]";
            return stream.str();
        }

        /// Returns the Vec2 as a Matrix
        /// @return Matrix
        [[nodiscard]] Matrix<double> toMatrix() const
        {
            Matrix returnArray = {x, y};
            return returnArray.transpose();
        }

        bool operator==(const Vec2 &rhs) const noexcept
        {
            return utils::essentiallyEqual(x, rhs.x) && utils::essentiallyEqual(y, rhs.y);
        }

        bool operator!=(const Vec2 &rhs) const noexcept
        {
            return !(*this == rhs);
        }

        Vec2 &operator+=(double scalar) noexcept
        {
            x += scalar;
            y += scalar;
            return *this;
        }

        Vec2 &operator+=(const Vec2 &rhs) noexcept
        {
            x += rhs.x;
            y += rhs.y;
            return *this;
        }

        Vec2 &operator-=(double scalar) noexcept
        {
            x -= scalar;
            y -= scalar;
            return *this;
        }

        Vec2 &operator-=(const Vec2 &rhs) noexcept
        {
            x -= rhs.x;
            y -= rhs.y;
            return *this;
        }

        Vec2 &operator*=(double scalar) noexcept
        {
            x *= scalar;
            y *= scalar;
            return *this;
        }

        Vec2 &operator/=(double scalar) noexcept
        {
            x /= scalar;
            y /= scalar;
            return *this;
        }
    };

    inline Vec2 operator+(const Vec2 &lhs, double rhs) noexcept
    {
        return Vec2(lhs) += rhs;
    }

    inline Vec2 operator+(double lhs, const Vec2 &rhs) noexcept
    {
        return Vec2(rhs) += lhs;
    }

    inline Vec2 operator+(const Vec2 &lhs, const Vec2 &rhs) noexcept
    {
        return Vec2(lhs) += rhs;
    }

    inline Vec2 operator-(const Vec2 &vec) noexcept
    {
        return Vec2(-vec.x, -vec.y); // NOLINT(modernize-return-braced-init-list)
    }

    inline Vec2 operator-(const Vec2 &lhs, double rhs) noexcept
    {
        return Vec2(lhs) -= rhs;
    }

    inline Vec2 operator-(double lhs, const Vec2 &rhs) noexcept
    {
        return -Vec2(rhs) += lhs;
    }

    inline Vec2 operator-(const Vec2 &lhs, const Vec2 &rhs) noexcept
    {
        return Vec2(lhs) -= rhs;
    }

    inline Vec2 operator*(const Vec2 &lhs, double rhs) noexcept
    {
        return Vec2(lhs) *= rhs;
    }

    inline Vec2 operator*(double lhs, const Vec2 &rhs) noexcept
    {
        return Vec2(rhs) *= lhs;
    }

    ///
    inline double operator*(const Vec2 &lhs, const Vec2 &rhs) noexcept
    {
        return lhs.dot(rhs);
    }

    inline Vec2 operator/(const Vec2 &lhs, double rhs) noexcept
    {
        return Vec2(lhs) /= rhs;
    }

    inline std::ostream &operator<<(std::ostream &stream, const Vec2 &vec)
    {
        stream << vec.toString() << std::endl;
        return stream;
    }
}
