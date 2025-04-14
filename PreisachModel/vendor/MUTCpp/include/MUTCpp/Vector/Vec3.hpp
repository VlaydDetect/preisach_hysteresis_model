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
#include "Functions/hypot.hpp"
#include "Matrix.hpp"
#include "Utils/essentiallyEqual.hpp"
#include "Utils/interp.hpp"
#include "Vector/Vec2.hpp"

namespace mc
{
    /// Holds a 3D vector
    class Vec3
    {
    public:
        double x{0.};
        double y{0.};
        double z{0.};

        constexpr Vec3() = default;

        constexpr Vec3(double inX, double inY, double inZ) noexcept :
            x(inX),
            y(inY),
            z(inZ)
        {
        }

        Vec3(const std::initializer_list<double> &inList)
        {
            if (inList.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("input initializer list must have a size = 3");
            }

            x = *inList.begin();
            y = *(inList.begin() + 1);
            z = *(inList.begin() + 2);
        }

        constexpr Vec3(const Vec2 &vec2) noexcept :
            x(vec2.x),
            y(vec2.y)
        {
        }

        Vec3(const Matrix<double> &ndArray)
        {
            if (ndArray.size() != 3)
            {
                THROW_INVALID_ARGUMENT_ERROR("input Matrix must have a size = 3");
            }

            x = ndArray[0];
            y = ndArray[1];
            z = ndArray[2];
        }

        std::vector<double> toVector()
        {
            return {x, y, z};
        }

        std::array<double, 3> toArray()
        {
            return {x, y, z};
        }
        
        [[nodiscard]] double angle(const Vec3 &otherVec) const noexcept
        {
            double dotProduct = dot(otherVec);
            dotProduct /= norm();
            dotProduct /= otherVec.norm();

            // clamp the value to the acos range just to be safe
            dotProduct = std::max(std::min(dotProduct, 1.), -1.);

            return std::acos(dotProduct);
        }
        
        /// Returns the unit vector [0, -1, 0]
        /// @return Vec3
        static constexpr Vec3 down() noexcept
        {
            return Vec3(0., -1., 0.); // NOLINT(modernize-return-braced-init-list)
        }
        
        /// Returns the unit vector [0, 0, 1]
        /// @return Vec3
        static constexpr Vec3 forward() noexcept
        {
            return Vec3(0., 0., 1.); // NOLINT(modernize-return-braced-init-list)
        }
        
        /// Returns the unit vector [-1, 0, 0]
        /// @return Vec3
        static constexpr Vec3 left() noexcept
        {
            return Vec3(-1., 0., 0.); // NOLINT(modernize-return-braced-init-list)
        }
        
        /// Returns the unit vector [0, 0, -1]
        /// @return Vec3
        static constexpr Vec3 back() noexcept
        {
            return Vec3(0., 0., -1.); // NOLINT(modernize-return-braced-init-list)
        }

        /// Returns the unit vector [1, 0, 0]
        /// @return Vec3
        static constexpr Vec3 right() noexcept
        {
            return Vec3(1., 0., 0.); // NOLINT(modernize-return-braced-init-list)
        }

        /// Returns the unit vector [0, 1, 0]
        /// @return Vec3
        static constexpr Vec3 up() noexcept
        {
            return Vec3(0., 1., 0.); // NOLINT(modernize-return-braced-init-list)
        }
        
        /// Returns a copy of the vector with its magnitude clamped
        /// to maxLength
        /// @param maxLength
        /// @return Vec3
        [[nodiscard]] Vec3 clampMagnitude(double maxLength) const noexcept
        {
            const double magnitude = norm();
            if (magnitude <= maxLength)
            {
                return *this;
            }

            Vec3 returnVec = Vec3(*this).normalize();
            returnVec *= maxLength;
            return returnVec;
        }
        
        /// Returns the cross-product of the two vectors
        [[nodiscard]] Vec3 cross(const Vec3 &otherVec) const noexcept
        {
            const double crossX = y * otherVec.z - z * otherVec.y;
            const double crossY = -(x * otherVec.z - z * otherVec.x);
            const double crossZ = x * otherVec.y - y * otherVec.x;

            return Vec3(crossX, crossY, crossZ); // NOLINT(modernize-return-braced-init-list)
        }

        /// Returns the distance between the two vectors
        /// @return the distance (equivalent to (a - b).norm()
        [[nodiscard]] double distance(const Vec3 &otherVec) const noexcept
        {
            return (Vec3(*this) -= otherVec).norm();
        }
        
        /// Returns the dot-product of the two vectors
        [[nodiscard]] double dot(const Vec3 &otherVec) const noexcept
        {
            return x * otherVec.x + y * otherVec.y + z * otherVec.z;
        }
        
        /// Linearly interpolates between two vectors
        /// @param otherVec
        /// @param t the amount to interpolate by (clamped from [0, 1]);
        /// @return Vec3
        [[nodiscard]] Vec3 lerp(const Vec3 &otherVec, double t) const noexcept
        {
            t = std::max(std::min(t, 1.), 0.);

            Vec3 trajectory = otherVec;
            trajectory -= *this;
            const double xInterp = utils::interp(0., trajectory.x, t);
            const double yInterp = utils::interp(0., trajectory.y, t);
            const double zInterp = utils::interp(0., trajectory.z, t);

            return Vec3(*this) += Vec3(xInterp, yInterp, zInterp);
        }
        
        /// Returns the norm of the vector
        /// @return norm of the vector
        [[nodiscard]] double norm() const noexcept
        {
            return hypot(x, y, z);
        }
        
        /// Returns a new normalized Vec3
        /// @return Vec3
        [[nodiscard]] Vec3 normalize() const noexcept
        {
            return Vec3(*this) /= norm();
        }
        
        /// Projects the vector onto the input vector
        /// @return Vec3
        [[nodiscard]] Vec3 project(const Vec3 &otherVec) const noexcept
        {
            const double projectedMagnitude = norm() * std::cos(angle(otherVec));
            return otherVec.normalize() *= projectedMagnitude;
        }
        
        /// Returns the Vec3 as a string
        /// @return std::string
        [[nodiscard]] std::string toString() const
        {
            std::stringstream stream;
            stream << "Vec3[" << x << ", " << y << ", " << z << "]";
            return stream.str();
        }
        
        /// Returns the Vec2 as a Matrix
        /// @return Matrix
        [[nodiscard]] Matrix<double> toMatrix() const
        {
            Matrix<double> returnArray = {x, y, z};
            return returnArray.transpose();
        }
        
        bool operator==(const Vec3 &rhs) const noexcept
        {
            return utils::essentiallyEqual(x, rhs.x) && utils::essentiallyEqual(y, rhs.y) &&
                utils::essentiallyEqual(z, rhs.z);
        }
        
        bool operator!=(const Vec3 &rhs) const noexcept
        {
            return !(*this == rhs);
        }
        
        Vec3 &operator+=(double scalar) noexcept
        {
            x += scalar;
            y += scalar;
            z += scalar;
            return *this;
        }
        
        Vec3 &operator+=(const Vec3 &rhs) noexcept
        {
            x += rhs.x;
            y += rhs.y;
            z += rhs.z;
            return *this;
        }
        
        Vec3 &operator-=(double scalar) noexcept
        {
            x -= scalar;
            y -= scalar;
            z -= scalar;
            return *this;
        }
        
        Vec3 &operator-=(const Vec3 &rhs) noexcept
        {
            x -= rhs.x;
            y -= rhs.y;
            z -= rhs.z;
            return *this;
        }
        
        Vec3 &operator*=(double scalar) noexcept
        {
            x *= scalar;
            y *= scalar;
            z *= scalar;
            return *this;
        }
        
        Vec3 &operator/=(double scalar) noexcept
        {
            x /= scalar;
            y /= scalar;
            z /= scalar;
            return *this;
        }
    };
    
    inline Vec3 operator+(const Vec3 &lhs, double rhs) noexcept
    {
        return Vec3(lhs) += rhs;
    }

    inline Vec3 operator+(double lhs, const Vec3 &rhs) noexcept
    {
        return Vec3(rhs) += lhs;
    }
    
    inline Vec3 operator+(const Vec3 &lhs, const Vec3 &rhs) noexcept
    {
        return Vec3(lhs) += rhs;
    }
    
    inline Vec3 operator-(const Vec3 &vec) noexcept
    {
        return Vec3(-vec.x, -vec.y, -vec.z); // NOLINT(modernize-return-braced-init-list)
    }
    
    inline Vec3 operator-(const Vec3 &lhs, double rhs) noexcept
    {
        return Vec3(lhs) -= rhs;
    }
    
    inline Vec3 operator-(double lhs, const Vec3 &rhs) noexcept
    {
        return -Vec3(rhs) += lhs;
    }
    
    inline Vec3 operator-(const Vec3 &lhs, const Vec3 &rhs) noexcept
    {
        return Vec3(lhs) -= rhs;
    }
    
    inline Vec3 operator*(const Vec3 &lhs, double rhs) noexcept
    {
        return Vec3(lhs) *= rhs;
    }
    
    inline Vec3 operator*(double lhs, const Vec3 &rhs) noexcept
    {
        return Vec3(rhs) *= lhs;
    }
    
    inline double operator*(const Vec3 &lhs, const Vec3 &rhs) noexcept
    {
        return lhs.dot(rhs);
    }
    
    inline Vec3 operator/(const Vec3 &lhs, double rhs) noexcept
    {
        return Vec3(lhs) /= rhs;
    }
    
    inline std::ostream &operator<<(std::ostream &stream, const Vec3 &vec)
    {
        stream << vec.toString() << std::endl;
        return stream;
    }
}
