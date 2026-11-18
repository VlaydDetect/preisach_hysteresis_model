// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include "Utils/DenseOperations.hpp"
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include "Utils/essentiallyEqual.hpp"
#include "Utils/num2str.hpp"

namespace mc::rotations
{
    class Quaternion
    {
    public:
        Quaternion() = default;

        Quaternion(double roll, double pitch, double yaw) noexcept
        {
            eulerToQuat(roll, pitch, yaw);
        }

        Quaternion(double i, double j, double k, double s) noexcept : m_Value(s, i, j, k)
        {
            normalize();
        }

        Quaternion(const std::array<double, 4> &components) noexcept :
            Quaternion(components[0], components[1], components[2], components[3])
        {
        }

        template <typename Derived>
        explicit Quaternion(const Eigen::DenseBase<Derived> &input)
        {
            const auto values = detail::rowMajor(input);
            if (input.size() == 3)
            {
                eulerToQuat(values(0), values(1), values(2));
            }
            else if (input.size() == 4)
            {
                m_Value.coeffs() << values(0), values(1), values(2), values(3);
                normalize();
            }
            else if (input.rows() == 3 && input.cols() == 3)
            {
                const auto at = [&](Eigen::Index row, Eigen::Index col)
                {
                    return static_cast<double>(values(3 * row + col));
                };
                // Use the largest component for the conversion, which also fixes the quaternion sign.
                Eigen::Vector4d squares(1 + at(0, 0) - at(1, 1) - at(2, 2),
                                       1 - at(0, 0) + at(1, 1) - at(2, 2),
                                       1 - at(0, 0) - at(1, 1) + at(2, 2),
                                       1 + at(0, 0) + at(1, 1) + at(2, 2));
                Eigen::Index largest = 3;
                if (squares.head<3>().maxCoeff() > squares.w()) squares.maxCoeff(&largest);
                const double component = 0.5 * std::sqrt(squares(largest));
                const double divisor = 4 * component;
                if (largest == 3)
                {
                    m_Value.coeffs() << (at(2, 1) - at(1, 2)) / divisor,
                                        (at(0, 2) - at(2, 0)) / divisor,
                                        (at(1, 0) - at(0, 1)) / divisor,
                                        component;
                }
                else
                {
                    const auto next = (largest + 1) % 3;
                    const auto last = (largest + 2) % 3;
                    m_Value.coeffs()(largest) = component;
                    m_Value.coeffs()(next) = (at(largest, next) + at(next, largest)) / divisor;
                    m_Value.coeffs()(last) = (at(largest, last) + at(last, largest)) / divisor;
                    m_Value.w() = (at(last, next) - at(next, last)) / divisor;
                }
            }
            else
            {
                throw std::invalid_argument("Quaternion requires three angles, four coefficients, or a 3x3 matrix");
            }
        }

        Quaternion(const Eigen::Vector3d &axis, double angle) noexcept :
            m_Value(Eigen::AngleAxisd(angle, axis / axis.norm()))
        {
        }

        [[nodiscard]] double i() const noexcept { return m_Value.x(); }
        [[nodiscard]] double j() const noexcept { return m_Value.y(); }
        [[nodiscard]] double k() const noexcept { return m_Value.z(); }
        [[nodiscard]] double s() const noexcept { return m_Value.w(); }

        static Quaternion identity() noexcept { return {}; }

        [[nodiscard]] double angleOfRotation() const noexcept { return 2.0 * std::acos(s()); }

        [[nodiscard]] Eigen::Vector3d axisOfRotation() const noexcept
        {
            const Eigen::Vector3d axis = m_Value.vec() / std::sin(angleOfRotation() * 0.5);
            return axis / axis.norm();
        }

        static Eigen::Vector3d angularVelocity(const Quaternion &first, const Quaternion &second, double time)
        {
            const Eigen::Vector4d derivative = (second.toMatrix() - first.toMatrix()) / time;
            const Eigen::Vector3d vector = second.m_Value.vec();
            return 2.0 * (second.s() * derivative.head<3>() -
                          vector.cross(derivative.head<3>()) - derivative.w() * vector);
        }

        [[nodiscard]] Eigen::Vector3d angularVelocity(const Quaternion &second, double time) const
        {
            return angularVelocity(*this, second, time);
        }

        [[nodiscard]] Quaternion conjugate() const noexcept { return {-i(), -j(), -k(), s()}; }
        [[nodiscard]] Quaternion inverse() const noexcept { return conjugate(); }

        static Quaternion nlerp(const Quaternion &first, const Quaternion &second, double fraction)
        {
            checkFraction(fraction);
            if (utils::essentiallyEqual(fraction, 0.0)) return first;
            if (utils::essentiallyEqual(fraction, 1.0)) return second;
            return Quaternion((1.0 - fraction) * first.toMatrix() + fraction * second.toMatrix());
        }

        [[nodiscard]] Quaternion nlerp(const Quaternion &second, double fraction) const
        {
            return nlerp(*this, second, fraction);
        }

        static Quaternion slerp(const Quaternion &first, const Quaternion &second, double fraction)
        {
            checkFraction(fraction);
            if (utils::essentiallyEqual(fraction, 0.0)) return first;
            if (utils::essentiallyEqual(fraction, 1.0)) return second;

            const double product = first.m_Value.dot(second.m_Value);
            if (std::abs(product) > 0.9995) return nlerp(first, second, fraction);
            const auto from = product < 0.0 ? -first : first;
            return Quaternion(from.m_Value.slerp(fraction, second.m_Value).coeffs());
        }

        [[nodiscard]] Quaternion slerp(const Quaternion &second, double fraction) const
        {
            return slerp(*this, second, fraction);
        }

        [[nodiscard]] double roll() const noexcept
        {
            return std::atan2(2.0 * (s() * i() + j() * k()), 1.0 - 2.0 * (i() * i() + j() * j()));
        }

        [[nodiscard]] double pitch() const noexcept { return std::asin(2.0 * (s() * j() - k() * i())); }

        [[nodiscard]] double yaw() const noexcept
        {
            return std::atan2(2.0 * (s() * k() + i() * j()), 1.0 - 2.0 * (j() * j() + k() * k()));
        }

        static Quaternion rollRotation(double angle) noexcept { return {angle, 0.0, 0.0}; }
        static Quaternion pitchRotation(double angle) noexcept { return {0.0, angle, 0.0}; }
        static Quaternion yawRotation(double angle) noexcept { return {0.0, 0.0, angle}; }
        static Quaternion xRotation(double angle) noexcept { return {Eigen::Vector3d::UnitX(), angle}; }
        static Quaternion yRotation(double angle) noexcept { return {Eigen::Vector3d::UnitY(), angle}; }
        static Quaternion zRotation(double angle) noexcept { return {Eigen::Vector3d::UnitZ(), angle}; }

        [[nodiscard]] Eigen::Vector3d rotate(const Eigen::Vector3d &vector) const { return *this * vector; }
        [[nodiscard]] Eigen::Matrix3d toDCM() const { return m_Value.toRotationMatrix(); }
        [[nodiscard]] Eigen::Vector4d toMatrix() const { return m_Value.coeffs(); }

        [[nodiscard]] std::string str() const
        {
            return "[" + utils::num2str(i()) + ", " + utils::num2str(j()) + ", " +
                   utils::num2str(k()) + ", " + utils::num2str(s()) + "]\n";
        }

        void print() const { std::cout << *this; }

        bool operator==(const Quaternion &other) const noexcept
        {
            return utils::essentiallyEqual(i(), other.i()) && utils::essentiallyEqual(j(), other.j()) &&
                   utils::essentiallyEqual(k(), other.k()) && utils::essentiallyEqual(s(), other.s());
        }

        Quaternion &operator+=(const Quaternion &other) noexcept
        {
            m_Value.coeffs() += other.m_Value.coeffs();
            normalize();
            return *this;
        }

        Quaternion &operator-=(const Quaternion &other) noexcept
        {
            m_Value.coeffs() -= other.m_Value.coeffs();
            normalize();
            return *this;
        }

        Quaternion &operator*=(const Quaternion &other) noexcept
        {
            // Preserve the composition order of MUTCpp's quaternion API.
            m_Value = other.m_Value * m_Value;
            normalize();
            return *this;
        }

        Quaternion &operator*=(double scalar) noexcept
        {
            m_Value.coeffs() *= scalar;
            normalize();
            return *this;
        }

        Quaternion &operator/=(const Quaternion &other) noexcept { return *this *= other.conjugate(); }
        Quaternion operator+(const Quaternion &other) const noexcept { return Quaternion(*this) += other; }
        Quaternion operator-(const Quaternion &other) const noexcept { return Quaternion(*this) -= other; }
        Quaternion operator-() const noexcept { return Quaternion(*this) *= -1.0; }
        Quaternion operator*(const Quaternion &other) const noexcept { return Quaternion(*this) *= other; }
        Quaternion operator*(double scalar) const noexcept { return Quaternion(*this) *= scalar; }
        Quaternion operator/(const Quaternion &other) const noexcept { return Quaternion(*this) /= other; }

        Eigen::Vector3d operator*(const Eigen::Vector3d &vector) const
        {
            if (utils::essentiallyEqual(vector.norm(), 0.0)) return vector;
            return m_Value.conjugate() * vector;
        }

        friend std::ostream &operator<<(std::ostream &stream, const Quaternion &quaternion)
        {
            return stream << quaternion.str();
        }

    private:
        Eigen::Quaterniond m_Value = Eigen::Quaterniond::Identity();

        void normalize() noexcept { m_Value.coeffs() /= m_Value.norm(); }

        static void checkFraction(double fraction)
        {
            if (!(fraction >= 0.0 && fraction <= 1.0))
            {
                throw std::invalid_argument("Interpolation fraction must be in [0, 1]");
            }
        }

        void eulerToQuat(double roll, double pitch, double yaw) noexcept
        {
            const auto halfPhi = roll / 2.;
            const auto halfTheta = pitch / 2.;
            const auto halfPsi = yaw / 2.;

            const auto sinHalfPhi = std::sin(halfPhi);
            const auto cosHalfPhi = std::cos(halfPhi);

            const auto sinHalfTheta = std::sin(halfTheta);
            const auto cosHalfTheta = std::cos(halfTheta);

            const auto sinHalfPsi = std::sin(halfPsi);
            const auto cosHalfPsi = std::cos(halfPsi);

            m_Value.x() = sinHalfPhi * cosHalfTheta * cosHalfPsi;
            m_Value.x() -= cosHalfPhi * sinHalfTheta * sinHalfPsi;

            m_Value.y() = cosHalfPhi * sinHalfTheta * cosHalfPsi;
            m_Value.y() += sinHalfPhi * cosHalfTheta * sinHalfPsi;

            m_Value.z() = cosHalfPhi * cosHalfTheta * sinHalfPsi;
            m_Value.z() -= sinHalfPhi * sinHalfTheta * cosHalfPsi;

            m_Value.w() = cosHalfPhi * cosHalfTheta * cosHalfPsi;
            m_Value.w() += sinHalfPhi * sinHalfTheta * sinHalfPsi;
        }


        /// Converts the direction cosine matrix to quaternion components
        ///
        /// @ param dcm: the direction cosine matrix
        ///
    };
}
