// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <string>

#include "Core/Error.hpp"
#include "Core/Algorithm.hpp"
#include "Core/Types.hpp"
#include "Functions/argmax.hpp"
#include "Functions/clip.hpp"
#include "Functions/dot.hpp"
#include "Functions/norm.hpp"
#include "Functions/square.hpp"
#include "Linalg/hat.hpp"
#include "Matrix.hpp"
#include "Utils/essentiallyEqual.hpp"
#include "Utils/num2str.hpp"
#include "Utils/mathOperations.hpp"
#include "Vector/Vec3.hpp"

namespace mc
{
    namespace rotations
    {
        /// Holds a unit quaternion
        class Quaternion
        {
        public:
            Quaternion() = default;

            /// Constructor
            /// @param roll: euler roll angle in radians
            /// @param pitch: euler pitch angle in radians
            /// @param yaw: euler yaw angle in radians
            Quaternion(double roll, double pitch, double yaw) noexcept
            {
                eulerToQuat(roll, pitch, yaw);
            }

            Quaternion(double inI, double inJ, double inK, double inS) noexcept :
                components_{inI, inJ, inK, inS}
            {
                normalize();
            }

            Quaternion(const std::array<double, 4> &components) noexcept :
                components_{components}
            {
                normalize();
            }

            /// Constructor
            /// @param inArray: if size = 3 the roll, pitch, yaw euler angles
            /// if size = 4 the i, j, k, s components
            /// if shape = [3, 3] then direction cosine matrix
            Quaternion(const Matrix<double> &inArray) :
                components_{0., 0., 0., 0.}
            {
                if (inArray.size() == 3)
                {
                    // euler angles
                    eulerToQuat(inArray[0], inArray[1], inArray[2]);
                }
                else if (inArray.size() == 4)
                {
                    // quaternion i, j, k, s components
                    algo::copy(inArray.cbegin(), inArray.cend(), components_.begin());
                    normalize();
                }
                else if (inArray.size() == 9)
                {
                    // direction cosine matrix
                    dcmToQuat(inArray);
                }
                else
                {
                    THROW_INVALID_ARGUMENT_ERROR("input array is not a valid size.");
                }
            }

            /// Constructor
            /// @param inAxis: Euler axis
            /// @param inAngle: Euler angle in radians
            Quaternion(const Vec3 &inAxis, double inAngle) noexcept
            {
                // normalize the input vector
                Vec3 normAxis = inAxis.normalize();

                const double halfAngle = inAngle / 2.;
                const double sinHalfAngle = std::sin(halfAngle);

                components_[0] = normAxis.x * sinHalfAngle;
                components_[1] = normAxis.y * sinHalfAngle;
                components_[2] = normAxis.z * sinHalfAngle;
                components_[3] = std::cos(halfAngle);
            }

            /// Constructor
            /// @param inAxis: Euler axis x,y,z vector components
            /// @param inAngle: Euler angle in radians
            Quaternion(const Matrix<double> &inAxis, double inAngle) :
                Quaternion(Vec3(inAxis), inAngle)
            {
            }

            /// The angle of rotation around the rotation axis that is described by the quaternion
            [[nodiscard]] double angleOfRotation() const noexcept
            {
                return 2. * std::acos(s());
            }

            /// Angular velocity vector between the two quaternions. The norm
            /// of the array is the magnitude
            /// @param inQuat1
            /// @param inQuat2
            /// @param inTime (seperation time)
            /// @return Matrix<double>
            static Matrix<double> angularVelocity(const Quaternion &inQuat1, const Quaternion &inQuat2, double inTime)
            {
                Matrix<double> q0 = inQuat1.toMatrix();
                Matrix<double> q1 = inQuat2.toMatrix();

                Matrix<double> qDot = q1 - q0;
                qDot /= inTime;

                Matrix<double> eyeTimesScalar(3);
                eyeTimesScalar.zeros();
                eyeTimesScalar(0, 0) = inQuat2.s();
                eyeTimesScalar(1, 1) = inQuat2.s();
                eyeTimesScalar(2, 2) = inQuat2.s();

                Matrix<double> epsilonHat = linalg::hat<double>(inQuat2.i(), inQuat2.j(), inQuat2.k());
                Matrix<double> q(4, 3);
                q.put(Slice(0, 3), Slice(0, 3), eyeTimesScalar + epsilonHat);
                q(3, 0) = -inQuat2.i();
                q(3, 1) = -inQuat2.j();
                q(3, 2) = -inQuat2.k();

                Matrix<double> omega = q.transpose().dot(qDot.transpose());
                return omega *= 2.;
            }

            /// Angular velocity vector between the two quaternions. The norm
            /// of the array is the magnitude
            /// @param inQuat2
            /// @param inTime (separation time)
            /// @return Matrix<double>
            [[nodiscard]] Matrix<double> angularVelocity(const Quaternion &inQuat2, double inTime) const
            {
                return angularVelocity(*this, inQuat2, inTime);
            }

            /// The axis of rotation described by the quaternion
            [[nodiscard]] Vec3 axisOfRotation() const noexcept
            {
                const auto halfAngle = angleOfRotation() / 2.;
                const auto sinHalfAngle = std::sin(halfAngle);
                auto axis = Vec3(i() / sinHalfAngle, j() / sinHalfAngle, k() / sinHalfAngle);

                // shouldn't be necessary, but let's be pedantic
                return axis.normalize();
            }

            /// Quaternion conjugate
            [[nodiscard]] Quaternion conjugate() const noexcept
            {
                return {-i(), -j(), -k(), s()};
            }

            /// returns the i component
            [[nodiscard]] double i() const noexcept
            {
                return components_[0];
            }

            /// quaternion identity (0,0,0,1)
            static Quaternion identity() noexcept
            {
                return {};
            }

            /// quaternion inverse
            [[nodiscard]] Quaternion inverse() const noexcept
            {
                /// for unit quaternions the inverse is equal to the conjugate
                return conjugate();
            }

            /// returns the j component
            [[nodiscard]] double j() const noexcept
            {
                return components_[1];
            }

            /// returns the k component
            [[nodiscard]] double k() const noexcept
            {
                return components_[2];
            }


            /// linearly interpolates between the two quaternions
            /// @param inQuat1
            /// @param inQuat2
            /// @param inPercent [0, 1]
            /// @return Quaternion
            static Quaternion nlerp(const Quaternion &inQuat1, const Quaternion &inQuat2, double inPercent)
            {
                if (inPercent < 0. || inPercent > 1.)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input percent must be of the range [0,1].");
                }

                if (utils::essentiallyEqual(inPercent, 0.))
                {
                    return inQuat1;
                }
                if (utils::essentiallyEqual(inPercent, 1.))
                {
                    return inQuat2;
                }

                const double oneMinus = 1. - inPercent;
                std::array<double, 4> newComponents{};

                algo::transform(inQuat1.components_.begin(),
                                inQuat1.components_.end(),
                                inQuat2.components_.begin(),
                                newComponents.begin(),
                                [inPercent, oneMinus](double component1, double component2) -> double
                                {
                                    return oneMinus * component1 + inPercent * component2;
                                });

                return {newComponents};
            }


            /// linearly interpolates between the two quaternions
            /// @param inQuat2
            /// @param inPercent (0, 1)
            /// @return Quaternion
            [[nodiscard]] Quaternion nlerp(const Quaternion &inQuat2, double inPercent) const
            {
                return nlerp(*this, inQuat2, inPercent);
            }


            /// The euler pitch angle in radians
            /// @return euler pitch angle in radians
            [[nodiscard]] double pitch() const noexcept
            {
                return std::asin(2 * (s() * j() - k() * i()));
            }


            /// returns a quaternion to rotate about the pitch axis
            ///
            /// @param inAngle (radians)
            /// @return Quaternion
            ///
            static Quaternion pitchRotation(double inAngle) noexcept
            {
                return {0., inAngle, 0.};
            }


            /// prints the Quaternion to the console
            ///
            void print() const
            {
                std::cout << *this;
            }


            /// The euler roll angle in radians
            ///
            /// @return euler roll angle in radians
            ///
            [[nodiscard]] double roll() const noexcept
            {
                return std::atan2(2. * (s() * i() + j() * k()), 1. - 2. * (utils::sqr(i()) + utils::sqr(j())));
            }


            /// returns a quaternion to rotate about the roll axis
            ///
            /// @param inAngle (radians)
            /// @return Quaternion
            ///
            static Quaternion rollRotation(double inAngle) noexcept
            {
                return {inAngle, 0., 0.};
            }


            /// rotate a vector using the quaternion
            ///
            /// @param inVector (cartesian vector with x,y,z components)
            /// @return Matrix<double> (cartesian vector with x,y,z components)
            ///
            [[nodiscard]] Matrix<double> rotate(const Matrix<double> &inVector) const
            {
                if (inVector.size() != 3)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input inVector must be a cartesion vector of length = 3.");
                }

                return *this * inVector;
            }


            /// rotate a vector using the quaternion
            ///
            /// @param inVec3
            /// @return Vec3
            ///
            [[nodiscard]] Vec3 rotate(const Vec3 &inVec3) const
            {
                return *this * inVec3;
            }


            /// returns the s component
            ///
            /// @return double
            ///
            [[nodiscard]] double s() const noexcept
            {
                return components_[3];
            }


            /// spherical linear interpolates between the two quaternions
            ///
            /// @param inQuat1
            /// @param inQuat2
            /// @param inPercent (0, 1)
            /// @return Quaternion
            ///
            static Quaternion slerp(const Quaternion &inQuat1, const Quaternion &inQuat2, double inPercent)
            {
                if (inPercent < 0 || inPercent > 1)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input percent must be of the range [0, 1]");
                }

                if (utils::essentiallyEqual(inPercent, 0.))
                {
                    return inQuat1;
                }
                if (utils::essentiallyEqual(inPercent, 1.))
                {
                    return inQuat2;
                }

                double dotProduct = dot<double>(inQuat1.toMatrix(), inQuat2.toMatrix()).item();

                // If the dot product is negative, the quaternions
                // have opposite handed-ness and slerp won't take
                // the shorter path. Fix by reversing one quaternion.
                Quaternion quat1Copy(inQuat1);
                if (dotProduct < 0.)
                {
                    quat1Copy *= -1.;
                    dotProduct *= -1.;
                }

                constexpr double DOT_THRESHOLD = 0.9995;
                if (dotProduct > DOT_THRESHOLD)
                {
                    // If the inputs are too close for comfort, linearly interpolate
                    // and normalize the result.
                    return nlerp(inQuat1, inQuat2, inPercent);
                }

                dotProduct = clip(dotProduct, -1., 1.); // Robustness: Stay within domain of acos()
                const double theta0 = std::acos(dotProduct); // angle between input vectors
                const double theta = theta0 * inPercent; // angle between v0 and result

                const double s0 = std::cos(theta) -
                    dotProduct * std::sin(theta) / std::sin(theta0); // == sin(theta_0 - theta) / sin(theta_0)
                const double s1 = std::sin(theta) / std::sin(theta0);

                Matrix<double> interpQuat = (quat1Copy.toMatrix() * s0) + (inQuat2.toMatrix() * s1);
                return Quaternion(interpQuat); // NOLINT(modernize-return-braced-init-list)
            }


            /// spherical linear interpolates between the two quaternions
            ///
            /// @param inQuat2
            /// @param inPercent (0, 1)
            /// @return Quaternion
            ///
            [[nodiscard]] Quaternion slerp(const Quaternion &inQuat2, double inPercent) const
            {
                return slerp(*this, inQuat2, inPercent);
            }


            /// returns the quaternion as a string representation
            ///
            /// @return std::string
            ///
            [[nodiscard]] std::string str() const
            {
                std::string output = "[" + utils::num2str(i()) + ", " + utils::num2str(j()) + ", " + utils::num2str(k())
                    +
                    ", " + utils::num2str(s()) + "]\n";

                return output;
            }


            /// returns the direction cosine matrix
            ///
            /// @return Matrix<double>
            ///
            [[nodiscard]] Matrix<double> toDCM() const
            {
                Matrix<double> dcm(3);

                const double q0 = i();
                const double q1 = j();
                const double q2 = k();
                const double q3 = s();

                const double q0sqr = utils::sqr(q0);
                const double q1sqr = utils::sqr(q1);
                const double q2sqr = utils::sqr(q2);
                const double q3sqr = utils::sqr(q3);

                dcm(0, 0) = q3sqr + q0sqr - q1sqr - q2sqr;
                dcm(0, 1) = 2. * (q0 * q1 - q3 * q2);
                dcm(0, 2) = 2. * (q0 * q2 + q3 * q1);
                dcm(1, 0) = 2. * (q0 * q1 + q3 * q2);
                dcm(1, 1) = q3sqr + q1sqr - q0sqr - q2sqr;
                dcm(1, 2) = 2. * (q1 * q2 - q3 * q0);
                dcm(2, 0) = 2. * (q0 * q2 - q3 * q1);
                dcm(2, 1) = 2. * (q1 * q2 + q3 * q0);
                dcm(2, 2) = q3sqr + q2sqr - q0sqr - q1sqr;

                return dcm;
            }


            /// returns the quaternion as an Matrix
            ///
            /// @return Matrix<double>
            ///
            [[nodiscard]] Matrix<double> toMatrix() const
            {
                auto componentsCopy = components_;
                return Matrix<double>(componentsCopy); // NOLINT(modernize-return-braced-init-list)
            }


            /// returns a quaternion to rotate about the x-axis by the input angle
            ///
            /// @param inAngle (radians)
            /// @return Quaternion
            ///
            static Quaternion xRotation(double inAngle) noexcept
            {
                const Vec3 eulerAxis = {1., 0., 0.};
                return Quaternion(eulerAxis, inAngle); // NOLINT(modernize-return-braced-init-list)
            }


            /// The euler yaw angle in radians
            ///
            /// @return euler yaw angle in radians
            ///
            [[nodiscard]] double yaw() const noexcept
            {
                return std::atan2(2. * (s() * k() + i() * j()), 1. - 2. * (utils::sqr(j()) + utils::sqr(k())));
            }


            /// returns a quaternion to rotate about the yaw axis
            ///
            /// @param inAngle (radians)
            /// @return Quaternion
            ///
            static Quaternion yawRotation(double inAngle) noexcept
            {
                return {0., 0., inAngle};
            }


            /// returns a quaternion to rotate about the y-axis by the input angle
            ///
            /// @param inAngle (radians)
            /// @return Quaternion
            ///
            static Quaternion yRotation(double inAngle) noexcept
            {
                const Vec3 eulerAxis = {0., 1., 0.};
                return Quaternion(eulerAxis, inAngle); // NOLINT(modernize-return-braced-init-list)
            }


            /// returns a quaternion to rotate about the y-axis by the input angle
            ///
            /// @param inAngle (radians)
            /// @return Quaternion
            ///
            static Quaternion zRotation(double inAngle) noexcept
            {
                const Vec3 eulerAxis = {0., 0., 1.};
                return Quaternion(eulerAxis, inAngle); // NOLINT(modernize-return-braced-init-list)
            }


            /// equality operator
            ///
            /// @param inRhs
            /// @return bool
            ///
            bool operator==(const Quaternion &inRhs) const noexcept
            {
                const auto comparitor = [](double value1, double value2) noexcept -> bool
                {
                    return utils::essentiallyEqual(value1, value2);
                };

                return algo::equal(components_.begin(), components_.end(), inRhs.components_.begin(),
                                   comparitor);
            }


            /// equality operator
            ///
            /// @param inRhs
            /// @return bool
            ///
            bool operator!=(const Quaternion &inRhs) const noexcept
            {
                return !(*this == inRhs);
            }


            /// addition assignment operator
            ///
            /// @param inRhs
            /// @return Quaternion
            ///
            Quaternion &operator+=(const Quaternion &inRhs) noexcept
            {
                algo::transform(components_.begin(),
                                components_.end(),
                                inRhs.components_.begin(),
                                components_.begin(),
                                std::plus<double>()); // NOLINT(modernize-use-transparent-functors)

                normalize();

                return *this;
            }


            /// addition operator
            ///
            /// @param inRhs
            /// @return Quaternion
            ///
            Quaternion operator+(const Quaternion &inRhs) const noexcept
            {
                return Quaternion(*this) += inRhs;
            }


            /// subtraction assignment operator
            ///
            /// @param inRhs
            /// @return Quaternion
            ///
            Quaternion &operator-=(const Quaternion &inRhs) noexcept
            {
                algo::transform(components_.begin(),
                                components_.end(),
                                inRhs.components_.begin(),
                                components_.begin(),
                                std::minus<double>()); // NOLINT(modernize-use-transparent-functors)

                normalize();

                return *this;
            }


            /// subtraction operator
            ///
            /// @param inRhs
            /// @return Quaternion
            ///
            Quaternion operator-(const Quaternion &inRhs) const noexcept
            {
                return Quaternion(*this) -= inRhs;
            }


            /// negative operator
            ///
            /// @return Quaternion
            ///
            Quaternion operator-() const noexcept
            {
                return Quaternion(*this) *= -1.;
            }


            /// multiplication assignment operator
            ///
            /// @param inRhs
            /// @return Quaternion
            ///
            Quaternion &operator*=(const Quaternion &inRhs) noexcept
            {
                double q0 = inRhs.s() * i();
                q0 += inRhs.i() * s();
                q0 -= inRhs.j() * k();
                q0 += inRhs.k() * j();

                double q1 = inRhs.s() * j();
                q1 += inRhs.i() * k();
                q1 += inRhs.j() * s();
                q1 -= inRhs.k() * i();

                double q2 = inRhs.s() * k();
                q2 -= inRhs.i() * j();
                q2 += inRhs.j() * i();
                q2 += inRhs.k() * s();

                double q3 = inRhs.s() * s();
                q3 -= inRhs.i() * i();
                q3 -= inRhs.j() * j();
                q3 -= inRhs.k() * k();

                components_[0] = q0;
                components_[1] = q1;
                components_[2] = q2;
                components_[3] = q3;

                normalize();

                return *this;
            }


            /// multiplication operator, only useful for multiplying
            /// by negative 1, all others will be renormalized back out
            ///
            /// @param inScalar
            /// @return Quaternion
            ///
            Quaternion &operator*=(double inScalar) noexcept
            {
                algo::for_each(components_.begin(),
                               components_.end(),
                               [inScalar](double &component) { component *= inScalar; });

                normalize();

                return *this;
            }


            /// multiplication operator
            ///
            /// @param inRhs
            /// @return Quaternion
            ///
            Quaternion operator*(const Quaternion &inRhs) const noexcept
            {
                return Quaternion(*this) *= inRhs;
            }


            /// multiplication operator, only useful for multiplying
            /// by negative 1, all others will be renormalized back out
            ///
            /// @param inScalar
            /// @return Quaternion
            ///
            Quaternion operator*(double inScalar) const noexcept
            {
                return Quaternion(*this) *= inScalar;
            }


            /// multiplication operator
            ///
            /// @param inVec
            /// @return Matrix<double>
            ///
            Matrix<double> operator*(const Matrix<double> &inVec) const
            {
                if (inVec.size() != 3)
                {
                    THROW_INVALID_ARGUMENT_ERROR("input vector must be a cartesion vector of length = 3.");
                }

                const auto vecNorm = norm(inVec).item();
                if (utils::essentiallyEqual(vecNorm, 0.))
                {
                    return inVec;
                }

                const auto p = Quaternion(inVec[0], inVec[1], inVec[2], 0.);
                const auto pPrime = *this * p * this->inverse();

                Matrix<double> rotatedVec = {pPrime.i(), pPrime.j(), pPrime.k()};
                rotatedVec *= vecNorm;
                return rotatedVec;
            }


            /// multiplication operator
            ///
            /// @param inVec3
            /// @return Vec3
            ///
            Vec3 operator*(const Vec3 &inVec3) const
            {
                return *this * inVec3.toMatrix();
            }


            /// division assignment operator
            ///
            /// @param inRhs
            /// @return Quaternion
            ///
            Quaternion &operator/=(const Quaternion &inRhs) noexcept
            {
                return *this *= inRhs.conjugate();
            }


            /// division operator
            ///
            /// @param inRhs
            /// @return Quaternion
            ///
            Quaternion operator/(const Quaternion &inRhs) const noexcept
            {
                return Quaternion(*this) /= inRhs;
            }


            /// IO operator for the Quaternion class
            ///
            /// @param inOStream
            /// @param inQuat
            /// @return std::ostream&
            ///
            friend std::ostream &operator<<(std::ostream &inOStream, const Quaternion &inQuat)
            {
                inOStream << inQuat.str();
                return inOStream;
            }

        private:
            //====================================Attributes==============================
            std::array<double, 4> components_{{0., 0., 0., 1.}};


            /// renormalizes the quaternion
            ///
            void normalize() noexcept
            {
                double sumOfSquares = 0.;
                std::for_each(components_.begin(),
                              components_.end(),
                              [&sumOfSquares](double component) noexcept -> void
                              {
                                  sumOfSquares += utils::sqr(component);
                              });

                const double norm = std::sqrt(sumOfSquares);
                algo::for_each(components_.begin(),
                               components_.end(),
                               [norm](double &component) noexcept -> void { component /= norm; });
            }


            /// Converts the euler roll, pitch, yaw angles to quaternion components
            ///
            /// @ param roll: the euler roll angle in radians
            /// @ param pitch: the euler pitch angle in radians
            /// @ param yaw: the euler yaw angle in radians
            ///
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

                components_[0] = sinHalfPhi * cosHalfTheta * cosHalfPsi;
                components_[0] -= cosHalfPhi * sinHalfTheta * sinHalfPsi;

                components_[1] = cosHalfPhi * sinHalfTheta * cosHalfPsi;
                components_[1] += sinHalfPhi * cosHalfTheta * sinHalfPsi;

                components_[2] = cosHalfPhi * cosHalfTheta * sinHalfPsi;
                components_[2] -= sinHalfPhi * sinHalfTheta * cosHalfPsi;

                components_[3] = cosHalfPhi * cosHalfTheta * cosHalfPsi;
                components_[3] += sinHalfPhi * sinHalfTheta * sinHalfPsi;
            }


            /// Converts the direction cosine matrix to quaternion components
            ///
            /// @ param dcm: the direction cosine matrix
            ///
            void dcmToQuat(const Matrix<double> &dcm)
            {
                const Shape inShape = dcm.shape();
                if (!(inShape.m_Rows == 3 && inShape.m_Cols == 3))
                {
                    THROW_INVALID_ARGUMENT_ERROR("input direction cosine matrix must have shape = (3,3).");
                }

                Matrix<double> checks(1, 4);
                checks[0] = 1 + dcm(0, 0) + dcm(1, 1) + dcm(2, 2);
                checks[1] = 1 + dcm(0, 0) - dcm(1, 1) - dcm(2, 2);
                checks[2] = 1 - dcm(0, 0) + dcm(1, 1) - dcm(2, 2);
                checks[3] = 1 - dcm(0, 0) - dcm(1, 1) + dcm(2, 2);

                const uint32 maxIdx = argmax(checks).item();

                switch (maxIdx)
                {
                case 0:
                {
                    components_[3] = 0.5 * std::sqrt(1 + dcm(0, 0) + dcm(1, 1) + dcm(2, 2));
                    components_[0] = (dcm(2, 1) - dcm(1, 2)) / (4 * components_[3]);
                    components_[1] = (dcm(0, 2) - dcm(2, 0)) / (4 * components_[3]);
                    components_[2] = (dcm(1, 0) - dcm(0, 1)) / (4 * components_[3]);

                    break;
                }
                case 1:
                {
                    components_[0] = 0.5 * std::sqrt(1 + dcm(0, 0) - dcm(1, 1) - dcm(2, 2));
                    components_[1] = (dcm(1, 0) + dcm(0, 1)) / (4 * components_[0]);
                    components_[2] = (dcm(2, 0) + dcm(0, 2)) / (4 * components_[0]);
                    components_[3] = (dcm(2, 1) - dcm(1, 2)) / (4 * components_[0]);

                    break;
                }
                case 2:
                {
                    components_[1] = 0.5 * std::sqrt(1 - dcm(0, 0) + dcm(1, 1) - dcm(2, 2));
                    components_[0] = (dcm(1, 0) + dcm(0, 1)) / (4 * components_[1]);
                    components_[2] = (dcm(2, 1) + dcm(1, 2)) / (4 * components_[1]);
                    components_[3] = (dcm(0, 2) - dcm(2, 0)) / (4 * components_[1]);

                    break;
                }
                case 3:
                {
                    components_[2] = 0.5 * std::sqrt(1 - dcm(0, 0) - dcm(1, 1) + dcm(2, 2));
                    components_[0] = (dcm(2, 0) + dcm(0, 2)) / (4 * components_[2]);
                    components_[1] = (dcm(2, 1) + dcm(1, 2)) / (4 * components_[2]);
                    components_[3] = (dcm(1, 0) - dcm(0, 1)) / (4 * components_[2]);

                    break;
                }
                }
            }
        };
    }
}
