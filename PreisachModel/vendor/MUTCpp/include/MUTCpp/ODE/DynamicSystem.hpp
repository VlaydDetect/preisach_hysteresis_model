// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <functional>
#include <utility>

#include <Eigen/Dense>

#include "DSArgs.hpp"

namespace mc
{
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;
    
    namespace ode
    {
        namespace detail
        {
            inline Vec rk4(
                const std::function<Vec(const Vec&, double, DSArgs &)> &fn,
                const Vec &x, double t, double dt, DSArgs &args)
            {
                const Vec k1 = fn(x, t, args);
                const Vec k2 = fn(x + k1 * dt / 2., t + dt / 2., args);
                const Vec k3 = fn(x + k2 * dt / 2., t + dt / 2., args);
                const Vec k4 = fn(x + k3 * dt, t + dt, args);

                Vec result = x;
                result.noalias() += dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
                return result;
            }
        }

        class DynamicalSystem : public RefCounted
        {
        public:
            using DSFunc = std::function<Mat(const Vec &, double, DSArgs &)>;
            using ResetFn = std::function<void(DSArgs &, DSArgs &)>;

        public:
            /**
         * Instantiation of a dynamical system.
         * @param x0 Initial condition.
         * @param t0 Initial time.
         * @param f Function of ẋ = f(x,t).
         * @param jac Jacobin of f with respect to x.
         * @param dt Time interval between two time steps.
         * @param args Additional arguments for f(x, t) and Jacobin
         * @param args2
         */
            DynamicalSystem(const DSFunc &f, const DSFunc &jac, const double dt,
                            const Vec &x0 = Eigen::Vector3d::Zero(), const double t0 = 0.0,
                            const DSArgs &args = {}, const DSArgs &args2 = {}) :
                m_X0(x0), m_X(x0), m_T0(t0), m_T(t0), m_Dimension(x0.size()), m_Function(std::move(f)),
                m_Jacobian(std::move(jac)), m_DeltaTime(dt), m_Args(std::move(args)), m_NextArgs(std::move(args))
            {
            }

            virtual ~DynamicalSystem() override = default;

            virtual void ShiftNext(Vec& x, double& t) = 0;

            /// Compute the state of the system after one time step.
            virtual void Next() = 0;

            /**
             * Compute the state of a deviation vector after one time step.
             * @param w Array of deviations vectors.
             * @return Array of deviation vectors at next time step.
             */
            virtual Vec NextLTM(Vec w) = 0;

            /**
             * Compute the state of the given system after one time step.
             * @param w Array of system vectors.
             * @return Array of system vectors at next time step.
             */
            virtual Vec NextTM(Vec w) = 0;

            Vec Shift(const Vec &x0, double period)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Shift");
                int numSteps = static_cast<int>(period / m_DeltaTime);

                Vec x = x0;
                double t = 0.0;
                while (--numSteps)
                {
                    ShiftNext(x, t);
                }

                return x;
            }
            
            /**
             * Forward the system for numSteps.
             * @param time Time of simulation to take.
             * @return Trajectory of the system of dimension (numSteps + 1, m_Dimension).
             */
            // template<typename dtype, std::enable_if_t<std::is_floating_point_v<dtype>, int> = 0>
            Mat Forward(double time)
            {
                return Forward(static_cast<uint32>(time / m_DeltaTime));
            }

            /**
             * Forward the system for numSteps.
             * @param numSteps Number of simulation steps to take.
             * @return Trajectory of the system of dimension (numSteps + 1, m_Dimension) if keepTraj.
             */
            template <typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int>  = 0>
            Mat Forward(dtype numSteps)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Forward");
                Mat traj = Mat::Zero(numSteps + 1, m_Dimension);
                traj.row(0) = m_X.transpose();
                
                for (dtype i = 1; i < numSteps + 1; ++i)
                {
                    Next();
                    traj.row(i) = m_X.transpose();
                }
                return traj;
            }

            int GetDimension() const { return m_Dimension; }
            double GetDeltaTime() const { return m_DeltaTime; }

            /// Reset a system solution to x0 and time to t0
            void Reset()
            {
                m_X = m_X0;
                m_T = m_T0;

                if (m_ResetFn.has_value())
                {
                    m_ResetFn.value()(m_Args, m_NextArgs);
                }
            }

            void SetX0(const Vec &x0)
            {
                m_X0 = x0;
            }

            void SetResetFn(const ResetFn &fn)
            {
                m_ResetFn = fn;
            }

            void CallResetFn()
            {
                if (m_ResetFn.has_value())
                {
                    m_ResetFn.value()(m_Args, m_NextArgs);
                }
            }

            void ResetSystemTime()
            {
                m_T = m_T0;
            }

            /// Update x0 and reset a system solution to x0 and time to t0
            void Reset(const Vec &x0)
            {
                m_X0 = x0;
                m_X = m_X0;
                m_T = m_T0;

                if (m_ResetFn.has_value())
                {
                    m_ResetFn.value()(m_Args, m_NextArgs);
                }
            }

            void AddAndSetArg(const std::string &name, const Vote &arg)
            {
                m_Args.insert_or_assign(name, arg);
            }

            void SetArg(const std::string &name, const Vote &arg)
            {
                m_Args.insert_or_assign(name, arg);
            }

            void SetNextArg(const std::string &name, const Vote &arg)
            {
                m_NextArgs.insert_or_assign(name, arg);
            }

            bool ContainsArg(const std::string &argName) const
            {
                return m_Args.contains(argName);
            }

            DSArgs GetArgs() const
            {
                return m_Args;
            }

            DSArgs GetNextArgs() const
            {
                return m_NextArgs;
            }

            Vec GetX() const
            {
                return m_X;
            }

        protected:
            Vec m_X0;
            Vec m_X;
            double m_T0 = 0;
            double m_T = 0;
            int m_Dimension = 1;
            DSFunc m_Function;
            DSFunc m_Jacobian;
            std::optional<ResetFn> m_ResetFn;
            double m_DeltaTime;
            DSArgs m_Args, m_NextArgs;

            friend std::vector<std::pair<double, int>> mLCE_DivergenceDegreeTest(DynamicalSystem *system,
                double timeForward, double e, double T, int M, uint8_t n, bool linearize, bool useMult);

            friend std::pair<double, Matrix<double>> Benettin_mLCE(DynamicalSystem *system, double timeForward,
                                                                   double e, double T, int M, bool linearize);
        };

        /// Continuous dynamical system
        class ContinuousDS : public DynamicalSystem
        {
        public:
            ContinuousDS(const DSFunc &f, const DSFunc &jac, const double dt, const DSArgs &args = {},
                         const DSArgs &args2 = {}, const Vec &x0 = Eigen::Vector3d::Zero(), const double t0 = 0.0) :
                DynamicalSystem(f, jac, dt, x0, t0, args, args2)
            {
            }

            virtual void ShiftNext(Vec& x, double& t) override
            {
                AL_PROFILE_FUNC("ContinuousDS::ShiftNext");
                x = detail::rk4(m_Function, x, t, m_DeltaTime, m_Args);
                t += m_DeltaTime;
            }

            /// Compute the state of the system after one time step with RK4 method.
            virtual void Next() override
            {
                AL_PROFILE_FUNC("ContinuousDS::Next");
                // auto k1 = m_Function(m_X, m_T, m_Args);
                // auto k2 = m_Function(m_X + (m_DeltaTime / 2.0) * k1, m_T + (m_DeltaTime / 2.0), m_Args);
                // auto k3 = m_Function(m_X + (m_DeltaTime / 2.0) * k2, m_T + (m_DeltaTime / 2.0), m_Args);
                // auto k4 = m_Function(m_X + m_DeltaTime * k3, m_T + m_DeltaTime, m_Args);
                // m_X += (m_DeltaTime / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                // m_T += m_DeltaTime;

                m_X = detail::rk4(m_Function, m_X, m_T, m_DeltaTime, m_Args);
                m_T += m_DeltaTime;
            }

            /**
             * Compute the state of a deviation vector after one time step with RK4 method.
             * @param w Array of deviations vectors.
             * @return Array of deviations vectors at next time step
             */
            virtual Vec NextLTM(Vec w) override
            {
                AL_PROFILE_FUNC("ContinuousDS::NextLTM");
                auto jac = m_Jacobian(m_X, m_T, m_Args);
                auto k1 = jac * w;
                auto k2 = jac * (w + (m_DeltaTime * 0.5) * k1);
                auto k3 = jac * (w + (m_DeltaTime * 0.5) * k2);
                auto k4 = jac * (w + m_DeltaTime * k3);
                auto res = w + (m_DeltaTime / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                return res;
            }

            virtual Vec NextTM(Vec w) override
            {
                AL_PROFILE_FUNC("ContinuousDS::NextTM");
                auto k1 = m_Function(w, m_T, m_NextArgs);
                auto k2 = m_Function(w + (m_DeltaTime / 2.0) * k1, m_T + (m_DeltaTime / 2.0), m_NextArgs);
                auto k3 = m_Function(w + (m_DeltaTime / 2.0) * k2, m_T + (m_DeltaTime / 2.0), m_NextArgs);
                auto k4 = m_Function(w + m_DeltaTime * k3, m_T + m_DeltaTime, m_NextArgs);
                auto res = w + (m_DeltaTime / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                return res;
                // return detail::rk4(m_Function, w, m_T, m_DeltaTime, m_NextArgs);
            }
        };
    }
}
