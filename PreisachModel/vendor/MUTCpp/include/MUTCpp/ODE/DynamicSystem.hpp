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
                const std::function<Vec(const Vec &, double, DSArgs &)> &fn,
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
            using ResetFn = std::function<void(DSArgs &, DSArgs &, uint32_t)>;

            struct SystemState
            {
                SystemState(const Vec &x0, const Vec &x,
                            double t0, double t,
                            const DSArgs &args, const DSArgs &default_args,
                            const DSArgs &next_args, const DSArgs &default_next_args) :
                    m_X0(x0), m_X(x), m_T0(t0), m_T(t),
                    m_Args(args), m_NextArgs(next_args), m_DefaultArgs(default_args),
                    m_DefaultNextArgs(default_next_args)
                {
                }

                Vec m_X0;
                Vec m_X;
                std::optional<Vec> m_MemoizedNextStep = std::nullopt;
                double m_T0 = 0;
                double m_T = 0;
                DSArgs m_Args, m_NextArgs, m_DefaultArgs, m_DefaultNextArgs;
            };

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
                m_State(x0, x0, t0, t0, args, args, args2, args2),
                m_Dimension(x0.size()), m_Function(std::move(f)), m_Jacobian(std::move(jac)), m_DeltaTime(dt)
            {
            }

            virtual ~DynamicalSystem() override = default;

            SystemState GetState() const noexcept
            {
                return m_State;
            }
            
            void SetState(const SystemState& state) noexcept
            {
                m_State = state;
            }

            /**
             * Shifts the trajectory of the system solution to the next point.
             */
            virtual void ShiftNext(Vec &x, double &t) = 0;

            /**
             * Shifts the trajectory of the system solution to the next point.
             * The function is not a `const` only because it changes the state of the `model' during calculations.
             * 
             * !!! DO NOW USE KeepDerivative in Preisach with ShiftTrajNext!!!
             */
            virtual Vec ShiftTrajNext(const Vec &x, double t, bool memoize = false) = 0;

            /// Compute the state of the system after one time step.
            virtual void Next() = 0;

            /**
             * Compute the state of a deviation vector after one time step.
             * @param w Array of deviations vectors.
             * @return Array of deviation vectors at next time step.
             */
            virtual Vec NextLTM(const Vec &w) = 0;

            /**
             * Compute the state of the given system after one time step.
             * @param w Array of system vectors.
             * @return Array of system vectors at next time step.
             */
            virtual Vec NextTM(const Vec &w) = 0;
            
            Vec Shift(const Vec &x0, double period)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Shift");
                
                const SystemState state = GetState();
                Reset();
                
                int numSteps = static_cast<int>(period / m_DeltaTime);

                Vec x = x0;
                double t = 0.0;
                while (--numSteps)
                {
                    ShiftNext(x, t);
                }
                
                SetState(state);

                return x;
            }

            Mat ShiftTraj(const Vec &x0, double period)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Shift");
                
                const SystemState state = GetState();
                Reset();
                
                int numSteps = static_cast<int>(period / m_DeltaTime);

                Mat traj = Mat::Zero(m_Dimension, numSteps + 1);
                traj.col(0) = x0;

                double t = 0.0;
                Vec x = x0;
                for (int i = 1; i < numSteps + 1; i++)
                {
                    ShiftNext(x, t);
                    traj.col(i) = x;
                }
                
                SetState(state);

                return traj;
            }

            /**
             * Forward the system for time.
             * @param time Time of simulation to take.
             * @return Trajectory of the system of dimension (numSteps + 1, m_Dimension).
             */
            Mat Forward(double time)
            {
                return Forward(static_cast<uint32>(time / m_DeltaTime));
            }
            
            Vec ForwardWithoutHistory(double time)
            {
                return ForwardWithoutHistory(static_cast<uint32>(time / m_DeltaTime));
            }

            /**
             * Forward two trajectories of the system for time with Args and NextArgs.
             * @param time Time of simulation to take.
             * @param next_x0 Initial value for second trajectory
             * @return Two trajectories (for Args and NextArgs) of the system of dimension (numSteps + 1, m_Dimension).
             */
            std::pair<Mat, Mat> ForwardTwoTrajs(double time, const Eigen::VectorXd &next_x0)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Forward");

                auto numSteps = static_cast<uint32>(time / m_DeltaTime);

                Mat traj1 = Mat::Zero(numSteps + 1, m_Dimension);
                Mat traj2 = Mat::Zero(numSteps + 1, m_Dimension);

                traj1.row(0) = m_State.m_X.transpose();
                traj2.row(0) = next_x0.transpose();

                for (uint32 i = 1; i < numSteps + 1; ++i)
                {
                    traj2.row(i) = NextTM(traj2.row(i - 1));

                    Next();
                    traj1.row(i) = m_State.m_X.transpose();
                }
                return {traj1, traj2};
            }

            /**
             * Forward the system for numSteps.
             * @param numSteps Number of simulation steps to take.
             * @return Trajectory of the system of dimension (numSteps + 1, m_Dimension) if keepTraj.
             */
            template <typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int> = 0>
            Mat Forward(dtype numSteps)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Forward");
                Mat traj = Mat::Zero(numSteps + 1, m_Dimension);
                traj.row(0) = m_State.m_X.transpose();

                for (dtype i = 1; i < numSteps + 1; ++i)
                {
                    Next();
                    traj.row(i) = m_State.m_X.transpose();
                }
                return traj;
            }
            
            template <typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int> = 0>
            Vec ForwardWithoutHistory(dtype numSteps)
            {
                AL_PROFILE_FUNC("DynamicalSystem::ForwardWithoutHistory");
               
                for (dtype i = 1; i < numSteps + 1; ++i)
                {
                    Next();
                }
                return m_State.m_X;
            }

            int GetDimension() const { return m_Dimension; }
            double GetDeltaTime() const { return m_DeltaTime; }

            /// Reset a system solution to x0 and time to t0
            void Reset(uint32_t reset_idx = 0)
            {
                m_State.m_X = m_State.m_X0;
                m_State.m_T = m_State.m_T0;
                m_State.m_MemoizedNextStep = std::nullopt;

                if (m_ResetFn.has_value())
                {
                    m_ResetFn.value()(m_State.m_Args, m_State.m_NextArgs, reset_idx);
                }
            }

            /// Update x0 and reset a system solution to x0 and time to t0
            void ResetTo(const Vec &x0, uint32_t reset_idx = 0)
            {
                m_State.m_X0 = x0;
                m_State.m_X = m_State.m_X0;
                m_State.m_T = m_State.m_T0;
                m_State.m_MemoizedNextStep = std::nullopt;

                if (m_ResetFn.has_value())
                {
                    m_ResetFn.value()(m_State.m_Args, m_State.m_NextArgs, reset_idx);
                }
            }

            void ResetArgs()
            {
                m_State.m_Args = m_State.m_DefaultArgs;
            }

            void ResetNextArgs()
            {
                m_State.m_NextArgs = m_State.m_DefaultNextArgs;
            }

            void SetX0(const Vec &x0)
            {
                m_State.m_X0 = x0;
            }

            void SetResetFn(const ResetFn &fn)
            {
                m_ResetFn = fn;
            }

            void CallResetFn(uint32_t reset_idx = 0)
            {
                if (m_ResetFn.has_value())
                {
                    m_ResetFn.value()(m_State.m_Args, m_State.m_NextArgs, reset_idx);
                }
            }

            void ResetSystemTime()
            {
                m_State.m_T = m_State.m_T0;
            }

            void SetArgs(const DSArgs &args, bool keepAsDefault = false)
            {
                SetDefaultArgs(args);
                m_State.m_Args = args;
            }

            void SetNextArgs(const DSArgs &args, bool keepAsDefault = false)
            {
                SetDefaultNextArgs(args);
                m_State.m_NextArgs = args;
            }

            void SetDefaultArgs(const DSArgs &args)
            {
                m_State.m_DefaultArgs = args;
            }

            void SetDefaultNextArgs(const DSArgs &args)
            {
                m_State.m_DefaultNextArgs = args;
            }

            void AddAndSetArg(const std::string &name, const Vote &arg)
            {
                m_State.m_Args.insert_or_assign(name, arg);
            }

            void SetArg(const std::string &name, const Vote &arg)
            {
                m_State.m_Args.insert_or_assign(name, arg);
            }

            void SetNextArg(const std::string &name, const Vote &arg)
            {
                m_State.m_NextArgs.insert_or_assign(name, arg);
            }

            bool ContainsArg(const std::string &argName) const
            {
                return m_State.m_Args.contains(argName);
            }

            DSArgs GetArgs() const
            {
                return m_State.m_Args;
            }

            DSArgs GetNextArgs() const
            {
                return m_State.m_NextArgs;
            }

            Vec GetX() const
            {
                return m_State.m_X;
            }

            double GetT() const
            {
                return m_State.m_T;
            }

            // mc::json::JsonDocument GetJson() const
            // {
            //     mc::json::JsonDocument doc;
            // }

        protected:
            SystemState m_State;

            int m_Dimension = 1;
            DSFunc m_Function;
            DSFunc m_Jacobian;
            std::optional<ResetFn> m_ResetFn;
            double m_DeltaTime;
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

            virtual void ShiftNext(Vec &x, double &t) override
            {
                AL_PROFILE_FUNC("ContinuousDS::ShiftNext");
                x = detail::rk4(m_Function, x, t, m_DeltaTime, m_State.m_Args);
                t += m_DeltaTime;
            }

            virtual Vec ShiftTrajNext(const Vec &x, double t, bool memoize) override
            {
                AL_PROFILE_FUNC("ContinuousDS::ShiftTrajNext");
                if (memoize)
                {
                    m_State.m_MemoizedNextStep = detail::rk4(m_Function, x, t, m_DeltaTime, m_State.m_Args);
                    return *m_State.m_MemoizedNextStep;
                }
                m_State.m_MemoizedNextStep = std::nullopt;
                return detail::rk4(m_Function, x, t, m_DeltaTime, m_State.m_Args);
            }

            /// Compute the state of the system after one time step with RK4 method.
            virtual void Next() override
            {
                AL_PROFILE_FUNC("ContinuousDS::Next");

                if (m_State.m_MemoizedNextStep.has_value())
                {
                    m_State.m_X = *m_State.m_MemoizedNextStep;
                    m_State.m_MemoizedNextStep = std::nullopt;
                }
                else
                {
                    m_State.m_X = detail::rk4(m_Function, m_State.m_X, m_State.m_T, m_DeltaTime, m_State.m_Args);
                }

                m_State.m_T += m_DeltaTime;
            }

            /**
             * Compute the state of a deviation vector after one time step with RK4 method.
             * @param w Array of deviations vectors.
             * @return Array of deviations vectors at next time step
             */
            virtual Vec NextLTM(const Vec &w) override
            {
                AL_PROFILE_FUNC("ContinuousDS::NextLTM");
                auto jac = m_Jacobian(m_State.m_X, m_State.m_T, m_State.m_Args);
                auto k1 = jac * w;
                auto k2 = jac * (w + (m_DeltaTime * 0.5) * k1);
                auto k3 = jac * (w + (m_DeltaTime * 0.5) * k2);
                auto k4 = jac * (w + m_DeltaTime * k3);
                auto res = w + (m_DeltaTime / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                return res;
            }

            virtual Vec NextTM(const Vec &w) override
            {
                AL_PROFILE_FUNC("ContinuousDS::NextTM");
                // auto k1 = m_Function(w, m_T, m_NextArgs);
                // auto k2 = m_Function(w + (m_DeltaTime / 2.0) * k1, m_T + (m_DeltaTime / 2.0), m_NextArgs);
                // auto k3 = m_Function(w + (m_DeltaTime / 2.0) * k2, m_T + (m_DeltaTime / 2.0), m_NextArgs);
                // auto k4 = m_Function(w + m_DeltaTime * k3, m_T + m_DeltaTime, m_NextArgs);
                // auto res = w + (m_DeltaTime / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                // return res;
                return detail::rk4(m_Function, w, m_State.m_T, m_DeltaTime, m_State.m_NextArgs);
            }
        };
    }
}
