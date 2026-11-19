// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <concepts>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

#include <Eigen/Dense>

#include "Core/Types.hpp"
#include "Utils/Profile.hpp"

namespace mc
{
    namespace ode
    {
        struct NoContext {};

        namespace detail
        {
            template <typename Function, typename State, typename Result, typename Context>
            concept SystemFunction = requires(Function &fn, const State &x, double t, const Context &args)
            {
                { std::invoke(fn, x, t, args) } -> std::same_as<Result>;
            };

            template <typename State, typename Context>
            State rk4(const std::function<State(const State &, double, const Context &)> &fn,
                      const State &x, double t, double dt, const Context &args)
            {
                const State k1 = fn(x, t, args);
                const State k2 = fn(x + k1 * dt / 2., t + dt / 2., args);
                const State k3 = fn(x + k2 * dt / 2., t + dt / 2., args);
                const State k4 = fn(x + k3 * dt, t + dt, args);

                State result = x;
                result.noalias() += dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
                return result;
            }

            template <typename State, typename Context, typename Jacobian, typename Basis>
            Basis rk4Tangent(const std::function<State(const State &, double, const Context &)> &fn,
                             const Jacobian &jac, const State &x, double t, double dt, const Context &args,
                             const Basis &basis, const Basis &forcing)
            {
                const double half = dt / 2.0;
                const State x2 = x + half * fn(x, t, args);
                const State x3 = x + half * fn(x2, t + half, args);
                const State x4 = x + dt * fn(x3, t + half, args);
                const Basis k1 = jac(x, t, args) * basis + forcing;
                const Basis k2 = jac(x2, t + half, args) * (basis + half * k1) + forcing;
                const Basis k3 = jac(x3, t + half, args) * (basis + half * k2) + forcing;
                const Basis k4 = jac(x4, t + dt, args) * (basis + dt * k3) + forcing;
                return basis + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            }
        }

        template <int N, typename Context = NoContext>
        class DynamicalSystem
        {
            static_assert(N > 0, "Dynamical system dimension must be positive");

        public:
            using State = Eigen::Vector<double, N>;
            using Jacobian = Eigen::Matrix<double, N, N>;
            using Trajectory = Eigen::Matrix<double, Eigen::Dynamic, N>;
            using ShiftedTrajectory = Eigen::Matrix<double, N, Eigen::Dynamic>;
            using RightHandSide = std::function<State(const State &, double, const Context &)>;
            using JacobianFunction = std::function<Jacobian(const State &, double, const Context &)>;
            using ResetFn = std::function<void(Context &, std::optional<Context> &, uint32_t)>;

            template <typename Derived>
            static State CheckedState(const Eigen::MatrixBase<Derived> &x)
            {
                static_assert(Derived::RowsAtCompileTime == N || Derived::RowsAtCompileTime == Eigen::Dynamic);
                static_assert(Derived::ColsAtCompileTime == 1 || Derived::ColsAtCompileTime == Eigen::Dynamic);
                if constexpr (Derived::RowsAtCompileTime == Eigen::Dynamic ||
                              Derived::ColsAtCompileTime == Eigen::Dynamic)
                {
                    if (x.rows() != N || x.cols() != 1)
                    {
                        throw std::invalid_argument("State must be a column vector matching the system dimension");
                    }
                }
                return x;
            }

            struct SystemState
            {
                SystemState(const State &x0, const State &x,
                            double t0, double t,
                            const Context &args, const Context &default_args,
                            const std::optional<Context> &next_args, const std::optional<Context> &default_next_args) :
                    m_X0(x0), m_X(x), m_T0(t0), m_T(t),
                    m_Args(args), m_NextArgs(next_args), m_DefaultArgs(default_args),
                    m_DefaultNextArgs(default_next_args)
                {
                }

                State m_X0;
                State m_X;
                std::optional<State> m_MemoizedNextStep = std::nullopt;
                double m_T0 = 0;
                double m_T = 0;
                Context m_Args;
                std::optional<Context> m_NextArgs;
                Context m_DefaultArgs;
                std::optional<Context> m_DefaultNextArgs;
            };

            /**
             * @param f Right-hand side of the differential equation.
             * @param jac Jacobian of f with respect to x.
             * @param dt Time interval between two integration steps.
             * @param x0 Initial condition.
             * @param t0 Initial time.
             * @param args Arguments for the primary trajectory.
             * @param args2 Arguments for the secondary trajectory.
             */
            template <typename Function, typename Derivative>
                 requires detail::SystemFunction<Function, State, State, Context> &&
                          detail::SystemFunction<Derivative, State, Jacobian, Context>
            DynamicalSystem(Function &&f, Derivative &&jac, const double dt,
                            const State &x0 = State::Zero(), const double t0 = 0.0,
                            const Context &args = {}, const std::optional<Context> &args2 = {}) :
                m_State(x0, x0, t0, t0, args, args, args2, args2),
                m_Function(std::forward<Function>(f)),
                m_Jacobian(std::forward<Derivative>(jac)),
                m_DeltaTime(dt)
            {
                ValidateContexts(args, args2);
            }

            virtual ~DynamicalSystem() = default;
            DynamicalSystem(const DynamicalSystem &) = delete;
            DynamicalSystem &operator=(const DynamicalSystem &) = delete;

            SystemState GetState() const
            {
                return m_State;
            }
            
            void SetState(const SystemState& state)
            {
                ValidateContexts(state.m_Args, state.m_NextArgs);
                ValidateContexts(state.m_DefaultArgs, state.m_DefaultNextArgs);
                m_State = state;
            }

            /**
             * Shifts the trajectory of the system solution to the next point.
             */
            virtual void ShiftNext(State &x, double &t) = 0;

            /**
             * Shifts the trajectory of the system solution to the next point.
             * The function is not a `const` only because it changes the state of the `model' during calculations.
             * 
             * !!! DO NOW USE KeepDerivative in Preisach with ShiftTrajNext!!!
             */
            virtual State ShiftTrajNext(const State &x, double t, bool memoize = false) = 0;

            template <typename Derived>
            State ShiftTrajNext(const Eigen::MatrixBase<Derived> &x, double t, bool memoize = false)
            {
                return ShiftTrajNext(CheckedState(x), t, memoize);
            }

            /// Compute the state of the system after one time step.
            virtual void Next() = 0;

            /**
             * Compute the state of a deviation vector after one time step.
             * @param w Array of deviations vectors.
             * @return Array of deviation vectors at next time step.
             */
            virtual State NextLTM(const State &w) = 0;

            template <typename Derived>
            State NextLTM(const Eigen::MatrixBase<Derived> &w)
            {
                return NextLTM(CheckedState(w));
            }

            /**
             * Compute the state of the given system after one time step.
             * @param w Array of system vectors.
             * @return Array of system vectors at next time step.
             */
            virtual State NextTM(const State &w) = 0;

            template <typename Derived>
            State NextTM(const Eigen::MatrixBase<Derived> &w)
            {
                return NextTM(CheckedState(w));
            }
            
            State Shift(const State &x0, double period)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Shift");
                
                const SystemState state = GetState();
                Reset();
                
                int numSteps = static_cast<int>(period / m_DeltaTime);

                State x = x0;
                double t = 0.0;
                while (--numSteps)
                {
                    ShiftNext(x, t);
                }
                
                SetState(state);

                return x;
            }

            template <typename Derived>
            State Shift(const Eigen::MatrixBase<Derived> &x0, double period)
            {
                return Shift(CheckedState(x0), period);
            }

            ShiftedTrajectory ShiftTraj(const State &x0, double period)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Shift");
                
                const SystemState state = GetState();
                Reset();
                
                int numSteps = static_cast<int>(period / m_DeltaTime);

                ShiftedTrajectory traj(N, numSteps + 1);
                traj.col(0) = x0;

                double t = 0.0;
                State x = x0;
                for (int i = 1; i < numSteps + 1; i++)
                {
                    ShiftNext(x, t);
                    traj.col(i) = x;
                }
                
                SetState(state);

                return traj;
            }

            template <typename Derived>
            ShiftedTrajectory ShiftTraj(const Eigen::MatrixBase<Derived> &x0, double period)
            {
                return ShiftTraj(CheckedState(x0), period);
            }

            /**
             * Forward the system for time.
             * @param time Time of simulation to take.
              * @return Trajectory of the system of dimension (numSteps + 1, N).
             */
            Trajectory Forward(double time)
            {
                return Forward(static_cast<uint32>(time / m_DeltaTime));
            }
            
            State ForwardWithoutHistory(double time)
            {
                return ForwardWithoutHistory(static_cast<uint32>(time / m_DeltaTime));
            }

            /**
             * Forward two trajectories of the system for time with Args and NextArgs.
             * @param time Time of simulation to take.
             * @param next_x0 Initial value for second trajectory
              * @return Two trajectories (for Args and NextArgs) of dimension (numSteps + 1, N).
             */
            std::pair<Trajectory, Trajectory> ForwardTwoTrajs(double time, const State &next_x0)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Forward");
                GetNextArgs();

                auto numSteps = static_cast<uint32>(time / m_DeltaTime);

                Trajectory traj1(numSteps + 1, N);
                Trajectory traj2(numSteps + 1, N);

                traj1.row(0) = m_State.m_X.transpose();
                traj2.row(0) = next_x0.transpose();

                for (uint32 i = 1; i < numSteps + 1; ++i)
                {
                    traj2.row(i) = NextTM(traj2.row(i - 1).transpose()).transpose();

                    Next();
                    traj1.row(i) = m_State.m_X.transpose();
                }
                return {traj1, traj2};
            }

            template <typename Derived>
            std::pair<Trajectory, Trajectory> ForwardTwoTrajs(double time, const Eigen::MatrixBase<Derived> &next_x0)
            {
                return ForwardTwoTrajs(time, CheckedState(next_x0));
            }

            /**
             * Forward the system for numSteps.
             * @param numSteps Number of simulation steps to take.
              * @return Trajectory of the system of dimension (numSteps + 1, N).
             */
            template <typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int> = 0>
            Trajectory Forward(dtype numSteps)
            {
                AL_PROFILE_FUNC("DynamicalSystem::Forward");
                Trajectory traj(numSteps + 1, N);
                traj.row(0) = m_State.m_X.transpose();

                for (dtype i = 1; i < numSteps + 1; ++i)
                {
                    Next();
                    traj.row(i) = m_State.m_X.transpose();
                }
                return traj;
            }
            
            template <typename dtype, std::enable_if_t<std::is_integral_v<dtype>, int> = 0>
            State ForwardWithoutHistory(dtype numSteps)
            {
                AL_PROFILE_FUNC("DynamicalSystem::ForwardWithoutHistory");
               
                for (dtype i = 1; i < numSteps + 1; ++i)
                {
                    Next();
                }
                return m_State.m_X;
            }

            static constexpr int GetDimension() { return N; }
            double GetDeltaTime() const { return m_DeltaTime; }

            /// Reset a system solution to x0 and time to t0
            void Reset(uint32_t reset_idx = 0)
            {
                m_State.m_X = m_State.m_X0;
                m_State.m_T = m_State.m_T0;
                m_State.m_MemoizedNextStep = std::nullopt;

                CallResetFn(reset_idx);
            }

            /// Update x0 and reset a system solution to x0 and time to t0
            void ResetTo(const State &x0, uint32_t reset_idx = 0)
            {
                m_State.m_X0 = x0;
                m_State.m_X = m_State.m_X0;
                m_State.m_T = m_State.m_T0;
                m_State.m_MemoizedNextStep = std::nullopt;

                CallResetFn(reset_idx);
            }

            template <typename Derived>
            void ResetTo(const Eigen::MatrixBase<Derived> &x0, uint32_t reset_idx = 0)
            {
                ResetTo(CheckedState(x0), reset_idx);
            }

            void ResetArgs()
            {
                ValidateContexts(m_State.m_DefaultArgs, m_State.m_NextArgs);
                m_State.m_Args = m_State.m_DefaultArgs;
                m_State.m_MemoizedNextStep.reset();
            }

            void ResetNextArgs()
            {
                ValidateContexts(m_State.m_Args, m_State.m_DefaultNextArgs);
                m_State.m_NextArgs = m_State.m_DefaultNextArgs;
            }

            void SetX0(const State &x0)
            {
                m_State.m_X0 = x0;
            }

            template <typename Derived>
            void SetX0(const Eigen::MatrixBase<Derived> &x0)
            {
                SetX0(CheckedState(x0));
            }

            /// Callbacks update the contexts explicitly associated with each trajectory.
            void SetResetFn(const ResetFn &fn)
            {
                m_ResetFn = fn;
            }

            void CallResetFn(uint32_t reset_idx = 0)
            {
                if (m_ResetFn.has_value())
                {
                    auto args = m_State.m_Args;
                    auto next = m_State.m_NextArgs;
                    m_ResetFn.value()(args, next, reset_idx);
                    SetContexts(args, next);
                }
            }

            void ResetSystemTime()
            {
                m_State.m_T = m_State.m_T0;
            }

            /// Replace current parameters. ResetArgs restores the separately stored defaults.
            void SetArgs(const Context &args)
            {
                ValidateContexts(args, m_State.m_NextArgs);
                m_State.m_Args = args;
                m_State.m_MemoizedNextStep.reset();
            }

            void SetNextArgs(const Context &args)
            {
                ValidateContexts(m_State.m_Args, args);
                m_State.m_NextArgs = args;
            }

            void SetDefaultArgs(const Context &args)
            {
                ValidateContexts(args, m_State.m_DefaultNextArgs);
                m_State.m_DefaultArgs = args;
            }

            void SetDefaultNextArgs(const Context &args)
            {
                ValidateContexts(m_State.m_DefaultArgs, args);
                m_State.m_DefaultNextArgs = args;
            }

            void SetContexts(const Context &primary, const std::optional<Context> &perturbed)
            {
                ValidateContexts(primary, perturbed);
                m_State.m_Args = primary;
                m_State.m_NextArgs = perturbed;
                m_State.m_MemoizedNextStep.reset();
            }

            static void ValidateContexts(const Context &primary, const std::optional<Context> &perturbed)
            {
                if constexpr (requires { Context::ValidatePair(primary, perturbed); })
                {
                    Context::ValidatePair(primary, perturbed);
                }
            }

            const Context &GetArgs() const
            {
                return m_State.m_Args;
            }

            const Context &GetNextArgs() const
            {
                if (!m_State.m_NextArgs) throw std::invalid_argument("A second trajectory context is required");
                return *m_State.m_NextArgs;
            }

            State GetX() const
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

            RightHandSide m_Function;
            JacobianFunction m_Jacobian;
            std::optional<ResetFn> m_ResetFn;
            double m_DeltaTime;
        };

        /// Continuous dynamical system
        template <int N, typename Context = NoContext>
        class ContinuousDS : public DynamicalSystem<N, Context>
        {
            using Base = DynamicalSystem<N, Context>;
            using Base::m_Function;
            using Base::m_Jacobian;
            using Base::m_DeltaTime;
            using Base::m_State;

        public:
            using typename Base::State;
            using typename Base::Jacobian;
            using Base::NextLTM;
            using Base::NextTM;
            using Base::ShiftTrajNext;

            template <typename Function, typename Derivative>
                 requires detail::SystemFunction<Function, State, State, Context> &&
                          detail::SystemFunction<Derivative, State, Jacobian, Context>
            ContinuousDS(Function &&f, Derivative &&jac, double dt, const Context &args = {},
                         const std::optional<Context> &args2 = {}, const State &x0 = State::Zero(), double t0 = 0.0) :
                Base(std::forward<Function>(f), std::forward<Derivative>(jac), dt, x0, t0, args, args2)
            {
            }

            template <typename Function, typename Derivative, typename Derived>
                 requires detail::SystemFunction<Function, State, State, Context> &&
                          detail::SystemFunction<Derivative, State, Jacobian, Context>
            ContinuousDS(Function &&f, Derivative &&jac, double dt, const Context &args,
                         const std::optional<Context> &args2, const Eigen::MatrixBase<Derived> &x0, double t0 = 0.0) :
                ContinuousDS(std::forward<Function>(f), std::forward<Derivative>(jac), dt, args, args2,
                             Base::CheckedState(x0), t0)
            {
            }

            void ShiftNext(State &x, double &t) override
            {
                AL_PROFILE_FUNC("ContinuousDS::ShiftNext");
                x = detail::rk4(m_Function, x, t, m_DeltaTime, m_State.m_Args);
                t += m_DeltaTime;
            }

            State ShiftTrajNext(const State &x, double t, bool memoize = false) override
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
            State NextLTM(const State &w) override
            {
                AL_PROFILE_FUNC("ContinuousDS::NextLTM");
                const State forcing = State::Zero();
                return detail::rk4Tangent(m_Function, m_Jacobian, m_State.m_X, m_State.m_T, m_DeltaTime,
                                          m_State.m_Args, w, forcing);
            }

            State NextTM(const State &w) override
            {
                AL_PROFILE_FUNC("ContinuousDS::NextTM");
                if (!m_State.m_NextArgs) throw std::invalid_argument("A second trajectory context is required");
                return detail::rk4(m_Function, w, m_State.m_T, m_DeltaTime, *m_State.m_NextArgs);
            }
        };
    }
}
