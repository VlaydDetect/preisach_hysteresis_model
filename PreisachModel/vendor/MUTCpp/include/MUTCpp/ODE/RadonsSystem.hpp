// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#include <memory>
#include <cmath>
#include <limits>

#include "DynamicSystem.hpp"
#include "EventStep.hpp"
#include "Hysteresis/ModelBase.hpp"

namespace mc
{
    namespace ode
    {
        struct RadonsParams
        {
            double dt;
            double gamma;
            double A;
            double w0;
            double w;
            double E;
            double phi = 0.0;
            double eps = 0.0;
        };

        struct RadonsContext
        {
            RadonsParams params;
            std::shared_ptr<PreisachModelBase> model;

            static void ValidatePair(const RadonsContext &primary, const std::optional<RadonsContext> &perturbed)
            {
                if (!primary.model || (perturbed && !perturbed->model))
                {
                    throw std::invalid_argument("Each Radons trajectory requires a Preisach model");
                }
                if (perturbed && primary.model == perturbed->model)
                {
                    throw std::invalid_argument("Radons trajectories require distinct Preisach models");
                }
            }
        };

        struct RadonsSnapshot
        {
            double time;
            Eigen::Vector2d x;
            RadonsParams params;
            std::shared_ptr<const PreisachModelBase> memory;
        };

        enum class RadonsIntegrator { RK4, EventAware };
        enum class RadonsOutputStrategy { Hold, PerStage };

        class RadonsSystem : public ContinuousDS<2, RadonsContext>
        {
            using Base = DynamicalSystem<2, RadonsContext>;

        public:
            using ContinuousDS::ContinuousDS;
            using ContinuousDS::NextLTM;
            using ContinuousDS::NextTM;
            using ContinuousDS::ShiftTrajNext;

            struct Trial
            {
                double time;
                State x;
                std::shared_ptr<PreisachModelBase> memory;
                std::optional<detail::LocatedEvent> event;
            };

            using TangentBasis = Eigen::Matrix<double, 2, Eigen::Dynamic>;

            struct TangentState
            {
                TangentBasis physical;
                Eigen::MatrixXd extrema;
                Eigen::RowVectorXd time;
            };

            static TangentState InitialTangents(const PreisachModelBase &memory, const TangentBasis &basis)
            {
                const Eigen::MatrixXd fixed = Eigen::MatrixXd::Zero(memory.ExtremaCount(), basis.cols());
                return {basis, memory.TransferExtrema(memory, fixed, basis.row(0)),
                        Eigen::RowVectorXd::Zero(basis.cols())};
            }

            void SetIntegrator(RadonsIntegrator integrator, EventTolerance tolerance = {})
            {
                if (!std::isfinite(tolerance.time) || tolerance.time <= 0.0 ||
                    !std::isfinite(tolerance.state) || tolerance.state <= 0.0 || tolerance.subdivisions < 1 ||
                    tolerance.subdivisions == std::numeric_limits<int>::max())
                {
                    throw std::invalid_argument("Invalid event tolerance");
                }
                if (integrator == RadonsIntegrator::EventAware && !GetArgs().model->SupportsContinuousEvents())
                {
                    throw std::invalid_argument("The model does not support continuous Preisach events");
                }
                m_Integrator = integrator;
                m_EventTolerance = tolerance;
            }

            RadonsIntegrator GetIntegrator() const { return m_Integrator; }

            /// Output policy for ordinary RK4. Event-aware integration always holds the output on each substep.
            void SetOutputStrategy(RadonsOutputStrategy strategy)
            {
                if (strategy == RadonsOutputStrategy::PerStage && !GetArgs().model->SupportsContinuousEvents())
                {
                    throw std::invalid_argument("Stagewise Preisach output requires a continuous model");
                }
                m_OutputStrategy = strategy;
            }

            RadonsOutputStrategy GetOutputStrategy() const { return m_OutputStrategy; }

            void Next() override
            {
                ShiftNext(Base::m_State.m_X, Base::m_State.m_T);
            }

            void ShiftNext(State &x, double &t) override
            {
                x = Advance(x, t, GetArgs(), true);
            }

            State ShiftTrajNext(const State &x, double t, bool = false) override
            {
                return Advance(x, t, GetArgs(), false);
            }

            State NextTM(const State &x) override
            {
                double time = GetT();
                return Advance(x, time, GetNextArgs(), true);
            }

            State NextLTM(const State &w) override
            {
                auto x = GetX();
                double time = GetT();
                auto args = PrepareContext(x, GetArgs());
                if (m_Integrator == RadonsIntegrator::EventAware)
                {
                    auto tangent = InitialTangents(*args.model, w);
                    State fixed = w;
                    VisitEventStep(x, time, args, [&](const State &before, double t, const RadonsContext &context,
                                                       const Trial &point, bool last)
                    {
                        DifferentiateSubstep(before, t, context, point, tangent);
                        if (last)
                        {
                            fixed = FixedTimeBasis(point, context.params, tangent).col(0);
                        }
                    });
                    return fixed;
                }
                if (m_OutputStrategy == RadonsOutputStrategy::PerStage)
                {
                    return StagewiseTangent(x, time, args, w);
                }
                State forcing = State::Zero();
                forcing(1) = args.params.E * args.model->CurrentSlope() * w(0);
                return detail::rk4Tangent(Base::m_Function, Base::m_Jacobian, x, time, GetDeltaTime(), args, w, forcing);
            }

            Jacobian EvaluateBranchJacobian(const State &x, double t) const
            {
                const auto args = PrepareContext(x, GetArgs());
                Jacobian result = Base::m_Jacobian(x, t, args);
                result(1, 0) += args.params.E * args.model->CurrentSlope();
                return result;
            }

            Trial ProbeSubstep(const State &x, double t, double dt, const RadonsContext &args, bool locate_events) const
            {
                auto working = PrepareContext(x, args);
                const auto sample = [&](double elapsed)
                {
                    return detail::rk4(Base::m_Function, x, t, elapsed, working);
                };
                std::optional<detail::LocatedEvent> event;
                if (locate_events)
                {
                    if (!working.model->SupportsContinuousEvents())
                    {
                        throw std::invalid_argument("The model does not support continuous Preisach events");
                    }
                    const bool increasing = x(1) > 0.0 || (x(1) == 0.0 && Base::m_Function(x, t, working)(1) >= 0.0);
                    std::vector<PreisachThreshold> thresholds;
                    working.model->AppendInputSurfaces(thresholds, increasing);
                    std::vector<detail::EventSurface> surfaces = {{1, 0.0}};
                    for (const auto &threshold : thresholds)
                    {
                        const detail::EventSurface surface{0, threshold.level, threshold.extremum};
                        if (std::ranges::find(surfaces, surface) == surfaces.end())
                        {
                            surfaces.push_back(surface);
                        }
                    }
                    event = detail::FindFirstCrossing(sample, dt, surfaces, m_EventTolerance);
                }
                const double elapsed = event && dt - event->elapsed > m_EventTolerance.time ? event->elapsed : dt;
                State next;
                if (!locate_events && m_OutputStrategy == RadonsOutputStrategy::PerStage)
                {
                    if (!working.model->SupportsContinuousEvents())
                    {
                        throw std::invalid_argument("Stagewise Preisach output requires a continuous model");
                    }
                    const RightHandSide staged = [this](const State &state, double time, const RadonsContext &initial)
                    {
                        return Base::m_Function(state, time, PrepareContext(state, initial));
                    };
                    next = detail::rk4(staged, x, t, elapsed, working);
                }
                else
                {
                    next = sample(elapsed);
                }
                if (event)
                {
                    for (const auto &surface : event->surfaces)
                    {
                        if (std::abs(surface.Value(next)) > m_EventTolerance.state)
                        {
                            throw std::runtime_error("Coincident Preisach event surfaces cannot be resolved together");
                        }
                        next(surface.coordinate) = surface.level;
                    }
                }
                working.model->AdvanceInput(next(0), next(1));
                return {t + elapsed, next, std::move(working.model), std::move(event)};
            }

            void DifferentiateSubstep(const State &x, double t, const RadonsContext &args,
                                      const Trial &trial, TangentState &tangent) const
            {
                const auto columns = tangent.physical.cols();
                const TangentBasis differential = HeldStepDifferential(x, t, trial.time - t, args, tangent);
                const State length_gradient = differential.col(columns);
                const auto fixed_length = differential.leftCols(columns);
                Eigen::RowVectorXd length = -tangent.time;
                if (trial.event)
                {
                    bool first = true;
                    for (const auto &surface : trial.event->surfaces)
                    {
                        const double transverse = length_gradient(surface.coordinate);
                        if (!std::isfinite(transverse) || std::abs(transverse) <= 1e-12)
                        {
                            throw std::runtime_error("Non-transverse Preisach event in tangent calculation");
                        }
                        Eigen::RowVectorXd candidate = -fixed_length.row(surface.coordinate);
                        if (surface.extremum)
                        {
                            candidate += tangent.extrema.row(*surface.extremum);
                        }
                        candidate /= transverse;
                        if (!first && (candidate - length).norm() > 1e-8 * (1.0 + length.norm()))
                        {
                            throw std::runtime_error("Coincident Preisach events have different time sensitivities");
                        }
                        length = std::move(candidate);
                        first = false;
                    }
                }
                tangent.physical = fixed_length + length_gradient * length;
                tangent.time += length;
                tangent.extrema = trial.memory->TransferExtrema(*args.model, tangent.extrema, tangent.physical.row(0));
            }

            TangentBasis FixedTimeBasis(const Trial &trial, const RadonsParams &params, const TangentState &tangent) const
            {
                const State field = Base::m_Function(trial.x, trial.time, RadonsContext{params, trial.memory});
                return tangent.physical - field * tangent.time;
            }

            static Eigen::VectorXd NormalizeTangents(const TangentBasis &fixed_time, TangentState &tangent)
            {
                if (!fixed_time.allFinite())
                {
                    throw std::runtime_error("Non-finite Preisach tangent basis");
                }
                Eigen::HouseholderQR<TangentBasis> qr(fixed_time);
                const auto columns = fixed_time.cols();
                const Eigen::MatrixXd r = qr.matrixQR().topLeftCorner(columns, columns).triangularView<Eigen::Upper>();
                const Eigen::VectorXd norms = r.diagonal().cwiseAbs();
                if (!norms.allFinite() || (norms.array() <= 0.0).any())
                {
                    throw std::runtime_error("Invalid QR diagonal in LCE calculation");
                }
                const auto lower = r.transpose().triangularView<Eigen::Lower>();
                tangent.physical = lower.solve(tangent.physical.transpose()).transpose().eval();
                tangent.extrema = lower.solve(tangent.extrema.transpose()).transpose().eval();
                tangent.time = lower.solve(tangent.time.transpose()).transpose().eval();
                return norms;
            }

            template <typename Observe>
            void VisitEventStep(State &x, double &time, RadonsContext &args, const Observe &observe) const
            {
                const double end = time + GetDeltaTime();
                for (int substeps = 0; substeps < 128; ++substeps)
                {
                    auto trial = ProbeSubstep(x, time, end - time, args, true);
                    if (!std::isfinite(trial.time) || !(trial.time > time))
                    {
                        throw std::runtime_error("Radons step did not advance time");
                    }
                    const bool finished = !trial.event || trial.time >= end;
                    if (finished)
                    {
                        trial.time = end;
                    }
                    observe(x, time, args, trial, finished);
                    x = trial.x;
                    time = trial.time;
                    args.model = std::move(trial.memory);
                    if (finished)
                    {
                        return;
                    }
                }
                throw std::runtime_error("Too many Preisach events in a single step");
            }

        private:
            State StagewiseTangent(const State &x, double t, const RadonsContext &args, const State &w) const
            {
                const auto initial = InitialTangents(*args.model, w);
                using StateAndTangent = Eigen::Matrix2d;
                StateAndTangent value;
                value.col(0) = x;
                value.col(1) = w;
                const std::function<StateAndTangent(const StateAndTangent &, double, const RadonsContext &)> staged =
                    [this, &initial](const StateAndTangent &stage, double time, const RadonsContext &start)
                {
                    const State state = stage.col(0);
                    const State direction = stage.col(1);
                    const auto at_stage = PrepareContext(state, start);
                    const Eigen::RowVectorXd input = Eigen::RowVectorXd::Constant(1, direction(0));
                    const auto extrema = at_stage.model->TransferExtrema(*start.model, initial.extrema, input);
                    StateAndTangent result;
                    result.col(0) = Base::m_Function(state, time, at_stage);
                    result.col(1) = Base::m_Jacobian(state, time, at_stage) * direction;
                    result(1, 1) += at_stage.params.E * at_stage.model->OutputVariation(extrema, input)(0);
                    return result;
                };
                return detail::rk4(staged, value, t, GetDeltaTime(), args).col(1);
            }

            TangentBasis HeldStepDifferential(const State &x, double t, double dt, const RadonsContext &args,
                                              const TangentState &tangent) const
            {
                const auto &params = args.params;
                const auto columns = tangent.physical.cols();
                TangentBasis basis = TangentBasis::Zero(2, columns + 1);
                basis.leftCols(columns) = tangent.physical;
                TangentBasis forcing = TangentBasis::Zero(2, columns + 1);
                forcing.row(1).head(columns) =
                    params.E * args.model->OutputVariation(tangent.extrema, tangent.physical.row(0));
                Eigen::RowVectorXd time = Eigen::RowVectorXd::Zero(columns + 1);
                time.head(columns) = tangent.time;
                Eigen::RowVectorXd length = Eigen::RowVectorXd::Zero(columns + 1);
                length(columns) = 1.0;
                const auto differentiate = [&](const State &at, double at_time, const TangentBasis &direction,
                                                const Eigen::RowVectorXd &time_direction) -> TangentBasis
                {
                    TangentBasis result = Base::m_Jacobian(at, at_time, args) * direction + forcing;
                    result.row(1) += (params.A * params.w * std::cos(params.w * at_time + params.phi)) * time_direction;
                    return result;
                };

                const double half = dt / 2.0;
                const State f1 = Base::m_Function(x, t, args);
                const TangentBasis k1 = differentiate(x, t, basis, time);
                const State x2 = x + half * f1;
                const State f2 = Base::m_Function(x2, t + half, args);
                const TangentBasis k2 = differentiate(x2, t + half, basis + half * k1 + 0.5 * f1 * length,
                                                       time + 0.5 * length);
                const State x3 = x + half * f2;
                const State f3 = Base::m_Function(x3, t + half, args);
                const TangentBasis k3 = differentiate(x3, t + half, basis + half * k2 + 0.5 * f2 * length,
                                                       time + 0.5 * length);
                const State x4 = x + dt * f3;
                const State f4 = Base::m_Function(x4, t + dt, args);
                const TangentBasis k4 = differentiate(x4, t + dt, basis + dt * k3 + f3 * length, time + length);
                return basis + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) +
                       ((f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0) * length;
            }

            static RadonsContext PrepareContext(const State &x, const RadonsContext &args)
            {
                RadonsContext working{args.params, args.model->CloneWorkingState()};
                if (working.model->GetCurrentInput() != x(0))
                {
                    working.model->AdvanceInput(x(0), x(1));
                }
                return working;
            }

            State Advance(State x, double &t, RadonsContext args, bool record) const
            {
                const bool locate_events = m_Integrator == RadonsIntegrator::EventAware;
                if (!locate_events)
                {
                    auto trial = ProbeSubstep(x, t, GetDeltaTime(), args, false);
                    if (record)
                    {
                        args.model->AcceptWorkingState(*trial.memory);
                    }
                    t = trial.time;
                    return trial.x;
                }
                const auto target = args.model;
                double time = t;
                std::vector<std::shared_ptr<PreisachModelBase>> accepted;
                VisitEventStep(x, time, args, [&](const State &, double, const RadonsContext &, const Trial &trial, bool)
                {
                    if (record)
                    {
                        accepted.push_back(trial.memory);
                    }
                });
                target->AcceptWorkingSequence(accepted);
                t = time;
                return x;
            }

            RadonsIntegrator m_Integrator = RadonsIntegrator::RK4;
            EventTolerance m_EventTolerance;
            RadonsOutputStrategy m_OutputStrategy = RadonsOutputStrategy::Hold;
        };

        inline RadonsSnapshot CaptureRadonsSnapshot(const RadonsSystem &system)
        {
            const auto x = system.GetX();
            auto memory = system.GetArgs().model->CloneWorkingState();
            if (memory->GetCurrentInput() != x(0))
            {
                memory->AdvanceInput(x(0), x(1));
            }
            return {system.GetT(), x, system.GetArgs().params, std::move(memory)};
        }

        inline RadonsContext ForkRadonsContext(const RadonsSnapshot &snapshot, const Eigen::Vector2d &x)
        {
            auto memory = snapshot.memory->CloneWorkingState();
            if (x(0) != snapshot.x(0))
            {
                memory->AdvanceInput(x(0), x(1));
            }
            return {snapshot.params, std::move(memory)};
        }

        /// Omit nextArgs for a single trajectory. A paired calculation requires
        /// a second context with a separately constructed Preisach model.
        inline std::unique_ptr<RadonsSystem> GetRadonsSystem(double dt, const RadonsContext &args,
            const std::optional<RadonsContext> &nextArgs = {}, const Eigen::Vector2d &x0 = {0.0, 0.0},
            bool use_atan = false, double t0 = 0.0)
        {
            auto func = [use_atan](const Eigen::Vector2d &x, double t, const RadonsContext &args) -> Eigen::Vector2d
            {
                AL_PROFILE_FUNC("Rodos::func");
                const auto &params = args.params;
                
                Eigen::Vector2d res;

                const double a = params.A * std::sin(params.w * t + params.phi);
                const double g = params.gamma * x[1];
                const double ww = params.w0 * x[0];
                const double atan = use_atan ? params.eps * std::atan(x[0]) : 0.0;
                const double p = params.E * args.model->GetCurrentOutput();
                const double v = a - g - ww + p + atan;

                res[0] = x[1];
                res[1] = v;

                return res;
            };

            auto jac = [use_atan](const Eigen::Vector2d &x, double, const RadonsContext &args) -> Eigen::Matrix2d
            {
                const auto &params = args.params;

                Eigen::Matrix2d res = Eigen::Matrix2d::Zero();

                res(0, 1) = 1.0;

                res(1, 0) = -params.w0;
                if (use_atan)
                {
                    res(1, 0) += params.eps / (1.0 + x[0] * x[0]);
                }
                
                res(1, 1) = -params.gamma;

                return res;
            };

            auto system = std::make_unique<RadonsSystem>(func, jac, dt, args, nextArgs, x0, t0);
            const auto reset = [](const std::shared_ptr<PreisachModelBase> &model)
            {
                std::weak_ptr<PreisachModelBase> source = model;
                std::shared_ptr<const PreisachModelBase> initial = model ? model->CloneWorkingState() : nullptr;
                return [source, initial](RadonsContext &context)
                {
                    context.model->ResetState();
                    if (source.lock() == context.model &&
                        initial->ConfigurationRevision() == context.model->ConfigurationRevision())
                    {
                        auto working = initial->CloneWorkingState();
                        context.model->AcceptWorkingState(*working, false);
                    }
                };
            };
            system->SetResetFn([primary = reset(args.model), secondary = reset(nextArgs ? nextArgs->model : nullptr)]
                               (RadonsContext &context, std::optional<RadonsContext> &next, uint32_t)
            {
                primary(context);
                if (next)
                {
                    secondary(*next);
                }
            });
            return system;
        }

        template <typename Derived>
        std::unique_ptr<RadonsSystem> GetRadonsSystem(double dt, const RadonsContext &args,
                                                         const std::optional<RadonsContext> &nextArgs,
                                                         const Eigen::MatrixBase<Derived> &x0, bool use_atan = false,
                                                         double t0 = 0.0)
        {
            return GetRadonsSystem(dt, args, nextArgs, RadonsSystem::CheckedState(x0), use_atan, t0);
        }
    }
}
