#include "MUTCpp/Hysteresis/ArealModel.hpp"
#include "MUTCpp/Hysteresis/SzaboEverett.hpp"
#include "MUTCpp/ODE/RadonsSystem.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
    void Require(bool condition, const char *message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    void TestRadonsSnapshot()
    {
        constexpr double dt = 0.125;
        constexpr double tol = 1e-12;
        auto model = std::make_shared<mc::ArealPreisachModel>(1.0);
        model->InitializeMemory(std::array{1.0, -0.4});
        mc::ArealPreisachModel legacy(1.0);
        legacy.P(1.0, 0.0, -2);
        legacy.P(-0.4, 0.0, -1);

        const Eigen::Vector2d x(-0.4, -0.25);
        const mc::ode::RadonsContext args{.params = {.dt = dt}, .model = model};
        const auto system = mc::ode::GetRadonsSystem(dt, args, {}, x);
        const auto snapshot = mc::ode::CaptureRadonsSnapshot(*system);
        Require(snapshot.time == system->GetT() && snapshot.x == x && snapshot.memory.get() != model.get(),
                "Snapshot must retain physical state with independent working memory");

        const auto velocity = mc::ode::ForkRadonsContext(snapshot, Eigen::Vector2d(x(0), x(1) + 0.1));
        const auto displacement = mc::ode::ForkRadonsContext(snapshot, x + Eigen::Vector2d(-0.02, 0.0));
        Require(velocity.model->GetCurrentInput() == x(0) && displacement.model->GetCurrentInput() == x(0) - 0.02,
                "Fork changed the wrong input");
        Require(std::abs(velocity.model->P(-0.6, 0.0) - legacy.P(-0.6, 0.0)) < tol,
                "Seed or velocity fork changed the continuation of the descending branch");
        Require(model->GetCurrentInput() == x(0) && snapshot.memory->GetCurrentInput() == x(0) &&
                    displacement.model->GetCurrentInput() == x(0) - 0.02,
                "Advancing a fork changed another trajectory");
        Require(std::get<0>(model->HysteresisLoop()).empty() &&
                    std::get<0>(snapshot.memory->HysteresisLoop()).empty(),
                "Snapshot or initialization polluted the original journal");

        mc::ArealPreisachModel ascending(1.0);
        ascending.InitializeMemory(std::array{-1.0, x(0)});
        auto descending = snapshot.memory->CloneWorkingState();
        Require(std::abs(ascending.P(x(0), 0.0) - descending->P(x(0), 0.0)) > 0.1,
                "An equal input erased the difference between two histories");
    }

    void TestDoubleSnapshot()
    {
        mc::DoubleArealPreisachModel model(1.0, 0.25, 0.7);
        model.InitializeMemory(std::array{-2.25, 2.25, 0.4});
        auto first = model.CloneWorkingState();
        auto second = model.CloneWorkingState();
        first->P(2.25, 0.0);
        Require(second->GetCurrentInput() == 0.4 && model.GetCurrentInput() == 0.4,
                "Double snapshot shares working memory with another trajectory");
        Require(std::abs(second->P(0.8, 0.0) - model.P(0.8, 0.0)) < 1e-12,
                "Double snapshot changed the continuation of a child model");
    }

    void TestAcceptedHistory()
    {
        mc::ArealPreisachModel model(1.0);
        constexpr std::array inputs{-1.0, 0.8, -0.6, 0.4, 0.4};
        for (double input : inputs)
        {
            model.P(input, 0.0, 0);
        }
        const auto before = model.DerivativeHistory();
        auto trial = model.CloneWorkingState();
        trial->AdvanceInput(0.9, 0.0);
        Require(model.DerivativeHistory() == before && model.GetCurrentInput() == 0.4,
                "Rejected trial changed the accepted history or memory");
        Require(std::get<0>(trial->HysteresisLoop()).empty(), "Trial created a history entry");

        model.P(0.9, 0.0, 0);
        const auto [u, p] = model.HysteresisLoop();
        const auto [du, slopes] = model.DerivativeHistory();
        Require(u == du && u.size() == inputs.size() + 1 && p.size() == u.size() && slopes.size() == u.size(),
                "Saturation, repeated input, or wiping desynchronized the history columns");
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            Require(u[i] == inputs[i] && slopes[i] == before[1][i], "A later input changed a recorded sample");
        }

        model.SetRecordHistory(false);
        model.P(0.6, 0.0);
        Require(model.DerivativeHistory()[0] == u && std::abs(model.CurrentSignedSlope() + 0.3) < 1e-12,
                "Disabling the journal disabled the current slope");
    }

    void TestHeldRK4()
    {
        class CountedModel : public mc::ArealPreisachModel
        {
        public:
            explicit CountedModel(int &updates) : mc::ArealPreisachModel(1.0), m_Updates(updates) {}
            double AdvanceInput(double u, double v) override
            {
                ++m_Updates;
                return mc::ArealPreisachModel::AdvanceInput(u, v);
            }
            std::unique_ptr<mc::SinglePreisachModel> CloneSingleWorkingState() const override
            {
                auto copy = std::make_unique<CountedModel>(m_Updates);
                CopyWorkingStateTo(*copy);
                return copy;
            }

        private:
            int &m_Updates;
        };

        constexpr double dt = 0.125;
        const Eigen::Vector2d x(0.2, 0.7);
        int updates = 0;
        auto model = std::make_shared<CountedModel>(updates);
        model->InitializeMemory(std::array{-1.0, x(0)});
        const double p = model->GetCurrentOutput();
        const Eigen::Vector2d expected(x(0) + dt * x(1) + 0.5 * dt * dt * p, x(1) + dt * p);
        const mc::ode::RadonsContext args{.params = {.dt = dt, .E = 1.0}, .model = model};
        auto system = mc::ode::GetRadonsSystem(dt, args, {}, x);
        system->ShiftTrajNext(x + Eigen::Vector2d(0.1, 0.0), 0.3, true);
        const auto trial = system->ShiftTrajNext(x, 0.0);
        Require(trial.isApprox(expected, 1e-12) && model->GetCurrentInput() == x(0) &&
                    std::get<0>(model->HysteresisLoop()).empty(),
                "Trial changed the accepted memory or mixed stage outputs");
        updates = 0;
        system->Next();
        Require(updates == 1 && system->GetX().isApprox(expected, 1e-12) &&
                    model->GetCurrentInput() == system->GetX()(0) &&
                    std::get<0>(model->HysteresisLoop()).size() == 1,
                "Accepted step must publish one update with the held initial output");
        const double next_p = model->GetCurrentOutput();
        system->Next();
        Require(std::abs(system->GetX()(1) - (expected(1) + dt * next_p)) < 1e-12 && updates == 2,
                "Next step did not use the preceding accepted output");
    }

    void TestForkedForcingPhase()
    {
        constexpr double dt = 0.125;
        const mc::ode::RadonsContext args{
            .params = {.dt = dt, .A = 1.0, .w = 1.0},
            .model = std::make_shared<mc::ArealPreisachModel>(1.0)
        };
        auto warmup = mc::ode::GetRadonsSystem(dt, args);
        warmup->ForwardWithoutHistory(4);
        const auto snapshot = mc::ode::CaptureRadonsSnapshot(*warmup);
        const double t = snapshot.time;
        const double dv = dt * (std::sin(t) + 4 * std::sin(t + dt / 2) + std::sin(t + dt)) / 6;
        const Eigen::Vector2d perturbations[] = {{0.0, 0.1}, {0.01, 0.0}};
        for (const auto &delta : perturbations)
        {
            const Eigen::Vector2d x = snapshot.x + delta;
            const auto primary = mc::ode::ForkRadonsContext(snapshot, snapshot.x);
            const auto perturbed = mc::ode::ForkRadonsContext(snapshot, x);
            auto pair = mc::ode::GetRadonsSystem(dt, primary, perturbed, snapshot.x, false, t);
            const auto [main, shifted] = pair->ForwardTwoTrajs(dt, x);
            Require(std::abs(main(1, 1) - main(0, 1) - dv) < 1e-12 &&
                        std::abs(shifted(1, 1) - shifted(0, 1) - dv) < 1e-12,
                    "Forked trajectories used different or incorrect forcing phases");
            Require(std::get<0>(primary.model->HysteresisLoop()).size() == 1 &&
                        std::get<0>(perturbed.model->HysteresisLoop()).size() == 1 &&
                        std::get<0>(args.model->HysteresisLoop()).size() == 4,
                    "Forked trajectories altered the warm-up journal");
        }
    }

    void TestPreisachMeasures()
    {
        const mc::UniformPreisachMeasure uniform;
        Require(uniform.TriangleMass(-1.0, 0.5) == 1.125 &&
                    uniform.BoundaryDerivatives(-1.0, 0.5) == std::array{-1.5, 1.5},
                "Uniform mass or its boundary derivative is incorrect");
        const auto uniform_surface = uniform.GetoMeasureSurfaceData(-1.0, 1.0, 3);
        Require(uniform_surface.z(2, 0) == 1.0 && std::isnan(uniform_surface.z(0, 2)),
                "Uniform surface mask is incorrect");

        const Eigen::Vector2d a(2.0, 3.0);
        const Eigen::Vector2d b(0.0, -0.4);
        const Eigen::Vector2d c(0.3, 0.7);
        const mc::SzaboEverett szabo(a, b, c);
        const mc::PreisachMeasure &measure = szabo;
        const std::array triangles{std::array{-1.0, 2.0}, std::array{0.25, 0.2502}, std::array{9.0, 11.0},
                                   std::array{-11.0, -9.0}};
        for (const auto &[lo, hi] : triangles)
        {
            const double h = std::min(1e-4, 1e-3 * (hi - lo));
            const auto derivative = measure.BoundaryDerivatives(lo, hi);
            const std::array reference{
                (measure.TriangleMass(lo + h, hi) - measure.TriangleMass(lo - h, hi)) / (2 * h),
                (measure.TriangleMass(lo, hi + h) - measure.TriangleMass(lo, hi - h)) / (2 * h)
            };
            for (int i = 0; i < 2; ++i)
            {
                if (std::abs(derivative[i] - reference[i]) > 2e-6 * std::abs(derivative[i]) + 1e-17)
                {
                    std::cerr << lo << ", " << hi << ", boundary " << i << ": "
                              << derivative[i] << " vs " << reference[i] << '\n';
                    throw std::runtime_error("Analytic Szabo boundary derivative differs from finite differences");
                }
            }
        }
        const auto surface = measure.GetoMeasureSurfaceData(-1.0, 1.0, 5, 2.5);
        for (Eigen::Index row = 0; row < surface.z.rows(); ++row)
        {
            for (Eigen::Index col = 0; col <= row; ++col)
            {
                Require(std::abs(surface.z(row, col) - 2.5 * measure.Density(surface.x(col), surface.y(row))) < 1e-12,
                        "Szabo surface differs from the scaled pointwise density");
            }
        }
    }

    void TestArealSlopes()
    {
        mc::ArealPreisachModel model(1.0);
        model.InitializeMemory(std::array{-1.0, 0.25});
        Require(std::abs(model.CurrentSlope() - 2.5) < 1e-12, "Wrong uniform ascending slope");
        model.InitializeMemory(std::array{-1.0, 0.75, 0.25});
        Require(std::abs(model.CurrentSlope() - 1.0) < 1e-12 &&
                    std::abs(model.CurrentSignedSlope() + 0.5) < 1e-12,
                "Wrong uniform descending slope or separate signed diagnostic");
        model.P(1.1, 0.0);
        Require(model.GetCurrentOutput() == 2.0 && model.CurrentSlope() == 0.0,
                "Saturation must use the full mass with zero slope");

        const Eigen::Vector2d a(2.0, 3.0);
        const Eigen::Vector2d b(0.1, -0.4);
        const Eigen::Vector2d c(0.3, 0.7);
        model.SetMeasure(std::make_shared<mc::SzaboEverett>(a, b, c));
        model.SetRecordHistory(false);
        model.InitializeMemory(std::array{-1.0, 0.6, -0.2});
        constexpr double h = 1e-6;
        for (const double delta : {-h, h})
        {
            if (delta > 0.0)
            {
                model.P(0.1, 0.0);
            }
            const auto center = model.GetCurrentOutput();
            auto branch = model.CloneWorkingState();
            const double sample = branch->AdvanceInput(model.GetCurrentInput() + delta, 0.0);
            Require(std::abs((sample - center) / delta - model.CurrentSlope()) < 2e-6,
                    "Weighted slope differs from the branch finite difference");
        }
        Require(model.DerivativeHistory()[0].empty(), "Reading a weighted slope required a journal");
    }

    void TestDoubleMeasureAndGap()
    {
        class AsymmetricMeasure final : public mc::PreisachMeasure
        {
        public:
            double Density(double lower, double) const override { return 2.0 + lower / 4; }
            double TriangleMass(double lower, double upper) const override
            {
                const double width = upper - lower;
                return Density(lower, upper) * width * width / 2 + width * width * width / 24;
            }
            std::array<double, 2> BoundaryDerivatives(double lower, double upper) const override
            {
                const double width = upper - lower;
                const double side = Density(lower, upper) * width;
                return {-side, side + width * width / 8};
            }
        };

        constexpr double K = 0.7;
        mc::DoubleArealPreisachModel model(0.5, 0.25, K);
        const auto check = [&](const mc::PreisachMeasure &measure)
        {
            const auto d = model.GetD();
            const auto bound = d + 1.0;
            const double lower_mass = measure.TriangleMass(-bound, -d);
            const double upper_mass = measure.TriangleMass(d, bound);
            model.InitializeMemory(std::array{-2.0});
            Require(std::abs(model.GetCurrentOutput() + 2 * lower_mass + K * d) < 1e-12 &&
                        model.CurrentSlope() == 0.0, "Incorrect lower saturation of shifted supports");
            for (const double seam : {-d, d})
            {
                constexpr double h = 1e-7;
                for (const auto direction : {-1.0, 1.0})
                {
                    model.InitializeMemory(std::array{-2.0 * direction, seam - h * direction});
                    const double before = model.GetCurrentOutput();
                    const double after = model.P(seam + h * direction, 0.0);
                    Require(std::abs(before - K * seam) < 1e-6 && std::abs(after - K * seam) < 1e-6,
                            "A double-model seam has discontinuous output");
                }
            }
            const double u = d + 0.25;
            model.InitializeMemory(std::array{-2.0, u});
            Require(std::abs(model.CurrentSlope() - 2 * measure.BoundaryDerivatives(d, u)[1]) < 1e-12,
                    "Outside the gap, the slope must use the child measures without K");
            if (d > 0.0)
            {
                Require(model.P(0.0, 0.0) == 0.0 && model.CurrentSlope() == K, "Wrong central branch");
            }
            Require(std::abs(model.P(2.0, 0.0) - 2 * upper_mass - K * d) < 1e-12 &&
                        model.CurrentSlope() == 0.0, "Incorrect upper saturation of shifted supports");
        };

        check(mc::UniformPreisachMeasure{});
        const auto asymmetric = std::make_shared<AsymmetricMeasure>();
        model.SetMeasure(asymmetric);
        check(*asymmetric);
        model.SetD(0.6);
        Require(std::isnan(model.GetCurrentInput()), "A changed gap kept incompatible memory");
        check(*asymmetric);
        model.SetD(0.0);
        check(*asymmetric);
    }

    void TestHeldStepJacobian()
    {
        constexpr double dt = 0.125;
        constexpr double t = 0.5;
        const Eigen::Vector2d x(0.4, 0.3);
        auto model = std::make_shared<mc::ArealPreisachModel>(1.0);
        model->InitializeMemory(std::array{-1.0, x(0)});
        const mc::ode::RadonsContext args{
            .params = {.dt = dt, .gamma = 0.2, .A = 0.8, .w0 = 0.6, .w = 1.0, .E = 1.3, .eps = 0.5},
            .model = model
        };
        const auto system = mc::ode::GetRadonsSystem(dt, args, {}, x, true, t);
        const auto snapshot = mc::ode::CaptureRadonsSnapshot(*system);
        const auto rhs = [&](const Eigen::Vector2d &state)
        {
            const auto context = mc::ode::ForkRadonsContext(snapshot, state);
            return Eigen::Vector2d(state(1), 0.8 * std::sin(t) - 0.2 * state(1) - 0.6 * state(0) +
                1.3 * context.model->GetCurrentOutput() + 0.5 * std::atan(state(0)));
        };
        const Eigen::Vector2d f0 = rhs(x);
        const Eigen::Vector2d step0 = system->ShiftTrajNext(x, t);
        const auto jac = system->EvaluateBranchJacobian(x, t);
        constexpr double h = 1e-7;
        for (int index = 0; index < 2; ++index)
        {
            const auto direction = Eigen::Vector2d::Unit(index).eval();
            const Eigen::Vector2d changed = x + h * direction;
            Require((jac.col(index) - (rhs(changed) - f0) / h).norm() < 2e-6,
                    "The right-hand side Jacobian disagrees with the inherited-branch perturbation");
            Require((system->NextLTM(direction) - (system->ShiftTrajNext(changed, t) - step0) / h).norm() < 2e-6,
                    "The one-step tangent map disagrees with the held RK4 finite difference");
        }
    }

    void TestEventBrackets()
    {
        using namespace mc::ode;
        const auto pair_of_crossings = [](double t)
        {
            return Eigen::Vector2d(0.0, (t - 0.25) * (t - 0.75));
        };
        const std::vector<detail::EventSurface> velocity = {{1, 0.0}};
        const auto first = detail::FindFirstCrossing(pair_of_crossings, 1.0, velocity, {});
        Require(first && std::abs(first->elapsed - 0.25) < 1e-9, "First of two interior roots was missed");
        const auto second = detail::FindFirstCrossing([&](double elapsed)
        {
            return pair_of_crossings(0.25 + elapsed);
        }, 0.75, velocity, {});
        Require(second && std::abs(second->elapsed - 0.5) < 1e-9, "Continuation missed the second root");
        const auto touch = detail::FindFirstCrossing([](double t)
        {
            return Eigen::Vector2d(0.0, (t - 0.5) * (t - 0.5));
        }, 1.0, velocity, {});
        Require(!touch, "A touch without a sign change produced an event");
        const auto coincident = detail::FindFirstCrossing([](double t)
        {
            return Eigen::Vector2d(t - 0.5, t - 0.5);
        }, 1.0, {{0, 0.0}, {1, 0.0}}, {});
        Require(coincident && coincident->elapsed == 0.5 && coincident->surfaces.size() == 2,
                "Coincident crossing times were not coalesced");
    }

    void TestRadonsEventIntegration()
    {
        using namespace mc::ode;
        auto model = std::make_shared<mc::ArealPreisachModel>(1.0);
        RadonsContext args{
            .params = {.dt = 1.0, .A = -2.0, .phi = std::acos(-1.0) / 2.0},
            .model = model
        };
        const Eigen::Vector2d x(0.1, 1.0);
        auto system = GetRadonsSystem(1.0, args, {}, x);
        system->SetIntegrator(RadonsIntegrator::EventAware);
        const auto turn = system->ProbeSubstep(x, 0.0, 1.0, args, true);
        Require(turn.event && std::abs(turn.time - 0.5) < 1e-9 && turn.x(1) == 0.0 &&
                    std::abs(turn.x(0) - 0.35) < 1e-9, "Reversal inside the step was localized incorrectly");
        system->ShiftTrajNext(x, 0.0, true);
        Require(std::get<0>(model->HysteresisLoop()).empty(), "Event trial added journal samples");
        system->Next();
        Require(system->GetX().isApprox(Eigen::Vector2d(0.1, -1.0), 1e-9) &&
                    std::get<0>(model->HysteresisLoop()).size() == 2,
                "Reversal skipped the remainder or was processed twice");

        args.params.A = 0.0;
        model->InitializeMemory(std::array{-1.0, 0.7, -0.5, 0.2});
        const auto wipe = system->ProbeSubstep(Eigen::Vector2d(0.2, 1.0), 0.0, 0.6, args, true);
        Require(wipe.event && std::abs(wipe.time - 0.5) < 1e-9 && wipe.x(0) == 0.7 && wipe.x(1) == 1.0 &&
                    std::abs(wipe.memory->CurrentSlope() - 3.4) < 1e-12,
                "Active return point did not wipe at nonzero velocity");

        auto double_model = std::make_shared<mc::DoubleArealPreisachModel>(0.5, 0.25, 0.7);
        args.model = double_model;
        auto double_system = GetRadonsSystem(3.0, args, {}, Eigen::Vector2d(-1.5, 1.0));
        double_system->SetIntegrator(RadonsIntegrator::EventAware);
        double_system->Next();
        const auto inputs = std::get<0>(double_model->HysteresisLoop());
        const std::vector<double> thresholds{-1.25, -0.25, 0.25, 1.25};
        Require(inputs.size() == 5 && std::equal(thresholds.begin(), thresholds.end(), inputs.begin()) &&
                    std::abs(inputs.back() - 1.5) < 1e-9,
                "Double supports or seams were missed or processed twice");
    }

    void TestEventTangents()
    {
        using namespace mc::ode;
        const Eigen::Matrix2d initial_basis = Eigen::Matrix2d::Identity();
        const auto compare = [&](bool wipe, double dt)
        {
            auto model = std::make_shared<mc::ArealPreisachModel>(1.0);
            const Eigen::Vector2d x = wipe ? Eigen::Vector2d(0.2, 1.0) : Eigen::Vector2d(0.25, 0.65);
            const RadonsContext args{
                .params = {.dt = dt, .A = wipe ? 0.0 : -1.6, .E = wipe ? 0.2 : 0.35, .phi = std::acos(-1.0) / 2},
                .model = model
            };
            if (wipe)
                model->InitializeMemory(std::array{-1.0, 0.7, -0.5, x(0)});
            else
                model->InitializeMemory(std::array{-1.0, x(0)});
            auto system = GetRadonsSystem(dt, args, {}, x);
            system->SetIntegrator(RadonsIntegrator::EventAware, {1e-13, 1e-11, 32});
            const auto snapshot = CaptureRadonsSnapshot(*system);

            const auto run = [&](const Eigen::Vector2d &initial, RadonsContext working,
                                 RadonsSystem::TangentState *directions)
            {
                double t = 0.0;
                auto state = initial;
                int events = 0;
                for (double end = dt; end <= 0.75; end += dt)
                {
                    while (t < end)
                    {
                        const auto point = system->ProbeSubstep(state, t, end - t, working, true);
                        if (directions)
                        {
                            system->DifferentiateSubstep(state, t, working, point, *directions);
                        }
                        events += point.event.has_value();
                        t = point.event ? point.time : end;
                        working.model = point.memory;
                        state = point.x;
                    }
                }
                Require(events == 1, "Perturbed event sequence changed");
                std::vector<double> memory;
                working.model->AppendExtrema(memory);
                return std::pair{state, memory};
            };
            auto directions = RadonsSystem::InitialTangents(*model, initial_basis);
            const auto [nominal, accepted] = run(x, ForkRadonsContext(snapshot, x), &directions);
            for (const double epsilon : {1e-5, 1e-6})
            {
                for (int i = 0; i < 2; ++i)
                {
                    const Eigen::Vector2d shifted = x + epsilon * initial_basis.col(i);
                    const auto [perturbed, memory] = run(shifted, ForkRadonsContext(snapshot, shifted), nullptr);
                    Require((directions.physical.col(i) - (perturbed - nominal) / epsilon).norm() < 6e-5,
                            "Physical tangent across event disagrees with the flow difference");
                    Require(memory.size() == accepted.size() && memory.size() == directions.extrema.rows(),
                            "Perturbed history has a different topology");
                    for (Eigen::Index row = 0; row < directions.extrema.rows(); ++row)
                    {
                        Require(std::abs(directions.extrema(row, i) - (memory[row] - accepted[row]) / epsilon) < 6e-5,
                                "Extremum tangent across event disagrees with the flow difference");
                    }
                }
            }
        };
        for (double step : {0.75, 0.375})
        {
            compare(false, step);
            compare(true, step);
        }

        auto model = std::make_shared<mc::DoubleArealPreisachModel>(0.5, 0.25, 0.7);
        model->InitializeMemory(std::array{-1.25, 0.1});
        const Eigen::Vector2d x(0.1, 1.0);
        const RadonsContext args{.params = {.dt = 0.5}, .model = model};
        auto system = GetRadonsSystem(0.5, args, {}, x);
        const auto seam = system->ProbeSubstep(x, 0.0, 0.5, args, true);
        Require(seam.event.has_value(), "Expected a transverse double-model seam");
        auto tangent = RadonsSystem::InitialTangents(*model, initial_basis);
        system->DifferentiateSubstep(x, 0.0, args, seam, tangent);
        Eigen::Matrix2d expected = Eigen::Matrix2d::Identity();
        expected(0, 1) = seam.time;
        const auto fixed = system->FixedTimeBasis(seam, args.params, tangent);
        Require(fixed.isApprox(expected, 1e-12), "Continuous seam added a physical jump");
        const auto old_times = tangent.time;
        const auto old_extrema = tangent.extrema;
        Eigen::HouseholderQR<Eigen::Matrix2d> qr(expected);
        const Eigen::Matrix2d r = qr.matrixQR().triangularView<Eigen::Upper>();
        RadonsSystem::NormalizeTangents(fixed, tangent);
        Require((system->FixedTimeBasis(seam, args.params, tangent).transpose() *
                    system->FixedTimeBasis(seam, args.params, tangent)).isIdentity(1e-12) &&
                    (tangent.time * r).isApprox(old_times, 1e-12) &&
                    (tangent.extrema * r).isApprox(old_extrema, 1e-12),
                "QR did not change physical state, extrema and time in the same basis");

        auto ambiguous = seam;
        ambiguous.event->surfaces.push_back({1, seam.x(1)});
        bool rejected = false;
        try
        {
            tangent = RadonsSystem::InitialTangents(*model, initial_basis);
            system->DifferentiateSubstep(x, 0.0, args, ambiguous, tangent);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        Require(rejected, "A non-transverse coincident event did not report an error");
    }

    void TestOutputAtRK4Stages()
    {
        using namespace mc::ode;
        const Eigen::Vector2d x(0.2, 0.2);
        const RadonsParams params{.gamma = 0.15, .A = 0.3, .w0 = 0.15, .w = 1.0, .E = 0.4, .phi = 0.5};
        const auto reference_rhs = [&params](const Eigen::Vector2d &p, double t, const NoContext &) -> Eigen::Vector2d
        {
            const double u = p(0) + 1.0;
            return {p(1), params.A * std::sin(t + params.phi) - params.gamma * p(1) - params.w0 * p(0) +
                          params.E * (u * u - 2.0)};
        };
        auto make = [](double step, const Eigen::Vector2d &initial, const RadonsParams &p)
        {
            auto model = std::make_shared<mc::ArealPreisachModel>(1.0);
            model->InitializeMemory(std::array{-1.0, initial(0)});
            return GetRadonsSystem(step, RadonsContext{p, std::move(model)}, {}, initial);
        };

        ContinuousDS<2> reference(reference_rhs,
            [](const Eigen::Vector2d &, double, const NoContext &) -> Eigen::Matrix2d
            {
                return Eigen::Matrix2d::Zero();
            }, 1.0 / 2048.0, {}, {}, x);
        reference.ForwardWithoutHistory(0.5);
        double previous_error = 1.0;
        for (const double h : {0.125, 0.0625, 0.03125})
        {
            auto held = make(h, x, params);
            auto staged = make(h, x, params);
            Require(held->GetOutputStrategy() == RadonsOutputStrategy::Hold, "Ordinary RK4 changed its default policy");
            staged->SetOutputStrategy(RadonsOutputStrategy::PerStage);
            const auto initial = CaptureRadonsSnapshot(*staged);
            const auto candidate = staged->ShiftTrajNext(x, 0.0);
            for (int col = 0; col < 2; ++col)
            {
                const Eigen::Vector2d direction = Eigen::Vector2d::Unit(col);
                constexpr double epsilon = 1e-7;
                const auto perturbed = staged->ShiftTrajNext(x + epsilon * direction, 0.0);
                Require((staged->NextLTM(direction) - (perturbed - candidate) / epsilon).norm() < 2e-6,
                        "Stagewise tangent did not differentiate its own RK4 map");
            }
            Require(staged->GetArgs().model->GetCurrentInput() == initial.x(0) &&
                        std::get<0>(staged->GetArgs().model->HysteresisLoop()).empty(),
                    "Stagewise probes touched the accepted memory or history");
            staged->ForwardWithoutHistory(0.5);
            held->ForwardWithoutHistory(0.5);
            const double error = (staged->GetX() - reference.GetX()).norm();
            Require(error < previous_error / 4.0 && error < (held->GetX() - reference.GetX()).norm() &&
                        staged->GetX()(1) > 0.0, "Stagewise RK4 did not converge on the reference monotone branch");
            Require(std::get<0>(staged->GetArgs().model->HysteresisLoop()).size() == 0.5 / h,
                    "Stagewise RK4 recorded trial stages as physical samples");
            previous_error = error;
        }

        const Eigen::Vector2d turning_x(0.25, 0.65);
        const RadonsParams turning_params{.A = -1.6, .E = 0.35, .phi = std::acos(-1.0) / 2};
        auto event_reference = make(1.0 / 1024.0, turning_x, turning_params);
        event_reference->SetIntegrator(RadonsIntegrator::EventAware);
        event_reference->ForwardWithoutHistory(0.75);
        auto coarse = make(0.125, turning_x, turning_params);
        auto fine = make(0.03125, turning_x, turning_params);
        for (auto *system : {coarse.get(), fine.get()})
        {
            system->SetOutputStrategy(RadonsOutputStrategy::PerStage);
            system->ForwardWithoutHistory(0.75);
            Require(system->GetX().allFinite() && system->GetX()(1) < 0.0,
                    "Stagewise experiment was unstable at the reversal");
        }
        Require((fine->GetX() - event_reference->GetX()).norm() < (coarse->GetX() - event_reference->GetX()).norm(),
                "Stagewise step refinement did not improve the reversal trajectory");
    }
}

int main() try
{
    TestRadonsSnapshot();
    TestDoubleSnapshot();
    TestAcceptedHistory();
    TestHeldRK4();
    TestForkedForcingPhase();
    TestPreisachMeasures();
    TestArealSlopes();
    TestDoubleMeasureAndGap();
    TestHeldStepJacobian();
    TestEventBrackets();
    TestRadonsEventIntegration();
    TestEventTangents();
    TestOutputAtRK4Stages();
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
