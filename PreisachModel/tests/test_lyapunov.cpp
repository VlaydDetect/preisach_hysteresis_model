#include "MUTCpp/ODE/Lyapunov.hpp"

#include <limits>
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

    void TestInvalidLCEParameters()
    {
        using System = mc::ode::ContinuousDS<2>;
        int resets = 0;
        int evaluations = 0;
        const auto rhs = [&evaluations](const System::State &, double, const mc::ode::NoContext &) -> System::State
        {
            ++evaluations;
            return System::State::Zero();
        };
        const auto jacobian = [](const System::State &, double, const mc::ode::NoContext &) -> System::Jacobian
        {
            return System::Jacobian::Zero();
        };
        const auto check = [&](double dt, double warmup, double duration, uint32_t period, int count, double t0 = 2.0)
        {
            System system(rhs, jacobian, dt, {}, {}, System::State::Ones().eval(), t0);
            system.SetResetFn([&resets](mc::ode::NoContext &, std::optional<mc::ode::NoContext> &, uint32_t)
            {
                ++resets;
            });
            try
            {
                mc::ode::ComputeLCEs(system, warmup, duration, period, count);
            }
            catch (const std::invalid_argument &)
            {
                Require(resets == 0 && evaluations == 0, "Invalid LCE call modified the system");
                Require(system.GetT() == t0 && system.GetX().isOnes(), "Invalid LCE call advanced the system");
                return;
            }
            throw std::runtime_error("Invalid LCE parameters were accepted");
        };

        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        check(0.0, 0.0, 1.0, 1, 2);
        check(-0.1, 0.0, 1.0, 1, 2);
        check(inf, 0.0, 1.0, 1, 2);
        check(nan, 0.0, 1.0, 1, 2);
        check(0.1, -1.0, 1.0, 1, 2);
        check(0.1, inf, 1.0, 1, 2);
        check(0.1, 0.0, nan, 1, 2);
        check(0.1, 0.0, 0.0, 1, 2);
        check(0.1, 0.0, 0.01, 1, 2);
        check(0.1, 0.0, 1.0, 0, 2);
        check(0.1, 0.0, 1.0, 1, 0);
        check(0.1, 0.0, 1.0, 1, 3);
        check(0.1, 1e10, 1.0, 1, 2);
        check(0.1, 0.0, 1e10, 1, 2);
        check(0.1, 0.0, 1.0, 1, 2, 1e20);
    }

    void TestLCEHistory()
    {
        using System = mc::ode::ContinuousDS<2>;
        constexpr double dt = 1.0 / 16.0;
        const System::State rates(0.5, -0.125);
        const auto rhs = [&rates](const System::State &x, double, const mc::ode::NoContext &) -> System::State
        {
            return rates.cwiseProduct(x);
        };
        const auto jacobian = [&rates](const System::State &, double, const mc::ode::NoContext &) -> System::Jacobian
        {
            return rates.asDiagonal();
        };
        System system(rhs, jacobian, dt, {}, {}, System::State::Zero().eval(), 5.0);
        auto result = mc::ode::ComputeLCEs(system, 3 * dt, 5 * dt, 2, std::nullopt, {});
        Require(result.history.rows() == 3 && result.history.cols() == 2, "Missing regular or final QR");
        Require(result.history_times == std::vector<double>{5 + 5 * dt, 5 + 7 * dt, 5 + 8 * dt},
                "LCE times do not match the QR schedule");
        Require(result.spectrum.isApprox(rates, 1e-8), "LCE normalization includes time outside observation");
        for (Eigen::Index row = 0; row < result.history.rows(); ++row)
        {
            Require(result.history.row(row).transpose().isApprox(rates, 1e-8), "Incorrect history normalization");
        }

        result = mc::ode::ComputeLCEs(system, 0.0, dt, 50, 1, {});
        Require(result.history.rows() == 1 && result.history.cols() == 1 && result.history_times.size() == 1,
                "An observation shorter than the QR period must finish with one recorded QR");

        const auto event = [](const System::State &, const System::State &) { return true; };
        result = mc::ode::ComputeLCEs(system, 0.0, 12 * dt, 4, std::nullopt, event);
        Require(result.singularity_count == 12 && result.history.rows() == 12 && result.history_times.size() == 12,
                "Events coinciding with regular or final QR must produce a single row");
        Require(result.history.bottomRows(1).transpose() == result.spectrum, "Final QR differs from the spectrum");
    }

    void TestQRNormsAndPartialBasis()
    {
        using System = mc::ode::ContinuousDS<2>;
        class ScaleSystem : public System
        {
        public:
            explicit ScaleSystem(double scale) :
                System(
                    [](const State &, double, const mc::ode::NoContext &) -> State { return State::Zero(); },
                    [](const State &, double, const mc::ode::NoContext &) -> Jacobian { return Jacobian::Zero(); },
                    1.0),
                m_Scale(scale)
            {
            }

            State NextLTM(const State &w) override
            {
                return m_Scale * w;
            }

        private:
            double m_Scale;
        };

        ScaleSystem small(1e-16);
        const auto result = mc::ode::ComputeLCEs(small, 0.0, 1.0, 1, std::nullopt, {});
        Require(result.spectrum.isApprox(System::State::Constant(std::log(1e-16)), 1e-12),
                "A positive small QR diagonal was discarded");
        for (const double scale : {0.0, std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity()})
        {
            ScaleSystem invalid(scale);
            bool rejected = false;
            try
            {
                mc::ode::ComputeLCEs(invalid, 0.0, 1.0, 1, std::nullopt, {});
            }
            catch (const std::runtime_error &)
            {
                rejected = true;
            }
            Require(rejected, "An invalid QR norm produced a spectrum");
        }

        const System::State rates(-1.0, 0.5);
        System diagonal(
            [&rates](const System::State &x, double, const mc::ode::NoContext &) -> System::State
            {
                return rates.cwiseProduct(x);
            },
            [&rates](const System::State &, double, const mc::ode::NoContext &) -> System::Jacobian
            {
                return rates.asDiagonal();
            },
            0.125);
        const auto leading = mc::ode::ComputeLCEs(diagonal, 0.0, 64.0, 8, 1, {});
        Require(std::abs(leading.spectrum(0) - rates(1)) < 0.06,
                "Partial basis missed a leading direction outside the first coordinate");
    }

    void TestSmoothTangentRK4()
    {
        using Scalar = mc::ode::ContinuousDS<1>;
        const auto rhs = [](const Scalar::State &x, double t, const mc::ode::NoContext &) -> Scalar::State
        {
            return t * x;
        };
        const auto jac = [](const Scalar::State &, double t, const mc::ode::NoContext &) -> Scalar::Jacobian
        {
            return Scalar::Jacobian::Constant(t);
        };
        constexpr double dt = 0.03125;
        constexpr double t = 0.5;
        Scalar system(rhs, jac, dt, {}, {}, Scalar::State::Ones().eval(), t);
        const double expected = std::exp(dt * (t + dt / 2));
        Require(std::abs(system.NextLTM(Scalar::State::Ones().eval())(0) - expected) < 2e-8,
                "Time-dependent Jacobian must be evaluated at every RK4 stage");

        constexpr double gamma = 0.3;
        const mc::ode::RadonsContext args{
            .params = {.dt = dt, .gamma = gamma, .A = 0.25, .w0 = 1.0, .w = 1.0},
            .model = std::make_shared<mc::ArealPreisachModel>(1.0)
        };
        auto radons = mc::ode::GetRadonsSystem(dt, args, {}, Eigen::Vector2d(0.2, 0.0));
        const auto result = mc::ode::ComputeLCEs(*radons, 0.125, 2.0, 4, std::nullopt, {});
        Require(std::abs(result.spectrum.sum() + gamma) < 1e-7, "The smooth Radons spectrum violates the trace identity");
    }

    void TestEventAwareLCEs()
    {
        using namespace mc::ode;
        const double phi = std::acos(-1.0) / 2;
        auto model = std::make_shared<mc::ArealPreisachModel>(2.0);
        model->InitializeMemory(std::array{-2.0, 0.1});
        RadonsContext args{.params = {.dt = 0.5, .A = -2.0, .phi = phi}, .model = model};
        auto system = GetRadonsSystem(0.5, args, {}, Eigen::Vector2d(0.1, 1.0));
        const auto result = ComputeLCEs(*system, 0.0, 1.0, 1);
        Require(result.singularity_count == 1 && result.history.rows() == 2 &&
                    result.history_times == std::vector<double>{0.5, 1.0} &&
                    result.history.bottomRows(1).transpose() == result.spectrum,
                "Coincident event and scheduled QR must give only one row");
        Require(system->GetIntegrator() == RadonsIntegrator::RK4 &&
                    std::get<0>(model->HysteresisLoop()).empty(),
                "LCE calculation changed the caller's integrator or added trial samples");
        system->SetOutputStrategy(RadonsOutputStrategy::PerStage);
        const auto staged_setting = ComputeLCEs(*system, 0.0, 1.0, 1);
        Require(staged_setting.history == result.history && staged_setting.history_times == result.history_times &&
                    system->GetOutputStrategy() == RadonsOutputStrategy::PerStage,
                "ComputeLCEs must hold the output even when ordinary RK4 is configured for per-stage evaluation");

        args.params = {.dt = 0.75, .A = -1.6, .E = 0.35, .phi = phi};
        const Eigen::Vector2d x(0.25, 0.65);
        model->InitializeMemory(std::array{-2.0, x(0)});
        system = GetRadonsSystem(0.75, args, {}, x);
        const auto warmed = ComputeLCEs(*system, 0.75, 0.75, 20);
        system->SetIntegrator(RadonsIntegrator::EventAware);
        system->Next();
        Require(std::get<0>(model->HysteresisLoop()).size() > 1, "Warm-up fixture did not contain an event");
        const auto snapshot = CaptureRadonsSnapshot(*system);
        auto continued = GetRadonsSystem(0.75, ForkRadonsContext(snapshot, snapshot.x), {}, snapshot.x, false, snapshot.time);
        const auto separately_warmed = ComputeLCEs(*continued, 0.0, 0.75, 20);
        Require(warmed.spectrum.isApprox(separately_warmed.spectrum, 1e-10) &&
                    warmed.singularity_count == separately_warmed.singularity_count &&
                    warmed.history_times == separately_warmed.history_times,
                "The LCE warm-up and observation used different integration schemes");

        bool rejected = false;
        const auto fail = [](const Eigen::Vector2d &, const Eigen::Vector2d &) -> bool
        {
            throw std::runtime_error("observer stopped");
        };
        try
        {
            ComputeLCEs(*continued, 0.0, 1.5, 20, std::nullopt, fail);
        }
        catch (const std::runtime_error &error)
        {
            rejected = std::string_view(error.what()) == "observer stopped";
        }
        Require(rejected && continued->GetIntegrator() == RadonsIntegrator::RK4,
                "Scoped LCE integration lost the caller's mode after an exception");
    }
}

int main()
{
    TestInvalidLCEParameters();
    TestLCEHistory();
    TestQRNormsAndPartialBasis();
    TestSmoothTangentRK4();
    TestEventAwareLCEs();
}
