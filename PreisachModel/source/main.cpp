// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <print>
#include <random>
#include <chrono>
#include <cmath>
#include <memory>

#include "MUTCpp/Utils/Eigen.hpp"

#include "FileWriter/ExperimentArchive.hpp"
#include "Server/Client.hpp"
#include "Utils/Math.hpp"
#include "Utils/doubleToString.hpp"

#include "MUTCpp/ODE.hpp"
#include "MUTCpp/Hysteresis.hpp"

#include "MUTCpp/Utils/Profile.hpp"
#include "ODE/DuffingSystem.hpp"
#include "ODE/FourierSpectrum.hpp"
#include "ODE/LorenzSystem.hpp"
#include "ODE/RadonsSystem.hpp"
#include "ODE/ZeroOneTest.hpp"

using json = nlohmann::json;

static bool g_SendResults = false;

static void PublishResult(Experiment::Archive &archive)
{
    const auto path = archive.Publish();
    std::println("Saved experiment archive: {}", path.string());
    if (g_SendResults)
    {
        try
        {
            Net::SendArchive(path);
            std::println("Python accepted archive: {}", path.string());
        }
        catch (const std::exception &error)
        {
            throw std::runtime_error(std::format("Delivery failed for saved archive {}: {}", path.string(), error.what()));
        }
    }
}

static json StoreTimes(Experiment::Archive &archive, Eigen::Index samples, double dt)
{
    Eigen::VectorXd times(samples);
    for (Eigen::Index i = 0; i < times.size(); ++i)
    {
        times(i) = i * dt;
    }
    return archive.AddArray("time", times, {"sample"});
}

static void StoreRadonsPlots(Experiment::Archive &archive,
                             const mc::ode::RadonsSystem::Trajectory &trajectory, double dt,
                             const std::vector<double> &inputs, const std::vector<double> &outputs)
{
    StoreTimes(archive, trajectory.rows(), dt);
    archive.AddArray("x", trajectory.col(0), {"sample"});
    archive.AddArray("v", trajectory.col(1), {"sample"});
    archive.AddSeries("loop_inputs", inputs, "sample");
    archive.AddSeries("loop_outputs", outputs, "sample");
    archive.AddPlot({{"kind", "displacement"}});
    archive.AddPlot({{"kind", "velocity"}});
    archive.AddPlot({{"kind", "phase_portrait"}, {"title", "Radons phase portrait"},
                     {"x_label", "Displacement x(t)"}, {"y_label", "Velocity v(t)"}});
    archive.AddPlot({{"kind", "hysteresis_loop"}, {"title", "Preisach hysteresis loop"}});
    archive.AddPlot({{"kind", "fourier_spectrum"}});
}

template <int N>
static json StoreTrajectory(Experiment::Archive &archive, const std::string &name,
                            const std::vector<Eigen::Vector<double, N>> &trajectory)
{
    Eigen::Matrix<double, Eigen::Dynamic, N, Eigen::RowMajor> samples(trajectory.size(), N);
    for (Eigen::Index i = 0; i < samples.rows(); ++i)
    {
        samples.row(i) = trajectory[i].transpose();
    }
    return archive.AddArray(name, samples, {"sample", "coordinate"});
}

template <int N>
static json StoreTrajectoryMap(Experiment::Archive &archive, const std::string &prefix,
                               const mc::ode::DivergenceTrajectories<N> &trajectories)
{
    auto entries = json::object();
    for (const auto &[interval, trajectory] : trajectories)
    {
        entries[interval] = StoreTrajectory(archive, std::format("{}_{}", prefix, entries.size()), trajectory);
    }
    return entries;
}

static json StorePowerLaw(Experiment::Archive &archive, const mc::ode::power_law::PowerLawResult &result)
{
    using mc::json::EncodeNumber;
    using mc::json::NonFinite;
    return {
        {"name", "DivergenceDegreeRegressionData"}, {"e", result.epsilon},
        {"nu", EncodeNumber(result.exponent, NonFinite::AsNull)},
        {"C", EncodeNumber(result.intercept, NonFinite::AsNull)},
        {"r2", EncodeNumber(result.r_squared, NonFinite::AsNull)},
        {"Ts", archive.AddArray("power_log_times", result.log_times, {"interval"})},
        {"ns", archive.AddArray("power_mean_log_distances", result.mean_log_distances, {"interval"})},
        {"trajs1", StoreTrajectoryMap(archive, "power_primary", result.primary)},
        {"trajs2", StoreTrajectoryMap(archive, "power_perturbed", result.perturbed)}
    };
}

void OperatorDerivativeTest()
{
    double L = 1.0;
    double h = 0.01;
    double dt = 0.5;

    std::vector<uint32_t> t0 = {0, 40, 72, 96, 115, 142};
    std::vector u0 = {-1.0, 0.9, -0.8, 0.5, -0.3, 1.0};
    auto [t, u] = ContinueFunction(dt, t0, u0);

    // auto *model = new mc::DiscretePreisachModel(1, h, true);
    // auto model = std::make_shared<mc::DiscretePreisachModel>(L, h, true);
    auto model = std::make_shared<mc::ArealPreisachModel>(L, true);

    for (int i = 0; i < t.size(); i++)
    {
        model->P(u[i], 0.0, i);
    }

    auto [inputs1, outputs] = model->HysteresisLoop();
    auto [inputs2, derivatives] = model->DerivativeHistory();
    // auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();

    Experiment::Archive archive("results", "operator_derivative_test");
    archive.Parameters() = {{"dt", dt}, {"h", h}, {"L", L}};
    const auto input = archive.AddSeries("x", u, "sample");
    archive.Results() = {
        {"name", "OperatorDerivativeTest"}, {"method", "plot"}, {"dt", dt}, {"h", h},
        {"times", archive.AddSeries("time", t, "sample")},
        {"loop", {{"inputs", input}, {"outputs", archive.AddSeries("loop_outputs", outputs, "sample")}}},
        {"results", {{"x", input}, {"derivatives", archive.AddSeries("derivative", derivatives, "sample")}}}
    };
    archive.AddPlot({{"kind", "displacement"}, {"title", "Operator derivative input"}});
    PublishResult(archive);
}

mc::json::JsonDocument DoubleLoopBiffurcationDiagram(
    mc::ode::RadonsSystem &system,
    std::vector<double> ds,
    double researchTime,
    double timeForward,
    uint32_t var_idx = 0
    )
{
    const auto model = std::dynamic_pointer_cast<mc::DoublePreisachModel>(
        system.GetArgs().model);
    if (!model)
    {
        THROW_INVALID_ARGUMENT_ERROR("Double-loop bifurcation requires a double Preisach model");
    }
    const double mem_d = model->GetD();

    system.SetResetFn([ds](mc::ode::RadonsContext &args, std::optional<mc::ode::RadonsContext> &, uint32_t reset_idx)
    {
        const auto model = std::dynamic_pointer_cast<mc::DoublePreisachModel>(args.model);
        if (!model)
        {
            THROW_INVALID_ARGUMENT_ERROR("Double-loop bifurcation requires a double Preisach model");
        }
        model->ResetState();
        model->SetD(ds[reset_idx]);
    });

    mc::json::JsonDocument message({"name", "d", "max_params", "max_values", "min_params", "min_values"});
    message.AddField("name", "BifurcationDiagram");
    message.AddField("d", ds);

    Eigen::MatrixXd trajs = Eigen::MatrixXd::Zero(ds.size(), researchTime / system.GetDeltaTime() + 1);
    for (uint32_t i = 0; i < ds.size(); ++i)
    {
        system.Reset(i);

        system.Forward(timeForward);

        const auto traj = system.Forward(researchTime);
        trajs.row(i) = traj.col(var_idx).transpose();
    }

    const auto biff_data = mc::ode::detail::extract_bifurcation_extrema(ds, trajs);
    message.AddField("max_params", biff_data.max_params);
    message.AddField("max_values", biff_data.max_values);
    message.AddField("min_params", biff_data.min_params);
    message.AddField("min_values", biff_data.min_values);

    model->SetD(mem_d);

    return message;
}

void JustSolveRodos()
{
    double dt = 0.001;
    double time = 500.0;

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    double A = 1.;

    double L = 2.0;
    double E = 1.35;

    auto model = std::make_shared<mc::ArealPreisachModel>(L, false, false);
    auto next_model = std::make_shared<mc::ArealPreisachModel>(L, false, false);
    // auto model = std::make_shared<mc::DoubleArealPreisachModel>(L, 1.0, 1.0, false);
    // auto next_model = std::make_shared<mc::DoubleArealPreisachModel>(L, 1.0, 1.0, false);

    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E, .phi = 0.0}, model
    };

    mc::ode::RadonsContext nextArgs = args;
    nextArgs.model = next_model;

    // Eigen::Vector2d x0 = {0.0, 0.0};

    std::vector<Eigen::Vector2d> coords = {
        {1., 0.},
        {-1., 0.},
        {0., 1.},
        {0., -1.},
        {sqrt(2.) / 2, sqrt(2.) / 2},
        {-sqrt(2.) / 2, -sqrt(2.) / 2},
        {sqrt(2.) / 2, -sqrt(2.) / 2},
        {-sqrt(2.) / 2, sqrt(2.) / 2},
    };
    // const Eigen::Vector2d x0 = {0.95, -2.0}; // {-sqrt(2.) / 2., -sqrt(2.) / 2.}, {-1., 0.}
    const Eigen::Vector2d x0 = {0.95, -2.5}; // {sqrt(2.) / 2., sqrt(2.) / 2.}, {-sqrt(2.) / 2., -sqrt(2.) / 2.}
    // const Eigen::Vector2d x0 = {-0.95, 1.5}; // {-1., 0.}, {sqrt(2.) / 2., sqrt(2.) / 2.}
    // const Eigen::Vector2d x0 = {-0.90, 1.7}; // {-sqrt(2.) / 2., -sqrt(2.) / 2.}, {sqrt(2.) / 2, sqrt(2.) / 2}

    auto system = mc::ode::GetRadonsSystem(dt, args, nextArgs, x0, false);
    auto traj = system->Forward(time);
    
    system->Reset();
    uint32_t steps_in_bounds = 0;
    for (int i = 0; i < static_cast<uint32_t>(time / dt); ++i)
    {
        system->Next();
        
        bool in_bounds = std::abs(system->GetX()(0)) <= L;
        if (!in_bounds) break;
        
        steps_in_bounds++;
    }
    std::println("in bounds: {}", steps_in_bounds * dt);

    auto [inputs1, outputs] = model->HysteresisLoop();
    auto [inputs2, derivatives] = model->DerivativeHistory();
    // auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();

    // const auto [LCEs, LCEs_history] = mc::ode::ComputeLCEs(*system, 50.0, time, 500);
    // Eigen::VectorXd Ts = Eigen::arange(1.0, 5.0, 0.25);
    // const auto doc = mc::ode::power_law::ComputePowerLawExponent(*system, 50.0, Eigen::Vector2d{1.0, 0.0}, 1.0, Ts, 500,
    //                                                              -0.9);
    const auto doc = mc::ode::power_law::ComputePowerLawExponent(*system, coords[4], 0.1, 0.1, 0.);

    Experiment::Archive archive("results", "just_solve_rodos");
    archive.Parameters() = {{"dt", dt}, {"duration", time}, {"gamma", gamma}, {"w0", w0},
                            {"w", w}, {"A", A}, {"L", L}, {"E", E}, {"x0", mc::json::EncodeVector(x0)}};
    StoreRadonsPlots(archive, traj, dt, inputs1, outputs);
    archive.AddSeries("derivative", derivatives, "sample");
    archive.Results()["power_law"] = StorePowerLaw(archive, doc);
    PublishResult(archive);
}

void TestTrajsCircle()
{
    auto func = [](const Eigen::Vector3d &x, double t, const mc::ode::RadonsContext &args) -> Eigen::Vector3d
    {
        AL_PROFILE_FUNC("Rodos::func");
        double dt = args.params.dt;
        double gamma = args.params.gamma;
        double A = args.params.A;
        double w0 = args.params.w0;
        double w = args.params.w;
        double E = args.params.E;
        auto &model = args.model;

        Eigen::Vector3d res;

        res[0] = x[1];
        res[1] = A * std::cos(w * t) - gamma * x[1] - mc::utils::power(w0, 2) * x[0] + E * model->P(
            x[0], x[1], static_cast<int>(t / dt));
        res[2] = mc::utils::power(w0, 2);

        return res;
    };

    auto jac = [](const Eigen::Vector3d &, double, const mc::ode::RadonsContext &) -> Eigen::Matrix3d
    {
        return Eigen::Matrix3d::Zero();
    };

    double dt = 0.01;

    double gamma = 0.1;
    double beta = 0.04;
    double w0 = 1.0;
    double w = 1.0;
    // double A = 0.5;
    double A = 1.0;

    double L = 1.0;
    double h = 0.05;
    double E = 1.35;

    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E}, std::make_shared<mc::ArealPreisachModel>(L, false, false)
    };
    mc::ode::RadonsContext args2{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E}, std::make_shared<mc::ArealPreisachModel>(L, false, false)
    };

    double areaCoeff = -0.9;
    double time = 3.0;

    auto system = std::make_unique<mc::ode::ContinuousDS<3, mc::ode::RadonsContext>>(func, jac, dt, args, args2);
    system->SetResetFn([areaCoeff, L](mc::ode::RadonsContext &args, std::optional<mc::ode::RadonsContext> &nextArgs, uint32_t)
    {
        auto model1 = std::make_shared<mc::ArealPreisachModel>(L, false, false);
        auto model2 = std::make_shared<mc::ArealPreisachModel>(L, false, false);
        if (!std::isnan(areaCoeff))
        {
            model2->P(L, 0.0, -2);
            model2->P(areaCoeff * L, 0.0, -1);
        }

        args.model = std::move(model1);
        nextArgs->model = std::move(model2);
    });

    // std::vector<double> es = mc::arange(0.05, 1.05, 0.05).toFlattenVector();
    const auto es = Eigen::arange(0., 1.1, 0.1);
    std::vector<mc::ode::CandidateValue> e_range;
    std::ranges::transform(es, std::back_inserter(e_range), [](const double e) { return mc::ode::CandidateValue(e); });

    const Eigen::VectorXd angels = Eigen::arange(0., mc::consts::twoPi, 0.5);
    std::vector<mc::ode::CandidateValue> coords;
    coords.reserve(angels.size());
    for (const auto &angel : angels)
    {
        Eigen::Vector3d m = {std::cos(angel), std::sin(angel), 0.};
        coords.emplace_back(Eigen::MatrixXd(m));
    }

    // mc::Matrix<double> x0_step = mc::arange(0.0, 0.2, 0.05);
    // std::vector<mc::ode::CandidateValue> x0_step_range;
    // std::ranges::transform(x0_step, std::back_inserter(x0_step_range), [](const double e) { return mc::ode::CandidateValue(e); });

    auto gen_random_points = [](int n, double side, uint32_t &seed_out)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        seed_out = gen();
        gen.seed(seed_out);

        std::uniform_real_distribution<double> dist(-side / 2.0, side / 2.0);

        std::vector<Eigen::Vector3d> points;
        points.reserve(n);

        for (int i = 0; i < n; ++i)
        {
            points.push_back({dist(gen), dist(gen), 0.});
        }

        return points;
    };

    uint32_t seed;
    std::vector<Eigen::Vector3d> points = gen_random_points(10, 1., seed);

    std::println("seed: {}", seed);
    std::println("random points: {}", points);

    std::vector<mc::ode::CandidateValue> x0s;
    std::ranges::transform(points, std::back_inserter(x0s),
                           [](const Eigen::VectorXd &point) { return mc::ode::CandidateValue(Eigen::MatrixXd(point)); });

    auto div_solution_fn = [time](mc::ode::DynamicalSystem<3, mc::ode::RadonsContext> &system,
                                  const mc::ode::DivergenceCandidate<3> &candidate)
    {
        const auto &[e, v] = candidate;
        // mc::Matrix<double> x0 = args.at("x0").toMatrix();
        // system.SetX0(x0);
        // double x0_step = args.at("x0_step").toDouble();

        // mc::Matrix<double> x0 = x0_step * mc::normalize(v);
        // system.SetX0(x0);

        uint32_t count = static_cast<uint32_t>(time / system.GetDeltaTime());
        auto mainTraj = mc::ode::DynamicalSystem<3>::Trajectory::Zero(count + 1, 3).eval();
        mainTraj.row(0) = system.GetX().transpose();

        Eigen::Vector3d w = system.GetX() + v.normalized() * e;

        auto secondTraj = mc::ode::DynamicalSystem<3>::Trajectory::Zero(count + 1, 3).eval();
        mainTraj.row(0) = w.transpose();

        for (uint32_t i = 1; i <= count; ++i)
        {
            system.Forward(1);
            mainTraj.row(i) = system.GetX().transpose();

            w = system.NextTM(w);
            secondTraj.row(i) = w.transpose();
        }

        Eigen::VectorXd x1 = mainTraj.col(0);
        Eigen::VectorXd y1 = mainTraj.col(1);

        Eigen::VectorXd x2 = secondTraj.col(0);
        Eigen::VectorXd y2 = secondTraj.col(1);

        Eigen::VectorXd div = ((x1 - x2).array().square() + (y1 - y2).array().square()).sqrt();
        return div;
    };

    std::unordered_map<std::string, mc::ode::CandidateParams> params = {
        {"e", mc::ode::CandidateParams(e_range, std::nullopt)},
        {"v", mc::ode::CandidateParams(coords, std::nullopt)},
        // {"x0", mc::ode::CandidateParams(x0s, std::nullopt)},
        // {"x0_step", mc::ode::CandidateParams(x0_step_range, std::nullopt)},
    };

    auto res = mc::ode::FindBestParams(*system, mc::ode::DecodeDivergenceCandidate<3>, div_solution_fn, params);

    for (const auto &param : res)
    {
        const auto [best_e, best_v] = mc::ode::DecodeDivergenceCandidate<3>(param.params);
        // mc::Matrix<double> best_x0 = param.params.at("x0").toMatrix();
        // double best_x0_step = param.params.at("x0_step").toDouble();
        auto best_metric = param.metric;

        std::print("Best e: {}\n", best_e);
        const double degrees = 180.0 / mc::consts::pi;
        const Eigen::Vector3d angles = {degrees * std::acos(best_v[0]), degrees * std::asin(best_v[0]), 0.0};
        std::print("Best v: {}\n", mc::doubleToString(angles, 6));
        // std::print("Best x0: {}\n", best_x0);
        // std::print("Best x0_step: {}\n", best_x0_step);
        std::print("Metric value: {}\n", best_metric.m_MetricValue / dt);
        std::print("Metric start: {}\n", best_metric.m_Start.value());
        std::print("Metric end: {}\n", best_metric.m_End.value());

        constexpr int M = 20;
        constexpr double d1 = 0.5;
        constexpr double d2 = -0.5;

        const double startT = best_metric.m_Start.value() * dt;
        const double endT = best_metric.m_End.value() * dt;

        const double minT = (startT + d1) / M;
        const double maxT = (endT - startT + d2) / M;

        Eigen::VectorXd Ts = Eigen::arange(minT, maxT, 0.005);

        auto doc = mc::ode::DivergenceDegreeRegressionData(*system, 0., 0.1, best_v, Ts, M);

        // for (const auto &T : Ts)
        // {
        //     // system->SetX0(best_x0);
        //     std::vector<mc::Matrix<double>> traj1, traj2;
        //     const auto [n, t] = mc::ode::DivergenceDegreeRegressionData(*system, 0., best_e, best_v, T, M, traj1, traj2,
        //                                                                 false);
        //
        //     const auto strT = mc::doubleToString(T, 2);
        //
        //     trajs1.insert({strT, traj1});
        //     trajs2.insert({strT, traj2});
        //     std::println("n: {}", n);
        //     ns.push_back(n);
        // }
        //
        // mc::json::JsonDocument message({"name", "method", "dt", "e", "M", "Ts", "ns", "trajs1", "trajs2"});
        // message.AddField("name", "RodosLCEs");
        // message.AddField("method", "plot");
        // message.AddField("dt", dt);
        // message.AddField("e", best_e);
        // message.AddField("M", M);
        // message.AddField("Ts", mc::log(Ts));
        // message.AddField("ns", ns);
        // message.AddField("trajs1", trajs1);
        // message.AddField("trajs2", trajs2);
        Experiment::Archive archive("results", "divergence_regression");
        archive.Parameters() = {{"dt", dt}, {"e", doc.epsilon}, {"M", doc.repetitions}};
        archive.Results() = {
            {"name", "DivergenceDegreeRegressionData"}, {"e", doc.epsilon}, {"M", doc.repetitions},
            {"Ts", archive.AddArray("log_times", doc.log_times, {"interval"})},
            {"ns", archive.AddSeries("mean_log_distances", doc.mean_log_distances, "interval")},
            {"trajs1", StoreTrajectoryMap(archive, "primary", doc.primary)},
            {"trajs2", StoreTrajectoryMap(archive, "perturbed", doc.perturbed)}
        };
        archive.AddPlot({{"kind", "divergence_regression"}, {"title", "Divergence regression"}});
        PublishResult(archive);
    }

    // mc::Matrix<double> es = {0.};

    // auto doc = mc::ode::TrajsCircle(*system, es.toFlattenVector(), 10., .0, -0.9);

    // FileWriter file("TestTrajsCircle.json");
    // file.Write(doc.ToString());
}

void TestEverettFunction()
{
    double dt = 0.01;
    double time = 300.0;
    auto steps = static_cast<uint32_t>(time / dt);

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    double A = 1.5;

    double L = 1.0;
    double h = 0.05;
    // from 0.67+ (maybe 0.63) to 1.918- 
    double E = 1.1;

    // measure-function mu(alpha, beta) = 1
    auto uniformEverett = [](double x, double y)
    {
        if (y <= x)
            return 0.0;

        const double d = y - x;
        return 0.5 * d * d;
    };

    auto model = std::make_shared<mc::ArealPreisachModel>(L, false, false);
    // auto surface = Everett.GetoOutputSurfaceData(model->GetBounds()[0], model->GetBounds()[1], 501).toJSON();
    
    constexpr int K = 1;
    Eigen::VectorXd a(K);
    Eigen::VectorXd b(K);
    Eigen::VectorXd c(K);
    a << 4299.43437;
    b << 1.5;
    c << 0.2;
    mc::SzaboEverett Everett(a, b, c);
    const auto surface = Everett.GetoMeasureSurfaceData(model->GetBounds()[0], model->GetBounds()[1], 501);
    model->SetEverettFunction([Everett = std::move(Everett)](double x, double y) -> double
    {
        return Everett(x, y);
    });
    // model->SetEverettFunction(uniformEverett);

    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E, .phi = 0.0}, model
    };

    auto system = mc::ode::GetRadonsSystem(dt, args);
    auto traj = system->Forward(steps);

    auto [inputs1, outputs] = model->HysteresisLoop();
    auto [inputs2, derivatives] = model->DerivativeHistory();

    Experiment::Archive archive("results", "just_solve_rodos");
    archive.Parameters() = {{"dt", dt}, {"duration", time}, {"gamma", gamma}, {"w0", w0},
                            {"w", w}, {"A", A}, {"L", L}, {"E", E}, {"h", h}};
    StoreRadonsPlots(archive, traj, dt, inputs1, outputs);
    archive.AddSeries("derivative", derivatives, "sample");
    archive.AddArray("everett_x", surface.x, {"alpha"});
    archive.AddArray("everett_y", surface.y, {"beta"});
    archive.AddArray("everett_z", surface.z, {"beta", "alpha"});
    PublishResult(archive);
}

void TwoTrajsOnCircle()
{
    constexpr double L = 1.0;
    auto model = std::make_shared<mc::ArealPreisachModel>(L);

    double phase = mc::consts::pi / 4;

    double dt = 0.01;
    double time = 200.0;

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    // double A = 0.5;
    double A = 1.5;

    double E = 1.35;

    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E}, model
    };

    auto args2 = args;
    args2.model = std::make_shared<mc::ArealPreisachModel>(L, false, false);
    auto system = mc::ode::GetRadonsSystem(dt, args, args2);
    constexpr double areaCoeff = -0.9;

    Experiment::Archive archive("results", "two_trajs_on_circle");
    archive.Parameters() = {{"dt", dt}, {"duration", time}, {"e", 0.01}};
    auto &results = archive.Results();
    results = {{"name", "TrajsCircle"}, {"time", time}, {"trajs", json::object()}, {"loops", json::object()}};

    std::vector<Eigen::Vector3d> coords = {
        {1., 0., 0.},
        {sqrt(2.) / 2, sqrt(2.) / 2, 0.},
        {0., 1., 0.},
        {-sqrt(2.) / 2, sqrt(2.) / 2, 0.},
        {-1., 0., 0.},
        {-sqrt(2.) / 2, -sqrt(2.) / 2, 0.},
        {0., -1., 0.},
        {sqrt(2.) / 2, -sqrt(2.) / 2, 0.},
    };
    std::vector<std::string> coordsStr(coords.size());
    std::ranges::transform(coords, coordsStr.begin(), [](const Eigen::Vector3d &elem)
    {
        return mc::doubleToString(elem, 4);
    });
    results["coords"] = coordsStr;

    for (const auto v : coords)
    {
        system->Reset();
        system->Forward(10. * mc::consts::twoPi / w);

        Eigen::Vector2d v1 = system->GetX() + v.head<2>().normalized() * 0.01;

        system->SetResetFn([v1, L](mc::ode::RadonsContext &args, std::optional<mc::ode::RadonsContext> &nextArgs, uint32_t)
        {
            auto model1 = std::make_shared<mc::ArealPreisachModel>(L, false, false);
            auto model2 = std::make_shared<mc::ArealPreisachModel>(L, false, false);
            if (!std::isnan(areaCoeff))
            {
                // model2->P(L, -2);
                model2->P(v1[0], 0.0, -1);
            }

            args.model = std::move(model1);
            nextArgs->model = std::move(model2);
        });
        system->CallResetFn();

        std::vector<Eigen::Vector2d> mainTraj{system->GetX()};

        std::vector<Eigen::Vector2d> secondTraj{v1};

        auto count = static_cast<mc::uint32>(time / system->GetDeltaTime());
        while (count--)
        {
            auto traj = system->Forward(1);
            mainTraj.push_back(system->GetX());

            v1 = system->NextTM(v1);
            secondTraj.push_back(v1);
        }

        auto [in1, out1] = system->GetArgs().model->HysteresisLoop();
        auto [in2, out2] = system->GetNextArgs().model->HysteresisLoop();

        const auto key = mc::doubleToString(v, 4);
        const auto id = std::to_string(results["trajs"].size());
        if (!results.contains("sample_times"))
        {
            results["sample_times"] = StoreTimes(archive, mainTraj.size(), dt);
        }
        results["trajs"][key] = json::array({
            StoreTrajectory(archive, "primary_" + id, mainTraj),
            StoreTrajectory(archive, "perturbed_" + id, secondTraj)
        });
        results["loops"][key] = json::array({
            archive.AddSeries("in_primary_" + id, in1, "sample"),
            archive.AddSeries("out_primary_" + id, out1, "sample"),
            archive.AddSeries("in_perturbed_" + id, in2, "sample"),
            archive.AddSeries("out_perturbed_" + id, out2, "sample")
        });
    }

    archive.AddPlot({{"kind", "directional_phase_pair"}, {"title", "Primary and perturbed trajectories"}});
    PublishResult(archive);
}

// void PoincareMapping()
// {
//     std::vector<Eigen::VectorXd> radonsPM = {};
//     std::vector<Eigen::VectorXd> lorenzPM = {};
//
//     // RADONS PM
//     {
//         mc::ode::DynamicalSystem::DSFunc func = [](const Eigen::VectorXd &x, const double t,
//                                                    mc::ode::DSArgs &args) -> Eigen::VectorXd
//         {
//             AL_PROFILE_FUNC("Rodos::func");
//             double dt = args.at("dt").toDouble();
//             double gamma = args.at("gamma").toDouble();
//             double A = args.at("A").toDouble();
//             double w0 = args.at("w0").toDouble();
//             double w = args.at("w").toDouble();
//             double E = args.at("E").toDouble();
//             auto &model = args.at("model").toPreisachModel();
//
//             Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());
//
//             const double rhs = A * std::sin(w * t) + E * model->P(
//                 x[0], static_cast<int>(t / dt));
//
//             res[0] = x[1];
//             res[1] = rhs - gamma * x[1] - mc::utils::power(w0, 2) * x[0];
//             res[2] = mc::utils::power(w0, 2);
//
//             return res;
//         };
//
//         auto jac = [](Eigen::VectorXd x, double t, const mc::ode::DSArgs &args) -> Eigen::MatrixXd { return {}; };
//
//         double dt = 0.05;
//         double time = 200.0;
//
//         double gamma = 0.1;
//         double w0 = 1.0;
//         double w = 1.0;
//         // double A = 0.5;
//         double A = 1.5;
//
//         double E = 1.35;
//
//         constexpr double L = 1.0;
//         auto model = std::make_shared<mc::ArealPreisachModel>(L);
//
//         mc::ode::DSArgs args = {
//             {"dt", dt},
//             {"gamma", gamma},
//             {"A", A},
//             {"w0", w0},
//             {"w", w},
//             {"E", E},
//             {"model", mc::ode::Vote(model)}
//         };
//
//         mc::ode::DSArgs args2 = {};
//
//         auto system = std::make_unique<mc::ode::ContinuousDS<3, mc::ode::RadonsContext>>(func, jac, dt, args, args2);
//
//         radonsPM = mc::ode::PoincareMapping(system, mc::consts::twoPi / w, 200);
//     }
//
//     // LORENZ PM
//     {
//         Eigen::Vector3d x0 = {1.5, -1.5, 20.0};
//         double t0 = 0.0;
//         double dt = 0.01;
//         double time = 100.0;
//         int steps = static_cast<int>(time / dt);
//
//         double sigma = 10.0;
//         double rho = 28.0;
//         double beta = 8.0 / 3.0;
//
//         bool usePreisach = true;
//         double L = 1.0;
//         double E = 1.35;
//         auto model = std::make_shared<mc::ArealPreisachModel>(L);
//
//         auto func = [&](Eigen::VectorXd x, double t, const mc::ode::DSArgs &args) -> Eigen::VectorXd
//         {
//             Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());
//             res[0] = sigma * (x[1] - x[0]) + (usePreisach ? E * model->P(x[0], static_cast<int>(t / dt)) : 0.0);
//             res[1] = x[0] * (rho - x[2]) - x[1];
//             res[2] = x[0] * x[1] - beta * x[2];
//             return res;
//         };
//
//         auto jac = [&](Eigen::VectorXd x, double t, const mc::ode::DSArgs &args) -> Eigen::MatrixXd
//         {
//             Eigen::MatrixXd res = Eigen::MatrixXd::Zero(x.size(), x.size());
//             res(0, 0) = -sigma + (usePreisach ? E * model->DerivativeOperator(t, dt) : 0.0);
//             res(0, 1) = sigma;
//             res(1, 0) = rho - x[2];
//             res(1, 1) = -1.0;
//             res(1, 2) = -x[0];
//             res(2, 0) = x[1];
//             res(2, 1) = x[0];
//             res(2, 2) = -beta;
//             return res;
//         };
//
//         mc::ode::DSArgs args = {};
//
//         auto lorenz = std::make_unique<mc::ode::ContinuousDS<3>>(func, jac, dt, args, args, x0, t0);
//
//         // lorenzPM = mc::ode::PoincareMapping(system, mc::consts::twoPi / w, 100);
//     }
//
//     mc::json::JsonDocument message({"name", "RadonsPoincareMapping", "LorenzPoincareMapping"});
//     message.AddField("name", "PoincareMapping");
//     message.AddField("RadonsPoincareMapping", radonsPM);
//     message.AddField("LorenzPoincareMapping", lorenzPM);
//
//     FileWriter file("PoincareMapping.json");
//     file.Write(message.ToString());
// }

void BifurcationDiagram()
{
    double dt = 0.01;
    double time = 200.0;

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    // double A = 0.5;
    double A = 1.5;

    double E = 1.35;

    auto phi = [](double x, double a, double b, double c)
    {
        // const double e = mc::exp((b - x) / c);
        // return a * e / mc::utils::power(1 + e, 2);
        return (a / 2.) / (1. + std::cosh((x - b) / c));
    };

    auto everett = [phi](double x, double y)
    {
        std::vector<double> a = {0.113, 2.224e-3, 1.812e-2};
        std::vector<double> b = {56.468, 30.197, 86.148};
        std::vector<double> c = {6.035, 284.54, 38.842};

        double integral = 0.0;
        for (int i = 0; i < a.size(); i++)
        {
            integral += phi(x, a[i], b[i], c[i]) * phi(-y, a[i], b[i], c[i]);
        }
        return integral;
    };

    constexpr double L = 1.0;
    auto model = std::make_shared<mc::ArealPreisachModel>(L);
    
    constexpr int K = 1;
    Eigen::VectorXd a(K);
    Eigen::VectorXd b(K);
    Eigen::VectorXd c(K);
    a << 4299.43437;
    b << 1.5;
    c << 0.2;
    mc::SzaboEverett Everett(a, b, c);
    auto surface = Everett.GetoMeasureSurfaceData(model->GetBounds()[0], model->GetBounds()[1], 501).toJSON();
    model->SetEverettFunction([Everett = std::move(Everett)](double x, double y) -> double
    {
        return Everett(x, y);
    });

    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E, .phi = 0.0, .eps = 0.05}, model
    };
    auto system = mc::ode::GetRadonsSystem(dt, args, {}, {0.0, 0.0}, false);
    const Eigen::VectorXd Es = Eigen::arange(0., 3., 0.01);
    // auto epss = mc::ode::utils::VoteRange(0., 3.0, 0.005);
    auto d = mc::ode::BifurcationDiagram(*system, "E", std::span(Es.data(), static_cast<std::size_t>(Es.size())),
        [](mc::ode::RadonsContext &context, double value) { context.params.E = value; },
        30. * mc::consts::twoPi / w, 5. * mc::consts::twoPi / w);

    // double sigma = 10.0;
    // double rho = 28.0;
    // double beta = 8.0 / 3.0;
    // mc::ode::DSArgs args = {
    //     {"sigma", sigma},
    //     {"rho", rho},
    //     {"beta", beta},
    // };
    // auto system = mc::ode::GetLorenzSystem(dt, args);
    // auto rs = mc::ode::utils::VoteRange(0., 30., 0.5);
    // auto d = mc::ode::BifurcationDiagram(*system, {"rho", rs}, 50., 100.);

    Experiment::Archive archive("results", "bifurcation_diagram");
    archive.Parameters()["parameter"] = d.parameter_name;
    archive.Results() = {
        {"name", "BifurcationDiagram"},
        {d.parameter_name, archive.AddSeries("parameter_values", d.parameter_values, "parameter")},
        {"max_params", archive.AddSeries("max_params", d.extrema.max_params, "extremum")},
        {"max_values", archive.AddSeries("max_values", d.extrema.max_values, "extremum")},
        {"min_params", archive.AddSeries("min_params", d.extrema.min_params, "extremum")},
        {"min_values", archive.AddSeries("min_values", d.extrema.min_values, "extremum")}
    };
    archive.AddPlot({{"kind", "bifurcation"}, {"title", "Bifurcation diagram"}, {"x_label", d.parameter_name}});
    PublishResult(archive);
}

#include "MUTCpp/ODE/shuttle.hpp"

void FindShuttlePointInitialValues()
{
    double dt = 0.001;
    double gamma = 4.;
    double w0 = 3.;
    double w = 1.0;
    double A = 1.5;
    double E = 1.35;

    constexpr double L = 1.0;

    auto model1 = std::make_shared<mc::ArealPreisachModel>(L);
    // model1->P(L, 0.0, -1);
    mc::ode::RadonsContext args1{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E}, model1
    };
    auto system_minus = mc::ode::GetRadonsSystem(dt, args1);
    system_minus->SetResetFn([](mc::ode::RadonsContext &args, std::optional<mc::ode::RadonsContext> &, uint32_t)
    {
        args.model->ResetState();
    });

    auto model2 = std::make_shared<mc::ArealPreisachModel>(L);
    mc::ode::RadonsContext args2{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E}, model2
    };
    auto system_plus = mc::ode::GetRadonsSystem(dt, args2);
    system_plus->SetResetFn([](mc::ode::RadonsContext &args, std::optional<mc::ode::RadonsContext> &, uint32_t)
    {
        args.model->ResetState();
    });

    const auto xs = Eigen::arange(-1.8, 1.9, 0.2);
    const auto vs = Eigen::arange(-1.8, 1.9, 0.2);
    const auto u0xs = Eigen::arange(0.0, 1.9, 0.2);
    const auto u0vs = Eigen::arange(0.0, 1.9, 0.2);

    const double period = mc::consts::twoPi / w;

    Eigen::Matrix2d M(2, 2);
    M << 0.0, 1.0,
        -w0, -gamma;
    Eigen::Vector2d b = {0.0, A};

    const auto cone = mc::SolidCone2d::fromCurve(M, b, mc::detail::Settings());

    std::vector<std::string> results;
    results.reserve(xs.size() * vs.size() * u0xs.size() * u0vs.size());

    for (const double u0x : u0xs)
    {
        for (const double u0v : u0vs)
        {
            for (const double x : xs)
            {
                for (const double v : vs)
                {
                    const Eigen::Vector2d z = {x, v};
                    const Eigen::Vector2d u0 = {u0x, u0v};

                    system_minus->ResetTo(z);
                    const Eigen::VectorXd z_minus_traj = system_minus->ForwardWithoutHistory(period);

                    system_plus->ResetTo(z);
                    const Eigen::VectorXd z_plus_traj = system_plus->ForwardWithoutHistory(period);

                    const Eigen::Vector2d z_minus_new = z_minus_traj - z - u0;
                    const Eigen::Vector2d z_plus_new = z - z_plus_traj - u0;

                    bool cond_z_minus = cone->contains(z_minus_new);
                    bool cond_z_plus = cone->contains(z_plus_new);

                    if (cond_z_minus || cond_z_plus)
                    {
                        results.push_back(std::format("{};{};{};{}", mc::doubleToString(u0, 3),
                                                      mc::doubleToString(z, 3), cond_z_minus, cond_z_plus));
                    }
                }
            }
        }
    }

    std::ofstream file;
    file.open("trace_zs.csv", std::ios::trunc);
    file << std::fixed << std::setprecision(8);
    for (const auto &str : results)
    {
        file << str << "\n";
    }
}

using Params = std::vector<
    std::pair<
        Eigen::Vector2d,
        std::pair<
            std::vector<Eigen::Vector2d>,
            std::vector<Eigen::Vector2d>
        >
    >
>;

void printParams(const Params &params)
{
    for (std::size_t i = 0; i < params.size(); ++i)
    {
        const auto &[u0, z] = params[i];
        const auto &[z_minus, z_plus] = z;

        std::cout << "u0[" << i << "] = ("
            << u0.x() << ", " << u0.y() << ")\n";

        std::cout << "  z-:\n";
        for (std::size_t j = 0; j < z_minus.size(); ++j)
        {
            const auto &v = z_minus[j];

            std::cout << "    [" << j << "] ("
                << v.x() << ", " << v.y() << ")\n";
        }

        std::cout << "  z+:\n";
        for (std::size_t j = 0; j < z_plus.size(); ++j)
        {
            const auto &v = z_plus[j];

            std::cout << "    [" << j << "] ("
                << v.x() << ", " << v.y() << ")\n";
        }

        std::cout << '\n';
    }
}

void addParams(
    Params &params,
    const Eigen::Vector2d &u0,
    const Eigen::Vector2d &z_minus,
    const Eigen::Vector2d &z_plus)
{
    for (auto &[existing_u0, z] : params)
    {
        // Для double лучше использовать isApprox(), а не ==
        if (existing_u0.isApprox(u0))
        {
            auto &[z_minus_list, z_plus_list] = z;

            z_minus_list.push_back(z_minus);
            z_plus_list.push_back(z_plus);

            return;
        }
    }

    // u0 не найден — создаём новую запись
    params.push_back({
        u0,
        {
            {z_minus},
            {z_plus}
        }
    });
}

void RadonsShuttlePoint()
{
    AL_PROFILE_FUNC("RadonsShuttlePoint");

    // u0 -> [z_{-}, z_{+}]
    // Params params = {
    //     {
    //         {0.6, 1.6},
    //         {
    //                     {{-1.8, -1.6}, {-1.8, -1.8}}, // z_{-}
    //                     {{-0.4, 1.6}, {-0.4, 1.6}} // z_{+}
    //         }
    //     },
    //     {
    //                 {0.6, 1.4},
    //                 {
    //                     {{-1.8, -1.6}, {-1.8, -1.4}}, // z_{-}
    //                     {{-0.2, 1.4}, {1.4, 1.6}, {1.4, 1.4}, {0.0, 1.4}, {-0.2, 1.2}, {-0.2, 1.6}} // z_{+}
    //                 }
    //     }
    // };

    double dt = 0.001;
    double gamma = 4.;
    double w0 = 3.;
    double w = 1.0;
    double A = 1.5;
    double E = 1.35;

    constexpr double L = 2.0;

    auto model = std::make_shared<mc::ArealPreisachModel>(L);
    // model1->P(L, 0.0, -1);

    // const Eigen::VectorXd w0s = Eigen::arange(0.0, 6.0, 0.3);
    // const Eigen::VectorXd gammas = Eigen::arange(0.0, 6.0, 0.3);
    //
    // std::vector<std::pair<std::pair<double, double>, Params>> allParams = {};
    // allParams.reserve(w0s.size() * gammas.size());

    // for (const double w0 : w0s)
    // {
    //     for (const double gamma : gammas)
    //     {
    //         const Eigen::VectorXd ab = Eigen::arange(-2.0, 2.0, 0.2);
    //
    //         Params params = {};
    //         params.reserve(mc::utils::power(ab.size(), 3));
    //
    //         mc::ode::DSArgs args1 = {
    //             {"dt", dt},
    //             {"gamma", gamma},
    //             {"A", A},
    //             {"w0", w0},
    //             {"w", w},
    //             {"E", E},
    //             {"model", mc::ode::Vote(model)}
    //         };
    //         auto system = mc::ode::GetRadonsSystem(dt, args1);
    //         system->SetResetFn([](mc::ode::DSArgs &args, mc::ode::DSArgs &, uint32_t)
    //         {
    //             args.at("model").toPreisachModel()->ResetState();
    //         });
    //
    //         Eigen::Matrix2d M(2, 2);
    //         M << 0.0, 1.0,
    //             -w0, -gamma;
    //         Eigen::Vector2d b = {0.0, 1.0};
    //
    //         for (const auto u0_1 : ab)
    //         {
    //             for (const auto u0_2 : ab)
    //             {
    //                 for (const auto z_minus_1 : ab)
    //                 {
    //                     for (const auto z_minus_2 : ab)
    //                     {
    //                         for (const auto z_plus_1 : ab)
    //                         {
    //                             for (const auto z_plus_2 : ab)
    //                             {
    //                                 const Eigen::Vector2d u0 = {u0_1, u0_2};
    //                                 const Eigen::Vector2d z_minus = {z_minus_1, z_minus_2};
    //                                 const Eigen::Vector2d z_plus = {z_plus_1, z_plus_2};
    //
    //                                 const auto p = mc::ShuttlePoint(
    //                                     system, mc::SolidCone2d::fromCurve(M, b, mc::detail::Settings()),
    //                                     u0, z_minus, z_plus, mc::consts::twoPi / w);
    //
    //                                 if (p.status == mc::ShuttlePointCoverageStatus::ConvergedTwoLimits)
    //                                 {
    //                                     addParams(params, u0, z_minus, z_plus);
    //                                 }
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //         params.shrink_to_fit();
    //         if (params.size() != 0)
    //         {
    //             allParams.push_back({
    //                 {w0, gamma},
    //                 params
    //             });
    //         }
    //     }
    // }
    // allParams.shrink_to_fit();
    //
    // for (const auto &[wg, params] : allParams)
    // {
    //     std::cout << "w0 = " << wg.first << ", gamma = " << wg.second << ":" << "\n";
    //     printParams(params);
    // }

    const Eigen::VectorXd phis = Eigen::arange(-mc::consts::pi / 2.0, 3.0 * mc::consts::pi / 2.0, 0.1);
    for (const double phi : phis)
    {
        std::cout << "phi = " << phi << "\n";
        mc::ode::RadonsContext args1{
            {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E, .phi = phi}, model
        };
        auto system = mc::ode::GetRadonsSystem(dt, args1);
        system->SetResetFn([](mc::ode::RadonsContext &args, std::optional<mc::ode::RadonsContext> &, uint32_t)
        {
            args.model->ResetState();
        });

        Eigen::Matrix2d M(2, 2);
        M << 0.0, 1.0,
            -w0, -gamma;
        Eigen::Vector2d b = {0.0, 1.0};

        const Eigen::Vector2d u0 = {0.6, 1.4};

        const auto cone = mc::SolidCone2d::fromCurve(M, b, mc::detail::Settings());
        auto p = mc::ShuttlePoint(*system, *cone,
                                  u0, {-1.8, -1.4}, {0.0, 1.2}, mc::consts::twoPi / w);

        if (p.status == mc::ShuttlePointCoverageStatus::ConvergedTwoLimits)
        {
            std::println("LIMITS: z_odd: {}, z_even: {}", *p.limits[0], *p.limits[1]);
        }

        mc::json::JsonDocument message({"name", "shuttle_logs", "results"});
        message.AddField("name", "ShuttlePoint");
        message.AddField("shuttle_logs", p.doc);
    }

    // system->ResetTo(p.limits[0]);
    // auto traj = system->Forward(500.0);
    // Eigen::VectorXd x = traj.col(0);
    // Eigen::VectorXd v = traj.col(1);
    // auto [in, out] = model->HysteresisLoop();
    // message.AddSubField({"results", "limit1", "x"}, x);
    // message.AddSubField({"results", "limit1", "v"}, v);
    // message.AddSubField({"results", "limit1", "loop", "in"}, in);
    // message.AddSubField({"results", "limit1", "loop", "out"}, out);
    //
    // system->ResetTo(p.limits[1]);
    // traj = system->Forward(500.0);
    // x = traj.col(0);
    // v = traj.col(1);
    // std::tie(in, out) = model->HysteresisLoop();
    // message.AddSubField({"results", "limit2", "x"}, x);
    // message.AddSubField({"results", "limit2", "v"}, v);
    // message.AddSubField({"results", "limit2", "loop", "in"}, in);
    // message.AddSubField({"results", "limit2", "loop", "out"}, out);

    // FileWriter file("ShuttlePointLog.json");
    // file.Write(message.ToString());
}

void ZeroOneRadonsTest()
{
    double dt = 0.01;

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    double A = 1.5;

    double E = 1.35;

    constexpr double L = 1.0;
    auto model = std::make_shared<mc::ArealPreisachModel>(L);

    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E, .eps = 0.05}, model
    };
    auto system = mc::ode::GetRadonsSystem(dt, args, {}, {0.0, 0.0}, false);

    // double sigma = 10.0;
    // double rho = 28.0;
    // double beta = 8.0 / 3.0;
    // mc::ode::DSArgs args = {
    //     {"sigma", sigma},
    //     {"rho", rho},
    //     {"beta", beta},
    // };
    // auto system = mc::ode::GetLorenzSystem(dt, args);
    // auto rs = mc::ode::utils::VoteRange(0., 200., 1.);

    std::vector<double> k_values;

    constexpr double time = 500.0;
    const Eigen::VectorXd amplitudes = Eigen::arange(0., 4., 0.005);
    for (double amplitude : amplitudes)
    {
        std::cout << "New" << std::endl;
        system->Reset();
        args.params.A = amplitude;
        system->SetArgs(args);

        const auto traj = system->Forward(time);
        const auto x = traj.col(0);

        const auto res = mc::ode::ZeroOneTest_Fast_New(x, 10);
        k_values.push_back(res);
    }

    Experiment::Archive archive("results", "zero_one_test");
    archive.Parameters() = {{"parameter", "A"}, {"dt", dt}, {"duration", time}};
    archive.Results() = {
        {"name", "ZeroOneTest"}, {"A", archive.AddArray("amplitudes", amplitudes, {"parameter"})},
        {"Ks", archive.AddSeries("ks", k_values, "parameter")}
    };
    archive.AddPlot({{"kind", "zero_one"}, {"title", "0-1 test"}, {"x_label", "A"}});
    PublishResult(archive);
}

void ZeroOneTest()
{
    double dt = 0.01;

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    double A = 1.5;

    double E = 1.35;

    constexpr double L = 1.0;
    auto model = std::make_shared<mc::ArealPreisachModel>(L);

    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = gamma, .A = A, .w0 = w0, .w = w, .E = E, .eps = 0.05}, model
    };
    auto system = mc::ode::GetRadonsSystem(dt, args, {}, {0.0, 0.0}, false);
    const double time = 20.0 * mc::consts::twoPi / w;
    const auto traj = system->Forward(time);
    const auto x = traj.col(0);
    const auto len = x.size();
    const auto skip = static_cast<Eigen::Index>(len * 0.05);

    // constexpr int stride = 40;
    // Eigen::VectorXd x_downsampled = x.tail(len - skip)(Eigen::seq(0, Eigen::last, stride));
    // double mean = x_downsampled.mean();
    // x_downsampled.array() -= mean;

    // mc::json::JsonDocument message({"name", "x", "v"});
    // message.AddField("name", "ZeroOneTestTraj");
    // message.AddField("x", x.eval());
    // message.AddField("v", traj.col(1).eval());
    //
    // FileWriter file("ZeroOneTestTraj.json");
    // file.Write(message.ToString());

    // Генерация тестовых данных (например, шум vs синус)
    int N = 2000;
    Eigen::VectorXd regular(N);
    Eigen::VectorXd chaotic(N); // Используем random как прокси хаоса для примера

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < N; ++i)
    {
        regular[i] = std::sin(i * 0.1);
        chaotic[i] = 4.0 * dist(gen) * (1.0 - dist(gen)); // Логистическое отображение-подобное
    }

    // Рекомендуемый размер окна n_cut <= N / 10
    int n_cut = N / 10;

    double k_reg = mc::ode::ZeroOneTest_Fast_New(regular, n_cut);
    double k_chaos = mc::ode::ZeroOneTest_Fast_New(chaotic, n_cut);
    double k_raodns = mc::ode::ZeroOneTest_Fast_New(x);

    std::cout << "Regular K (expect ~0): " << k_reg << "\n";
    std::cout << "Chaotic K (expect ~1): " << k_chaos << "\n";
    std::cout << "Chaotic Radons K (expect ~1): " << k_raodns << "\n";
}

void ComplexRadonsResearch()
{
    std::vector<Eigen::Vector2d> x0s = {
        {0.0, 0.0},

        {0.1, 0.0},
        {-0.1, 0.0},

        {0.5, 0.0},
        {-0.5, 0.0},

        {1.0, 0.0},
        {-1.0, 0.0},

        {0.0, 0.1},
        {0.0, -0.1},

        {0.0, 0.5},
        {0.0, -0.5},

        {0.0, 1.0},
        {0.0, -1.0},

        {0.1, 0.1},
        {-0.1, 0.1},
        {0.1, -0.1},
        {-0.1, -0.1},

        {0.5, 0.5},
        {-0.5, 0.5},
        {0.5, -0.5},
        {-0.5, -0.5},

        {1.0, 1.0},
        {-1.0, 1.0},
        {1.0, -1.0},
        {-1.0, -1.0},
    };

    constexpr double solve_time = 500.0;
    constexpr double afc_time = 500.0;
    constexpr double dt = 0.01;
    constexpr double L = 1.0;

    constexpr double researchTime = 100.;
    constexpr double timeForward = 50.;
    constexpr double poincarePeriodMultiplier = 50.0;

    constexpr uint32_t M = 500;
    constexpr double e = 0.5;
    Eigen::VectorXd Ts = Eigen::arange(1.0, 5.0, 0.25);
    Eigen::Vector2d v = {1.0, 0.0};
    constexpr double areaCoeff = -0.9;

    Eigen::VectorXd freqs = Eigen::arange(0.0, 2.5, 0.05);

    auto model1 = std::make_shared<mc::ArealPreisachModel>(L, true, false);
    auto model2 = std::make_shared<mc::DoubleArealPreisachModel>(L, 1.0, 1.0);
    auto model3 = std::make_shared<mc::DoubleArealPreisachModel>(L, 0.7, 1.0);

    auto next_model1 = std::make_shared<mc::ArealPreisachModel>(L, true, false);
    auto next_model2 = std::make_shared<mc::DoubleArealPreisachModel>(L, 1.0, 1.0, true);
    auto next_model3 = std::make_shared<mc::DoubleArealPreisachModel>(L, 0.7, 1.0, true);

    const std::unordered_map<std::string, std::array<std::shared_ptr<mc::PreisachModelBase>, 2>> models = {
        {"single_loop", {model1, next_model1}},
        // {"double_loop_1.0", {model2, next_model2}},
        // {"double_loop_0.7", {model3, next_model3}},
    };

    const auto resetFn = [](mc::ode::RadonsContext &args, std::optional<mc::ode::RadonsContext> &, uint32_t)
    {
        auto &model = args.model;
        model->ResetState();
    };

    for (const auto &[model_name, models] : models)
    {
        auto &[model, next_model] = models;

        mc::ode::RadonsContext first_set{
            {.dt = dt, .gamma = 0.1, .A = 1.5, .w0 = 1.0, .w = 1.0, .E = 1.35}, model
        };
        mc::ode::RadonsContext second_set{
            {.dt = dt, .gamma = 0.5, .A = 1.5, .w0 = 1.0, .w = 1.0, .E = 1.35}, model
        };
        mc::ode::RadonsContext third_set{
            {.dt = dt, .gamma = 0.1, .A = 1.5, .w0 = 1.0, .w = 1.0, .E = 1.6}, model
        };
        mc::ode::RadonsContext fourth_set{
            {.dt = dt, .gamma = 0.1, .A = 1.5, .w0 = 1.0, .w = 1.0, .E = 1.5}, model
        };
        mc::ode::RadonsContext fifth_set{
            {.dt = dt, .gamma = 0.1, .A = 1.0, .w0 = 1.0, .w = 1.0, .E = 1.35}, model
        };
        mc::ode::RadonsContext sixth_set{
            {.dt = dt, .gamma = 0.1, .A = 0.5, .w0 = 1.0, .w = 1.0, .E = 1.35}, model
        };
        mc::ode::RadonsContext seventh_set{
            {.dt = dt, .gamma = 0.1, .A = 0.5, .w0 = 0.5, .w = 1.0, .E = 1.35}, model
        };
        mc::ode::RadonsContext eighth_set{
            {.dt = dt, .gamma = 0.1, .A = 0.5, .w0 = 1.0, .w = 0.5, .E = 1.35}, model
        };
        mc::ode::RadonsContext ninth_set{
            {.dt = dt, .gamma = 0.0, .A = 0.5, .w0 = 1.0, .w = 1.0, .E = 1.35}, model
        };
        mc::ode::RadonsContext tenth_set{
            {.dt = dt, .gamma = 0.1, .A = 0.5, .w0 = 1.3, .w = 1.0, .E = 1.35}, model
        };
        mc::ode::RadonsContext eleventh_set{
            {.dt = dt, .gamma = 0.1, .A = 0.5, .w0 = 1.0, .w = 1.3, .E = 1.35}, model
        };

        const std::unordered_map<std::string, mc::ode::RadonsContext &> params_sets = {
            {"first_set", first_set},
            // {"second_set", second_set},
            // {"third_set", third_set},
            // {"fourth_set", fourth_set},
            // {"fifth_set", fifth_set},
            // {"sixth_set", sixth_set},
            // {"seventh_set", seventh_set},
            // {"eighth_set", eighth_set},
            // {"ninth_set", ninth_set},
            // {"tenth_set", tenth_set},
            // {"eleventh_set", eleventh_set},
        };

        for (const auto &[args_set, args] : params_sets)
        {
            for (const auto &x0 : x0s)
            {
                auto nextArgs = args;
                nextArgs.model = next_model;
                auto system = mc::ode::GetRadonsSystem(dt, args, nextArgs, x0);
                system->SetResetFn(resetFn);
                system->Reset();

                Experiment::Archive archive("results", "complex_radons_research");
                const auto &p = args.params;
                archive.Parameters() = {
                    {"dt", dt}, {"time", solve_time}, {"model", model_name}, {"args_set", args_set},
                    {"args", {{"dt", p.dt}, {"gamma", p.gamma}, {"A", p.A}, {"w0", p.w0},
                              {"w", p.w}, {"E", p.E}, {"phi", p.phi}, {"eps", p.eps}}},
                    {"x0", mc::doubleToString(x0, 1)}
                };
                archive.Results()["name"] = "JustSolveRodos";
                archive.Results()["params"] = archive.Parameters();

                const auto full_name = std::format("{}/{}/{}", model_name, args_set, mc::doubleToString(x0, 1));

                std::cout << "start " << full_name << "\n";

                // auto traj = system->Forward(solve_time);
                // message.AddSubField({"results", "x"}, Eigen::VectorXd(traj.col(0)));
                // message.AddSubField({"results", "v"}, Eigen::VectorXd(traj.col(1)));
                //
                // const auto spectrum = mc::ode::FourierSpectrum(traj.col(0), dt);
                // message.AddSubField({"results", "fourier"}, spectrum);
                //
                // const auto [inputs1, outputs] = model->HysteresisLoop();
                // message.AddSubField({"results", "loop", "inputs"}, inputs1);
                // message.AddSubField({"results", "loop", "outputs"}, outputs);
                //
                // const auto [inputs2, derivatives] = model->DerivativeHistory();
                // message.AddSubField({"results", "derivatives"}, derivatives);
                //
                // const auto biff_A = mc::ode::BifurcationDiagram(*system, {"A", As}, researchTime, timeForward);
                // message.AddSubField({"results", "biff", "A"}, biff_A);
                //
                // const auto biff_E = mc::ode::BifurcationDiagram(*system, {"E", Es}, researchTime, timeForward);
                // message.AddSubField({"results", "biff", "E"}, biff_E);
                //
                // const auto biff_w = mc::ode::BifurcationDiagram(*system, {"w", ws}, researchTime, timeForward);
                // message.AddSubField({"results", "biff", "w"}, biff_w);
                //
                // // const auto [mLCE, mLCE_history] = mc::ode::Benettin_mLCE(*system, timeForward, e, T, M);
                // const auto [LCEs, LCEs_history] = mc::ode::ComputeLCEs(*system, timeForward, solve_time, M, 2);
                // message.AddSubField({"results", "LCEs", "LCEs"}, LCEs);
                // message.AddSubField({"results", "LCEs", "history"}, LCEs_history);

                // const auto div_doc = mc::ode::DivergenceDegreeRegressionData(*system, timeForward, e, v, Ts, M);
                // message.AddSubField({"results", "mLCE", "div_reg_data"}, div_doc);

                // auto power_law = mc::ode::power_law::ComputePowerLawExponent(*system, v, e, Ts, M);
                auto power_law = mc::ode::power_law::ComputePowerLawExponent(*system, v, e, 0.1);

                // const auto es = Eigen::arange(0.1, 1.2, 0.1);
                // const std::vector<Eigen::Vector2d> vs = {
                //     {1.0, 0.0},
                //     {1. / std::sqrt(2.), 0.0},
                //     {1. / std::sqrt(2.), 1. / std::sqrt(2.)},
                //     {1.0, 1. / std::sqrt(2.)},
                //     {1.0, 1.0}
                // };
                // const auto coeffs = Eigen::arange(-1.0, -0.5, 0.1);
                // for (const auto v1 : vs)
                // {
                //     
                //     
                //     for (const auto e1 : es)
                //     {
                //         for (const auto areaCoeff1 : coeffs)
                //         {
                //             mc::ode::power_law::ComputePowerLawExponent(*system, 0.0, v1, e1, Ts, M, areaCoeff1);
                //             // const mc::ode::power_law::regression::ContinuousPowerLawResult power_law = mc::ode::power_law::regression::ComputeContinuousPowerLawExponent(
                //             //     system, v, timeForward, 100.0 * mc::consts::twoPi / args.at("w").toDouble(), e1, areaCoeff1);
                //
                //             // const mc::ode::power_law::window::AdaptivePowerLawResult power_law =
                //             //     mc::ode::power_law::window::ComputeContinuousPowerLawExponent(
                //             //         system, v, timeForward, 100.0 * mc::consts::twoPi / args.at("w").toDouble(), e1,
                //             //         areaCoeff1, static_cast<uint32_t>(mc::consts::twoPi / args.at("w").toDouble() / dt));
                //
                //             // std::println("v: {}, e: {}, ac: {};  avg_nu: {}, num_segments: {}",
                //             //              mc::doubleToString(v1, 3),
                //             //              mc::doubleToString(e1, 2), mc::doubleToString(areaCoeff1, 2),
                //             //              power_law.average_nu, power_law.segments.size());
                //         }
                //     }
                // }
                archive.Results()["results"]["LCEs"]["power_law"] = StorePowerLaw(archive, power_law);
                archive.AddPlot({{"kind", "power_law"}, {"title", "Power law of trajectory divergence"}});
                // system->SetResetFn(resetFn); // restore reset fn

                // at period
                // const auto psm = mc::ode::PoincareStroboscopicMapping(
                //     system, poincarePeriodMultiplier, 2, mc::consts::twoPi / args.at("w").toDouble(), dt);
                // message.AddSubField({"results", "poincare_mapping", "psm"}, psm);
                //
                // // Plane v = 0
                // Eigen::Vector2d n = {0.0, 1.0};
                // Eigen::Vector2d p = {0.0, 0.0};
                // const auto pm = mc::ode::PoincareMapping(
                //     system, poincarePeriodMultiplier * mc::consts::twoPi / args.at("w").toDouble(),
                //     2, n, p);
                // message.AddSubField({"results", "poincare_mapping", "pm"}, pm);
                //
                // const auto afc = mc::ode::AFC(*system, afc_time, freqs);
                // message.AddSubField({"results", "AFC"}, afc);
                //
                // // Do this in the end, cause algorithm changes model d param
                // if (model_name == "double_loop_1.0")
                // {
                //     auto biff_d = DoubleLoopBiffurcationDiagram(*system, ds(args), researchTime, timeForward);
                //     message.AddSubField({"results", "biff", "d"}, biff_d);
                // }

                std::cout << "write " << full_name << "\n";

                PublishResult(archive);

                std::cout << "end" << "\n";
                std::cout << "\n";
                exit(0);
            }
        }
    }
}

void TwoTrajs()
{
    constexpr double dt = 0.01;
    constexpr double time = 300.;
    auto model = std::make_shared<mc::ArealPreisachModel>(1.0, true, false);
    auto next_model = std::make_shared<mc::ArealPreisachModel>(1.0, true, false);
    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = 0.1, .A = 1.5, .w0 = 1.0, .w = 1.0, .E = 1.35}, model
    };
    mc::ode::RadonsContext nextArgs = args;
    nextArgs.model = next_model;

    const Eigen::Vector2d v = {1., 0.};
    const double eps = 0.1;

    const Eigen::Vector2d x0 = {1., 0.};
    const auto pert_x0 = x0 + eps * v;

    auto system1 = mc::ode::GetRadonsSystem(dt, args, nextArgs, x0);
    auto system2 = mc::ode::GetRadonsSystem(dt, args, {}, pert_x0);

    auto traj1 = system1->ForwardTwoTrajs(time, pert_x0);
    model->ResetState();
    auto traj2 = system2->Forward(time);

    Experiment::Archive archive("results", "two_trajs");
    archive.Parameters() = {{"dt", dt}, {"duration", time}, {"e", eps}};
    archive.Results() = {
        {"name", "TrajsCircle"}, {"time", time},
        {"sample_times", StoreTimes(archive, traj1.first.rows(), dt)},
        {"traj1", archive.AddArray("primary", traj1.first, {"sample", "coordinate"})},
        {"traj1_next", archive.AddArray("perturbed", traj1.second, {"sample", "coordinate"})},
        {"traj2", archive.AddArray("reference", traj2, {"sample", "coordinate"})}
    };
    archive.AddPlot({{"kind", "phase_pair"}, {"title", "Primary and perturbed trajectories"}});
    PublishResult(archive);
}

void TwoTrajsOnPhaseTrajectory()
{
    constexpr double dt = 0.001;
    constexpr double time = 50.;

    const double eps = 0.1;

    // const double alpha = -1.;
    // const double beta = 1.;
    // const double delta = 0.2;
    // const double gamma = 0.5;
    // const double omega = 1.2;
    //
    // mc::ode::DSArgs args = {
    //     {"alpha", alpha},
    //     {"beta", beta},
    //     {"delta", delta},
    //     {"gamma", gamma},
    //     {"w", omega},
    // };
    //
    // const Eigen::Vector2d v = {1., 0.};
    // auto system = mc::ode::GetDuffingSystem(dt, args, args);


    // double sigma = 10.0;
    // double rho = 28.0;
    // double beta = 8.0 / 3.0;
    // mc::ode::DSArgs args = {
    //     {"sigma", sigma},
    //     {"rho", rho},
    //     {"beta", beta},
    // };
    // const Eigen::Vector3d v = {0., 0., 1.};
    // auto system = mc::ode::GetLorenzSystem(dt, args, args);


    auto model = std::make_shared<mc::ArealPreisachModel>(1.0, true, false);
    auto next_model = std::make_shared<mc::ArealPreisachModel>(1.0, true, false);
    mc::ode::RadonsContext args{
        {.dt = dt, .gamma = 0.1, .A = 1.5, .w0 = 1.0, .w = 1.0, .E = 1.35}, model
    };
    mc::ode::RadonsContext nextArgs = args;
    nextArgs.model = next_model;
    const Eigen::Vector2d v = {1., 0.};

    auto system = mc::ode::GetRadonsSystem(dt, args, nextArgs, {1., 0.});

    std::vector<Eigen::Vector2d> coords = {
        {1., 0.},
        {-1., 0.},
        {0., 1.},
        {0., -1.},
        {sqrt(2.) / 2, sqrt(2.) / 2},
        {-sqrt(2.) / 2, -sqrt(2.) / 2},
        {sqrt(2.) / 2, -sqrt(2.) / 2},
        {-sqrt(2.) / 2, sqrt(2.) / 2},
    };
    std::vector<std::string> coordsStr(coords.size());
    std::ranges::transform(coords, coordsStr.begin(), [](const Eigen::Vector2d &elem)
    {
        return mc::doubleToString(elem, 4);
    });

    // constexpr uint32_t N = 10;
    // constexpr double TEST_TIME = 100.;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<uint32_t> distrib(1, static_cast<uint32_t>(TEST_TIME / dt) - 1);
    // std::vector<Eigen::VectorXd> points;
    // std::vector<double> times;
    // points.reserve(N);
    // times.reserve(N);
    // const auto text_traj = system->Forward(TEST_TIME);
    // for (uint32_t i = 0; i < N; ++i)
    // {
    //     const uint32_t j = distrib(gen);
    //     points.push_back(text_traj.row(j));
    //     times.push_back(j * dt);
    // }

    std::unordered_map<uint32_t, std::vector<Eigen::MatrixXd>> trajs;

    // for (uint32_t i = 0; i < N; ++i)
    // {
    //     const auto x0 = points[i];
    //     system->ResetTo(x0);
    //     model->ResetState();
    //     next_model->ResetState();
    //     
    //     const auto pert_x0 = x0 + eps * v;
    //     auto traj = system->ForwardTwoTrajs(time, pert_x0);
    //     
    //     trajs.insert({i, {traj.first, traj.second}});
    // }

    // const Eigen::Vector2d x0 = {0.95, -2.0}; // {-sqrt(2.) / 2., -sqrt(2.) / 2.}, {-1., 0.}
    const Eigen::Vector2d x0 = {0.95, -2.5}; // {sqrt(2.) / 2., sqrt(2.) / 2.}, {-sqrt(2.) / 2., -sqrt(2.) / 2.}
    // const Eigen::Vector2d x0 = {-0.95, 1.5}; // {-1., 0.}, {sqrt(2.) / 2., sqrt(2.) / 2.}
    // const Eigen::Vector2d x0 = {-0.90, 1.7}; // {-sqrt(2.) / 2., -sqrt(2.) / 2.}, {sqrt(2.) / 2, sqrt(2.) / 2}

    for (uint32_t i = 0; i < coords.size(); ++i)
    {
        const auto v1 = coords[i];
        system->ResetTo(x0);
        model->ResetState();
        next_model->ResetState();

        const auto pert_x0 = x0 + eps * v1;
        auto traj = system->ForwardTwoTrajs(time, pert_x0);

        trajs.insert({i, {traj.first, traj.second}});
    }

    Experiment::Archive archive("results", "two_trajs_on_phase_trajectory");
    archive.Parameters() = {{"dt", dt}, {"duration", time}, {"e", eps}};
    auto &results = archive.Results();
    results = {{"name", "TrajsCircle"}, {"time", time}, {"coords", coordsStr}, {"trajs", json::object()}};
    if (!trajs.empty())
    {
        results["sample_times"] = StoreTimes(archive, trajs.begin()->second[0].rows(), dt);
    }
    for (const auto &[index, pair] : trajs)
    {
        const auto id = std::to_string(index);
        results["trajs"][id] = json::array({
            archive.AddArray("primary_" + id, pair[0], {"sample", "coordinate"}),
            archive.AddArray("perturbed_" + id, pair[1], {"sample", "coordinate"})
        });
    }
    archive.AddPlot({{"kind", "directional_phase_pair"}, {"title", "Primary and perturbed trajectories"}});
    PublishResult(archive);
}

int main(int argc, char **argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--send")
    {
        g_SendResults = true;
    }
    else if (argc != 1)
    {
        std::println(stderr, "Usage: PreisachModel.exe [--send]");
        return 2;
    }
    // TwoTrajsOnPhaseTrajectory();
    // ComplexRadonsResearch();
    // JustSolveRodos();

    // ZeroOneTest();

    // ZeroOneRadonsTest();

    // BifurcationDiagram();

    // DoublePreisachModelTest();

    // mc::run_example_2d();

    // FindShuttlePointInitialValues();
    // RadonsShuttlePoint();
    // ShiftTest();


    // CourseWorkModelsDiff();

    // Initial duration for profiler
    // Sleep(1000);
    // SinusTests();

    // DiscreteModelChaosTestByStepSize();

    // MainResearch();
    // PLot_mLCE_HysteronsStep_E();


    // SearchForDegreeOfNonLinearizedTrajsDivergence();

    // ArealModelTest();
    // OperatorDerivativeTest();

    // twoTrajs();

    // AFC();
    // AFC_test();

    // JustSolveRodos();
    // RodosLCEs();

    // DivergenceDegreeTable();


    // TwoTrajsOnCircle();
    // TestTrajsCircle();
    try
    {
        TestEverettFunction();
    }
    catch (const std::exception &error)
    {
        std::println(stderr, "{}", error.what());
        return 1;
    }

    // PoincareMapping();
    // BifurcationDiagram();
}
