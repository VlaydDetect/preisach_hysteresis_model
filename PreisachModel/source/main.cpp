// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <print>
#include <random>
#include <chrono>

#include "Eigen/utils.hpp"

#include "FileWriter/FileWriter.hpp"
#include "Server/Server.hpp"
#include "Utils/Math.hpp"

#include "MUTCpp/ODE.hpp"
#include "MUTCpp.hpp"

#include "Debug/Profile.hpp"
#include "ODE/DuffingSystem.hpp"
#include "ODE/FourierSpectrum.hpp"
#include "ODE/LorenzSystem.hpp"
#include "ODE/RadonsSystem.hpp"
#include "ODE/ZeroOneTest.hpp"

using json = nlohmann::json;

static Net::Server g_Server(8888, "127.0.0.1");

void OperatorDerivativeTest()
{
    double L = 1.0;
    double h = 0.01;
    double dt = 0.5;

    std::vector<uint32_t> t0 = {0, 40, 72, 96, 115, 142};
    std::vector u0 = {-1.0, 0.9, -0.8, 0.5, -0.3, 1.0};
    auto [t, u] = ContinueFunction(dt, t0, u0);

    // auto *model = new mc::DiscretePreisachModel(1, h, true);
    // auto model = mc::Ref<mc::DiscretePreisachModel>::Create(L, h, true);
    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L, true);

    for (int i = 0; i < t.size(); i++)
    {
        model->P(u[i], 0.0, i);
    }

    auto [inputs1, outputs] = model->HysteresisLoop();
    auto [inputs2, derivatives] = model->DerivativeHistory();
    // auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();

    mc::json::JsonDocument message({"name", "method", "dt", "h", "times", "loop", "results"});
    message.AddField("name", "OperatorDerivativeTest");
    message.AddField("method", "plot");
    message.AddField("dt", dt);
    message.AddField("h", h);
    message.AddField("times", t);
    message.AddSubField({"results", "x"}, u);
    message.AddSubField({"results", "derivatives"}, derivatives);
    message.AddSubField({"loop", "inputs"}, u);
    message.AddSubField({"loop", "outputs"}, outputs);
    // message.AddSubField({"anim", "in"}, inHist);
    // message.AddSubField({"anim", "x"}, xHist);
    // message.AddSubField({"anim", "y"}, yHist);
    // message.AddSubField({"anim", "out"}, outHist);
    // message.AddSubField({"anim", "save"}, true);

    // g_Server.SendDataMessage(message);
    mc::Ref file = mc::Ref<FileWriter>::Create("OperatorDerivativeTest.json");
    file->Write(message.ToString());
}

mc::json::JsonDocument DoubleLoopBiffurcationDiagram(
    mc::Ref<mc::ode::ContinuousDS> &system,
    std::vector<mc::ode::Vote> ds,
    double researchTime,
    double timeForward,
    uint32_t var_idx = 0
    )
{
    const double mem_d = static_cast<mc::DoublePreisachModel *>(system->GetArgs().at("model").toPreisachModel().Raw())->
        GetD();

    system->SetResetFn([&ds](mc::ode::DSArgs &args, mc::ode::DSArgs &, uint32_t reset_idx)
    {
        auto &model = args.at("model").toPreisachModel();
        model->ResetState();
        static_cast<mc::DoublePreisachModel *>(model.Raw())->SetD(ds[reset_idx].toDouble());
    });

    mc::json::JsonDocument message({"name", "d", "max_params", "max_values", "min_params", "min_values"});
    message.AddField("name", "BifurcationDiagram");
    mc::ode::WriteVotesToDoc(message, "d", ds, mc::ode::VoteDataType::Double);

    Eigen::MatrixXd trajs = Eigen::MatrixXd::Zero(ds.size(), researchTime / system->GetDeltaTime() + 1);
    for (uint32_t i = 0; i < ds.size(); ++i)
    {
        system->Reset(i);

        system->Forward(timeForward);

        const auto traj = system->Forward(researchTime);
        trajs.row(i) = traj.col(var_idx);
    }

    const auto biff_data = mc::ode::detail::extract_bifurcation_extrema(ds, trajs);
    message.AddField("max_params", biff_data.max_params);
    message.AddField("max_values", biff_data.max_values);
    message.AddField("min_params", biff_data.min_params);
    message.AddField("min_values", biff_data.min_values);

    static_cast<mc::DoublePreisachModel *>(system->GetArgs().at("model").toPreisachModel().Raw())->SetD(mem_d);

    return message;
}

void JustSolveRodos()
{
    double dt = 0.001;
    double time = 500.0;

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    double A = 1.5;

    double L = 1.0;
    double E = 1.35;

    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
    auto next_model = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
    // auto model = mc::Ref<mc::DoubleArealPreisachModel>::Create(L, 1.0, 1.0, false);
    // auto next_model = mc::Ref<mc::DoubleArealPreisachModel>::Create(L, 1.0, 1.0, false);

    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model)}
    };

    mc::ode::DSArgs nextArgs = args;
    nextArgs.at("model") = mc::ode::Vote(next_model);

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

    mc::Ref system = mc::ode::GetRadonsSystem(dt, args, nextArgs, x0, false);
    auto traj = system->Forward(time);

    auto [inputs1, outputs] = model->HysteresisLoop();
    auto [inputs2, derivatives] = model->DerivativeHistory();
    // auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();

    // const auto [LCEs, LCEs_history] = mc::ode::ComputeLCEs(system, 50.0, time, 500);
    // Eigen::VectorXd Ts = Eigen::arange(1.0, 5.0, 0.25);
    // const auto doc = mc::ode::power_law::ComputePowerLawExponent(system, 50.0, Eigen::Vector2d{1.0, 0.0}, 1.0, Ts, 500,
    //                                                              -0.9);
    const auto doc = mc::ode::power_law::ComputePowerLawExponent(system, coords[4], 0.1, 0.1, 0.);

    // mc::json::JsonDocument message({"name", "method", "dt", "h", "E", "time", "loop", "anim", "results"});
    mc::json::JsonDocument message({"name", "method", "dt", "time", "results", "loop"});
    message.AddField("name", "JustSolveRodos");
    message.AddField("method", "plot");
    message.AddField("dt", dt);
    message.AddField("time", time);
    message.AddSubField({"results", "x"}, Eigen::VectorXd(traj.col(0)));
    message.AddSubField({"results", "v"}, Eigen::VectorXd(traj.col(1)));
    message.AddSubField({"results", "power_law"}, doc);
    message.AddSubField({"results", "derivatives"}, derivatives);
    message.AddSubField({"results", "derivatives"}, derivatives);
    message.AddSubField({"loop", "inputs"}, inputs1);
    message.AddSubField({"loop", "outputs"}, outputs);

    // message.AddSubField({"anim", "in"}, inHist);
    // message.AddSubField({"anim", "x"}, xHist);
    // message.AddSubField({"anim", "y"}, yHist);
    // message.AddSubField({"anim", "out"}, outHist);

    // g_Server.SendDataMessage(message);
    mc::Ref file = mc::Ref<FileWriter>::Create("JustSolveRodos.json");
    file->Write(message.ToString());
}

void TestTrajsCircle()
{
    mc::ode::DynamicalSystem::DSFunc func = [](const Eigen::VectorXd &x, const double t,
                                               mc::ode::DSArgs &args) -> Eigen::VectorXd
    {
        AL_PROFILE_FUNC("Rodos::func");
        double dt = args.at("dt").toDouble();
        double gamma = args.at("gamma").toDouble();
        double A = args.at("A").toDouble();
        double w0 = args.at("w0").toDouble();
        double w = args.at("w").toDouble();
        double E = args.at("E").toDouble();
        auto &model = args.at("model").toPreisachModel();

        Eigen::VectorXd res = Eigen::VectorXd::Zero(x.size());

        res[0] = x[1];
        res[1] = A * mc::cos(w * t) - gamma * x[1] - mc::power(w0, 2) * x[0] + E * model->P(
            x[0], x[1], static_cast<int>(t / dt));
        res[2] = mc::power(w0, 2);

        return res;
    };

    mc::ode::DynamicalSystem::DSFunc jac = [](const Eigen::VectorXd &x, const double t,
                                              mc::ode::DSArgs &args) -> Eigen::MatrixXd
    {
        return {};
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

    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"beta", beta},
        {"w0", w0},
        {"w", w},
        {"E", E},
        // {"model", model}
    };
    mc::ode::DSArgs args2 = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"beta", beta},
        {"w0", w0},
        {"w", w},
        {"E", E},
        // {"model", model2}
    };

    double areaCoeff = -0.9;
    double time = 3.0;

    mc::Ref system = mc::Ref<mc::ode::ContinuousDS>::Create(func, jac, dt, args, args2);
    system->SetResetFn([areaCoeff, L](mc::ode::DSArgs &args, mc::ode::DSArgs &nextArgs, uint32_t)
    {
        auto model1 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
        auto model2 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
        if (!isnan(areaCoeff))
        {
            model2->P(L, 0.0, -2);
            model2->P(areaCoeff * L, 0.0, -1);
        }

        args.insert_or_assign("model", mc::ode::Vote(model1));
        nextArgs.insert_or_assign("model", mc::ode::Vote(model2));
    });

    // std::vector<double> es = mc::arange(0.05, 1.05, 0.05).toFlattenVector();
    std::vector<double> es = mc::arange(0., 1.1, 0.1).toFlattenVector();
    std::vector<mc::ode::Vote> e_range;
    std::ranges::transform(es, std::back_inserter(e_range), [](const double e) { return mc::ode::Vote(e); });

    const Eigen::VectorXd angels = Eigen::arange(0., mc::consts::twoPi, 0.5);
    std::vector<mc::ode::Vote> coords;
    coords.reserve(angels.size());
    for (const auto &angel : angels)
    {
        Eigen::Vector3d m = {mc::cos(angel), mc::sin(angel), 0.};
        coords.emplace_back(m);
    }

    // mc::Matrix<double> x0_step = mc::arange(0.0, 0.2, 0.05);
    // std::vector<mc::ode::Vote> x0_step_range;
    // std::ranges::transform(x0_step, std::back_inserter(x0_step_range), [](const double e) { return mc::ode::Vote(e); });

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

    std::vector<mc::ode::Vote> x0s;
    std::ranges::transform(points, std::back_inserter(x0s),
                           [](const Eigen::VectorXd &point) { return mc::ode::Vote(point); });

    auto div_solution_fn = [time](mc::Ref<mc::ode::DynamicalSystem> system,
                                  const std::unordered_map<mc::ode::DSArgs::key_type, mc::ode::Vote> &args)
    {
        double e = args.at("e").toDouble();
        Eigen::VectorXd v = args.at("v").toMatrix();
        // mc::Matrix<double> x0 = args.at("x0").toMatrix();
        // system->SetX0(x0);
        // double x0_step = args.at("x0_step").toDouble();

        // mc::Matrix<double> x0 = x0_step * mc::normalize(v);
        // system->SetX0(x0);

        uint32_t count = static_cast<uint32_t>(time / system->GetDeltaTime());
        Eigen::MatrixXd mainTraj = Eigen::MatrixXd::Zero(count + 1, system->GetDimension());
        mainTraj.row(0) = system->GetX().transpose();

        Eigen::VectorXd w = system->GetX() + v.normalized() * e;

        Eigen::MatrixXd secondTraj = Eigen::MatrixXd::Zero(count + 1, system->GetDimension());
        mainTraj.row(0) = w.transpose();

        for (uint32_t i = 1; i <= count; ++i)
        {
            system->Forward(1);
            mainTraj.row(i) = system->GetX().transpose();

            w = system->NextTM(w);
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

    auto res = mc::ode::FindBestParams(system, div_solution_fn, params);

    for (const auto &param : res)
    {
        double best_e = param.params.at("e").toDouble();
        Eigen::VectorXd best_v = param.params.at("v").toVector();
        // mc::Matrix<double> best_x0 = param.params.at("x0").toMatrix();
        // double best_x0_step = param.params.at("x0_step").toDouble();
        auto best_metric = param.metric;

        std::print("Best e: {}\n", best_e);
        mc::Matrix v = {mc::rad2deg(mc::arccos(best_v[0])), mc::rad2deg(mc::arcsin(best_v[0])), 0.0};
        std::print("Best v: {}\n", v);
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

        auto doc = mc::ode::DivergenceDegreeRegressionData(system, 0., 0.1, best_v, Ts, M);

        // for (const auto &T : Ts)
        // {
        //     // system->SetX0(best_x0);
        //     std::vector<mc::Matrix<double>> traj1, traj2;
        //     const auto [n, t] = mc::ode::DivergenceDegreeRegressionData(system, 0., best_e, best_v, T, M, traj1, traj2,
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
        mc::Ref file = mc::Ref<FileWriter>::Create("RodosLCEs_Regression.json");
        file->Write(doc.ToString());
    }

    // mc::Matrix<double> es = {0.};

    // auto doc = mc::ode::TrajsCircle(system, es.toFlattenVector(), 10., .0, -0.9);

    // mc::Ref file = mc::Ref<FileWriter>::Create("TestTrajsCircle.json");
    // file->Write(doc.ToString());
}

void TestEverettFunction()
{
    double dt = 0.01;
    double time = 300.0;
    auto steps = static_cast<uint32_t>(time / dt);

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    double A = 0.5;

    double L = 1.0;
    double h = 0.05;
    double E = 1.35;

    auto phi = [](double x, double a, double b, double c)
    {
        // const double e = mc::exp((b - x) / c);
        // return a * e / mc::power(1 + e, 2);
        return (a / 2.) / (1 + std::cosh((x - b) / c));
    };

    auto everett = [phi](double x, double y)
    {
        // return 1.;

        // std::vector<double> a = {0.113, 2.224e-3, 1.812e-2};
        // std::vector<double> b = {56.468, 30.197, 86.148};
        // std::vector<double> c = {6.035, 284.54, 38.842};

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

    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
    model->SetEverettFunction(everett);

    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model)}
    };

    mc::ode::DSArgs args2 = {};

    mc::Ref system = mc::ode::GetRadonsSystem(dt, args, args2);
    auto traj = system->Forward(steps);

    auto [inputs1, outputs] = model->HysteresisLoop();
    auto [inputs2, derivatives] = model->DerivativeHistory();

    mc::json::JsonDocument message({"name", "method", "dt", "h", "E", "time", "loop", "anim", "results"});
    message.AddField("name", "JustSolveRodos");
    message.AddField("method", "plot");
    message.AddField("dt", dt);
    message.AddField("h", h);
    message.AddField("E", E);
    message.AddField("time", time);
    message.AddSubField({"results", "x"}, traj.col(0).eval());
    message.AddSubField({"results", "v"}, traj.col(1).eval());
    message.AddSubField({"results", "derivatives"}, derivatives);
    message.AddSubField({"loop", "inputs"}, inputs1);
    message.AddSubField({"loop", "outputs"}, outputs);

    // g_Server.SendDataMessage(message);
    mc::Ref file = mc::Ref<FileWriter>::Create("JustSolveRodos.json");
    file->Write(message.ToString());
}

void CourseWorkModelsDiff()
{
    constexpr double L = 1.0;
    constexpr double h1 = 0.05;
    // constexpr double h2 = 0.005;

    // auto model = mc::Ref<mc::DiscretePreisachModel>::Create(L, h1, true);
    // auto model = mc::Ref<mc::DiscretePreisachModel>::Create(L, h1, true);
    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L, true);

    double dt = 0.01;
    double time = 1000.0;

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    // double A = 0.5;
    double A = 1.0;

    double E = 1.35;

    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model)}
    };

    mc::ode::DSArgs args2 = {};

    mc::Ref system = mc::ode::GetRadonsSystem(dt, args, {}, {0.0, 0.0}, false);
    // constexpr double areaCoeff = -0.9;
    // system->SetResetFn([areaCoeff, L](mc::ode::DSArgs &args, mc::ode::DSArgs &nextArgs)
    // {
    //     auto model1 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
    //     auto model2 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
    //     if (!isnan(areaCoeff))
    //     {
    //         model2->P(L, -2);
    //         model2->P(areaCoeff * L, -1);
    //     }
    //
    //     args.insert_or_assign("model", mc::ode::Vote(model1));
    //     nextArgs.insert_or_assign("model", mc::ode::Vote(model2));
    // });

    std::iostream::sync_with_stdio(false);
    auto start_bench = std::chrono::high_resolution_clock::now();
    auto traj = system->Forward(time);
    auto stop_bench = std::chrono::high_resolution_clock::now();

    auto duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(stop_bench - start_bench).
        count()) * 1e-9;
    std::cout << std::fixed << duration << std::setprecision(9);
    std::cout << " sec\n";

    // auto [mLCE, hist] = mc::ode::mLCE(system, 1, 500000);
    // std::print("{}", mLCE);

    // mc::json::JsonDocument message({"name", "method", "mLCE", "hist"});
    // message.AddField("name", "RodosLCEs");
    // message.AddField("method", "plot");
    // message.AddField("mLCE", mLCE);
    // message.AddField("hist", hist);
    // mc::Ref file = mc::Ref<FileWriter>::Create("RodosLCEs.json");
    // file->Write(message.ToString());


    auto [inputs, outputs] = model->HysteresisLoop();

    mc::json::JsonDocument message({"name", "method", "dt", "h", "E", "time", "results", "loop"});
    message.AddField("name", "JustSolveRodos");
    message.AddField("method", "plot");
    message.AddField("dt", dt);
    message.AddField("h", h1);
    message.AddField("E", E);
    message.AddField("time", time);
    message.AddSubField({"results", "x"}, traj.col(0).eval());
    message.AddSubField({"results", "v"}, traj.col(1).eval());
    message.AddSubField({"loop", "inputs"}, inputs);
    message.AddSubField({"loop", "outputs"}, outputs);
    mc::Ref file = mc::Ref<FileWriter>::Create("JustSolveRodos.json");
    file->Write(message.ToString());

    // std::unordered_map<std::string, std::vector<Eigen::MatrixXd>> trajs1, trajs2;
    // std::vector<double> ns;
    //
    // constexpr int M = 400;
    //
    // Eigen::VectorXd Ts = Eigen::arange(0.1, 2.35, 0.25);
    //
    // auto doc = mc::ode::DivergenceDegreeRegressionData(system, 0., 1., Eigen::Vector3d(0.7, -0.7, 0.0), Ts, M, trajs1,
    //                                                    trajs2, ns);
    // mc::Ref file = mc::Ref<FileWriter>::Create("RodosLCEs_Regression.json");
    // file->Write(doc.ToString());

    // std::vector<mc::Matrix<double>> traj1 = {};
    // std::vector<mc::Matrix<double>> traj2 = {};
    // auto n = mc::ode::FindDivergenceDegree(system, 0., 1., {0.7, -0.7, 0.0}, .5, M, traj1, traj2);
    // std::print("{}", n);
}

void TwoTrajsOnCircle()
{
    constexpr double L = 1.0;
    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L);

    double phase = mc::consts::pi / 4;

    double dt = 0.01;
    double time = 200.0;

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    // double A = 0.5;
    double A = 1.5;

    double E = 1.35;

    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model)}
    };

    mc::ode::DSArgs args2 = {};

    mc::Ref system = mc::ode::GetRadonsSystem(dt, args, args2);
    constexpr double areaCoeff = -0.9;

    mc::json::JsonDocument message({"name", "coords", "time", "trajs"});
    message.AddField("name", "TrajsCircle");
    message.AddField("time", time);

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
        return Eigen::DoubleVectorToString(elem, 4);
    });
    message.AddField("coords", coordsStr);

    std::unordered_map<std::string,
                       std::array<std::vector<std::vector<double>>, 2>> trajs = {};

    std::unordered_map<std::string, std::array<std::vector<double>, 4>> loops = {};

    for (const auto v : coords)
    {
        system->Reset();
        system->Forward(10. * mc::consts::twoPi / w);

        Eigen::VectorXd v1 = system->GetX() + v.normalized() * 0.01;

        system->SetResetFn([v1, L](mc::ode::DSArgs &args, mc::ode::DSArgs &nextArgs, uint32_t)
        {
            auto model1 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
            auto model2 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
            if (!isnan(areaCoeff))
            {
                // model2->P(L, -2);
                model2->P(v1[0], 0.0, -1);
            }

            args.insert_or_assign("model", mc::ode::Vote(model1));
            nextArgs.insert_or_assign("model", mc::ode::Vote(model2));
        });
        system->CallResetFn();

        std::vector<std::vector<double>> mainTraj = {};
        mainTraj.push_back(Eigen::to_std(system->GetX()));

        std::vector<std::vector<double>> secondTraj = {};
        secondTraj.push_back(Eigen::to_std(v1));

        auto count = static_cast<mc::uint32>(time / system->GetDeltaTime());
        while (count--)
        {
            auto traj = system->Forward(1);
            mainTraj.push_back(Eigen::to_std(system->GetX()));

            v1 = system->NextTM(v1);
            secondTraj.push_back(Eigen::to_std(v1));
        }

        auto [in1, out1] = system->GetArgs().at("model").toPreisachModel()->HysteresisLoop();
        auto [in2, out2] = system->GetNextArgs().at("model").toPreisachModel()->HysteresisLoop();

        trajs.insert({Eigen::DoubleVectorToString(v, 4), {mainTraj, secondTraj}});
        loops.insert({Eigen::DoubleVectorToString(v, 4), {in1, out1, in2, out2}});
    }

    auto &doc = message.GetDoc();

    doc["trajs"] = trajs;
    doc["loops"] = loops;

    mc::Ref file = mc::Ref<FileWriter>::Create("TwoTrajsOnCircle.json");
    file->Write(message.ToString());
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
//             const double rhs = A * mc::sin(w * t) + E * model->P(
//                 x[0], static_cast<int>(t / dt));
//
//             res[0] = x[1];
//             res[1] = rhs - gamma * x[1] - mc::power(w0, 2) * x[0];
//             res[2] = mc::power(w0, 2);
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
//         auto model = mc::Ref<mc::ArealPreisachModel>::Create(L);
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
//         mc::Ref system = mc::Ref<mc::ode::ContinuousDS>::Create(func, jac, dt, args, args2);
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
//         auto model = mc::Ref<mc::ArealPreisachModel>::Create(L);
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
//         mc::Ref lorenz = mc::Ref<mc::ode::ContinuousDS>::Create(func, jac, dt, args, args, x0, t0);
//
//         // lorenzPM = mc::ode::PoincareMapping(system, mc::consts::twoPi / w, 100);
//     }
//
//     mc::json::JsonDocument message({"name", "RadonsPoincareMapping", "LorenzPoincareMapping"});
//     message.AddField("name", "PoincareMapping");
//     message.AddField("RadonsPoincareMapping", radonsPM);
//     message.AddField("LorenzPoincareMapping", lorenzPM);
//
//     mc::Ref file = mc::Ref<FileWriter>::Create("PoincareMapping.json");
//     file->Write(message.ToString());
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
        // return a * e / mc::power(1 + e, 2);
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
    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L);
    // model->SetEverettFunction(everett);

    // mc::ode::DSArgs args = {
    //     {"dt", dt},
    //     {"gamma", gamma},
    //     {"A", A},
    //     {"w0", w0},
    //     {"w", w},
    //     {"E", E},
    //     {"model", mc::ode::Vote(model)},
    //     {"eps", 0.05}
    // };
    // auto system = mc::ode::GetRadonsSystem(dt, args, {}, {0.0, 0.0}, false);
    // auto As = mc::ode::utils::VoteRange(0.0, 4., 0.005);
    // auto Es = mc::ode::utils::VoteRange(0., 3., 0.005);
    // auto ws = mc::ode::utils::VoteRange(0., 2., 0.05);
    // auto epss = mc::ode::utils::VoteRange(0., 3.0, 0.005);
    // auto d = mc::ode::BifurcationDiagram(system, {"w", ws}, 60. * mc::consts::twoPi / w, 5. * mc::consts::twoPi / w);

    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;
    mc::ode::DSArgs args = {
        {"sigma", sigma},
        {"rho", rho},
        {"beta", beta},
    };
    auto system = mc::ode::GetLorenzSystem(dt, args);
    auto rs = mc::ode::utils::VoteRange(0., 30., 0.5);
    auto d = mc::ode::BifurcationDiagram(system, {"rho", rs}, 50., 100.);

    mc::Ref file = mc::Ref<FileWriter>::Create("BifurcationDiagram.json");
    file->Write(d.ToString());
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

    auto model1 = mc::Ref<mc::ArealPreisachModel>::Create(L);
    // model1->P(L, 0.0, -1);
    mc::ode::DSArgs args1 = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model1)}
    };
    mc::Ref<mc::ode::DynamicalSystem> system_minus = mc::ode::GetRadonsSystem(dt, args1);
    system_minus->SetResetFn([](mc::ode::DSArgs &args, mc::ode::DSArgs &, uint32_t)
    {
        args.at("model").toPreisachModel()->ResetState();
    });

    auto model2 = mc::Ref<mc::ArealPreisachModel>::Create(L);
    mc::ode::DSArgs args2 = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model2)}
    };
    mc::Ref<mc::ode::DynamicalSystem> system_plus = mc::ode::GetRadonsSystem(dt, args1);
    system_plus->SetResetFn([](mc::ode::DSArgs &args, mc::ode::DSArgs &, uint32_t)
    {
        args.at("model").toPreisachModel()->ResetState();
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
                        results.push_back(std::format("{};{};{};{}", Eigen::DoubleVectorToString(u0, 3),
                                                      Eigen::DoubleVectorToString(z, 3), cond_z_minus, cond_z_plus));
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

void RadonsShuttlePoint()
{
    AL_PROFILE_FUNC("RadonsShuttlePoint");

    double dt = 0.001;
    double gamma = 4.;
    double w0 = 3.;
    double w = 1.0;
    double A = 1.5;
    double E = 1.35;

    constexpr double L = 1.0;

    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L);
    // model1->P(L, 0.0, -1);
    mc::ode::DSArgs args1 = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model)}
    };
    mc::Ref<mc::ode::DynamicalSystem> system = mc::ode::GetRadonsSystem(dt, args1);
    system->SetResetFn([](mc::ode::DSArgs &args, mc::ode::DSArgs &, uint32_t)
    {
        args.at("model").toPreisachModel()->ResetState();
    });

    // u0 -> [z_{-}, z_{+}]
    std::vector<std::pair<Eigen::Vector2d, std::pair<std::vector<Eigen::Vector2d>, std::vector<Eigen::Vector2d>>>>
        params = {
            {
                {0.6, 1.6},
                {
                    {{-1.8, -1.6}, {-1.8, -1.8}}, // z_{-}
                    {{-0.4, 1.6}, {-0.4, 1.6}} // z_{+}
                }
            },
            {
                {0.6, 1.4},
                {
                    {{-1.8, -1.6}, {-1.8, -1.4}}, // z_{-}
                    {{-0.2, 1.4}, {1.4, 1.6}, {1.4, 1.4}, {0.0, 1.4}, {-0.2, 1.2}, {-0.2, 1.6}} // z_{+}
                }
            }
        };

    Eigen::Matrix2d M(2, 2);
    M << 0.0, 1.0,
        -w0, -gamma;
    Eigen::Vector2d b = {0.0, A};

    const Eigen::Vector2d u0 = {0.6, 1.4};

    auto p = mc::ShuttlePoint(system, mc::SolidCone2d::fromCurve(M, b, mc::detail::Settings()),
                              u0, {-1.8, -1.4}, {0.0, 1.2}, mc::consts::twoPi / w);
    
    std::println("LIMITS: z_odd: {}, z_even: {}", p.limits[0], p.limits[1]);

    mc::json::JsonDocument message({"name", "shuttle_logs", "results"});
    message.AddField("name", "ShuttlePoint");
    message.AddField("shuttle_logs", p.doc);
    
    system->ResetTo(p.limits[0]);
    auto traj = system->Forward(500.0);
    Eigen::VectorXd x = traj.col(0);
    Eigen::VectorXd v = traj.col(1);
    auto [in, out] = model->HysteresisLoop();
    message.AddSubField({"results", "limit1", "x"}, x);
    message.AddSubField({"results", "limit1", "v"}, v);
    message.AddSubField({"results", "limit1", "loop", "in"}, in);
    message.AddSubField({"results", "limit1", "loop", "out"}, out);
    
    system->ResetTo(p.limits[1]);
    traj = system->Forward(500.0);
    x = traj.col(0);
    v = traj.col(1);
    std::tie(in, out) = model->HysteresisLoop();
    message.AddSubField({"results", "limit2", "x"}, x);
    message.AddSubField({"results", "limit2", "v"}, v);
    message.AddSubField({"results", "limit1", "loop", "in"}, in);
    message.AddSubField({"results", "limit2", "loop", "out"}, out);
    
    mc::Ref file = mc::Ref<FileWriter>::Create("ShuttlePointLog.json");
    file->Write(message.ToString());
}

void ShiftTest()
{
    double dt = 0.01;
    double gamma = 4.;
    double w0 = 3.;
    double w = 1.0;
    double A = 1.5;
    double E = 1.35;

    constexpr double L = 1.0;
    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L);
    model->P(L, 0.0, -1);

    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model)}
    };

    // Eigen::Matrix2d Am(2, 2);
    // Am << 0.0, 1.0,
    //     -w0, -gamma;
    // Eigen::Vector2d b(2);
    // b << 0.0, A;
    // mc::SolidCone2d::fromCurve(Am, b, mc::detail::Settings());

    Eigen::Vector2d x0 = {-1.5, -1.1};
    mc::Ref<mc::ode::DynamicalSystem> system = mc::ode::GetRadonsSystem(dt, args, {}, x0);
    system->SetResetFn([](mc::ode::DSArgs &args, mc::ode::DSArgs &, uint32_t)
    {
        auto model = args.at("model").toPreisachModel();
        model->ResetState();
        args.insert_or_assign("model", mc::ode::Vote(model));
    });
    double period = 1.0 * mc::consts::twoPi / w;

    const auto traj1 = system->Forward(2. * period);
    Eigen::VectorXd x1 = traj1.col(0);
    Eigen::VectorXd v1 = traj1.col(1);

    x0 = {1.6, 0.9};
    system->ResetTo(x0);
    const auto traj2 = system->Forward(10.0 * period);
    Eigen::VectorXd x2 = traj2.col(0);
    Eigen::VectorXd v2 = traj2.col(1);

    // const Eigen::Vector2d u0 = {0.5, 0.5};
    //
    // const auto traj = system->ShiftTraj(u0, mc::consts::twoPi / w);
    // Eigen::VectorXd x = traj.row(0);
    // Eigen::VectorXd v = traj.row(1);

    mc::json::JsonDocument message({"name", "z+", "z-"});
    message.AddField("name", "ShuttlePoint");
    message.AddSubField({"z-", "x"}, x1);
    message.AddSubField({"z-", "v"}, v1);
    message.AddSubField({"z+", "x"}, x2);
    message.AddSubField({"z+", "v"}, v2);

    mc::Ref file = mc::Ref<FileWriter>::Create("ShiftTest.json");
    file->Write(message.ToString());
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
    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L);

    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model)},
        {"eps", 0.05}
    };
    auto system = mc::ode::GetRadonsSystem(dt, args, {}, {0.0, 0.0}, false);
    auto As = mc::ode::utils::VoteRange(0.0, 4., 0.005);
    auto Es = mc::ode::utils::VoteRange(0., 3., 0.005);
    auto ws = mc::ode::utils::VoteRange(0., 4., 0.005);
    auto epss = mc::ode::utils::VoteRange(0., 3.0, 0.005);

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
    const std::pair<std::string, std::vector<mc::ode::Vote>> param = {"A", As};
    for (const auto &val : param.second)
    {
        std::cout << "New" << std::endl;
        system->Reset();
        system->SetArg(param.first, val);

        const auto traj = system->Forward(time);
        const auto x = traj.col(0);

        const auto res = mc::ode::ZeroOneTest_Fast_New(x, 10);
        k_values.push_back(res);
    }

    mc::json::JsonDocument message({"name", param.first, "Ks"});
    message.AddField("name", "ZeroOneTest");
    mc::ode::WriteVotesToDoc(message, param.first, param.second, param.second[0].Type());
    message.AddField("Ks", k_values);

    mc::Ref file = mc::Ref<FileWriter>::Create("ZeroOneTest.json");
    file->Write(message.ToString());
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
    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L);

    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", gamma},
        {"A", A},
        {"w0", w0},
        {"w", w},
        {"E", E},
        {"model", mc::ode::Vote(model)},
        {"eps", 0.05}
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
    // mc::Ref file = mc::Ref<FileWriter>::Create("ZeroOneTestTraj.json");
    // file->Write(message.ToString());

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
    const auto As = mc::ode::utils::VoteRange(0., 4., 0.01);
    const auto Es = mc::ode::utils::VoteRange(0., 3., 0.01);
    const auto ws = mc::ode::utils::VoteRange(0., 2.5, 0.01);
    const auto ds = [](const mc::ode::DSArgs &args)
    {
        return mc::ode::utils::VoteRange(0., args.at("A").toDouble() / args.at("w").toDouble() + 1.5, 0.01);
    };

    constexpr uint32_t M = 500;
    constexpr double e = 0.5;
    Eigen::VectorXd Ts = Eigen::arange(1.0, 5.0, 0.25);
    Eigen::Vector2d v = {1.0, 0.0};
    constexpr double areaCoeff = -0.9;

    Eigen::VectorXd freqs = Eigen::arange(0.0, 2.5, 0.05);

    auto model1 = mc::Ref<mc::ArealPreisachModel>::Create(L, true, false);
    auto model2 = mc::Ref<mc::DoubleArealPreisachModel>::Create(L, 1.0, 1.0);
    auto model3 = mc::Ref<mc::DoubleArealPreisachModel>::Create(L, 0.7, 1.0);

    auto next_model1 = mc::Ref<mc::ArealPreisachModel>::Create(L, true, false);
    auto next_model2 = mc::Ref<mc::DoubleArealPreisachModel>::Create(L, 1.0, 1.0, true);
    auto next_model3 = mc::Ref<mc::DoubleArealPreisachModel>::Create(L, 0.7, 1.0, true);

    const std::unordered_map<std::string, std::array<mc::Ref<mc::PreisachModelBase>, 2>> models = {
        {"single_loop", {model1, next_model1}},
        // {"double_loop_1.0", {model2, next_model2}},
        // {"double_loop_0.7", {model3, next_model3}},
    };

    const auto resetFn = [](mc::ode::DSArgs &args, mc::ode::DSArgs &, uint32_t)
    {
        auto &model = args.at("model").toPreisachModel();
        model->ResetState();
    };

    mc::Ref system = mc::ode::GetRadonsSystem(dt, {}, {}, {0.0, 0.0}, false);

    mc::json::JsonDocument message({"name", "params", "results"});

    for (const auto &[model_name, models] : models)
    {
        auto &[model, next_model] = models;

        mc::ode::DSArgs first_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 1.5},
            {"w0", 1.0},
            {"w", 1.0},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs second_set = {
            {"dt", dt},
            {"gamma", 0.5},
            {"A", 1.5},
            {"w0", 1.0},
            {"w", 1.0},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs third_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 1.5},
            {"w0", 1.0},
            {"w", 1.0},
            {"E", 1.6},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs fourth_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 1.5},
            {"w0", 1.0},
            {"w", 1.0},
            {"E", 1.5},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs fifth_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 1.0},
            {"w0", 1.0},
            {"w", 1.0},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs sixth_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 0.5},
            {"w0", 1.0},
            {"w", 1.0},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs seventh_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 0.5},
            {"w0", 0.5},
            {"w", 1.0},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs eighth_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 0.5},
            {"w0", 1.0},
            {"w", 0.5},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs ninth_set = {
            {"dt", dt},
            {"gamma", 0.0},
            {"A", 0.5},
            {"w0", 1.0},
            {"w", 1.0},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs tenth_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 0.5},
            {"w0", 1.3},
            {"w", 1.0},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };
        mc::ode::DSArgs eleventh_set = {
            {"dt", dt},
            {"gamma", 0.1},
            {"A", 0.5},
            {"w0", 1.0},
            {"w", 1.3},
            {"E", 1.35},
            {"model", mc::ode::Vote(model)}
        };

        const std::unordered_map<std::string, mc::ode::DSArgs &> params_sets = {
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
                system->SetResetFn(resetFn);
                system->SetArgs(args, true);

                mc::ode::DSArgs nextArgs = args;
                nextArgs.at("model") = mc::ode::Vote(next_model);
                system->SetNextArgs(nextArgs, true);

                system->ResetTo(x0);

                message.AddField("name", "JustSolveRodos");
                message.AddSubField({"params", "dt"}, dt);
                message.AddSubField({"params", "time"}, solve_time);
                message.AddSubField({"params", "model"}, model_name);
                message.AddSubField({"params", "args_set"}, args_set);
                auto args_doc = dsArgsToJsonDoc(args);
                message.AddSubField({"params", "args"}, args_doc);
                message.AddSubField({"params", "x0"}, DoubleVectorToString(x0, 1));

                const auto full_name = std::format("{}/{}/{}", model_name, args_set, DoubleVectorToString(x0, 1));

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
                // const auto biff_A = mc::ode::BifurcationDiagram(system, {"A", As}, researchTime, timeForward);
                // message.AddSubField({"results", "biff", "A"}, biff_A);
                //
                // const auto biff_E = mc::ode::BifurcationDiagram(system, {"E", Es}, researchTime, timeForward);
                // message.AddSubField({"results", "biff", "E"}, biff_E);
                //
                // const auto biff_w = mc::ode::BifurcationDiagram(system, {"w", ws}, researchTime, timeForward);
                // message.AddSubField({"results", "biff", "w"}, biff_w);
                //
                // // const auto [mLCE, mLCE_history] = mc::ode::Benettin_mLCE(system, timeForward, e, T, M);
                // const auto [LCEs, LCEs_history] = mc::ode::ComputeLCEs(system, timeForward, solve_time, M, 2);
                // message.AddSubField({"results", "LCEs", "LCEs"}, LCEs);
                // message.AddSubField({"results", "LCEs", "history"}, LCEs_history);

                // const auto div_doc = mc::ode::DivergenceDegreeRegressionData(system, timeForward, e, v, Ts, M);
                // message.AddSubField({"results", "mLCE", "div_reg_data"}, div_doc);

                // auto power_law = mc::ode::power_law::ComputePowerLawExponent(system, v, e, Ts, M);
                auto power_law = mc::ode::power_law::ComputePowerLawExponent(system, v, e, 0.1);

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
                //             mc::ode::power_law::ComputePowerLawExponent(system, 0.0, v1, e1, Ts, M, areaCoeff1);
                //             // const mc::ode::power_law::regression::ContinuousPowerLawResult power_law = mc::ode::power_law::regression::ComputeContinuousPowerLawExponent(
                //             //     system, v, timeForward, 100.0 * mc::consts::twoPi / args.at("w").toDouble(), e1, areaCoeff1);
                //
                //             // const mc::ode::power_law::window::AdaptivePowerLawResult power_law =
                //             //     mc::ode::power_law::window::ComputeContinuousPowerLawExponent(
                //             //         system, v, timeForward, 100.0 * mc::consts::twoPi / args.at("w").toDouble(), e1,
                //             //         areaCoeff1, static_cast<uint32_t>(mc::consts::twoPi / args.at("w").toDouble() / dt));
                //
                //             // std::println("v: {}, e: {}, ac: {};  avg_nu: {}, num_segments: {}",
                //             //              Eigen::DoubleVectorToString(v1, 3),
                //             //              mc::doubleToString(e1, 2), mc::doubleToString(areaCoeff1, 2),
                //             //              power_law.average_nu, power_law.segments.size());
                //         }
                //     }
                // }
                message.AddSubField({"results", "LCEs", "power_law"}, power_law);
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
                // const auto afc = mc::ode::AFC(system, afc_time, freqs);
                // message.AddSubField({"results", "AFC"}, afc);
                //
                // // Do this in the end, cause algorithm changes model d param
                // if (model_name == "double_loop_1.0")
                // {
                //     auto biff_d = DoubleLoopBiffurcationDiagram(system, ds(args), researchTime, timeForward);
                //     message.AddSubField({"results", "biff", "d"}, biff_d);
                // }

                std::cout << "write " << full_name << "\n";

                mc::Ref file = mc::Ref<FileWriter>::Create(
                    // std::format("../python/article_plots/{}/ComplexRadonsResearch.json", full_name));
                    std::format("../python/pl/PowerLaw_0.5_50.0.json", full_name));
                file->Write(message.ToString());
                message.ClearDoc();

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
    auto model = mc::Ref<mc::ArealPreisachModel>::Create(1.0, true, false);
    auto next_model = mc::Ref<mc::ArealPreisachModel>::Create(1.0, true, false);
    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", 0.1},
        {"A", 1.5},
        {"w0", 1.0},
        {"w", 1.0},
        {"E", 1.35},
        {"model", mc::ode::Vote(model)}
    };
    mc::ode::DSArgs nextArgs = args;
    nextArgs.at("model") = mc::ode::Vote(next_model);

    const Eigen::Vector2d v = {1., 0.};
    const double eps = 0.1;

    const Eigen::Vector2d x0 = {1., 0.};
    const auto pert_x0 = x0 + eps * v;

    mc::Ref system1 = mc::ode::GetRadonsSystem(dt, args, nextArgs, x0);
    mc::Ref system2 = mc::ode::GetRadonsSystem(dt, args, {}, pert_x0);

    auto traj1 = system1->ForwardTwoTrajs(time, pert_x0);
    model->ResetState();
    auto traj2 = system2->Forward(time);

    mc::json::JsonDocument message({"name", "time", "traj1", "traj1_next", "traj2"});
    message.AddField("name", "TrajsCircle");
    message.AddField("time", time);

    message.AddField("traj1", traj1.first);
    message.AddField("traj1_next", traj1.second);
    message.AddField("traj2", traj2);

    mc::Ref file = mc::Ref<FileWriter>::Create("TwoTrajs.json");
    file->Write(message.ToString());
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


    auto model = mc::Ref<mc::ArealPreisachModel>::Create(1.0, true, false);
    auto next_model = mc::Ref<mc::ArealPreisachModel>::Create(1.0, true, false);
    mc::ode::DSArgs args = {
        {"dt", dt},
        {"gamma", 0.1},
        {"A", 1.5},
        {"w0", 1.0},
        {"w", 1.0},
        {"E", 1.35},
        {"model", mc::ode::Vote(model)}
    };
    mc::ode::DSArgs nextArgs = args;
    nextArgs.at("model") = mc::ode::Vote(next_model);
    const Eigen::Vector2d v = {1., 0.};

    mc::Ref system = mc::ode::GetRadonsSystem(dt, args, nextArgs, {1., 0.});

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
        return Eigen::DoubleVectorToString(elem, 4);
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

    mc::json::JsonDocument message({"name", "time", "times", "coords", "x0s", "trajs"});
    message.AddField("name", "TrajsCircle");
    message.AddField("time", time);

    // message.AddField("x0s", points);
    // message.AddField("times", times);
    message.AddField("coords", coordsStr);
    message.AddField("trajs", trajs);

    mc::Ref file = mc::Ref<FileWriter>::Create("TwoTrajsOnPhaseTrajectory.json");
    file->Write(message.ToString());
}

int main()
{
    // TwoTrajsOnPhaseTrajectory();
    // ComplexRadonsResearch();
    // JustSolveRodos();

    // ZeroOneTest();

    // ZeroOneRadonsTest();

    // BifurcationDiagram();

    // DoublePreisachModelTest();

    // mc::run_example_2d();

    // FindShuttlePointInitialValues();
    RadonsShuttlePoint();
    // ShiftTest();


    // CourseWorkModelsDiff();

    // Initial duration for profiler
    // Sleep(1000);
    // SinusTests();

    // DiscreteModelChaosTestByStepSize();

    // g_Server.Connect();
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
    // TestEverettFunction();

    // PoincareMapping();
    // BifurcationDiagram();

    // auto g = [](std::vector<double> const & x)
    // {
    //     constexpr const double A = 1.0 / (mc::consts::pi * mc::consts::pi * mc::consts::pi);
    //     return A / (1.0 - cos(x[0])*cos(x[1])*cos(x[2]));
    // };
    // std::vector<std::pair<double, double>> bounds{{0, mc::consts::pi}, {0, mc::consts::pi}, {0, mc::consts::pi}};
    // double error_goal = 0.001;
    // mc::integrate::naive_monte_carlo<double, decltype(g)> mc(g, bounds, error_goal);
    //
    // std::future<double> task = mc.integrate();
    // // while (task.wait_for(std::chrono::seconds(1)) != std::future_status::ready)
    // //
}
