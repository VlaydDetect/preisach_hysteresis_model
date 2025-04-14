// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include <print>
#include <map>
#include <random>
#include <chrono>

// #include "Model/DiscreteModel.hpp"
#include "FileWriter/FileWriter.hpp"
#include "Server/Server.hpp"
#include "Utils/Math.hpp"
#include "Utils/Array.hpp"

#include "MUTCpp/ODE.hpp"
#include "MUTCpp.hpp"

#include "Debug/Profile.hpp"

using json = nlohmann::json;

static Net::Server g_Server(8888, "127.0.0.1");

// void OperatorDerivativeTest()
// {
//     double h = 0.01;
//     double dt = 0.5;
//
//     std::vector<uint32_t> t0 = {0, 40, 72, 96, 115, 142};
//     std::vector u0 = {-1.0, 0.9, -0.8, 0.5, -0.3, 1.0};
//     auto [t, u] = ContinueFunction(dt, t0, u0);
//
//     // auto *model = new mc::DiscretePreisachModel(1, h, true);
//     auto *model = new mc::ArealPreisachModel(1.0, true);
//
//     for (int i = 0; i < t.size(); i++)
//     {
//         model->P(u[i], i);
//     }
//
//     auto [inputs1, outputs] = model->HysteresisLoop();
//     auto [inputs2, derivatives] = model->DerivativeHistory();
//     auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();
//
//     mc::json::JsonDocument message({"name", "method", "dt", "h", "times", "loop", "anim", "results"});
//     message.AddField("name", "OperatorDerivativeTest");
//     message.AddField("method", "plot");
//     message.AddField("dt", dt);
//     message.AddField("h", h);
//     message.AddField("times", t);
//     message.AddSubField({"results", "x"}, u);
//     message.AddSubField({"results", "derivatives"}, derivatives);
//     message.AddSubField({"loop", "inputs"}, u);
//     message.AddSubField({"loop", "outputs"}, outputs);
//     message.AddSubField({"anim", "in"}, inHist);
//     message.AddSubField({"anim", "x"}, xHist);
//     message.AddSubField({"anim", "y"}, yHist);
//     message.AddSubField({"anim", "out"}, outHist);
//     message.AddSubField({"anim", "save"}, true);
//
//     // g_Server.SendDataMessage(message);
//     auto *file = new FileWriter("ArealModelTest.json");
//     file->Write(message.ToString());
// }
//
// void LyapunovExponentsTest()
// {
//     mc::Matrix x0 = {1.5, -1.5, 20.0};
//     double t0 = 0.0;
//     double dt = 0.01;
//     double time = 100.0;
//     int steps = static_cast<int>(time / dt);
//
//     double sigma = 10.0;
//     double rho = 28.0;
//     double beta = 8.0 / 3.0;
//
//     bool usePreisach = true;
//     double L = 1.0;
//     double h = 0.05;
//     double E = 1.35;
//     auto *model = new mc::DiscretePreisachModel(L, h, true);
//
//     auto func = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         auto res = mc::zeros_like<double>(x);
//         res[0] = sigma * (x[1] - x[0]) + (usePreisach ? E * model->P(x[0], static_cast<int>(t / dt)) : 0.0);
//         res[1] = x[0] * (rho - x[2]) - x[1];
//         res[2] = x[0] * x[1] - beta * x[2];
//         return res;
//     };
//
//     auto jac = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         auto res = mc::zeros<double>(x.shape().m_Cols);
//         res(0, 0) = -sigma + (usePreisach ? E * model->DerivativeOperator(t, dt) : 0.0);
//         res(0, 1) = sigma;
//         res(1, 0) = rho - x[2];
//         res(1, 1) = -1.0;
//         res(1, 2) = -x[0];
//         res(2, 0) = x[1];
//         res(2, 1) = x[0];
//         res(2, 2) = -beta;
//         return res;
//     };
//
//     auto *lorenz = new mc::ode::ContinuousDS(func, jac, dt, {}, {}, x0, t0);
//     lorenz->Forward(steps);
//
//     // auto [LCEs, history_LCE] = LCE(lorenz, 3, 0, steps);
//     // printf("LCE-1 = %lf\n", LCEs[0]);
//     // printf("LCE-2 = %lf\n", LCEs[1]);
//     // printf("LCE-3 = %lf\n", LCEs[2]);
//
//     // auto [max1, history_mLCE1] = mLCE(lorenz, 0, steps);
//     // std::println("mLCE = {:f}", max1);
//     // mc::json::JsonDocument message1({"name", "method", "x"});
//     // message1.AddField("name", "2e2");
//     // message1.AddField("method", "simplePlot");
//     // message1.AddField("x", history_mLCE1.toStlVector());
//     // g_Server.SendDataMessage(message1);
//
//     auto [max2, history_mLCE2] = mLCE_Kuznetsov(lorenz, 1, 150.0, 150, true);
//     std::println("mLCE (by Kuznetsov) = {:f}", max2);
//     // mc::json::JsonDocument message2({"name", "method", "x"});
//     // message2.AddField("name", "2e2");
//     // message2.AddField("method", "simplePlot");
//     // message2.AddField("x", history_mLCE2.toStlVector());
//     // g_Server.SendDataMessage(message2);
// }
//
// void PLot_mLCE_HysteronsStep_E()
// {
//     mc::Matrix x0 = {0.0, 0.0, 0.0};
//     double t0 = 0.0;
//     double dt = 0.01;
//     double time = 800.0;
//     int steps = static_cast<int>(time / dt);
//
//     double gamma = 0.1;
//     double w02 = 1.0;
//     double w = 1.0;
//     double A = 0.5;
//
//     bool usePreisach = true;
//     double L = 1.0;
//     // std::vector hSteps = GenerateSequenceByCount(0.1, 2, 5);
//     // std::vector EVals = GenerateSequenceByDelta(2.2, 2.5, 0.1);
//     std::vector EVals = {2.4};
//
//     std::vector hSteps = {0.0125};
//
//     std::vector<double> mLCEs = {};
//
//     for (const auto &h : hSteps)
//     {
//         auto *model = new mc::DiscretePreisachModel(L, h, true);
//         for (const auto &E : EVals)
//         {
//             auto func = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//             {
//                 auto res = mc::zeros_like<double>(x);
//
//                 res[0] = x[1];
//                 res[1] = A * mc::sin(x[2]) - gamma * x[1] - w02 * x[0] + (usePreisach
//                     ? E * model->P(x[0], static_cast<int>(t / dt))
//                     : 0.0);
//                 res[2] = w;
//
//                 return res;
//             };
//
//             auto jac = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//             {
//                 auto res = mc::zeros<double>(x.shape().m_Cols);
//
//                 res(0, 1) = 1.0;
//
//                 res(1, 0) = -w02 + (usePreisach ? E * model->DerivativeOperator(t, dt) : 0.0);
//                 res(1, 1) = -gamma;
//                 res(1, 2) = A * mc::cos(x[2]);
//
//                 return res;
//             };
//
//             auto *rodos = new mc::ode::ContinuousDS(func, jac, dt, {}, {}, x0, t0);
//             rodos->Forward(steps);
//
//             auto [maxLCE, history_mLCE] = mc::ode::mLCE(rodos, 0, steps);
//             // auto [LCEs, history_LCE] = LCE(rodos, 3, 0, steps);
//             // double maxLCE = LCEs[0];
//
//             mc::json::JsonDocument message({"name", "method", "mLCE", "E", "h"});
//             message.AddField("name", "Oscillator mLCE(h, E)");
//             message.AddField("method", "3d");
//             message.AddField("mLCE", maxLCE);
//             message.AddField("E", E);
//             message.AddField("h", h);
//
//             const std::string name = std::format("tests/LCEs/mLCE_E={0:.3}_h={1:}.txt", E, h);
//             auto *file = new FileWriter(name);
//             file->Write(message.ToString());
//
//             printf("mLCE(h=%lf, E=%lf) = %lf\n", h, E, maxLCE);
//             mLCEs.emplace_back(maxLCE);
//         }
//     }
//
//     mc::json::JsonDocument message({"name", "method", "mLCEs", "E_vals", "h_steps"});
//     message.AddField("name", "Oscillator mLCE(h, E)");
//     message.AddField("method", "3d");
//     message.AddField("mLCEs", mLCEs);
//     message.AddField("E_vals", EVals);
//     message.AddField("h_steps", hSteps);
//
//     auto *file = new FileWriter("LCEs.txt");
//     file->Write(message.ToString());
//
//     // g_Server.SendDataMessage(message);
// }
//
// // void SearchForDegreeOfNonLinearizedTrajsDivergence()
// // {
// //     mc::Matrix x0 = {0.2, 0.2, 0.0};
// //     double t0 = 0.0;
// //     double dt = 0.01;
// //     double time = 800.0;
// //     int steps = static_cast<int>(time / dt);
// //
// //     double gamma = 0.1;
// //     double w02 = 1.0;
// //     double w = 1.0;
// //     double A = 0.5;
// //
// //     bool usePreisach = true;
// //     double L = 1.0;
// //     double h = 0.05;
// //     double E = 1.35;
// //     auto *model = new mc::DiscretePreisachModel(L, h, true);
// //
// //     auto func = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
// //     {
// //         auto res = mc::zeros_like<double>(x);
// //
// //         res[0] = x[1];
// //         res[1] = A * mc::sin(x[2]) - gamma * x[1] - w02 * x[0] + (usePreisach
// //             ? E * model->P(x[0], static_cast<int>(t / dt))
// //             : 0.0);
// //         res[2] = w;
// //
// //         return res;
// //     };
// //
// //     auto jac = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
// //     {
// //         auto res = mc::zeros<double>(x.shape().m_Cols);
// //
// //         res(0, 1) = 1.0;
// //
// //         res(1, 0) = -w02 + (usePreisach ? E * model->DerivativeOperator(t, dt) : 0.0);
// //         res(1, 1) = -gamma;
// //         res(1, 2) = A * mc::cos(x[2]);
// //
// //         return res;
// //     };
// //
// //     auto *rodos = new mc::ode::ContinuousDS(func, jac, dt, {}, x0, t0);
// //     // rodos->Forward(steps);
// //
// //     mc::Matrix<double> traj1 = {};
// //     mc::Matrix<double> traj2 = {};
// //     auto div = mLCE_DivergenceDegree(rodos, 0.1, traj1, traj2, 1.0, 90, false, 3, false, true);
// //     for (const auto &[divergence, degree] : div)
// //     {
// //         std::println(std::cout, "Degree: {}, Divergence: {:f}", degree, divergence);
// //     }
// //
// //     mc::json::JsonDocument message({"name", "method", "traj1", "traj2"});
// //     message.AddField("name", "Divergence Degree");
// //     message.AddField("method", "plot");
// //     message.AddField("traj1", traj1);
// //     message.AddField("traj2", traj2);
// //
// //     auto *file = new FileWriter("LCEs.txt");
// //     file->Write(message.ToString());
// //
// //     // g_Server.SendDataMessage(message);
// //
// //     // auto [max2, history_mLCE2] = mLCE_Kuznetsov(rodos, 0.1, 6, 80, true);
// // }
//
// void twoTrajs()
// {
//     mc::Matrix x01 = {-0.2, -0.2, 0.0};
//     mc::Matrix x02 = {-0.23, -0.2, 0.0};
//     double t0 = 0.0;
//     double dt = 0.01;
//     double time = 100.0;
//     int steps = static_cast<int>(time / dt);
//
//     double gamma = 0.0;
//     double w02 = 1.0;
//     double w = 1.0;
//     double A = 0.5;
//
//     bool usePreisach = true;
//     double L = 1.0;
//     double h = 0.05;
//     double E = 1.35;
//
//
//     auto *model1 = new mc::DiscretePreisachModel(L, h, true);
//
//     auto func1 = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         auto res = mc::zeros_like<double>(x);
//
//         res[0] = x[1];
//         res[1] = A * mc::sin(x[2]) - gamma * x[1] - w02 * x[0] + (usePreisach
//             ? E * model1->P(x[0], static_cast<int>(t / dt))
//             : 0.0);
//         res[2] = w;
//
//         return res;
//     };
//
//     auto jac1 = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         auto res = mc::zeros<double>(x.shape().m_Cols);
//
//         res(0, 1) = 1.0;
//
//         res(1, 0) = -w02 + (usePreisach ? E * model1->DerivativeOperator(t, dt) : 0.0);
//         res(1, 1) = -gamma;
//         res(1, 2) = A * mc::cos(x[2]);
//
//         return res;
//     };
//
//
//     auto *model2 = new mc::DiscretePreisachModel(L, h, true);
//
//     auto func2 = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         auto res = mc::zeros_like<double>(x);
//
//         res[0] = x[1];
//         res[1] = A * mc::sin(x[2]) - gamma * x[1] - w02 * x[0] + (usePreisach
//             ? E * model2->P(x[0], static_cast<int>(t / dt))
//             : 0.0);
//         res[2] = w;
//
//         return res;
//     };
//
//     auto jac2 = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         auto res = mc::zeros<double>(x.shape().m_Cols);
//
//         res(0, 1) = 1.0;
//
//         res(1, 0) = -w02 + (usePreisach ? E * model2->DerivativeOperator(t, dt) : 0.0);
//         res(1, 1) = -gamma;
//         res(1, 2) = A * mc::cos(x[2]);
//
//         return res;
//     };
//
//     mc::Matrix<double> traj1 = {};
//     mc::Matrix<double> traj2 = {};
//
//     auto *rodos1 = new mc::ode::ContinuousDS(func1, jac1, dt, {}, {}, x01, t0);
//     auto *rodos2 = new mc::ode::ContinuousDS(func2, jac2, dt, {}, {}, x02, t0);
//
//     for (int i = 0; i < steps; i++)
//     {
//         auto traj1_vec = rodos1->Forward(1);
//         auto traj2_vec = rodos2->Forward(1);
//
//         traj1.append(traj1_vec(-1, traj1_vec.cSlice()), mc::Axis::ROW);
//         traj2.append(traj2_vec(-1, traj2_vec.cSlice()), mc::Axis::ROW);
//     }
//
//     auto loop1 = model1->HysteresisLoop();
//     auto loop2 = model2->HysteresisLoop();
//
//     mc::json::JsonDocument message({"name", "method", "traj1", "traj2", "loop1", "loop2"});
//     message.AddField("name", "Divergence Degree");
//     message.AddField("method", "plot");
//     message.AddField("traj1", traj1);
//     message.AddField("traj2", traj2);
//     message.AddSubField({"loop1", "in"}, loop1[0]);
//     message.AddSubField({"loop1", "out"}, loop1[1]);
//     message.AddSubField({"loop2", "in"}, loop2[0]);
//     message.AddSubField({"loop2", "out"}, loop2[1]);
//
//     auto *file = new FileWriter("trajs.txt");
//     file->Write(message.ToString());
// }
//
// void AFC()
// {
//     double t0 = 0.0;
//     double dt = 0.01;
//     double time = 100.0;
//     int steps = static_cast<int>(time / dt);
//
//     double gamma = 0.1;
//     double beta = 0.04;
//     double w0 = 1.0;
//     double A = 1.0;
//
//     bool usePreisach = false;
//     double L = 1.0;
//     double h = 0.05;
//     double E = 1.35;
//
//     mc::Matrix x0 = {0.0, 0.0, 0.0};
//
//     // auto *model = new mc::DiscretePreisachModel(L, h, true);
//     auto *model = new mc::ArealPreisachModel(L, true);
//
//     auto freqLoop = [t0, dt, gamma, beta, w0, A, steps, usePreisach, E, model](
//         const std::vector<double> &freqs, std::vector<mc::Matrix<double>> &trajs, mc::Matrix<double> &x0)
//     {
//         for (auto w : freqs)
//         {
//             auto func = [&](const mc::Matrix<double> &x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//             {
//                 auto res = mc::zeros_like<double>(x);
//
//                 res[0] = x[1];
//                 res[1] = A * mc::cos(w * t) - gamma * x[1] - w0 * x[0] + (usePreisach
//                     ? E * model->P(x[0], static_cast<int>(t / dt))
//                     : -beta * mc::power(x[0], 3));
//                 res[2] = w0;
//
//                 return res;
//             };
//
//             auto *rodos = new mc::ode::ContinuousDS(func, {}, dt, {}, {}, x0, t0);
//
//             auto traj = rodos->Forward(steps);
//
//             trajs.push_back(traj);
//             x0 = {traj(-1, 0), traj(-1, 1), x0[2]};
//         }
//     };
//
//     std::vector<double> freqs_inc = GenerateSequenceByDelta(0.0, 2.5, 0.05);
//     std::vector<double> freqs_dec(freqs_inc.size());
//     std::ranges::transform(freqs_inc.begin(), freqs_inc.end(), freqs_dec.rbegin(), [](auto val) { return val; });
//
//     std::vector<mc::Matrix<double>> trajs_inc = {};
//     std::vector<mc::Matrix<double>> trajs_dec = {};
//
//     freqLoop(freqs_inc, trajs_inc, x0);
//     mc::print(x0);
//     freqLoop(freqs_dec, trajs_dec, x0);
//
//     mc::json::JsonDocument message({"name", "method", "freqs", "trajs_inc", "trajs_dec"});
//     message.AddField("name", "AFC");
//     message.AddField("method", "plot");
//     message.AddField("freqs", freqs_inc);
//     message.AddField("trajs_inc", trajs_inc);
//     message.AddField("trajs_dec", trajs_dec);
//
//     auto *file = new FileWriter("AFC.txt");
//     file->Write(message.ToString());
// }
//
// void RodosAFCs()
// {
//     auto func = [](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         AL_PROFILE_FUNC("Rodos::func");
//         double dt = args.at("dt").toDouble();
//         double gamma = args.at("gamma").toDouble();
//         double A = args.at("A").toDouble();
//         double beta = args.at("beta").toDouble();
//         double w0 = args.at("w0").toDouble();
//         double w = args.at("w").toDouble();
//         double E = args.at("E").toDouble();
//         double usePreisach = args.at("usePreisach").toBool();
//         const auto& model = args.at("model").toPreisachModel();
//
//         auto res = mc::zeros_like<double>(x);
//
//         res[0] = x[1];
//         res[1] = A * mc::cos(w * t) - gamma * x[1] - w0 * x[0] + (usePreisach
//             ? E * model->P(x[0], static_cast<int>(t / dt))
//             : -beta * mc::power(x[0], 3));
//         res[2] = w0;
//
//         return res;
//     };
//
//     double dt = 0.01;
//     double time = 100.0;
//
//     double gamma = 0.1;
//     double beta = 0.04;
//     double w0 = 1.0;
//     double A = 0.5;
//
//     bool usePreisach = true;
//     double L = 1.0;
//     double h = 0.05;
//     double E = 1.35;
//
//     auto As = mc::byDelta(0.5, 3.0, 0.5);
//
//     // auto *model = new mc::DiscretePreisachModel(L, h, true);
//     auto *model = new mc::ArealPreisachModel(L);
//
//     for (int i = 0; i < As.size(); i++)
//     {
//         mc::ode::DSArgs args = {
//             {"dt", dt},
//             {"gamma", gamma},
//             {"A", As[i]},
//             {"beta", beta},
//             {"w0", w0},
//             {"E", E},
//             {"usePreisach", usePreisach},
//             {"model", model}
//         };
//
//         auto *system = new mc::ode::ContinuousDS(func, {}, dt, args);
//
//         auto doc = mc::ode::AFC(system, time, 2.5, 0.05);
//         auto *file = new FileWriter(std::format("tests/AFCs/AFC{}.txt", i));
//         file->Write(doc.ToString());
//     }
// }
//
// void RodosLCEs()
// {
//     auto func = [](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         AL_PROFILE_FUNC("Rodos::func");
//         double dt = args.at("dt").toDouble();
//         double gamma = args.at("gamma").toDouble();
//         double A = args.at("A").toDouble();
//         double beta = args.at("beta").toDouble();
//         double w0 = args.at("w0").toDouble();
//         double w = args.at("w").toDouble();
//         double E = args.at("E").toDouble();
//         double usePreisach = args.at("usePreisach").toBool();
//         const auto& model = args.at("model").toPreisachModel();
//
//         auto res = mc::zeros_like<double>(x);
//
//         res[0] = x[1];
//         res[1] = A * mc::cos(w * t) - gamma * x[1] - w0 * x[0] + (usePreisach
//             ? E * model->P(x[0], static_cast<int>(t / dt))
//             : -beta * mc::power(x[0], 3));
//         res[2] = w0;
//
//         return res;
//     };
//
//     auto jac = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         double w0 = args.at("w0").toDouble();
//         double gamma = args.at("gamma").toDouble();
//         double usePreisach = args.at("usePreisach").toBool();
//         double E = args.at("E").toDouble();
//         const auto& model = args.at("model").toPreisachModel();
//         double dt = args.at("dt").toDouble();
//         double A = args.at("A").toDouble();
//
//         auto res = mc::zeros<double>(x.shape().m_Cols);
//
//         res(0, 1) = 1.0;
//
//         res(1, 0) = -w0 + (usePreisach ? E * model->DerivativeOperator(t, dt) : 0.0);
//         res(1, 1) = -gamma;
//         res(1, 2) = A * mc::cos(x[2]);
//
//         return res;
//     };
//
//     double dt = 0.01;
//
//     double gamma = 0.1;
//     double beta = 0.04;
//     double w0 = 1.0;
//     double w = 1.0;
//     double A = 0.5;
//
//     bool usePreisach = true;
//     double L = 1.0;
//     double h = 0.05;
//     double E = 1.4;
//
//     auto *model = new mc::ArealPreisachModel(L, false, false);
//     auto *model2 = new mc::ArealPreisachModel(L, false, false);
//
//     mc::ode::DSArgs args = {
//         {"dt", dt},
//         {"gamma", gamma},
//         {"A", A},
//         {"beta", beta},
//         {"w0", w0},
//         {"w", w},
//         {"E", E},
//         {"usePreisach", usePreisach},
//         {"model", model}
//     };
//     mc::ode::DSArgs args2 = {
//         {"dt", dt},
//         {"gamma", gamma},
//         {"A", A},
//         {"beta", beta},
//         {"w0", w0},
//         {"w", w},
//         {"E", E},
//         {"usePreisach", usePreisach},
//         {"model", model2}
//     };
//
//     // mc::Matrix x0 = {1.5, -1.5, 20.0};
//     // double t0 = 0.0;
//     // double dt = 0.01;
//     // double time = 100.0;
//     // int steps = static_cast<int>(time / dt);
//     //
//     // double sigma = 10.0;
//     // double rho = 28.0;
//     // double beta = 8.0 / 3.0;
//     //
//     // bool usePreisach = true;
//     // double L = 1.0;
//     // double h = 0.05;
//     // double E = 1.35;
//     // auto *model = new mc::ArealPreisachModel(L, true, false);
//     //
//     // mc::ode::DSArgs args = {
//     //     {"dt", dt},
//     //     {"sigma", sigma},
//     //     {"rho", rho},
//     //     {"beta", beta},
//     //     {"E", E},
//     //     {"usePreisach", usePreisach},
//     //     {"model", model}
//     // };
//     //
//     // auto func = [](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     // {
//     //     auto sigma = args.at("sigma").toDouble();
//     //     auto rho = args.at("rho").toDouble();
//     //     auto beta = args.at("beta").toDouble();
//     //     auto usePreisach = args.at("usePreisach").toBool();
//     //     auto E = args.at("E").toDouble();
//     //     auto model = args.at("model").toPreisachModel();
//     //     auto dt = args.at("dt").toDouble();
//     //     
//     //     auto res = mc::zeros_like<double>(x);
//     //     res[0] = sigma * (x[1] - x[0]) + (usePreisach ? E * model->P(x[0], static_cast<int>(t / dt)) : 0.0);
//     //     res[1] = x[0] * (rho - x[2]) - x[1];
//     //     res[2] = x[0] * x[1] - beta * x[2];
//     //     return res;
//     // };
//     //
//     // auto jac = [](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     // {
//     //     auto sigma = args.at("sigma").toDouble();
//     //     auto rho = args.at("rho").toDouble();
//     //     auto beta = args.at("beta").toDouble();
//     //     auto usePreisach = args.at("usePreisach").toBool();
//     //     auto E = args.at("E").toDouble();
//     //     auto model = args.at("model").toPreisachModel();
//     //     auto dt = args.at("dt").toDouble();
//     //     
//     //     auto res = mc::zeros<double>(x.shape().m_Cols);
//     //     res(0, 0) = -sigma + (usePreisach ? E * model->DerivativeOperator(t, dt) : 0.0);
//     //     res(0, 1) = sigma;
//     //     res(1, 0) = rho - x[2];
//     //     res(1, 1) = -1.0;
//     //     res(1, 2) = -x[0];
//     //     res(2, 0) = x[1];
//     //     res(2, 1) = x[0];
//     //     res(2, 2) = -beta;
//     //     return res;
//     // };
//
//     auto *system = new mc::ode::ContinuousDS(func, jac, dt, args, args2);
//
//     // auto div = mc::ode::mLCE_DivergenceDegreeTest(system, 100., 0.5, .6, 500, 3, false, true);
//     // for (const auto &[divergence, degree] : div)
//     // {
//     //     std::println(std::cout, "Degree: {}, Divergence: {:f}", degree, divergence);
//     // }
//
//     // std::vector<double> ns = {};
//     // double T = 0.5;
//     // double e = 0.15;
//     // auto Ms = mc::arange(1, 1500, 100);
//     // for (const auto& M : Ms)
//     // {
//     //     auto n = mc::ode::FindDivergenceDegree(system, 10., e, T, M, traj1, traj2, false, true);
//     //     ns.push_back(n);
//     // }
//     //
//     // mc::json::JsonDocument message({"name", "method", "dt", "T", "Ms","e","ns"});
//     // message.AddField("name", "RodosLCEs");
//     // message.AddField("method", "plot");
//     // message.AddField("dt", dt);
//     // message.AddField("e", e);
//     // message.AddField("Ms", Ms);
//     // message.AddField("T", T);
//     // message.AddField("ns", ns);
//     // auto *file = new FileWriter("RodosLCEs_Regression.json");
//     // file->Write(message.ToString());
//
//     mc::Matrix<double> Ts = {};
//     Ts = mc::append(Ts, mc::arange(0.5, 3.25, 0.25));
//     int M = 400;
//     double e = 0.5;
//     double areaCoeff = -0.9;
//
//     std::unordered_map<std::string, std::vector<mc::Matrix<double>>> trajs1, trajs2;
//     std::vector<double> ns;
//
//     for (const auto &T : Ts)
//     {
//         std::vector<mc::Matrix<double>> traj1, traj2;
//         const auto [n, t] = mc::ode::DivergenceDegreeRegressionData(system, 0., e, T, M, traj1, traj2, -0.9, false);
//
//         const auto strT = mc::doubleToString(T, 2);
//
//         trajs1.insert({strT, traj1});
//         trajs2.insert({strT, traj2});
//         std::println("n: {}", n);
//         ns.push_back(n);
//     }
//
//     mc::json::JsonDocument message({"name", "method", "dt", "e", "M", "Ts", "ns", "trajs1", "trajs2"});
//     message.AddField("name", "RodosLCEs");
//     message.AddField("method", "plot");
//     message.AddField("dt", dt);
//     message.AddField("e", e);
//     message.AddField("M", M);
//     message.AddField("Ts", mc::log(Ts));
//     message.AddField("ns", ns);
//     message.AddField("trajs1", trajs1);
//     message.AddField("trajs2", trajs2);
//     auto *file = new FileWriter("RodosLCEs_Regression.json");
//     file->Write(message.ToString());
//
//     // double T = 1.5;
//     // int M = 500;
//     // double e = 0.005;
//     //
//     // auto n = mc::ode::FindDivergenceDegree(system, 0., e, T, M, traj1, traj2, false, true);
//     // std::println(std::cout, "Degree: {:f}", n);
//     //
//     // mc::json::JsonDocument message({"name", "method", "dt", "e", "M","T","traj1", "traj2"});
//     // message.AddField("name", "RodosLCEs");
//     // message.AddField("method", "plot");
//     // message.AddField("dt", dt);
//     // message.AddField("e", e);
//     // message.AddField("M", M);
//     // message.AddField("T", T);
//     // message.AddField("traj1", traj1);
//     // message.AddField("traj2", traj2);
//     // auto *file = new FileWriter("RodosLCEs_TwoTrajs.json");
//     // file->Write(message.ToString());
//
//
//     //
//     // mc::json::JsonDocument message({"name", "method", "dt","time","results"});
//     // message.AddField("name", "RodosLCEs");
//     // message.AddField("method", "plot");
//     // message.AddField("dt", dt);
//     // message.AddField("time", time);
//     // message.AddSubField("results", "traj1", traj1);
//     // message.AddSubField("results", "traj2", traj2);
//     // auto *file = new FileWriter("RodosLCEs_TwoTrajs.json");
//     // file->Write(message.ToString());
//
//     // // auto [mLCE, history] = mc::ode::mLCE(system, static_cast<uint32_t>(50. / dt), steps);
//     // auto [b_mLCE, b_history] = mc::ode::Benettin_mLCE(system, 10., 1., 1.5, 500, false);
//     // std::print("{}", b_mLCE);
//     //
//     // mc::json::JsonDocument message({"name", "method", "dt", "h", "E", "time","results"});
//     // message.AddField("name", "RodosLCEs");
//     // message.AddField("method", "plot");
//     // message.AddField("dt", dt);
//     // message.AddField("h", h);
//     // message.AddField("E", E);
//     // message.AddField("time", time);
//     // // message.AddSubField("results", "mLCE", mLCE);
//     // message.AddSubField("results", "Benettin_mLCE", b_mLCE);
//     // // message.AddSubField("results", "mLCE_history", history);
//     // message.AddSubField("results", "Benettin_mLCE_history", b_history);
//     //
//     // // g_Server.SendDataMessage(message);
//     // auto *file = new FileWriter("RodosLCEs.json");
//     // file->Write(message.ToString());
// }
//
// void DivergenceDegreeTable()
// {
//     auto func = [](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         AL_PROFILE_FUNC("Rodos::func");
//         double dt = args.at("dt").toDouble();
//         double gamma = args.at("gamma").toDouble();
//         double A = args.at("A").toDouble();
//         double w0 = args.at("w0").toDouble();
//         double w = args.at("w").toDouble();
//         double E = args.at("E").toDouble();
//         const auto& model = args.at("model").toPreisachModel();
//
//         auto res = mc::zeros_like<double>(x);
//
//         res[0] = x[1];
//         res[1] = A * mc::cos(w * t) - gamma * x[1] - w0 * x[0] + E * model->P(x[0], static_cast<int>(t / dt));
//         res[2] = w0;
//
//         return res;
//     };
//
//     double dt = 0.01;
//
//     double gamma = 0.1;
//     double w0 = 1.0;
//     double w = 1.0;
//     double A = 0.5;
//
//     double L = 1.0;
//     double E = 1.4;
//
//     mc::Matrix<double> Ts = {};
//     Ts = mc::append(Ts, mc::arange(0.5, 3.25, 0.25));
//     int M = 400;
//     double e = 0.5;
//     double areaCoeff = -0.9;
//
//     // ------------ Loop by areaCoeffs ------------
//     const auto areaCoeffsLoop = [Ts, e, M, dt, gamma, A, w0, w, E, L, func](mc::json::JsonDocument &message)
//     {
//         auto *model = new mc::ArealPreisachModel(L, false, false);
//         auto *model2 = new mc::ArealPreisachModel(L, false, false);
//
//         mc::ode::DSArgs args = {
//             {"dt", dt},
//             {"gamma", gamma},
//             {"A", A},
//             {"w0", w0},
//             {"w", w},
//             {"E", E},
//             {"model", model}
//         };
//         mc::ode::DSArgs args2 = {
//             {"dt", dt},
//             {"gamma", gamma},
//             {"A", A},
//             {"w0", w0},
//             {"w", w},
//             {"E", E},
//             {"model", model2}
//         };
//
//         auto *system = new mc::ode::ContinuousDS(func, {}, dt, args, args2);
//
//         std::println("AREAL_COEFFS_LOOP");
//         const std::vector areaCoeffs = {-0.98, -0.96, -0.95, -0.9, -0.85, -0.8};
//         std::vector<std::string> areaCoeffsStr(areaCoeffs.size());
//         std::ranges::transform(areaCoeffs, areaCoeffsStr.begin(),
//                                [](const double &elem) { return mc::doubleToString(elem, 2); });
//
//         message.ExtendHeader("areaCoeffsLoop");
//         message.AddSubField({"areaCoeffsLoop", "areaCoeffs"}, areaCoeffsStr);
//
//         for (const auto &areaCoeff : areaCoeffs)
//         {
//             std::unordered_map<std::string, std::vector<mc::Matrix<double>>> trajs1, trajs2;
//             std::vector<double> ns = {};
//             for (const auto &T : Ts)
//             {
//                 std::vector<mc::Matrix<double>> traj1, traj2;
//                 const auto n = mc::ode::DivergenceDegreeRegressionData(system, 0., e, T, M, traj1, traj2, areaCoeff,
//                                                                        false)[0];
//
//                 const auto strT = mc::doubleToString(T, 2);
//
//                 trajs1.insert({strT, traj1});
//                 trajs2.insert({strT, traj2});
//                 std::println("n: {}", n);
//                 ns.push_back(n);
//             }
//
//             const auto areaCoeffStr = mc::doubleToString(areaCoeff, 2);
//             message.AddSubField({"areaCoeffsLoop", areaCoeffStr, "ns"}, ns);
//             message.AddSubField({"areaCoeffsLoop", areaCoeffStr, "trajs1"}, trajs1);
//             message.AddSubField({"areaCoeffsLoop", areaCoeffStr, "trajs2"}, trajs2);
//         }
//
//         message.AddSubField({"areaCoeffsLoop", "dt"}, dt);
//         message.AddSubField({"areaCoeffsLoop", "e"}, e);
//         message.AddSubField({"areaCoeffsLoop", "M"}, M);
//         message.AddSubField({"areaCoeffsLoop", "Ts"}, mc::log(Ts));
//     };
//     // --------------------------------------------
//
//     // ------------ Loop by epsilon ------------
//     const auto epsilonLoop = [Ts, areaCoeff, M, dt, gamma, A, w0, w, E, L, func](mc::json::JsonDocument &message)
//     {
//         auto *model = new mc::ArealPreisachModel(L, false, false);
//         auto *model2 = new mc::ArealPreisachModel(L, false, false);
//
//         mc::ode::DSArgs args = {
//             {"dt", dt},
//             {"gamma", gamma},
//             {"A", A},
//             {"w0", w0},
//             {"w", w},
//             {"E", E},
//             {"model", model}
//         };
//         mc::ode::DSArgs args2 = {
//             {"dt", dt},
//             {"gamma", gamma},
//             {"A", A},
//             {"w0", w0},
//             {"w", w},
//             {"E", E},
//             {"model", model2}
//         };
//
//         auto *system = new mc::ode::ContinuousDS(func, {}, dt, args, args2);
//
//         std::println("EPSILON_LOOP");
//         const std::vector es = {0.1, 0.2, 0.5, 0.7, 1.0};
//         std::vector<std::string> esStr(es.size());
//         std::ranges::transform(es, esStr.begin(), [](const double &elem) { return mc::doubleToString(elem, 2); });
//
//         message.ExtendHeader("epsilonLoop");
//         message.AddSubField({"epsilonLoop", "es"}, esStr);
//
//         for (const auto &e : es)
//         {
//             std::unordered_map<std::string, std::vector<mc::Matrix<double>>> trajs1, trajs2;
//             std::vector<double> ns = {};
//             for (const auto &T : Ts)
//             {
//                 std::vector<mc::Matrix<double>> traj1, traj2;
//                 const auto n = mc::ode::DivergenceDegreeRegressionData(system, 0., e, T, M, traj1, traj2, areaCoeff,
//                                                                        false)[0];
//
//                 const auto strT = mc::doubleToString(T, 2);
//
//                 trajs1.insert({strT, traj1});
//                 trajs2.insert({strT, traj2});
//                 std::println("n: {}", n);
//                 ns.push_back(n);
//             }
//
//             const auto epsilonStr = mc::doubleToString(e, 2);
//
//             message.AddSubField({"epsilonLoop", epsilonStr, "ns"}, ns);
//             message.AddSubField({"epsilonLoop", epsilonStr, "trajs1"}, trajs1);
//             message.AddSubField({"epsilonLoop", epsilonStr, "trajs2"}, trajs2);
//         }
//
//         message.AddSubField({"epsilonLoop", "dt"}, dt);
//         message.AddSubField({"epsilonLoop", "areaCoeff"}, areaCoeff);
//         message.AddSubField({"epsilonLoop", "M"}, M);
//         message.AddSubField({"epsilonLoop", "Ts"}, mc::log(Ts));
//     };
//     // --------------------------------------------
//
//     // ------------ Loop by M ------------
//     const auto MLoop = [Ts, areaCoeff, e, dt, gamma, A, w0, w, E, L, func](mc::json::JsonDocument &message)
//     {
//         auto *model = new mc::ArealPreisachModel(L, false, false);
//         auto *model2 = new mc::ArealPreisachModel(L, false, false);
//
//         mc::ode::DSArgs args = {
//             {"dt", dt},
//             {"gamma", gamma},
//             {"A", A},
//             {"w0", w0},
//             {"w", w},
//             {"E", E},
//             {"model", model}
//         };
//         mc::ode::DSArgs args2 = {
//             {"dt", dt},
//             {"gamma", gamma},
//             {"A", A},
//             {"w0", w0},
//             {"w", w},
//             {"E", E},
//             {"model", model2}
//         };
//
//         auto *system = new mc::ode::ContinuousDS(func, {}, dt, args, args2);
//
//         std::println("M_LOOP");
//         const std::vector Ms = {200, 300, 400, 500};
//         std::vector<std::string> MsStr(Ms.size());
//         std::ranges::transform(Ms, MsStr.begin(), [](const double &elem) { return std::to_string(elem); });
//
//         message.ExtendHeader("MLoop");
//         message.AddSubField({"MLoop", "Ms"}, MsStr);
//
//         for (const auto &M : Ms)
//         {
//             std::unordered_map<std::string, std::vector<mc::Matrix<double>>> trajs1, trajs2;
//             std::vector<double> ns = {};
//             for (const auto &T : Ts)
//             {
//                 std::vector<mc::Matrix<double>> traj1, traj2;
//                 const auto n = mc::ode::DivergenceDegreeRegressionData(system, 0., e, T, M, traj1, traj2, areaCoeff,
//                                                                        false)[0];
//
//                 const auto strT = mc::doubleToString(T, 2);
//
//                 trajs1.insert({strT, traj1});
//                 trajs2.insert({strT, traj2});
//                 std::println("n: {}", n);
//                 ns.push_back(n);
//             }
//
//             const auto MStr = std::to_string(M);
//
//             message.AddSubField({"MLoop", MStr, "ns"}, ns);
//             message.AddSubField({"MLoop", MStr, "trajs1"}, trajs1);
//             message.AddSubField({"MLoop", MStr, "trajs2"}, trajs2);
//         }
//
//         message.AddSubField({"MLoop", "dt"}, dt);
//         message.AddSubField({"MLoop", "areaCoeff"}, areaCoeff);
//         message.AddSubField({"MLoop", "e"}, e);
//         message.AddSubField({"MLoop", "Ts"}, mc::log(Ts));
//     };
//     // --------------------------------------------
//
//     mc::json::JsonDocument message({"name"});
//     message.AddField("name", "RodosLCEs_DivergenceDegreeTable");
//
//     // areaCoeffsLoop(message);
//     epsilonLoop(message);
//     // MLoop(message);
//
//     auto *file = new FileWriter("RodosLCEs_DivergenceDegreeTable.json");
//     file->Write(message.ToString());
// }
//
// void ArealModelTest()
// {
//     double dt = 0.01;
//     double time = 100.0;
//     int steps = static_cast<int>(time / dt);
//
//     double gamma = 0.1;
//     double w02 = 1.0;
//     double w = 1.0;
//     double A = 0.5;
//
//     double L = 1.0;
//     double E = 1.35;
//
//     auto model = std::make_unique<mc::ArealPreisachModel>(L, true);
//     // auto* model = new mc::DiscretePreisachModel(L, 0.01, true);
//
//     auto func = [&](mc::Matrix<double> x, double t, const mc::ode::DSArgs &args) -> mc::Matrix<double>
//     {
//         auto res = mc::zeros_like<double>(x);
//
//         res[0] = x[1];
//         res[1] = A * mc::sin(w * t) - gamma * x[1] - w02 * x[0] + E * model->P(x[0], static_cast<int>(t / dt));
//         res[2] = w;
//
//         return res;
//     };
//
//     auto *system = new mc::ode::ContinuousDS(func, {}, dt);
//     auto traj = system->Forward(steps);
//
//     auto [inputs1, outputs] = model->HysteresisLoop();
//     auto [inputs2, derivatives] = model->DerivativeHistory();
//     auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();
//
//     auto x = traj(traj.rSlice(), 0);
//     auto v = traj(traj.rSlice(), 1);
//
//     mc::json::JsonDocument message({"name", "method", "dt", "time", "loop", "anim", "results"});
//     message.AddField("name", "ArealModelTest");
//     message.AddField("method", "plot");
//     message.AddField("dt", dt);
//     message.AddField("time", time);
//     message.AddSubField({"results", "x"}, x);
//     message.AddSubField({"results", "v"}, v);
//     message.AddSubField({"results", "derivatives"}, derivatives);
//     message.AddSubField({"loop", "inputs"}, x);
//     message.AddSubField({"loop", "outputs"}, outputs);
//     message.AddSubField({"anim", "in"}, inHist);
//     message.AddSubField({"anim", "x"}, xHist);
//     message.AddSubField({"anim", "y"}, yHist);
//     message.AddSubField({"anim", "out"}, outHist);
//     message.AddSubField({"anim", "save"}, true);
//
//     auto *file = new FileWriter("ArealModelTest.json");
//     file->Write(message.ToString());
// }

void JustSolveRodos()
{
    mc::ode::DynamicalSystem::DSFunc func = [](const mc::Matrix<double> &x, const double t,
                                               mc::ode::DSArgs &args) -> mc::Matrix<double>
    {
        AL_PROFILE_FUNC("Rodos::func");
        double dt = args.at("dt").toDouble();
        double gamma = args.at("gamma").toDouble();
        double A = args.at("A").toDouble();
        double w0 = args.at("w0").toDouble();
        double w = args.at("w").toDouble();
        double E = args.at("E").toDouble();
        auto &model = args.at("model").toPreisachModel();

        auto res = mc::zeros_like<double>(x);

        res[0] = x[1];
        res[1] = A * mc::cos(w * t) - gamma * x[1] - mc::power(w0, 2) * x[0] + E * model->P(
            x[0], static_cast<int>(t / dt));
        res[2] = mc::power(w0, 2);

        return res;
    };

    mc::ode::DynamicalSystem::DSFunc jac = [](const mc::Matrix<double> &x, const double t,
                                              mc::ode::DSArgs &args) -> mc::Matrix<double>
    {
        return {};
    };

    double dt = 0.01;
    double time = 500.0;
    auto steps = static_cast<uint32_t>(time / dt);

    double gamma = 0.1;
    double w0 = 1.0;
    double w = 1.0;
    double A = 0.5;

    double L = 1.0;
    // double h = 0.05;
    double E = 1.35;

    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);

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

    mc::Matrix<double> x0 = {0.0, 0.0, 0.0};

    mc::Ref system = mc::Ref<mc::ode::ContinuousDS>::Create(func, jac, dt, args, args2, x0);
    auto traj = system->Forward(steps);

    auto [inputs1, outputs] = model->HysteresisLoop();
    // auto [inputs2, derivatives] = model->DerivativeHistory();
    // auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();

    // mc::json::JsonDocument message({"name", "method", "dt", "h", "E", "time", "loop", "anim", "results"});
    mc::json::JsonDocument message({"name", "method", "dt", "E", "time", "results", "loop"});
    message.AddField("name", "JustSolveRodos");
    message.AddField("method", "plot");
    message.AddField("dt", dt);
    // message.AddField("h", h);
    message.AddField("E", E);
    message.AddField("time", time);
    message.AddSubField({"results", "x"}, traj(traj.rSlice(), 0));
    message.AddSubField({"results", "v"}, traj(traj.rSlice(), 1));
    // message.AddSubField({"results", "derivatives"}, derivatives);
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
    mc::ode::DynamicalSystem::DSFunc func = [](const mc::Matrix<double> &x, const double t,
                                               mc::ode::DSArgs &args) -> mc::Matrix<double>
    {
        AL_PROFILE_FUNC("Rodos::func");
        double dt = args.at("dt").toDouble();
        double gamma = args.at("gamma").toDouble();
        double A = args.at("A").toDouble();
        double w0 = args.at("w0").toDouble();
        double w = args.at("w").toDouble();
        double E = args.at("E").toDouble();
        auto &model = args.at("model").toPreisachModel();

        auto res = mc::zeros_like<double>(x);

        res[0] = x[1];
        res[1] = A * mc::cos(w * t) - gamma * x[1] - mc::power(w0, 2) * x[0] + E * model->P(
            x[0], static_cast<int>(t / dt));
        res[2] = mc::power(w0, 2);

        return res;
    };

    mc::ode::DynamicalSystem::DSFunc jac = [](const mc::Matrix<double> &x, const double t,
                                              mc::ode::DSArgs &args) -> mc::Matrix<double>
    {
        return {};
    };

    double dt = 0.01;

    double gamma = 0.1;
    double beta = 0.04;
    double w0 = 1.0;
    double w = 1.0;
    double A = 0.5;

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
    system->SetResetFn([areaCoeff, L](mc::ode::DSArgs &args, mc::ode::DSArgs &nextArgs)
    {
        auto model1 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);

        auto model2 = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);
        if (!isnan(areaCoeff))
        {
            model2->P(L, -2);
            model2->P(areaCoeff * L, -1);
        }

        args.insert_or_assign("model", mc::ode::Vote(model1));
        nextArgs.insert_or_assign("model", mc::ode::Vote(model2));
    });

    // std::vector<double> es = mc::arange(0.05, 1.05, 0.05).toFlattenVector();
    std::vector<double> es = mc::arange(0., 1.05, 0.05).toFlattenVector();
    std::vector<mc::ode::Vote> e_range;
    std::ranges::transform(es, std::back_inserter(e_range), [](const double e) { return mc::ode::Vote(e); });

    const mc::Matrix<double> angels = mc::arange(0., mc::consts::twoPi, 0.5);
    std::vector<mc::ode::Vote> coords = {
        // mc::Matrix({1., 0., 0.}),
        // mc::Matrix({sqrt(2.) / 2, sqrt(2.) / 2, 0.}),
        // mc::Matrix({0., 1., 0.}),
        // mc::Matrix({-sqrt(2.) / 2, sqrt(2.) / 2, 0.}),
        // mc::Matrix({-1., 0., 0.}),
        // mc::Matrix({-sqrt(2.) / 2, -sqrt(2.) / 2, 0.}),
        // mc::Matrix({0., -1., 0.}),
        // mc::Matrix({sqrt(2.) / 2, -sqrt(2.) / 2, 0.}),
    };
    for (const auto &angel : angels)
    {
        mc::Matrix m = {mc::cos(angel), mc::sin(angel), 0.};
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

        std::vector<mc::Matrix<double>> points;
        points.reserve(n);

        for (int i = 0; i < n; ++i)
        {
            points.push_back({dist(gen), dist(gen), 0.});
        }

        return points;
    };

    uint32_t seed;
    std::vector<mc::Matrix<double>> points = gen_random_points(10, 1., seed);

    std::println("seed: {}", seed);
    std::println("random points: {}", points);

    std::vector<mc::ode::Vote> x0s;
    std::ranges::transform(points, std::back_inserter(x0s),
                           [](const mc::Matrix<double> &point) { return mc::ode::Vote(point); });

    auto div_solution_fn = [time](mc::Ref<mc::ode::DynamicalSystem> system,
                                  const std::unordered_map<mc::ode::DSArgs::key_type, mc::ode::Vote> &args)
    {
        double e = args.at("e").toDouble();
        mc::Matrix<double> v = args.at("v").toMatrix();
        mc::Matrix<double> x0 = args.at("x0").toMatrix();
        system->SetX0(x0);
        // double x0_step = args.at("x0_step").toDouble();

        // mc::Matrix<double> x0 = x0_step * mc::normalize(v);
        // system->SetX0(x0);

        std::vector<std::vector<double>> mainTraj = {};
        mainTraj.push_back(system->GetX().toFlattenVector());

        auto w = system->GetX() + normalize(v) * e;

        std::vector<std::vector<double>> secondTraj = {};
        secondTraj.push_back(w.toFlattenVector());

        auto count = static_cast<mc::uint32>(time / system->GetDeltaTime());
        while (count--)
        {
            auto traj = system->Forward(1);
            mainTraj.push_back(system->GetX().toFlattenVector());

            w = system->NextTM(w);
            secondTraj.push_back(w.toFlattenVector());
        }

        auto main = mc::Matrix(mainTraj);
        auto x1 = main(main.rSlice(), 0);
        auto y1 = main(main.rSlice(), 1);

        auto other = mc::Matrix(secondTraj);
        auto x2 = other(other.rSlice(), 0);
        auto y2 = other(other.rSlice(), 1);

        auto div = mc::sqrt(mc::square(x1 - x2) + mc::square(y1 - y2));
        return div;
    };

    std::unordered_map<std::string, mc::ode::CandidateParams> params = {
        {"e", mc::ode::CandidateParams(e_range, std::nullopt)},
        {"v", mc::ode::CandidateParams(coords, std::nullopt)},
        {"x0", mc::ode::CandidateParams(x0s, std::nullopt)},
        // {"x0_step", mc::ode::CandidateParams(x0_step_range, std::nullopt)},
    };

    auto res = mc::ode::FindBestParams(system, div_solution_fn, params);

    for (const auto &param : res)
    {
        double best_e = param.params.at("e").toDouble();
        auto best_v = param.params.at("v").toMatrix();
        mc::Matrix<double> best_x0 = param.params.at("x0").toMatrix();
        // double best_x0_step = param.params.at("x0_step").toDouble();
        double best_metric = param.metric_value;

        std::print("Best e: {}\n", best_e);
        mc::Matrix v = {mc::rad2deg(mc::arccos(best_v[0])), mc::rad2deg(mc::arcsin(best_v[0])), 0.0};
        std::print("Best v: {}\n", v);
        std::print("Best x0: {}\n", best_x0);
        // std::print("Best x0_step: {}\n", best_x0_step);
        std::print("Metric value: {}\n", best_metric / dt);

        std::unordered_map<std::string, std::vector<mc::Matrix<double>>> trajs1, trajs2;
        std::vector<double> ns;

        mc::Matrix<double> Ts = {};
        Ts = mc::append(Ts, mc::arange(0.5, 3.25, 0.25));
        int M = 400;

        for (const auto &T : Ts)
        {
            system->SetX0(best_x0);
            std::vector<mc::Matrix<double>> traj1, traj2;
            const auto [n, t] = mc::ode::DivergenceDegreeRegressionData(system, 0., best_e, best_v, T, M, traj1, traj2,
                                                                        false);

            const auto strT = mc::doubleToString(T, 2);

            trajs1.insert({strT, traj1});
            trajs2.insert({strT, traj2});
            std::println("n: {}", n);
            ns.push_back(n);
        }

        mc::json::JsonDocument message({"name", "method", "dt", "e", "M", "Ts", "ns", "trajs1", "trajs2"});
        message.AddField("name", "RodosLCEs");
        message.AddField("method", "plot");
        message.AddField("dt", dt);
        message.AddField("e", best_e);
        message.AddField("M", M);
        message.AddField("Ts", mc::log(Ts));
        message.AddField("ns", ns);
        message.AddField("trajs1", trajs1);
        message.AddField("trajs2", trajs2);
        mc::Ref file = mc::Ref<FileWriter>::Create("RodosLCEs_Regression.json");
        file->Write(message.ToString());
    }

    // mc::Matrix<double> es = {0.};

    // auto doc = mc::ode::TrajsCircle(system, es.toFlattenVector(), 10., .0, -0.9);

    // mc::Ref file = mc::Ref<FileWriter>::Create("TestTrajsCircle.json");
    // file->Write(doc.ToString());
}

void TestEverettFunction()
{
    mc::ode::DynamicalSystem::DSFunc func = [](const mc::Matrix<double> &x, const double t,
                                               mc::ode::DSArgs &args) -> mc::Matrix<double>
    {
        AL_PROFILE_FUNC("Rodos::func");
        double dt = args.at("dt").toDouble();
        double gamma = args.at("gamma").toDouble();
        double A = args.at("A").toDouble();
        double w0 = args.at("w0").toDouble();
        double w = args.at("w").toDouble();
        double E = args.at("E").toDouble();
        auto &model = args.at("model").toPreisachModel();

        auto res = mc::zeros_like<double>(x);

        res[0] = x[1];
        res[1] = A * mc::cos(w * t) - gamma * x[1] - mc::power(w0, 2) * x[0] + E * model->P(
            x[0], static_cast<int>(t / dt));
        res[2] = mc::power(w0, 2);

        return res;
    };

    mc::ode::DynamicalSystem::DSFunc jac = [](const mc::Matrix<double> &x, const double t,
                                              mc::ode::DSArgs &args) -> mc::Matrix<double>
    {
        return {};
    };

    double dt = 0.01;
    double time = 200.0;
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
        const double e = mc::exp((x - b) / c);
        return a * e / mc::power(1 + e, 2);
    };

    auto everett = [phi](double x, double y)
    {
        mc::Matrix<double> a = {0.113, 2.224e-3, 1.812e-2};
        mc::Matrix<double> b = {56.468, 30.197, 86.148};
        mc::Matrix<double> c = {6.035, 284.54, 38.842};

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

    mc::Ref system = mc::Ref<mc::ode::ContinuousDS>::Create(func, jac, dt, args, args2);
    auto traj = system->Forward(steps);

    auto [inputs1, outputs] = model->HysteresisLoop();
    auto [inputs2, derivatives] = model->DerivativeHistory();
    auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();

    mc::json::JsonDocument message({"name", "method", "dt", "h", "E", "time", "loop", "anim", "results"});
    message.AddField("name", "JustSolveRodos");
    message.AddField("method", "plot");
    message.AddField("dt", dt);
    message.AddField("h", h);
    message.AddField("E", E);
    message.AddField("time", time);
    message.AddSubField({"results", "x"}, traj(traj.rSlice(), 0));
    message.AddSubField({"results", "v"}, traj(traj.rSlice(), 1));
    message.AddSubField({"results", "derivatives"}, derivatives);
    message.AddSubField({"loop", "inputs"}, inHist);
    message.AddSubField({"loop", "outputs"}, outputs);
    message.AddSubField({"anim", "in"}, inHist);
    message.AddSubField({"anim", "x"}, xHist);
    message.AddSubField({"anim", "y"}, yHist);
    message.AddSubField({"anim", "out"}, outHist);

    // g_Server.SendDataMessage(message);
    mc::Ref file = mc::Ref<FileWriter>::Create("JustSolveRodos.json");
    file->Write(message.ToString());
}

void SecondAttractor()
{
    mc::ode::DynamicalSystem::DSFunc func = [](const mc::Matrix<double> &x, const double t,
                                               mc::ode::DSArgs &args) -> mc::Matrix<double>
    {
        AL_PROFILE_FUNC("Rodos::func");
        double dt = args.at("dt").toDouble();
        double gamma = args.at("gamma").toDouble();
        double A = args.at("A").toDouble();
        double w0 = args.at("w0").toDouble();
        double w = args.at("w").toDouble();
        double E = args.at("E").toDouble();
        auto &model = args.at("model").toPreisachModel();

        auto res = mc::zeros_like<double>(x);

        if (x[0] > 0)
        {
            E = 3.;
        }
        else if (x[0] < 0)
        {
            E = 1.;
        }

        res[0] = x[1];
        res[1] = A * mc::cos(w * t) - gamma * x[1] - mc::power(w0, 2) * x[0] + E * model->P(
            x[0], static_cast<int>(t / dt));
        res[2] = mc::power(w0, 2);

        return res;
    };

    mc::ode::DynamicalSystem::DSFunc jac = [](const mc::Matrix<double> &x, const double t,
                                              mc::ode::DSArgs &args) -> mc::Matrix<double>
    {
        return {};
    };

    double dt = 0.1;
    double time = 200.0;
    auto steps = static_cast<uint32_t>(time / dt);

    double L = 1.0;
    double E = 1.7;

    double gamma = 0.01;
    double w0 = 2.0;
    // double w0 = 1.0;
    // double w = mc::sqrt(mc::power(w0, 2) - E);
    double w = 2.;
    // double A = 0.5;
    double A = 1.5;

    auto model = mc::Ref<mc::ArealPreisachModel>::Create(L, false, false);

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

    mc::Matrix<double> x0 = {0.0, 0.0, 0.0};

    mc::Ref system = mc::Ref<mc::ode::ContinuousDS>::Create(func, jac, dt, args, args2, x0);
    auto traj = system->Forward(steps);

    auto [inputs1, outputs] = model->HysteresisLoop();
    // auto [inputs2, derivatives] = model->DerivativeHistory();
    // auto [inHist, xHist, yHist, outHist] = model->GetAnimationData();

    // mc::json::JsonDocument message({"name", "method", "dt", "h", "E", "time", "loop", "anim", "results"});
    mc::json::JsonDocument message({"name", "method", "dt", "E", "time", "results", "loop"});
    message.AddField("name", "JustSolveRodos");
    message.AddField("method", "plot");
    message.AddField("dt", dt);
    // message.AddField("h", h);
    message.AddField("E", E);
    message.AddField("time", time);
    message.AddSubField({"results", "x"}, traj(traj.rSlice(), 0));
    message.AddSubField({"results", "v"}, traj(traj.rSlice(), 1));
    // message.AddSubField({"results", "derivatives"}, derivatives);
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

int main()
{

    SecondAttractor();

    // auto model = mc::Ref<mc::ArealPreisachModel>::Create(1., false, false);
    //
    // mc::Matrix<double> t = mc::arange(0., 5. * mc::consts::twoPi, 0.01);
    // mc::Matrix<double> f = 2. * mc::sin(t);
    //
    // std::vector<double> res = {};
    //
    // for (int i = 0; i < f.size(); i++)
    // {
    //     res.push_back(model->P(f[i] - 2.0, i));
    // }
    //
    //
    // mc::json::JsonDocument message({"name", "in", "out"});
    // message.AddField("name", "sin");
    // message.AddField("in", f);
    // message.AddField("out", res);
    // mc::Ref file = mc::Ref<FileWriter>::Create("TEST.json");
    // file->Write(message.ToString());


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


    // TestTrajsCircle();
    // TestEverettFunction();


    // RodosAFCs();

    // SolutionBehaviorResearchDependingOnH();

    // OperatorDerivativeTest();

    // LyapunovExponentsTest();    

    // _sleep(10000);
}
