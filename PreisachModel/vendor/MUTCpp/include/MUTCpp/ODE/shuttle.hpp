// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <vector>
#include <random>
#include <expected>
#include <stdexcept>

#include "Ref/Ref.hpp"
#include "DynamicSystem.hpp"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include "vendor/json/nlohmann_json.hpp"

namespace Eigen
{
    template <typename Derived>
    void to_json(nlohmann::json &j, const MatrixBase<Derived> &matrix)
    {
        j = nlohmann::json::array();
        for (int i = 0; i < matrix.rows(); ++i)
        {
            nlohmann::json row = nlohmann::json::array();
            for (int k = 0; k < matrix.cols(); ++k)
            {
                row.push_back(matrix(i, k));
            }
            j.push_back(row);
        }
    }

    template <typename Derived>
    void from_json(const nlohmann::json &j, MatrixBase<Derived> &matrix)
    {
        if (!j.is_array() || j.empty() || !j.front().is_array())
        {
            matrix.resize(0, 0);
            return;
        }
        const auto rows = j.size();
        const auto cols = j.front().size();
        matrix.resize(rows, cols);
        for (size_t i = 0; i < rows; ++i)
        {
            const auto &row = j.at(i);
            if (row.size() != cols)
                THROW_RUNTIME_ERROR("Non-rectangular matrix in JSON.");
            for (size_t k = 0; k < cols; ++k)
            {
                matrix(i, k) = row.at(k).get<double>();
            }
        }
    }
}

namespace mc
{
    namespace detail
    {
        /**
         * @struct Settings
         * @param T: Curve simulation time.
         * @param dt: Delta time step.
         * @param tol: Difference between adjacent points on the curve for premature simulation stopping.
         * @param patience: Number of consecutive simulation steps with a change < tol.
         */
        struct Settings
        {
            double T = 40.0;
            double dt = 0.05;
            double tol = 1e-5; // изменение для досрочной остановки
            int patience = 3; // сколько последовательных шагов с изменением < tol
        };

        inline void to_json(nlohmann::json &j, const Settings &settings)
        {
            j = nlohmann::json::object();
            j["T"] = settings.T;
            j["dt"] = settings.dt;
            j["tol"] = settings.tol;
            j["patience"] = settings.patience;
        }

        inline void from_json(const nlohmann::json &j, Settings &settings)
        {
            if (!j.is_object() || j.empty())
                return;
            settings.T = j["T"].get<double>();
            settings.dt = j["dt"].get<double>();
            settings.tol = j["tol"].get<double>();
            settings.patience = j["patience"].get<int>();
        }

        struct CurveResult
        {
            std::vector<double> t;
            Eigen::Matrix2d A;
            Eigen::Vector2d b;
            std::vector<Eigen::Vector2d> pts; // size m x n
        };

        struct MatrixExp
        {
            Eigen::Matrix2d m_A;
            Eigen::Vector2d m_b;

            MatrixExp(const Eigen::Matrix2d &A, const Eigen::Vector2d &b) :
                m_A(A), m_b(b)
            {
            }

            Eigen::Vector2d operator()(double t) const
            {
                const Eigen::Matrix2d At = m_A * t;
                const Eigen::Matrix2d expAt = At.exp();
                return expAt * m_b;
            }
        };

        inline CurveResult generate_curve(const Eigen::Matrix2d &A, const Eigen::Vector2d &b, const Settings &s)
        {
            AL_PROFILE_FUNC("generate_curve");
            CurveResult res;
            res.A = A;
            res.b = b;
            const int maxSteps = static_cast<int>(std::ceil(s.T / s.dt));
            res.t.reserve(std::min(maxSteps + 1, 1'000'000));
            res.pts.reserve(std::min(maxSteps + 1, 1'000'000));

            auto expm = MatrixExp(A, b);

            double t = 0.0;
            Eigen::Vector2d prev = expm(t);
            res.t.push_back(t);
            res.pts.push_back(prev);

            int small_count = 0;
            for (int i = 1; i <= maxSteps; ++i)
            {
                t = i * s.dt;
                auto cur = expm(t);
                if (!std::isfinite(cur.array().abs().maxCoeff()))
                {
                    std::cout << "[WARN] Non-finite result at t=" << t << "; stopping.\n";
                    break;
                }
                double delta = (cur - prev).norm();
                if (delta < s.tol)
                {
                    ++small_count;
                    if (small_count >= s.patience)
                    {
                        std::cout << "[INFO] Early stop at t=" << t << " after " << small_count <<
                            " small steps (delta=" << delta << ").\n";
                        break;
                    }
                }
                else
                {
                    small_count = 0;
                }
                res.t.push_back(t);
                res.pts.push_back(cur);
                prev = cur;
            }
            return res;
        }

        // struct TwoRaysResult
        // {
        //     Eigen::Matrix2d rays; // 2 x 2 (rows are unit rays)
        //     std::array<int, 2> idxs; // sample indices
        //     double arc_len;
        //     std::string info;
        // };

        // TwoRaysResult two_rays_by_angle_2d(const CurveResult &cr, Eigen::Matrix2d A,
        //                                           double min_norm_ratio = 1e-6, double eps_tangent = 1e-12)
        // {
        //     const auto &pts = cr.pts;
        //     const int m = static_cast<int>(pts.size());
        //     if (m < 2)
        //     {
        //         THROW_INVALID_ARGUMENT_ERROR("Cure is too small (< 2 points).");
        //     }
        //
        //     std::vector<double> norms(m);
        //     double max_norm = 0.0;
        //     for (int i = 0; i < m; ++i)
        //     {
        //         norms[i] = pts[i].norm();
        //         if (norms[i] > max_norm)
        //         {
        //             max_norm = norms[i];
        //         }
        //     }
        //     if (max_norm == 0.0)
        //     {
        //         TwoRaysResult r;
        //         r.rays = Eigen::Matrix2d::Zero();
        //         r.idxs = {-1, -1};
        //         r.arc_len = consts::pi;
        //         r.info = "all samples zero - fallback";
        //         return r;
        //     }
        //
        //     double thr = max_norm * min_norm_ratio;
        //     std::vector<int> valid;
        //     valid.reserve(m);
        //     for (int i = 0; i < m; ++i)
        //     {
        //         if (norms[i] >= thr)
        //         {
        //             valid.push_back(i);
        //         }
        //     }
        //
        //     if (valid.empty())
        //     {
        //         // fallback to eigenvector + perp
        //         Eigen::Vector2d v = dominant_real_eigvec(A);
        //         Eigen::Vector2d perp(2);
        //         perp << -v(1), v(0);
        //         Eigen::Matrix2d R;
        //         R.col(0) = v / v.norm();
        //         R.col(1) = perp / perp.norm();
        //         TwoRaysResult r;
        //         r.rays = R;
        //         r.idxs = {-1, -1};
        //         r.arc_len = consts::pi;
        //         r.info = "no valid points; fallback eigvec";
        //         return r;
        //     }
        //
        //     // compute angles for valid points
        //     std::vector<std::pair<double, int>> ang_idx;
        //     ang_idx.reserve(valid.size());
        //     for (size_t k = 0; k < valid.size(); ++k)
        //     {
        //         const auto &pt = pts[valid[k]];
        //         double th = std::atan2(pt.y(), pt.x());
        //         ang_idx.emplace_back(th, static_cast<int>(k)); // store index into 'valid'
        //     }
        //
        //     // sort by angle
        //     std::ranges::sort(ang_idx, [](auto const &a, auto const &b) { return a.first < b.first; });
        //
        //     // create sorted angles array and mapping to valid indices
        //     int M = static_cast<int>(ang_idx.size());
        //     std::vector<double> ang_sorted(M);
        //     std::vector<int> ang_sorted_idx(M); // indices into 'valid'
        //     for (int j = 0; j < M; ++j)
        //     {
        //         ang_sorted[j] = ang_idx[j].first;
        //         ang_sorted_idx[j] = ang_idx[j].second;
        //     }
        //
        //     // find largest circular gap
        //     double largest_gap = -1.0;
        //     int kmax = 0;
        //     for (int j = 0; j < M; ++j)
        //     {
        //         double a = ang_sorted[j];
        //         double bval = ang_sorted[(j + 1) % M];
        //         if (j == M - 1)
        //             bval += 2.0 * mc::consts::pi;
        //         double gap = bval - a;
        //         if (gap > largest_gap)
        //         {
        //             largest_gap = gap;
        //             kmax = j;
        //         }
        //     }
        //     double arc_len = 2.0 * consts::pi - largest_gap;
        //
        //     // endpoints indices in ang_sorted: start = kmax+1, end = kmax
        //     int pos_start_sorted = (kmax + 1) % M;
        //     int pos_end_sorted = kmax;
        //
        //     // map to original sample indices:
        //     int idx_start = valid[ang_sorted_idx[pos_start_sorted]];
        //     int idx_end = valid[ang_sorted_idx[pos_end_sorted]];
        //
        //     Eigen::Vector2d pos_start = pts[idx_start];
        //     Eigen::Vector2d pos_end = pts[idx_end];
        //
        //     auto choose_ray = [&](const Eigen::Vector2d &pos)-> std::pair<Eigen::Vector2d, std::string>
        //     {
        //         double pos_norm = pos.norm();
        //         Eigen::Vector2d ray;
        //         if (pos_norm >= thr)
        //         {
        //             // radial
        //             ray = pos / pos_norm;
        //             // ensure outward
        //             if (ray.dot(pos) < 0)
        //                 ray = -ray;
        //             return {ray, "radial"};
        //         }
        //         Eigen::Vector2d tang = A * pos;
        //         if (tang.norm() >= eps_tangent)
        //         {
        //             ray = tang / tang.norm();
        //             if (pos_norm > 1e-12 && ray.dot(pos) < 0)
        //                 ray = -ray;
        //             return {ray, "tangent"};
        //         }
        //         Eigen::Vector2d eig = dominant_real_eigvec(A);
        //         if (eig.norm() < 1e-12)
        //             eig(0) = 1.0;
        //         eig.normalize();
        //         if (pos_norm > 1e-12 && eig.dot(pos) < 0)
        //             eig = -eig;
        //         return {eig, "eigvec_fallback"};
        //     };
        //
        //     auto [r1, reason1] = choose_ray(pos_start);
        //     auto [r2, reason2] = choose_ray(pos_end);
        //
        //     Eigen::Matrix2d rays;
        //     rays.col(0) = r1;
        //     rays.col(1) = r2;
        //
        //     TwoRaysResult res;
        //     res.rays = rays;
        //     res.idxs = {idx_start, idx_end};
        //     res.arc_len = arc_len;
        //     res.info = std::string("start: ") + reason1 + ", end: " + reason2;
        //     return res;
        // }

        struct Tangent
        {
            double k; // slope
            double b; // intercept

            [[nodiscard]] constexpr double operator()(double x) const noexcept
            {
                return k * x + b;
            }
        };

        [[nodiscard]] inline std::expected<Tangent, std::string> tangent_from_table(
            const std::vector<Eigen::Vector2d> &data, size_t idx)
        {
            AL_PROFILE_FUNC("tangent_from_table");
            const auto n = data.size();
            if (idx >= n)
            {
                return std::unexpected("Index out of bounds.");
            }

            const double x0 = data[idx][0];
            const double y0 = data[idx][1];

            double m = 0.0;

            if (idx > 0 && idx < n - 1)
            {
                // central difference
                const double x_prev = data[idx - 1][0];
                const double y_prev = data[idx - 1][1];
                const double x_next = data[idx + 1][0];
                const double y_next = data[idx + 1][1];
                m = (y_next - y_prev) / (x_next - x_prev);
            }
            else if (idx == 0 && n > 1)
            {
                // forward difference
                const double x1 = data[1][0];
                const double y1 = data[1][1];
                m = (y1 - y0) / (x1 - x0);
            }
            else if (idx == n - 1 && n > 1)
            {
                // backward difference
                const double x1 = data[n - 2][0];
                const double y1 = data[n - 2][1];
                m = (y0 - y1) / (x0 - x1);
            }
            else
            {
                return std::unexpected("Insufficient data for tangent computation.");
            }

            const double b = y0 - m * x0;
            return Tangent{m, b};
        }

        inline Eigen::Matrix2d pick_two_support_rays_2d(const CurveResult &c)
        {
            AL_PROFILE_FUNC("pick_two_support_rays_2d");
            auto tan = tangent_from_table(c.pts, c.pts.size() - 1);
            if (!tan.has_value())
            {
                THROW_RUNTIME_ERROR(tan.error());
            }

            const Eigen::Vector2d r1 = c.b;

            auto tan_v = tan.value();
            double x = -1 / tan_v.k;
            double y = tan_v(x);
            const Eigen::Vector2d r2 = {x, y};

            Eigen::Matrix2d rays;
            // rays.col(0) = r1.normalized();
            // rays.col(1) = r2.normalized();
            rays.col(0) = r1;
            rays.col(1) = r2;
            return rays;
        }
    }

    inline void write_curve_csv(const std::string &prefix, const detail::CurveResult &cr)
    {
        std::string fname = prefix + "_curve.csv";
        std::ofstream ofs(fname);
        ofs << std::setprecision(12);
        ofs << "t,x,y\n";
        for (size_t i = 0; i < cr.t.size(); ++i)
        {
            ofs << cr.t[i] << "," << cr.pts[i](0) << "," << cr.pts[i](1) << "\n";
        }
        ofs.close();
        std::cout << "[INFO] Curve written to: " << fname << " (" << cr.t.size() << " samples)\n";
    }


    inline void write_rays_csv(const std::string &prefix, const Eigen::Matrix2d &rays)
    {
        std::string fname = prefix + "_rays.csv";
        std::ofstream ofs(fname);
        ofs << std::setprecision(12);
        ofs << "idx,x,y\n";
        ofs << rays(0, 0) << "," << rays(1, 0) << "\n";
        ofs << rays(0, 1) << "," << rays(1, 1) << "\n";
        ofs.close();
        std::cout << "[INFO] Rays written to: " << fname << "\n";
    }

    inline void run_example_2d()
    {
        Eigen::Matrix2d A(2, 2);
        A << 0.0, 1.0,
            -0.1, -0.7;
        Eigen::Vector2d b(2);
        b << 0.0, 1.5;

        detail::Settings s = detail::Settings();
        s.T = 100000000.0;

        detail::CurveResult cr = detail::generate_curve(A, b, s);
        if (cr.t.empty())
        {
            std::cerr << "[ERROR] No samples generated.\n";
            return;
        }

        auto rays = detail::pick_two_support_rays_2d(cr);
        write_curve_csv("out", cr);
        write_rays_csv("out", rays);

        // auto tr = detail::two_rays_by_angle_2d(cr, A);
        //
        // if (tr.idxs[0] == -1)
        // {
        //     THROW_RUNTIME_ERROR(std::format("[ERROR]: {}", tr.info));
        // }
        // else
        // {
        //     std::cout << "[INFO]" << " (" << tr.info << ") " << "Two rays chosen: indices " << tr.idxs[0] << ", " << tr.
        //         idxs[1] << ", arc (rad)=" << tr.
        //         arc_len << "\n";
        //     write_rays_csv("out", tr);
        // }
        //
        // if (const auto trv = tr.value(); !trv.ok)
        // {
        //     THROW_RUNTIME_ERROR(
        //         std::format("[ERROR] Two rays insufficient: angular span = {} rad ({}°) (> pi).", trv.arc_len, mc::
        //             rad2deg(trv.arc_len)));
        // }
        // else
        // {
        //     std::cout << "[INFO] Two rays chosen: indices " << trv.idx1 << ", " << trv.idx2 << ", arc (rad)=" << trv.
        //         arc_len << "\n";
        //     write_rays_csv("out", trv);
        // }

        std::cout << "[DONE] Outputs written with prefix '" << "out" << "'.\n";
        return;
    }

    struct GenerationCurve
    {
        Eigen::Matrix2d m_A;
        Eigen::Vector2d m_b;
        detail::Settings settings;
    };

    /**
     * @class SolidCone2d
     * @brief A class that describes a solid cone in n-dimensional Euclidean space.
     *
     * A cone is defined by a set of boundary rays.
     * It can be built in two ways:
     * 1. By explicitly specifying the boundary rays.
     * 2. By approximating it using the generating curve exp(A * t), where A is the Hurwitz matrix.
     */
    class SolidCone2d : public RefCounted
    {
    public:
        enum class ConstructionType { FROM_RAYS, FROM_CURVE };

        virtual ~SolidCone2d() override = default;

        /**
        * @brief Construct a cone from a set of boundary rays.
        * @param rays A matrix whose columns are the vectors of the boundary rays.
        */
        static Ref<SolidCone2d> fromRays(const Eigen::Matrix2d &rays)
        {
            AL_PROFILE_FUNC("SolidCone2d::fromRays");
            SolidCone2d *c = new SolidCone2d();
            c->m_Dimension = 2;
            c->m_BoundaryRays = rays;
            c->m_GeneratingMatrix = std::nullopt;
            c->m_ConstructionType = ConstructionType::FROM_RAYS;

            // Проверка на полноразмерность: ядро дополняется
            if (const Eigen::FullPivLU<Eigen::Matrix2d> lu(c->m_BoundaryRays); lu.rank() < 2)
            {
                // Если не полноразмерные генераторы, конус не содержит шар -> не телесный
                THROW_INVALID_ARGUMENT_ERROR("given rays do not span R^2; cone is not solid");
            }

            for (int j = 0; j < c->m_BoundaryRays.cols(); ++j)
            {
                const double nm = c->m_BoundaryRays.col(j).norm();
                if (nm <= 1e-9)
                {
                    THROW_INVALID_ARGUMENT_ERROR(std::format("zero ray provided at {}", j));
                }
                c->m_BoundaryRays.col(j) /= nm;
            }

            c->m_Rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

            return Ref(c);
        }

        /**
         * @brief Конструктор конуса из матрицы Гурвица.
         *
         * В этом методе мы аппроксимируем границу конуса. Граница конуса
         * строится как выпуклая оболочка касательных к образующим кривым вида
         * x(t) = exp(A*t)*x0. Касательная в точке t равна A*exp(A*t)*x0.
         *
         * @param A Гурвицева матрица.
         * @param b вектор
         * @param s настройки генерации конуса
         * @param t "Большое" время для матричной экспоненты.
         */
        static Ref<SolidCone2d> fromCurve(const Eigen::Matrix2d &A, const Eigen::Vector2d &b, const detail::Settings &s)
        {
            AL_PROFILE_FUNC("SolidCone2d::fromCurve");
            SolidCone2d *c = new SolidCone2d();
            c->m_Dimension = 2;
            c->m_GeneratingMatrix = GenerationCurve(A, b, s);
            c->m_ConstructionType = ConstructionType::FROM_CURVE;

            const detail::CurveResult cr = detail::generate_curve(A, b, s);
            write_curve_csv("out", cr);
            // if (cr.t.empty())
            // {
            //     THROW_RUNTIME_ERROR("[ERROR] No samples generated.");
            // }
            //
            // if (auto tr = two_rays_by_angle_2d(cr); !tr.has_value())
            // {
            //     THROW_RUNTIME_ERROR("[ERROR] Cannot chose rays.");
            // }
            // else if (const auto trv = tr.value(); !trv.ok)
            // {
            //     THROW_RUNTIME_ERROR(
            //         std::format("[ERROR] Two rays insufficient: angular span = {} rad (> pi).", trv.arc_len));
            // }
            // else
            // {
            //     std::cout << "[INFO] Two rays chosen: indices " << trv.idx1 << ", " << trv.idx2 << ", arc (rad)=" << trv
            //         .
            //         arc_len << "\n";
            //     c.m_BoundaryRays.col(0) = trv.ray1;
            //     c.m_BoundaryRays.col(1) = trv.ray2;
            // }

            auto rays = detail::pick_two_support_rays_2d(cr);
            c->m_BoundaryRays = rays;
            write_rays_csv("out", rays);

            c->m_Rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

            return Ref(c);
        }

        // Проверка принадлежности точки конусу (внутри или на границе)
        bool contains(const Eigen::Vector2d &x, double tol = 1e-8) const
        {
            AL_PROFILE_FUNC("SolidCone2d::contains");
            Eigen::Vector2d coeffs = m_BoundaryRays.inverse() * x;
            return coeffs[0] >= tol && coeffs[1] >= tol;
        }

        /// Генерация случайной внутренней точки (не на границе)
        Eigen::Vector2d sampleInterior(double minCoeff = 0.1, double maxCoeff = 1.0)
        {
            AL_PROFILE_FUNC("SolidCone2d::sampleInterior");
            // thread_local std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<double> dist(minCoeff, maxCoeff);

            double a = dist(m_Rng);
            double b = dist(m_Rng);
            return a * m_BoundaryRays.col(0) + b * m_BoundaryRays.col(1);
        }

        nlohmann::json ToJson() const
        {
            nlohmann::json j;
            j["dimension"] = m_Dimension;
            j["rays_count"] = 2;
            j["rays"] = nlohmann::json::array();
            for (int jcol = 0; jcol < m_BoundaryRays.cols(); ++jcol)
            {
                std::vector<double> v(m_Dimension);
                for (int i = 0; i < m_Dimension; ++i)
                    v[i] = m_BoundaryRays(i, jcol);
                j["rays"].push_back(v);
            }
            if (m_GeneratingMatrix.has_value())
            {
                const auto curve = m_GeneratingMatrix.value();
                j["curve_A"] = nlohmann::json::array();
                for (int i = 0; i < curve.m_A.rows(); ++i)
                {
                    std::vector<double> row(curve.m_A.cols());
                    for (int k = 0; k < curve.m_A.cols(); ++k)
                        row[k] = curve.m_A(i, k);
                    j["curve_A"].push_back(row);
                }
                j["curve_b"] = curve.m_b;
                j["curve_settings"] = curve.settings;
            }
            return j;
        }

        static Ref<SolidCone2d> FromJson(const nlohmann::json &j)
        {
            int n = j.at("dimension").get<int>();

            int m = j.at("rays_count").get<int>();
            Eigen::MatrixXd R(n, m);
            auto arr = j.at("rays");
            for (int col = 0; col < m; ++col)
            {
                auto v = arr.at(col).get<std::vector<double>>();
                if (static_cast<int>(v.size()) != n)
                    THROW_INVALID_ARGUMENT_ERROR("rays size mismatch");
                for (int i = 0; i < n; ++i)
                    R(i, col) = v[i];
            }
            auto c = fromRays(R);

            if (j.contains("curve_A"))
            {
                auto Aarr = j.at("curve_A").get<std::vector<std::vector<double>>>();
                Eigen::MatrixXd A(n, n);
                for (int i = 0; i < n; ++i)
                {
                    for (int k = 0; k < n; ++k)
                    {
                        A(i, k) = Aarr[i][k];
                    }
                }
                Eigen::Vector2d b;
                Eigen::from_json(j["curve_b"], b);
                detail::Settings s = j["curve_settings"];
                c->m_GeneratingMatrix = GenerationCurve(A, b, s);
                c->m_ConstructionType = ConstructionType::FROM_CURVE;
            }

            return c;
        }

    private:
        SolidCone2d() :
            m_Dimension(0), m_ConstructionType(ConstructionType::FROM_RAYS)
        {
        }

        // Оценка максимального собственного значения R^T R (power method) => для шага градиента
        // double estimate_max_eig_RtR(int max_iters = 200, double tol = 1e-9) const
        // {
        //     Eigen::VectorXd b = Eigen::VectorXd::Random(2);
        //     b.normalize();
        //     Eigen::VectorXd y(2);
        //     for (int it = 0; it < max_iters; ++it)
        //     {
        //         y = m_BoundaryRays.transpose() * (m_BoundaryRays * b);
        //         double norm_y = y.norm();
        //         if (norm_y == 0)
        //             return 0.0;
        //         Eigen::VectorXd b_next = y / norm_y;
        //         if ((b_next - b).norm() < tol)
        //         {
        //             return norm_y;
        //         }
        //         b = b_next;
        //     }
        //     // Rayleigh quotient
        //     const Eigen::VectorXd Rb = m_BoundaryRays * b;
        //     return b.dot(m_BoundaryRays.transpose() * Rb);
        // }

        // Решение NNLS: min ||R c - x||^2 s.t. c >= 0
        // Возвращает optional вектора коэффициентов (если не сошлось, optional пуст).
        // std::optional<Eigen::VectorXd> solve_nnls_projected_gradient(const Eigen::VectorXd &x) const
        // {
        //     int m = m_BoundaryRays.cols();
        //     if (m == 0)
        //         return std::nullopt;
        //     // Начальная оценка: R^T x, обрезаем
        //     Eigen::VectorXd c = m_BoundaryRays.transpose() * x;
        //     for (int i = 0; i < m; ++i)
        //         if (c(i) < 0)
        //             c(i) = 0;
        //     // Step size
        //     double L = estimate_max_eig_RtR();
        //     if (L <= 0)
        //         L = 1.0;
        //     double alpha = 1.0 / L;
        //     Eigen::VectorXd grad(m);
        //     for (int iter = 0; iter < m_NNLS_MaxIters; ++iter)
        //     {
        //         Eigen::VectorXd Rc = m_BoundaryRays * c;
        //         grad = m_BoundaryRays.transpose() * (Rc - x); // gradient of 0.5||R c - x||^2
        //         Eigen::VectorXd c_next = c - alpha * grad;
        //         // projection onto nonnegatives
        //         for (int i = 0; i < m; ++i)
        //         {
        //             if (c_next(i) < 0)
        //             {
        //                 c_next(i) = 0;
        //             }
        //         }
        //         double diff = (c_next - c).norm();
        //         c.swap(c_next);
        //         if (diff < 1e-12)
        //             break;
        //         // early exit if residual small
        //         double res = (m_BoundaryRays * c - x).norm();
        //         if (res <= m_NNLS_Tol)
        //             return c;
        //     }
        //     double final_res = (m_BoundaryRays * c - x).norm();
        //     if (final_res <= 1e-6)
        //         return c; // tolerant
        //     return std::nullopt;
        // }

    private:
        uint32_t m_Dimension;
        Eigen::Matrix2d m_BoundaryRays;
        ConstructionType m_ConstructionType;
        std::optional<GenerationCurve> m_GeneratingMatrix;

        std::mt19937_64 m_Rng;
    };

    struct CSVWriter
    {
        std::ofstream file;

        CSVWriter(const std::string &filename)
        {
            file.open(filename, std::ios::trunc);
            file << std::fixed << std::setprecision(8);
        }

        void writePoint(int step, const Eigen::Vector2d &v)
        {
            file << step << "," << v.x() << "," << v.y() << "\n";
        }
    };

    struct ShuttlePointResult
    {
        bool converged = false;
        std::vector<Eigen::Vector2d> limits = {}; // либо {A}, либо {A,B}
    };

    inline ShuttlePointResult ShuttlePoint(Ref<ode::DynamicalSystem> system_minus, Ref<ode::DynamicalSystem> system_plus, const Ref<SolidCone2d> &cone,
                                           const Eigen::Vector2d &u0, const Eigen::Vector2d &z_minus,
                                           const Eigen::Vector2d &z_plus, double period)
    {
        AL_PROFILE_FUNC("ShuttlePoint");

        CSVWriter trace_even("trace_even.csv");
        CSVWriter trace_odd("trace_odd.csv");
        CSVWriter trace_all("trace_all.csv");

        mc::json::JsonDocument message({"name", "z+", "z-"});
        
        std::println("z-: {}, z+: {}", z_minus, z_plus);

        // const Eigen::Vector2d z_minus = -40.0 * u0;
        // const Eigen::Vector2d z_plus = 40.0 * u0;

        const Eigen::Vector2d z_minus_new = system_minus->Shift(z_minus, period) - z_minus - u0;
        const Eigen::Vector2d z_plus_new = z_plus - system_plus->Shift(z_plus, period) - u0;

        bool cond_z_minus = cone->contains(z_minus_new);
        bool cond_z_plus = cone->contains(z_plus_new);

        // Проверка условий для z- и z+
        if (!cond_z_minus)
        {
            THROW_RUNTIME_ERROR("[ERROR] z- does not satisfy cone condition.");
        }
        if (!cond_z_plus)
        {
            THROW_RUNTIME_ERROR("[WARN] z+ does not satisfy cone condition.");
        }

        // 3) Последовательность sigma_n
        auto sigma_n = [](int n)
        {
            double v = 1.0 / std::pow(2.0, n);
            std::println("sigma_{}: {}", n, v);
            return v;
        };

        // Настройки итераций
        constexpr int maxIter = 20;
        constexpr int maxYIter = 1000;

        std::vector<Eigen::Vector2d> z_odd;
        z_odd.push_back(z_minus);
        std::vector<Eigen::Vector2d> z_even;
        z_even.push_back(z_plus);

        // Вспомогательные последовательности
        auto build_sequence = [maxYIter, &system_minus, &system_plus, period, sigma_n, u0, &cone](
            const Eigen::Vector2d &y0, bool add, int sig_idx, CSVWriter &trace) -> std::optional<Eigen::Vector2d>
        {
            AL_PROFILE_FUNC("ShuttlePoint::build_sequence");
            Eigen::Vector2d y_prev = y0;
            std::println("y{}_0: {}", sig_idx, y_prev);
            for (int n = 1; n < maxYIter; ++n)
            {
                Eigen::Vector2d y_next;
                if (add)
                {
                    y_next = system_minus->Shift(y_prev, period);
                    y_next += sigma_n(sig_idx) * u0;
                }
                else
                {
                    y_next = system_plus->Shift(y_prev, period);
                    y_next -= sigma_n(sig_idx) * u0;
                }

                std::println("y{}_{}: {}", sig_idx, n, y_next);

                trace.writePoint(n, y_next);

                // проверка условий выбора z
                if (add)
                {
                    if (cone->contains(y_prev - y_next - sigma_n(sig_idx + 2) * u0))
                    {
                        std::println("y{}_{} that satisfies condition: {}", sig_idx, n, y_next);
                        return y_next;
                    }
                }
                else
                {
                    if (cone->contains(y_next - y_prev - sigma_n(sig_idx + 2) * u0))
                    {
                        std::println("y{}_{} that satisfies condition: {}", sig_idx, n, y_next);
                        return y_next;
                    }
                }

                y_prev = y_next;
            }
            return std::nullopt;
        };

        // Основной цикл построения z_n
        ShuttlePointResult result;
        {
            AL_PROFILE_FUNC("ShuttlePoint::<z_odd, z_event search>");
            bool stop = false;
            for (int iter = 1; iter < maxIter + 1 && !stop; ++iter)
            {
                if (iter % 2 != 0) // нечётный шаг, строим "возрастающую"
                {
                    system_minus->CallResetFn();
                    std::println("z_odd back: {}", z_odd.back());
                    auto z1 = build_sequence(z_odd.back(), true, iter, trace_odd);
                    if (z1.has_value())
                    {
                        std::println("z_odd new: {}", *z1);
                        z_odd.push_back(z1.value());
                        trace_all.writePoint(iter, *z1);
                    }
                    else
                    {
                        std::cerr << std::format("[WARN]: the odd sequence y{}_n has no z element", iter);
                        break;
                    }
                }
                else // чётный шаг, строим "убывающую"
                {
                    system_plus->CallResetFn();
                    std::println("z_even back: {}", z_even.back());
                    auto z2 = build_sequence(z_even.back(), false, iter, trace_even);
                    if (z2.has_value())
                    {
                        std::println("z_even new: {}", *z2);
                        z_even.push_back(z2.value());
                        trace_all.writePoint(iter, *z2);
                    }
                    else
                    {
                        std::cerr << std::format("[WARN]: the even sequence y{}_n has no z element", iter);
                        break;
                    }
                }

                // Проверка сходимости
                // if (!z_even.empty() && !z_odd.empty())
                // {
                //     const Eigen::Vector2d &a = z_even.back();
                //     const Eigen::Vector2d &b = z_odd.back();
                //
                //     double dist = (a - b).norm();
                //     if (dist < tol)
                //     {
                //         result.converged = true;
                //         result.limits = {0.5 * (a + b)};
                //         stop = true;
                //     }
                //     else if (z_even.size() > 2 && z_odd.size() > 2)
                //     {
                //         double d1 = (z_even.back() - z_even[z_even.size() - 2]).norm();
                //         double d2 = (z_odd.back() - z_odd[z_odd.size() - 2]).norm();
                //         if (d1 < tol && d2 < tol)
                //         {
                //             result.converged = true;
                //             result.limits = {a, b};
                //             stop = true;
                //         }
                //     }
                // }
            }
        }

        result.converged = true;
        result.limits = {z_odd.back(), z_even.back()};

        if (!result.converged)
            std::cerr << "[WARN] ShuttlePoint did not converge.\n";

        return result;
    }
}
