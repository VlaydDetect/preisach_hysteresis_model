// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once
#include <vector>
#include <random>

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
            CurveResult res;
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

        struct TwoRaysResult
        {
            bool ok;
            Eigen::Vector2d ray1;
            Eigen::Vector2d ray2;
            double arc_len;
            int idx1;
            int idx2;
        };

        inline double wrap_to_2pi(double ang)
        {
            // результат в [0, 2pi)
            double x = std::fmod(ang, 2.0 * consts::pi);
            if (x < 0)
            {
                x += 2.0 * consts::pi;
            }
            return x;
        }

        inline std::optional<TwoRaysResult> two_rays_by_angle_2d(const CurveResult &cr, double eps = 1e-12)
        {
            const auto &pts = cr.pts;
            const int m = static_cast<int>(pts.size());
            if (m < 2)
            {
                return std::nullopt;
            }

            std::vector<double> ang(m);
            std::vector<double> norms(m);
            for (int i = 0; i < m; ++i)
            {
                norms[i] = pts[i].norm();
                if (norms[i] < eps)
                {
                    std::cerr << "[WARN] Point " << i << " is too close to origin (norm=" << norms[i] << ").\n";
                    return std::nullopt;
                }
                ang[i] = std::atan2(pts[i](1), pts[i](0)); // (-pi, pi]
            }


            // sort angles
            std::vector<std::pair<double, int>> arr(m);
            for (int i = 0; i < m; ++i)
            {
                arr[i] = {ang[i], i};
            }
            std::ranges::sort(arr, [](auto &a, auto &b) { return a.first < b.first; });


            // compute gaps (including wrap-around)
            double largest_gap = -1.0;
            int largest_idx = -1; // index i such that gap between arr[i] and arr[i+1]
            for (int i = 0; i < m; ++i)
            {
                double a1 = arr[i].first;
                double a2;
                if (i + 1 == m)
                {
                    // last to first wrap
                    a2 = arr[0].first + 2.0 * consts::pi;
                }
                else
                {
                    a2 = arr[i + 1].first;
                }
                double gap = a2 - a1;
                if (gap > largest_gap)
                {
                    largest_gap = gap;
                    largest_idx = i;
                }
            }


            double min_cover_arc = 2.0 * consts::pi - largest_gap;
            // endpoints of minimal arc are arr[largest_idx+1] .. arr[largest_idx]
            double start = arr[(largest_idx + 1) % m].first;
            double end = arr[largest_idx].first;
            if (end < start)
            {
                end += 2.0 * consts::pi;
            }
            double arc_len = end - start;


            // numerical safety
            if (arc_len < 0)
            {
                arc_len = min_cover_arc; // fallback
            }


            if (arc_len > consts::pi + 1e-12)
            {
                TwoRaysResult out;
                out.ok = false;
                out.arc_len = arc_len;
                out.idx1 = -1;
                out.idx2 = -1;
                return out;
            }


            // find original indices closest to start and to start+arc_len
            double start_mod = start; // in (-pi, 3pi), but shifts handled below
            double target1 = start_mod;
            double target2 = start_mod + arc_len;

            int best1 = -1, best2 = -1;
            double bestd1 = 1e300, bestd2 = 1e300;
            for (int i = 0; i < m; ++i)
            {
                // normalize ang[i] to be comparable with start_mod
                double a = ang[i];
                // try shifts of +-2pi to get near targets
                double candidates[3] = {a, a + 2.0 * consts::pi, a - 2.0 * consts::pi};
                for (double ac : candidates)
                {
                    double d1 = std::fabs(ac - target1);
                    double d2 = std::fabs(ac - target2);
                    if (d1 < bestd1)
                    {
                        bestd1 = d1;
                        best1 = i;
                    }
                    if (d2 < bestd2)
                    {
                        bestd2 = d2;
                        best2 = i;
                    }
                }
            }

            TwoRaysResult out;
            out.ok = true;
            out.idx1 = best1;
            out.idx2 = best2;
            out.ray1 = pts[best1];
            out.ray2 = pts[best2];
            out.arc_len = arc_len;
            return out;
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


    inline void write_rays_csv(const std::string &prefix, const detail::TwoRaysResult &r)
    {
        std::string fname = prefix + "_rays.csv";
        std::ofstream ofs(fname);
        ofs << std::setprecision(12);
        ofs << "idx,x,y\n";
        ofs << r.idx1 << "," << r.ray1(0) << "," << r.ray1(1) << "\n";
        ofs << r.idx2 << "," << r.ray2(0) << "," << r.ray2(1) << "\n";
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

        write_curve_csv("out", cr);

        if (auto tr = two_rays_by_angle_2d(cr); !tr.has_value())
        {
            THROW_RUNTIME_ERROR("[ERROR] Cannot chose rays.");
        }
        else if (const auto trv = tr.value(); !trv.ok)
        {
            THROW_RUNTIME_ERROR(
                std::format("[ERROR] Two rays insufficient: angular span = {} rad ({}°) (> pi).", trv.arc_len, mc::
                    rad2deg(trv.arc_len)));
        }
        else
        {
            std::cout << "[INFO] Two rays chosen: indices " << trv.idx1 << ", " << trv.idx2 << ", arc (rad)=" << trv.
                arc_len << "\n";
            write_rays_csv("out", trv);
        }

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
            SolidCone2d c;
            c.m_Dimension = 2;
            c.m_BoundaryRays = rays;
            c.m_GeneratingMatrix = std::nullopt;
            c.m_ConstructionType = ConstructionType::FROM_RAYS;

            // Проверка на полноразмерность: ядро дополняется
            if (const Eigen::FullPivLU<Eigen::Matrix2d> lu(c.m_BoundaryRays); lu.rank() < 2)
            {
                // Если не полноразмерные генераторы, конус не содержит шар -> не телесный
                THROW_INVALID_ARGUMENT_ERROR("given rays do not span R^2; cone is not solid");
            }

            for (int j = 0; j < c.m_BoundaryRays.cols(); ++j)
            {
                const double nm = c.m_BoundaryRays.col(j).norm();
                if (nm <= 1e-9)
                {
                    THROW_INVALID_ARGUMENT_ERROR(std::format("zero ray provided at {}", j));
                }
                c.m_BoundaryRays.col(j) /= nm;
            }

            c.m_Rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

            return Ref(&c);
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
            SolidCone2d c;
            c.m_Dimension = 2;
            c.m_GeneratingMatrix = GenerationCurve(A, b, s);
            c.m_ConstructionType = ConstructionType::FROM_CURVE;

            const detail::CurveResult cr = detail::generate_curve(A, b, s);
            if (cr.t.empty())
            {
                THROW_RUNTIME_ERROR("[ERROR] No samples generated.");
            }

            if (auto tr = two_rays_by_angle_2d(cr); !tr.has_value())
            {
                THROW_RUNTIME_ERROR("[ERROR] Cannot chose rays.");
            }
            else if (const auto trv = tr.value(); !trv.ok)
            {
                THROW_RUNTIME_ERROR(
                    std::format("[ERROR] Two rays insufficient: angular span = {} rad (> pi).", trv.arc_len));
            }
            else
            {
                std::cout << "[INFO] Two rays chosen: indices " << trv.idx1 << ", " << trv.idx2 << ", arc (rad)=" << trv
                    .
                    arc_len << "\n";
                c.m_BoundaryRays.col(0) = trv.ray1;
                c.m_BoundaryRays.col(1) = trv.ray2;
            }

            c.m_Rng.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());

            return Ref(&c);
        }

        // Проверка принадлежности точки x к замыканию конуса (closure). Возвращает pair{belongs, coefficients}
        // std::pair<bool, Eigen::VectorXd> ContainsWithCoeffs(const Eigen::VectorXd &x, double tol = 1e-8) const
        // {
        //     if (x.size() != m_Dimension)
        //         throw std::invalid_argument("Dimension mismatch in contains");
        //     const auto opt = solve_nnls_projected_gradient(x);
        //     if (!opt.has_value())
        //     {
        //         return {false, Eigen::VectorXd()};
        //     }
        //     Eigen::VectorXd c = *opt;
        //     const double res = (m_BoundaryRays * c - x).norm();
        //     bool ok = res <= std::max(tol, 1e-8);
        //     return {ok, c};
        // }

        // true если точка находится во внутренности конуса (не на границе).
        // Критерий: есть представление x = R*c, с c_i >= interior_eps (для как минимум n линейно независимых положительных коэффициентов)
        // bool ContainsInterior(const Eigen::VectorXd &x, double tol = 1e-8) const
        // {
        //     auto [ok, c] = ContainsWithCoeffs(x, tol);
        //     if (!ok)
        //         return false;
        //     // Считаем внутренней, если существуют >= n линейно независимых положительных коэффициентов,
        //     // но для практичности: требуем, чтобы хотя бы n коэффициентов > interior_eps и R(selection) имела ранк n.
        //     std::vector<int> idx;
        //     for (int i = 0; i < c.size(); ++i)
        //     {
        //         if (c(i) > m_InteriorEps)
        //         {
        //             idx.push_back(i);
        //         }
        //     }
        //     if (static_cast<int>(idx.size()) < m_Dimension)
        //         return false;
        //     Eigen::MatrixXd M(m_Dimension, static_cast<int>(idx.size()));
        //     for (int j = 0; j < static_cast<int>(idx.size()); ++j)
        //     {
        //         M.col(j) = m_BoundaryRays.col(idx[j]);
        //     }
        //     // ранг
        //     Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
        //     return lu.rank() == m_Dimension;
        // }

        // Проверка принадлежности точки конусу (внутри или на границе)
        bool contains(const Eigen::Vector2d &x, double tol = 1e-8) const
        {
            // return ContainsWithCoeffs(x, tol).first;
            if (x.isZero())
                return true;
            Eigen::Vector2d coeffs = m_BoundaryRays.inverse() * x;
            return coeffs[0] >= tol && coeffs[1] >= tol;
        }

        /// Генерация случайной внутренней точки (не на границе)
        Eigen::Vector2d sampleInterior(double minCoeff = 0.1, double maxCoeff = 1.0)
        {
            // thread_local std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<double> dist(minCoeff, maxCoeff);

            double a = dist(m_Rng);
            double b = dist(m_Rng);
            return a * m_BoundaryRays.col(0) + b * m_BoundaryRays.col(1);
        }

        // // Сэмплинг случайной внутренней точки.
        // // Стратегия: берем m = R.cols() > = n. Генерируем положительные коэффициенты exp(1) (или uniform>0),
        // // обеспечивающие положительность; если полученная точка не интериорная, повторяем попытки.
        // Eigen::VectorXd sample_interior_point(double scale = 1.0, int max_tries = 200)
        // {
        //     std::exponential_distribution expd(1.0);
        //     int m = m_BoundaryRays.cols();
        //     if (m == 0)
        //         throw std::runtime_error("No rays to sample from");
        //     for (int attempt = 0; attempt < max_tries; ++attempt)
        //     {
        //         Eigen::VectorXd coeffs(m);
        //         for (int i = 0; i < m; ++i)
        //         {
        //             // ensure strictly positive
        //             double v = expd(m_Rng);
        //             coeffs(i) = std::max(v, 1e-6);
        //         }
        //         Eigen::VectorXd x = m_BoundaryRays * coeffs * scale;
        //         if (ContainsInterior(x))
        //             return x;
        //     }
        //     // fallback: take mean of rays (strictly positive combination)
        //     Eigen::VectorXd coeffs = Eigen::VectorXd::Ones(m) * 0.5;
        //     Eigen::VectorXd x = m_BoundaryRays * coeffs * scale;
        //     return x;
        // }

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

        double m_NNLS_Tol = 1e-8;
        int m_NNLS_MaxIters = 2000;
        double m_InteriorEps = 1e-9; // минимальное значение коэффициента для внутренней точки

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

    inline ShuttlePointResult ShuttlePoint(Ref<ode::DynamicalSystem> system, Ref<SolidCone2d> cone,
                                           const Eigen::Vector2d &u0, double period)
    {
        CSVWriter trace_even("trace_even.csv");
        CSVWriter trace_odd("trace_odd.csv");
        CSVWriter trace_all("trace_all.csv");

        // 1) Поиск начальных точек z- и z+
        // Простая эвристика: берём две случайные точки в конусе, увеличиваем их норму
        Eigen::Vector2d z_minus = -10.0 * u0;
        Eigen::Vector2d z_plus = 10.0 * u0;

        bool cond_z_minus = cone->contains(system->Shift(z_minus, period) - z_minus);
        bool cond_z_plus = cone->contains(z_plus - system->Shift(z_plus, period));

        // Проверка условий для z- и z+
        if (cond_z_minus)
        {
            THROW_RUNTIME_ERROR("[ERROR] z- does not satisfy cone condition.");
        }
        if (!cond_z_plus)
        {
            THROW_RUNTIME_ERROR("[WARN] z+ does not satisfy cone condition.");
        }

        // 3) Последовательность sigma_n
        auto sigma_n = [](int n) { return 1.0 / std::pow(2.0, n); };

        // Настройки итераций
        constexpr int maxIter = 20;
        constexpr double tol = 1e-6;

        std::vector<Eigen::Vector2d> z_odd;
        z_odd.push_back(z_minus);
        std::vector<Eigen::Vector2d> z_even;
        z_even.push_back(z_plus);

        // Вспомогательные последовательности
        auto build_sequence = [maxIter, &system, period, sigma_n, u0, &cone](
            const Eigen::Vector2d &y0, bool add, int sig_idx, CSVWriter &trace) -> std::optional<Eigen::Vector2d>
        {
            Eigen::Vector2d y_prev = y0;
            for (int n = 1; n < maxIter; ++n)
            {
                Eigen::Vector2d y_next = system->Shift(y_prev, period);
                if (add)
                {
                    y_next += sigma_n(sig_idx) * u0;
                }
                else
                {
                    y_next -= sigma_n(sig_idx) * u0;
                }

                trace.writePoint(n, y_next);

                // проверка условий выбора z
                if (add)
                {
                    if (cone->contains(y_prev - y_next - sigma_n(sig_idx + 2) * u0))
                        return y_next;
                }
                else
                {
                    if (cone->contains(y_next - y_prev - sigma_n(sig_idx + 2) * u0))
                        return y_next;
                }

                y_prev = y_next;
            }
            return std::nullopt;
        };

        // Основной цикл построения z_n
        bool stop = false;
        ShuttlePointResult result;
        for (int iter = 1; iter < maxIter + 1 && !stop; ++iter)
        {
            if (iter % 2 != 0) // нечётный шаг, строим "возрастающую"
            {
                auto z1 = build_sequence(z_odd.back(), true, iter, trace_odd);
                if (z1.has_value())
                {
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
                auto z2 = build_sequence(z_even.back(), false, iter, trace_even);
                if (z2.has_value())
                {
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

        result.converged = true;
        result.limits = {z_odd.back(), z_even.back()};

        if (!result.converged)
            std::cerr << "[WARN] ShuttlePoint did not converge.\n";

        return result;
    }
}
