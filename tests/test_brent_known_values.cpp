#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../src/disper26.cpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

template <typename T, typename Func>
T legacy_brents_method(Func func, T min, T max, T tol = 1e-6,
                       int max_iter = 100) {
    T a = min;
    T b = max;
    T fa = func(a);
    T fb = func(b);
    if (fa * fb > 0) return T(-1.0);
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }
    T c = a;
    T fc = fa;
    bool mflag = true;
    T s = 0;
    T fs = 0;
    T d = 0;
    for (int i = 0; i < max_iter; ++i) {
        if (std::abs(b - a) < tol) return b;
        if (std::abs(fb) < 1e-12) return b;
        if (fa != fc && fb != fc) {
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                (c * fa * fb) / ((fc - fa) * (fc - fb));
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }
        const T lower_bound = std::min((3 * a + b) / 4, b);
        const T upper_bound = std::max((3 * a + b) / 4, b);
        const bool s_outside = s < lower_bound || s > upper_bound;
        if (s_outside ||
            (mflag && std::abs(s - b) >= std::abs(b - c) / 2) ||
            (!mflag && std::abs(s - b) >= std::abs(c - d) / 2) ||
            (mflag && std::abs(b - c) < tol) ||
            (!mflag && std::abs(c - d) < tol)) {
            s = (a + b) / 2;
            mflag = true;
        } else {
            mflag = false;
        }
        fs = func(s);
        d = c;
        c = b;
        fc = fb;
        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }
    }
    return b;
}

template <typename Func>
std::vector<double> legacy_search_roots(Func func, double min_c, double max_c,
                                        double step_c, double tol) {
    std::vector<double> roots;
    double c1 = min_c;
    double f1 = func(c1);
    for (double c2 = min_c + step_c; c2 <= max_c; c2 += step_c) {
        double f2 = func(c2);
        if (!std::isnan(f1) && !std::isnan(f2) && f1 * f2 < 0.0) {
            const double root = legacy_brents_method(func, c1, c2, tol);
            if (root > 0.0 && std::abs(func(root)) < 0.1) roots.push_back(root);
        }
        c1 = c2;
        f1 = f2;
    }
    if (std::abs(f1) < 1e-12 &&
        (roots.empty() || std::abs(roots.back() - c1) > 1e-4)) {
        roots.push_back(c1);
    }
    return roots;
}

}  // namespace

int main() {
    const auto mesh_node_root = searchRoots([](double x) {return x - 1.75;},
                                          1., 1.8, .25);
    require(mesh_node_root == std::vector<double>{1.75}, "Lost exact last interior root");
    const auto polynomial = [](double x) {
        return (x - 1.23) * (x - 1.87);
    };

    int legacy_count = 0;
    const auto legacy_func = [&](double x) {
        ++legacy_count;
        return polynomial(x);
    };
    const auto legacy_roots = legacy_search_roots(
        legacy_func, 0.5, 2.5, 0.25, 1e-10);

    int optimized_count = 0;
    const auto optimized_func = [&](double x) {
        ++optimized_count;
        return polynomial(x);
    };
    const auto optimized_roots = searchRoots(
        optimized_func, 0.5, 2.5, 0.25, 1e-10);

    require(optimized_roots == legacy_roots,
            "searchRoots changed the returned roots");
    require(optimized_roots.size() == 2, "expected two accepted brackets");
    require(legacy_count - optimized_count ==
                3 * static_cast<int>(optimized_roots.size()),
            "searchRoots did not save three evaluations per accepted bracket");

    int legacy_brent_count = 0;
    const auto counted_legacy = [&](double x) {
        ++legacy_brent_count;
        return polynomial(x);
    };
    const double legacy_root = legacy_brents_method(
        counted_legacy, 1.0, 1.5, 1e-12, 100);

    int compatible_count = 0;
    const auto counted_compatible = [&](double x) {
        ++compatible_count;
        return polynomial(x);
    };
    const double compatible_root = brents_method(
        counted_compatible, 1.0, 1.5, 1e-12, 100);
    require(compatible_root == legacy_root,
            "legacy brents_method wrapper changed the root");
    require(compatible_count == legacy_brent_count,
            "legacy brents_method wrapper changed evaluation count");

    int known_count = 0;
    const auto counted_known = [&](double x) {
        ++known_count;
        return polynomial(x);
    };
    const double fa = counted_known(1.0);
    const double fb = counted_known(1.5);
    const auto result = brents_method_with_values(
        counted_known, 1.0, 1.5, fa, fb, 1e-12, 100);
    require(result.root == legacy_root, "known-value Brent changed the root");
    require(result.function_value == polynomial(result.root),
            "known-value Brent did not return final fb");
    require(legacy_brent_count - known_count == 0,
            "known-value Brent core changed iteration evaluations");

    const auto no_bracket = brents_method_with_values(
        polynomial, 0.0, 0.5, polynomial(0.0), polynomial(0.5), 1e-10, 100);
    require(no_bracket.root == -1.0,
            "known-value Brent changed the unbracketed sentinel");

    int nan_count = 0;
    const auto nan_func = [&](double x) {
        ++nan_count;
        return x >= 0.9 && x <= 1.1
            ? std::numeric_limits<double>::quiet_NaN()
            : x - 1.0;
    };
    const auto nan_roots = searchRoots(nan_func, 0.5, 1.5, 0.2, 1e-10);
    require(nan_roots.empty(), "searchRoots changed NaN-gap handling");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    require(brents_method_with_values(polynomial, 0., 1., nan, 0.01).root < 0.,
            "Nonfinite bracket endpoint accepted");
    require(brents_method_with_values([nan](double) { return nan; },
            0., 1., -0.01, 0.01).root < 0., "Nonfinite trial point accepted");
    const auto last = searchRoots([](double x) { return x - 4.499; },
                                  1., 4.5, 3.5 / 200., 1e-10);
    require(last.size() == 1 && std::abs(last.front() - 4.499) < 1e-10,
            "Root lost in final scan interval");

    const auto pole = [](double x) { return 1.0 / (x - 1.23); };
    require(searchRoots(pole, 0.5, 2.0, 0.25, 1e-10).empty(),
            "cached residual accepted a pole");
    const auto exhausted = brents_method_with_values(
        polynomial, 1.0, 1.5, polynomial(1.0), polynomial(1.5), 1e-12, 0);
    require(exhausted.root == legacy_brents_method(polynomial, 1.0, 1.5, 1e-12, 0) &&
            exhausted.function_value == polynomial(exhausted.root),
            "iteration limit changed the returned root/residual pair");

    std::cout << "Brent known-endpoint tests passed: roots="
              << optimized_roots.size() << " evaluations_saved="
              << legacy_count - optimized_count << '\n';
    return 0;
}
