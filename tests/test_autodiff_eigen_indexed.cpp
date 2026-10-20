#include <autodiff/forward/dual/eigen.hpp>
#include <array>
#include <cmath>
#include <iostream>

template<typename View>
bool check_gradient(View view)
{
    static_assert(autodiff::detail::isVector<View>);
    const auto squared_norm = [](const auto& x) -> autodiff::dual {
        return x.squaredNorm();
    };
    autodiff::dual value;
    Eigen::VectorXd gradient;
    // The explicit-output overload also works without AUTODIFF_EIGEN_FOUND.
    autodiff::gradient(
        squared_norm, autodiff::wrt(view), autodiff::at(view), value, gradient);
    for (Eigen::Index i = 0; i < view.size(); ++i) {
        if (!(std::abs(gradient[i] - 2.0 * autodiff::val(view(i))) <= 1e-12))
            return false;
    }
    return true;
}

int main()
{
    Eigen::Matrix<autodiff::dual, 3, 3> matrix;
    matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    const std::array<int, 2> indices{0, 2};
    // Exercise both indexed-view orientations supported by autodiff v1.1.2.
    if (!check_gradient(matrix(indices, 1)) ||
        !check_gradient(matrix(1, indices))) {
        std::cerr << "Indexed-view gradient mismatch\n";
        return 1;
    }
    std::cout << "Indexed-view gradients passed (tolerance 1e-12)\n";
}
