/**
 * test_autodiff_basic.cpp
 * 
 * Minimal test to verify autodiff library integration works correctly
 * with Eigen and the Disper26 project.
 * 
 * Tests:
 * 1. Forward mode AD (autodiff::dual)
 * 2. Reverse mode AD (autodiff::var)
 * 3. Higher-order derivatives (autodiff::dual2nd)
 * 4. Eigen integration
 */

#include <iostream>
#include <cmath>
#include <iomanip>

// autodiff headers - include forward and reverse separately to avoid ambiguity
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

// --- Test 1: Forward Mode Basic ---
void test_forward_mode_basic() {
    std::cout << "=== Test 1: Forward Mode Basic ===" << std::endl;
    
    using namespace autodiff;
    
    dual x = 2.0;
    dual y = 3.0;
    
    // f(x, y) = x^2 * y + sin(x)
    auto f = [](dual x, dual y) -> dual {
        return x * x * y + sin(x);
    };
    
    // Compute df/dx at (2, 3) - use forward mode namespace explicitly
    double dfdx = autodiff::detail::derivative(f, autodiff::detail::wrt(x), autodiff::detail::at(x, y));
    // Analytical: df/dx = 2*x*y + cos(x) = 2*2*3 + cos(2) = 12 + cos(2)
    double expected = 12.0 + std::cos(2.0);
    
    std::cout << "  f(x,y) = x^2*y + sin(x)" << std::endl;
    std::cout << "  df/dx at (2,3) = " << dfdx << std::endl;
    std::cout << "  Expected:       " << expected << std::endl;
    std::cout << "  Error:          " << std::abs(dfdx - expected) << std::endl;
    
    if (std::abs(dfdx - expected) < 1e-10) {
        std::cout << "  [PASS]" << std::endl;
    } else {
        std::cout << "  [FAIL]" << std::endl;
    }
}

// --- Test 2: Reverse Mode Basic ---
void test_reverse_mode_basic() {
    std::cout << "\n=== Test 2: Reverse Mode Basic ===" << std::endl;
    
    using namespace autodiff;
    namespace rv = autodiff::reverse::detail;
    
    var x = 2.0;
    var y = 3.0;
    
    // f(x, y) = x^2 * y + sin(x)
    var u = x * x * y + sin(x);
    
    // Compute all gradients in one backward pass - use reverse mode namespace
    auto [ux, uy] = rv::derivatives(u, rv::wrt(x, y));
    
    // Analytical: df/dx = 2*x*y + cos(x), df/dy = x^2
    double expected_ux = 12.0 + std::cos(2.0);
    double expected_uy = 4.0;
    
    std::cout << "  f(x,y) = x^2*y + sin(x)" << std::endl;
    std::cout << "  df/dx at (2,3) = " << ux << " (expected " << expected_ux << ")" << std::endl;
    std::cout << "  df/dy at (2,3) = " << uy << " (expected " << expected_uy << ")" << std::endl;
    
    bool pass = std::abs(ux - expected_ux) < 1e-10 && std::abs(uy - expected_uy) < 1e-10;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
}

// --- Test 3: Higher-Order Derivatives (dual2nd) ---
void test_second_order() {
    std::cout << "\n=== Test 3: Second-Order Derivatives ===" << std::endl;
    
    using namespace autodiff;
    namespace fwd = autodiff::detail;
    
    dual2nd x = 2.0;
    
    // f(x) = x^3
    auto f = [](dual2nd x) -> dual2nd {
        return x * x * x;
    };
    
    // Compute d²f/dx² at x=2
    double d2fdx2 = fwd::derivative(f, fwd::wrt(x, x), fwd::at(x));
    // Analytical: f' = 3x^2, f'' = 6x = 12
    double expected = 12.0;
    
    std::cout << "  f(x) = x^3" << std::endl;
    std::cout << "  d²f/dx² at x=2 = " << d2fdx2 << std::endl;
    std::cout << "  Expected:        " << expected << std::endl;
    
    bool pass = std::abs(d2fdx2 - expected) < 1e-10;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
}

// --- Test 4: Eigen Integration (Forward Mode) ---
void test_eigen_integration() {
    std::cout << "\n=== Test 4: Eigen Integration ===" << std::endl;
    
    using namespace autodiff;
    namespace fwd = autodiff::detail;
    
    Eigen::VectorXdual x(3);
    x << 1.0, 2.0, 3.0;
    
    // f(x) = ||x||^2 = x1^2 + x2^2 + x3^2
    auto f = [](const Eigen::VectorXdual& v) -> dual {
        return v.squaredNorm();
    };
    
    dual value;
    Eigen::VectorXd grad;
    fwd::gradient(f, fwd::wrt(x), fwd::at(x), value, grad);
    // Analytical: grad = 2*x = [2, 4, 6]
    
    std::cout << "  f(x) = ||x||^2" << std::endl;
    std::cout << "  grad at [1,2,3] = [" << grad.transpose() << "]" << std::endl;
    std::cout << "  Expected:        [2, 4, 6]" << std::endl;
    
    bool pass = std::abs(grad[0] - 2.0) < 1e-10 && 
                std::abs(grad[1] - 4.0) < 1e-10 && 
                std::abs(grad[2] - 6.0) < 1e-10;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
}

// --- Test 5: Implicit Function Theorem (for dispersion) ---
void test_implicit_function() {
    std::cout << "\n=== Test 5: Implicit Function (dc/dω simulation) ===" << std::endl;
    
    using namespace autodiff;
    namespace fwd = autodiff::detail;
    
    // Simulating dispersion equation: F(c, ω) = c^2 - ω (simple test case)
    // Root: c = sqrt(ω)
    // dc/dω = -∂F/∂ω / ∂F/∂c = -(-1) / (2c) = 1/(2c) = 1/(2*sqrt(ω))
    
    dual c = 2.0;  // c value (should be sqrt(4))
    dual omega = 4.0;
    
    auto F = [](dual c, dual omega) -> dual {
        return c * c - omega;  // F(c, omega) = 0 at root
    };
    
    double dF_dc = fwd::derivative(F, fwd::wrt(c), fwd::at(c, omega));
    double dF_domega = fwd::derivative(F, fwd::wrt(omega), fwd::at(c, omega));
    
    double dc_domega = -dF_domega / dF_dc;
    double expected = 1.0 / (2.0 * 2.0);  // 1/(2*sqrt(4)) = 0.25
    
    std::cout << "  F(c, ω) = c² - ω = 0  =>  c = √ω" << std::endl;
    std::cout << "  At ω=4: c=2" << std::endl;
    std::cout << "  dc/dω (implicit) = " << dc_domega << std::endl;
    std::cout << "  Expected:          " << expected << std::endl;
    
    bool pass = std::abs(dc_domega - expected) < 1e-10;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "    Autodiff Library Integration Test   " << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_forward_mode_basic();
    test_reverse_mode_basic();
    test_second_order();
    test_eigen_integration();
    test_implicit_function();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "      All basic tests completed!        " << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
