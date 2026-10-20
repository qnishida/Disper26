#include <iostream>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include "../src/disper26.cpp"

void verify_double() {
    double vs = 1.0;
    double vp = 2.0;
    double rho = 3.0;
    
    auto p = createIsotropicParameters(vs, vp, rho);
    
    double expected_c44 = 3.0 * 1.0;
    double expected_c11 = 3.0 * 4.0;
    double expected_c13 = expected_c11 - 2.0 * expected_c44;
    
    assert(std::abs(p.rho - 3.0) < 1e-9);
    assert(std::abs(p.c44 - expected_c44) < 1e-9);
    assert(std::abs(p.c66 - expected_c44) < 1e-9);
    assert(std::abs(p.c11 - expected_c11) < 1e-9);
    assert(std::abs(p.c33 - expected_c11) < 1e-9);
    assert(std::abs(p.c13 - expected_c13) < 1e-9); // 12 - 6 = 6
    
    std::cout << "Double Verification Passed." << std::endl;
}

void verify_ads() {
    using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    ADS vs; vs.value() = 1.0; vs.derivatives() = Eigen::VectorXd::Ones(1);
    ADS vp; vp.value() = 2.0; vp.derivatives() = Eigen::VectorXd::Zero(1);
    ADS rho; rho.value() = 3.0; rho.derivatives() = Eigen::VectorXd::Zero(1);
    
    auto p = createIsotropicParameters(vs, vp, rho);
    
    assert(std::abs(p.c66.value() - 3.0) < 1e-9);
    // d(c44)/dvs = d(rho*vs^2)/dvs = 2*rho*vs = 6.0
    assert(std::abs(p.c44.derivatives()[0] - 6.0) < 1e-9);
    assert(std::abs(p.c66.derivatives()[0] - 6.0) < 1e-9);
    
    std::cout << "ADS Verification Passed." << std::endl;
}

int main() {
    verify_double();
    verify_ads();
    std::cout << "All Isotropic Helper tests passed." << std::endl;
    return 0;
}
