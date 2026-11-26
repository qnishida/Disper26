// Legacy single-layer phase diagnostics, not production integer mode labels.
// The whole-model crossing counter is covered by test_bran_mode_count.py.
// Both legacy helpers ignore their top-state argument; pass the bottom state
// twice. Rayleigh is compared with an independent isotropic ODE solution.
// See docs/mode_counting.md for the removed singular reference fixture.

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <cassert>
#include "../src/disper26.cpp"

bool approxEqual(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

// Isotropic reference material: rho=mu=lambda=1.
static PhysicalParameters<double> solid() {
    return createIsotropicParameters<double>(1.0, std::sqrt(3.0), 1.0);
}

// r_beta * thickness for the isotropic solid above, at omega = 1.
static double thickness_for_s_phase(double k, double phase) {
    return phase / std::sqrt(1.0 - k * k);
}

// Independent isotropic reference: spectral projectors of the 4-state
// displacement/traction ODE, followed by its second compound. No production
// layer propagator, modal coefficients or phase-unwrapping helper is used.
using RefMatrix = Eigen::Matrix<long double, 4, 4>;
using RefState = Eigen::Matrix<long double, 5, 1>;
RefState reference_state(long double k, long double z, const RefState& initial) {
    const RefMatrix identity = RefMatrix::Identity();
    RefMatrix g = RefMatrix::Zero();
    // rho=mu=1, lambda=1; ordering [uz,szz,ur,szr].
    g(0,1)=1.L/3; g(0,2)=k/3;
    g(1,0)=-1; g(1,3)=k;
    g(2,0)=-k; g(2,3)=1;
    g(3,1)=-k/3; g(3,2)=8.L*k*k/3-1;
    const long double qp=1.L/3-k*k, qs=1-k*k;
    auto oscillator = [&](long double q) -> RefMatrix {
        const long double r=std::sqrt(std::abs(q));
        if (q == 0) return (identity + z*g).eval();
        return q>0 ? (std::cos(r*z)*identity + std::sin(r*z)/r*g).eval()
                   : (std::cosh(r*z)*identity + std::sinh(r*z)/r*g).eval();
    };
    const RefMatrix p = ((g*g+qs*identity)/(qs-qp)*oscillator(qp)
                      +(g*g+qp*identity)/(qp-qs)*oscillator(qs)).eval();
    constexpr int pairs[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    RefState out=RefState::Zero();
    for(int r=0;r<5;r++) for(int c=0;c<6;c++) {
        const auto* a=pairs[r]; const auto* b=pairs[c];
        const long double minor=p(a[0],b[0])*p(a[1],b[1])-p(a[0],b[1])*p(a[1],b[0]);
        out(r)+=minor*(c==5 ? -initial(0):initial(c));
    }
    return out;
}
long double reference_phase(long double k,long double d,const RefState& initial,int steps) {
    const long double pi=std::acos(-1.L);
    auto angle=[](const RefState& y){return std::atan2(y(4),-(y(2)+y(3)));};
    long double previous=angle(initial),total=0;
    for(int i=1;i<=steps;i++) {
        const RefState y=reference_state(k,d*i/steps,initial);
        const long double next=angle(y);
        const long double delta = std::remainder(next-previous,2*pi);
        // Keep the reference samples well inside the unwrap branch boundary.
        assert(std::abs(delta) < pi/2);
        total += delta;
        previous=next;
    }
    return total;
}

void testRayleigh() {
    std::cout << "Testing legacy Rayleigh layer phase diagnostics..." << std::endl;
    const double pi = std::acos(-1.0);
    const auto params = solid();

    // The old Y_12=1 fixture has x=y=0 initially. At k=0.5 and
    // r_s*d=6*pi, r_p*d=2*pi: the exact 4-state propagator is the identity,
    // so the endpoint phase is undefined too. Its former decimal expectation
    // measured atan2 of roundoff, not a reference solution.
    RefState singular = RefState::Zero();
    singular(I12) = 1;
    const long double exact_d = 6*std::acos(-1.L)/std::sqrt(.75L);
    const RefState returned = reference_state(.5L, exact_d, singular);
    assert(std::abs(returned(I24)) < 1e-12L);
    assert(std::abs(returned(I14)+returned(I23)) < 1e-12L);

    // Decomposable, nonzero phase state: (.3*e1+e2) wedge e4.
    // The decomposable state satisfies the Plücker relation
    // Y_12*Y_34-Y_13*Y_24+Y_14*Y_23=0, with the reduction Y_34=-Y_12.
    RefState initial = RefState::Zero();
    initial(I14) = .3L;
    initial(I24) = 1;
    const Eigen::Matrix<double,5,1> y = initial.cast<double>();
    double worst_error = 0;
    long double worst_refinement = 0;
    auto check = [&](double k, double d) {
        const long double coarse = reference_phase(k,d,initial,2048);
        const long double fine = reference_phase(k,d,initial,4096);
        worst_refinement = std::max(worst_refinement,std::abs(coarse-fine));
        // The phase is unwrapped on independently refined direct-depth grids.
        // This tighter reference check supports the existing 1e-9 phase
        // tolerance; no production subdivision or normalizePhase is reused.
        assert(std::abs(coarse-fine) < 1e-11L);
        const auto result = countModesLayerRayleigh(y,y,1.,k,params,d);
        const double error = std::abs(result.second-static_cast<double>(fine));
        worst_error = std::max(worst_error,error);
        assert(error < 1e-9);
        const long double cycles = std::abs(fine)/std::acos(-1.L);
        // At an exact integer the floor is roundoff-dependent. Verify phase
        // there, and verify the integer on both sides of the boundary.
        if(std::abs(cycles-std::round(cycles)) > 1e-6L)
            assert(result.first == static_cast<int>(std::floor(cycles)));
    };
    check(1.5, .5);  // Both vertical waves evanescent.
    check(1.5, 5.);
    check(.987, thickness_for_s_phase(.987,pi/2)); // Only S propagates.
    for(double phase_pi : {.5,4.99,5.,5.01,5.99,6.,6.01,7.})
        check(.5,thickness_for_s_phase(.5,phase_pi*pi)); // P and S propagate.
    std::cout << "  Independent phase reference: max error=" << worst_error
              << ", refinement=" << worst_refinement << std::endl;
}

void testLove() {
    std::cout << "Testing legacy Love layer phase diagnostics..." << std::endl;

    const double omega = 1.0;
    const PhysicalParameters<double> params = solid();

    Eigen::Matrix<double, 2, 1> y_bot;
    y_bot << 1.0, 0.0;

    // The integer half of countModesLayerLove is currently hard-wired to zero:
    // the loop declares int n_sub = 0, never assigns it, and accumulates
    // total_N += n_sub. Only the phase it returns carries information, and the
    // only caller in production, the legacy counter in
    // calculateTotalPhaseAccumulationWithDebug, derives its integer from the
    // summed phase rather than from this field, so nothing downstream is wrong
    // today. Asserting the zeros here pins that gap: if the integer is ever
    // implemented, this test says so.
    //
    // At these quarter/full-turn endpoints the Love state ellipse has an
    // analytically known angle. At general thickness its atan2 angle need
    // not equal the vertical S phase r_beta * thickness.

    // Case 1: evanescent, k = 1.5. Early return, exactly {0, 0}.
    {
        auto res = countModesLayerLove(y_bot, y_bot, omega, 1.5, params, 10.0);
        assert(res.first == 0);
        assert(res.second == 0.0);
        std::cout << "  Passed: Evanescent Case" << std::endl;
    }

    // Case 2: quarter turn.
    {
        const double k = 0.9;
        const double d = thickness_for_s_phase(k, M_PI / 2.0);
        auto res = countModesLayerLove(y_bot, y_bot, omega, k, params, d);
        assert(res.first == 0); // structurally zero, see note above
        assert(approxEqual(std::abs(res.second), M_PI / 2.0, 1e-9));
        std::cout << "  Passed: Standard Case (PI/2, phase=" << res.second
                  << ")" << std::endl;
    }

    // Case 3: five half-wavelengths.
    {
        const double k = 0.5;
        const double d = thickness_for_s_phase(k, 5.0 * M_PI);
        auto res = countModesLayerLove(y_bot, y_bot, omega, k, params, d);
        assert(res.first == 0); // structurally zero, see note above
        assert(approxEqual(std::abs(res.second), 5.0 * M_PI, 1e-9));
        std::cout << "  Passed: High Freq Case (5PI, phase=" << res.second
                  << ")" << std::endl;
    }
}

int main() {
    testRayleigh();
    testLove();
    std::cout << "All legacy layer phase diagnostic tests passed!" << std::endl;
    return 0;
}
