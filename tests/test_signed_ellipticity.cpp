#include "../src/dispersion_interface.hpp"
#include <cassert>
#include <cmath>

int main() {
    // Construct two independent states with zero bilinear invariant.
    // The first has zero shear traction and the prescribed displacement ratio.
    using AD = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    for (double e : {-0.5, 0.0, 0.5}) {
        Eigen::Matrix<AD, 4, 1> a,b;
        auto constant=[](double v) { AD x; x.value()=v; x.derivatives()=Eigen::VectorXd::Zero(2); return x; };
        a << constant(1),constant(0),constant(e),constant(0);
        a(2).derivatives()(0)=1;
        b << constant(0),-a(2),constant(0),constant(1);
        Eigen::Matrix<AD, 5, 1> y;
        const int pairs[5][2]={{0,1},{0,2},{0,3},{1,2},{1,3}};
        for (int n=0;n<5;++n) {
            int i=pairs[n][0],j=pairs[n][1];
            y(n)=a(i)*b(j)-a(j)*b(i);
        }
        auto ratio=calculateHVRatio(y);
        assert(std::abs(ratio.value()-e)<1e-14);
        assert(std::abs(ratio.derivatives()(0)-1)<1e-14);
        assert(ratio.derivatives()(1)==0);
        // A vertical-displacement zero is undefined, with preserved AD shape.
        y(I14)=constant(0);
        auto pole=calculateHVRatio(y);
        assert(std::isnan(pole.value()));
        assert(pole.derivatives().size()==2);
        assert(std::isnan(pole.derivatives()(0)) && std::isnan(pole.derivatives()(1)));
    }
}
