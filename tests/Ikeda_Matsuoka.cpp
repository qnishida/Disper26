#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Indices for the reduced 5x5 matrix. Y_34 is recovered from Y_12.
enum ReducedIndex {
    I12 = 0,
    I13 = 1,
    I14 = 2,
    I23 = 3,
    I24 = 4
};

struct Parameters {
    double eps1, eps3;
    double d1, d2, d3, d4, d5, d6;
    double c33, c44;
};

struct Variables {
    double C1, C3;
    double S1, S3;
};

// Five-component vector and 5x5 matrix.
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;

Vector5d computeInitialVectorReduced(const Parameters& p) {
    Vector5d Y;
    
    Y(I12) = p.d2 - p.d1;
    Y(I13) = p.eps3 - p.eps1;
    Y(I14) = -p.c44 * p.d5;
    Y(I23) = -p.c33 * p.d6;
    Y(I24) = p.c44 * (p.d1 * p.d4 - p.d2 * p.d3);
    
    // Y_12 determines Y_34 through Y_34 = -Y_12.

    return Y;
}

Matrix5d computeMatrixReduced(const Parameters& p, const Variables& v) {
    // Use the 6x6 entry names when assembling the reduced 5x5 matrix.
    
    Matrix5d B = Matrix5d::Zero();

    // Common factors from the full 6x6 formulation.
    double C1C3 = v.C1 * v.C3;
    double S1S3 = v.S1 * v.S3;
    double inv_d5 = 1.0 / p.d5;
    double inv_d6 = 1.0 / p.d6;
    double inv_c33 = 1.0 / p.c33;
    double inv_c44 = 1.0 / p.c44;
    double k_33_d5d6 = inv_c33 * inv_d5 * inv_d6;
    double k_33_d6 = inv_c33 * inv_d6;
    double k_44_d5 = inv_c44 * inv_d5;
    double k_all   = inv_c33 * inv_c44 * inv_d5 * inv_d6;

    // Entries needed from the full matrix.
    // Col 12 and Col 34 components
    double b1234 = k_all * (2.0 * p.d1 * p.d2 * (C1C3 - 1.0) - (p.d1*p.d1 + p.d2*p.d2) * S1S3);
    double b1212 = 1.0 - b1234;
    
    // Reduced B(0,0) = b1212 - b1234
    //                = (1 - b1234) - b1234 = 1 - 2*b1234
    B(I12, I12) = 1.0 - 2.0 * b1234;

    // For the remaining rows, form the first column as Col_12 - Col_34.
    // Use the identities for each row's entry in column 34.
    
    // Row 13: b1312 - b1334 = 2*b1312 because b1334 = -b1312.
    double b1224 = k_all * ((p.eps1 * p.d2 + p.eps3 * p.d1)*(1.0 - C1C3) + (p.eps1*p.d1 + p.eps3*p.d2)*S1S3);
    double b1312 = b1224; 
    B(I13, I12) = 2.0 * b1312;

    // Row 14: b1412 - b1434 = 2*b1412 because b1434 = -b1412.
    double b1223 = k_33_d6 * (p.d1 * v.C1 * v.S3 - p.d2 * v.C3 * v.S1);
    double b1412 = b1223;
    B(I14, I12) = 2.0 * b1412;

    // Row 23: b2312 - b2334 = 2*b2312 because b2334 = -b2312.
    double b1214 = k_44_d5 * (-p.d2 * v.C1 * v.S3 + p.d1 * v.C3 * v.S1);
    double b2312 = b1214;
    B(I23, I12) = 2.0 * b2312;

    // Row 24: b2412 - b2434 = b2412 + b1213 because b2434 = -b1213.
    double b1213 = k_33_d5d6 * (p.d1*p.d2*(p.d3+p.d4)*(C1C3-1.0) - (p.d2*p.d2*p.d3+p.d1*p.d1*p.d4)*S1S3);
    double b2412 = b1213; // Equal by definition.
    B(I24, I12) = 2.0 * b1213;

    // Columns I13, I14, I23 and I24 retain their full-matrix entries.
    // Use symmetry to fill in repeated entries.

    // Col 13 (Original Index 13)
    B(I12, I13) = b1213;
    B(I13, I13) = C1C3 + b1234; // b1313
    // b1413
    B(I14, I13) = (p.c44 * k_33_d6) * (-p.d1*p.d4*v.C1*v.S3 + p.d2*p.d3*v.C3*v.S1);
    // b2313
    B(I23, I13) = inv_d5 * (p.d2*p.d3*v.C1*v.S3 - p.d1*p.d4*v.C3*v.S1);
    // b2413
    B(I24, I13) = (p.c44 * k_33_d5d6) * (2.0*p.d1*p.d2*p.d3*p.d4*(1.0-C1C3) + (p.d1*p.d1*p.d4*p.d4 + p.d2*p.d2*p.d3*p.d3)*S1S3);

    // Col 14 (Original Index 14)
    B(I12, I14) = b1214;
    double b1314 = k_44_d5 * (-p.eps3 * v.C1 * v.S3 + p.eps1 * v.C3 * v.S1);
    B(I13, I14) = b1314;
    B(I14, I14) = C1C3; // b1414
    B(I23, I14) = (p.c33 * p.d6 * k_44_d5) * S1S3; // b2314
    B(I24, I14) = B(I23, I13); // b2414 = b2313

    // Col 23 (Original Index 23)
    B(I12, I23) = b1223;
    double b1323 = k_33_d6 * (p.eps1 * v.C1 * v.S3 - p.eps3 * v.C3 * v.S1);
    B(I13, I23) = b1323;
    B(I14, I23) = (p.c44 * p.d5 * k_33_d6) * S1S3; // b1423
    B(I23, I23) = B(I14, I14); // b2323 = b1414
    B(I24, I23) = B(I14, I13); // b2423 = b1413

    // Col 24 (Original Index 24)
    B(I12, I24) = b1224;
    B(I13, I24) = k_all * (2.0 * p.eps1 * p.eps3 * (1.0 - C1C3) + (p.eps1*p.eps1 + p.eps3*p.eps3) * S1S3); // b1324
    B(I14, I24) = B(I13, I23); // b1424 = b1323
    B(I23, I24) = B(I13, I14); // b2324 = b1314
    B(I24, I24) = B(I13, I13); // b2424 = b1313

    return B;
}

int main() {
    // Example parameters.
    Parameters p = { 1.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 10.0, 20.0 };
    Variables v = { 0.8, 0.8, 0.6, 0.6 };

    Vector5d Y0 = computeInitialVectorReduced(p);
    Matrix5d B_reduced = computeMatrixReduced(p, v);

    std::cout << "Reduced Matrix B (5x5):\n" << B_reduced << std::endl;
    
    // The reduced representation implicitly preserves Y_34 = -Y_12.
    
    return 0;
}
