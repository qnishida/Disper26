import disper26py
import sys
import math

def approx_equal(a, b, tol=1e-5):
    return abs(a - b) < tol

def test_wrapper():
    print("Testing disper26py wrapper...")
    
    # 1. Test PhysicalParameters Creation
    # Use helper
    layer = disper26py.create_isotropic_parameters(vs=1.0, vp=1.732, rho=1.0)
    print(f"Created Isotropic Layer: {layer}")
    
    # Verify c44 = rho * vs^2 = 1.0 * 1.0 = 1.0
    # Verify c33 = rho * vp^2 = 1.0 * 3.0 = 3.0
    if not approx_equal(layer.c44, 1.0):
        print(f"FAILED: c44 mismatch. Got {layer.c44}, expected 1.0")
        sys.exit(1)
    
    # 2. Test Mode Counting
    #
    # Single layer over a fast half-space, tuned so the layer holds exactly five
    # half-wavelengths of vertical S motion: r_beta * thickness = 5*pi.
    #
    # The expected counts used to be 2 here, taken from Case 3 of the C++ unit
    # test tests/verify_mode_count.cpp. That case only borrowed these same
    # omega/k/thickness numbers -- what it actually checks is
    # countModesLayerRayleigh(), a per-layer phase-unwrapping helper fed
    # hand-built endpoint state vectors, and its 2 is the number of 2*pi
    # rotations in the unwrapped phase, round((5*pi - pi) / 2*pi). That is a
    # different quantity from the mode count of a physical model, and
    # countModesLayerRayleigh() is not exposed to Python at all, so 2 was never
    # a valid expectation for calculate_mode_count().
    #
    # The counts below are the mode count of this model at this (omega, c),
    # which is what calculate_mode_count() returns. Love: the layer holds
    # 5 half-wavelengths of S, so 5. Rayleigh: P contributes as well
    # (r_alpha * thickness = 5*pi/3), giving 6. Both agree with
    # calculate_continuous_mode_count() -- 6.487 and 5.000.
    #
    # This is a wrapper smoke test: it checks the binding is reachable and
    # returns the right number on one known model. The counter itself is
    # validated properly in test_bran_mode_count.py, which walks the actual
    # dispersion roots and checks the label increments 0, 1, 2, ... across them.
    omega = 1.0
    k = 0.5
    c_phase = omega / k # 2.0 km/s

    r_beta = math.sqrt((omega/1.0)**2 - k**2) # sqrt(1 - 0.25) = sqrt(0.75) = 0.866
    thickness = (5.0 * math.pi) / r_beta

    layers = [layer]
    thicknesses = [thickness]
    half_space = disper26py.create_isotropic_parameters(vs=3.0, vp=5.2, rho=1.0) # High velocity half-space to ensure exponential decay (trapped mode)

    count = disper26py.compute_mode_count(
        layers, thicknesses, half_space, omega, c_phase, "Rayleigh"
    )

    print(f"Rayleigh Mode Count (Expect 6): {count}")
    if count != 6:
        print(f"FAILED: Expected 6, got {count}")
        sys.exit(1)

    count_love = disper26py.compute_mode_count(
        layers, thicknesses, half_space, omega, c_phase, "Love"
    )
    print(f"Love Mode Count (Expect 5): {count_love}")
    if count_love != 5:
        print(f"FAILED: Expected 5, got {count_love}")
        sys.exit(1)

    print("All Python Wrapper Tests Passed!")

if __name__ == "__main__":
    test_wrapper()
