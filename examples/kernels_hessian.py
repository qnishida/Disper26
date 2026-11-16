"""Use a phase-velocity gradient and Hessian for a local Taylor prediction."""
import disper26 as d26


def model(vs_layer=3.0):
    return d26.create_model(
        [0.0, 10.0], kind="layer_cake",
        vs=[vs_layer, 4.5], vp=[5.2, 8.0], rho=[2.5, 3.3],
    )


def run():
    solution = d26.compute_dispersion(model(), period=5.0, mode=0)
    kernels = d26.compute_kernels(solution, basis="isotropic")
    hessian = d26.compute_hessian(solution, basis="isotropic")
    # Select the physical coordinate from the returned axis metadata.
    axis = next(i for i, parameter in enumerate(kernels.parameters)
                if (parameter.entity, parameter.index, parameter.name) == ("layer", 0, "Vs"))
    rows = []
    for delta_vs in (0.01, 0.02):  # km/s; Vp and rho stay fixed.
        exact = d26.compute_dispersion(model(3.0 + delta_vs), period=5.0).phase_velocity
        linear = solution.phase_velocity + kernels.gradient[axis] * delta_vs
        quadratic = linear + 0.5 * hessian.hessian[axis, axis] * delta_vs**2
        rows.append((delta_vs, exact, linear, quadratic))
    return solution, kernels, hessian, rows


def main():
    solution, kernels, hessian, rows = run()
    print(f"Reference phase velocity: {solution.phase_velocity:.9f} km/s")
    print("Gradient axes:", [(p.entity, p.index, p.name) for p in kernels.parameters])
    print("Hessian shape:", hessian.hessian.shape)
    print("Delta Vs  Recomputed c  Linear error  Quadratic error (km/s)")
    for delta_vs, exact, linear, quadratic in rows:
        print(f"{delta_vs:8.3f}  {exact:.9f}  {abs(linear-exact):12.3e}  "
              f"{abs(quadratic-exact):20.3e}")
    print("Derivative convergence:", hessian.diagnostics.convergence)


if __name__ == "__main__":
    main()
