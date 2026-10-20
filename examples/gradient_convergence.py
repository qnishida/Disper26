"""Inspect Romberg convergence for a model with depth-varying stiffness."""
import warnings

import disper26 as d26


def run():
    model = d26.create_model(
        [0.0, 10.0, 30.0], kind="grid", interpolation="stiffness",
        vs=[3.0, 3.4, 4.5], vp=[5.2, 6.0, 8.0], rho=[2.5, 2.8, 3.3],
    )
    solutions = []
    for limit in range(1, 9):
        # Low caps deliberately stop before convergence. The table reports
        # that status explicitly instead of printing a warning for every cap.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", d26.ConvergenceWarning)
            solution = d26.compute_dispersion(
                model, period=5.0, wave="Rayleigh", mode=0,
                options=d26.Options(max_steps=limit),
            )
        solutions.append(solution)
    return solutions


def main():
    print("Level cap  Accepted  c (km/s)    Estimated relative error  Convergence")
    for solution in run():
        diagnostic = solution.diagnostics
        print(f"{solution.options.max_steps:9d}  {diagnostic.accepted_level:8d}  "
              f"{solution.phase_velocity:.9f}  {diagnostic.relative_error:24.3e}  "
              f"{diagnostic.convergence}")


if __name__ == "__main__":
    main()
