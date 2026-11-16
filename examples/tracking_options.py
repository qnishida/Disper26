"""Choose the start period, initial speed range and mode verification policy."""
import disper26 as d26

def run():
    model = d26.create_model(
        [0., 20., 60.], kind="layer_cake",
        vs=[3., 3.5, 4.5], vp=[5.2, 6., 8.], rho=[2.5, 2.8, 3.3])
    return d26.compute_dispersion_curve(
        model, periods=[8., 8.2, 8.4, 8.6],
        options=d26.Options(start="long", c_range=(2.6, 3.),
                            verify="endpoints", on_violation="fallback"))


def main():
    for solution in run().solutions:
        print(solution.period, solution.phase_velocity,
              "mode verified:", solution.diagnostics.mode_verified)

# Select verify="every" when every period must have the requested mode number.
# Select start="short" to retain the previous ascending-period calculation order.


if __name__ == "__main__":
    main()
