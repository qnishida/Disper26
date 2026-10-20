"""Compute Rayleigh and Love dispersion for a layer over a halfspace."""
import disper26 as d26


def run():
    # Depths: km; velocities: km/s; densities: g/cm^3.
    # The last point starts the halfspace.
    model = d26.create_model(
        [0.0, 10.0], kind="pancake",
        vs=[3.0, 4.5], vp=[5.2, 8.0], rho=[2.5, 3.3],
    )
    periods = [2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    curves = {
        wave: d26.compute_dispersion_curve(model, periods=periods, wave=wave, mode=0)
        for wave in ("Rayleigh", "Love")
    }
    return curves


def main():
    curves = run()
    print("Period (s)  Rayleigh c (km/s)  Love c (km/s)")
    for period, rayleigh, love in zip(
        curves["Rayleigh"].periods,
        curves["Rayleigh"].phase_velocity,
        curves["Love"].phase_velocity,
    ):
        print(f"{period:10.2f}  {rayleigh:17.9f}  {love:13.9f}")


if __name__ == "__main__":
    main()
