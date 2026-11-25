"""Compare Rayleigh and Love dispersion for isotropic and VTI layers."""
import numpy as np

import disper26 as d26


def run(*, vti=False):
    # Depths: km; velocities: km/s; densities: g/cm^3.
    # Only the upper layer's Vsh differs between the models.
    vs = [3.0, 4.5]
    vp = [5.2, 8.0]
    if vti:
        velocities = dict(vsh=[3.15, 4.5], vsv=vs, vph=vp, vpv=vp, eta=[1., 1.])
    else:
        velocities = dict(vs=vs, vp=vp)
    model = d26.create_model([0., 10.], kind="layer_cake", rho=[2.5, 3.3], **velocities)
    periods = np.arange(2., 20.5, 0.5)
    curves = {
        wave: d26.compute_dispersion_curve(model, periods=periods, wave=wave, mode=0)
        for wave in ("Rayleigh", "Love")
    }
    return curves


def main():
    isotropic = run()
    vti = run(vti=True)
    print("Phase velocity (km/s)")
    print("Period (s)  Rayleigh (iso)  Rayleigh (VTI)  Love (iso)  Love (VTI)")
    for period, rayleigh_iso, rayleigh_vti, love_iso, love_vti in zip(
        isotropic["Rayleigh"].periods,
        isotropic["Rayleigh"].phase_velocity, vti["Rayleigh"].phase_velocity,
        isotropic["Love"].phase_velocity, vti["Love"].phase_velocity,
    ):
        print(f"{period:10.1f}  {rayleigh_iso:14.6f}  {rayleigh_vti:14.6f}"
              f"  {love_iso:10.6f}  {love_vti:10.6f}")


if __name__ == "__main__":
    main()
