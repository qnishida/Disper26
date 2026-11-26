# Reduced Delta Matrix (P-SV Problem)

This document describes the Reduced Delta Matrix (Compound Matrix) formulation used in `Disper26` for computing P-SV wave dispersion.

## 1. Solid Layer Formulation

For a solid layer with vertical transverse isotropy (VTI), including the
isotropic limit, Disper26 uses the $5\times5$ reduced delta matrix of
[Ikeda & Matsuoka (2013)](https://doi.org/10.1785/0120120207).

### 1.1 Physical Parameters
*   $\rho$: Density
*   $c_{11}, c_{13}, c_{33}, c_{44}$: Elastic moduli
    *   For isotropic media: $c_{11} = c_{33} = \rho V_p^2$, $c_{44} = \rho V_s^2$, $c_{13} = c_{33} - 2c_{44}$.
*   $\omega$: Angular frequency
*   $k$: Horizontal wavenumber, with phase velocity $c=\omega/k$
*   $z$ is the vertical coordinate, positive upward.
*   $H=z_{\rm top}-z_{\rm bottom}>0$ is the layer thickness.

### 1.2 Eigenvalue Problem (Vertical Propagation Exponents)

The vertical propagation exponents $\nu_1$ and $\nu_3$ enter the layer
solutions through $e^{\pm\nu_j z}$. For a plane wave with vertical phase
factor $e^{ik_z z}$, each exponent satisfies $\nu_j^2=-k_z^2$ for the
corresponding vertical wavenumber $k_z$.

Define auxiliary terms $M_1, M_2$:
$$
M_1 = c_{44}(\rho\omega^2 - c_{44}k^2) + c_{33}(\rho\omega^2 - c_{11}k^2) + (c_{13} + c_{44})^2 k^2
$$
$$
M_2 = (\rho\omega^2 - c_{44}k^2)(\rho\omega^2 - c_{11}k^2)
$$

The squared exponents are the roots of the Christoffel polynomial $c_{33}c_{44}(\nu^2)^2 + M_1(\nu^2) + M_2 = 0$.
$$
\nu_{1,3}^2 = \frac{-M_1 \pm \sqrt{M_1^2 - 4 c_{33} c_{44} M_2}}{2 c_{33} c_{44}}
$$
where $\nu_1$ corresponds to the $+$ sign and $\nu_3$ to the $-$ sign. $\nu_{1,3}$ are taken as the complex square roots of $\nu_{1,3}^2$.

### 1.3 Eigenvector Parameters

The displacement-stress eigenvectors are characterized by the parameters $\epsilon_1, \epsilon_3$:
$$
\epsilon_i = \frac{(c_{13} + c_{44})k \nu_i}{(c_{11}k^2 - \rho\omega^2) - c_{44}\nu_i^2} \quad (i=1,3)
$$
These modal expressions have removable singularities at critical speeds.
The [finite-layer implementation](../src/dispersion_rayleigh.hpp) cancels
factors of $\nu_i$ before evaluation. For degenerate roots or an
unusable denominator it forms the reduced matrix directly from the regular
four-component state propagator.

### 1.4 Intermediate Constants ($d_1 \dots d_6$)

Define the following constants used in the matrix calculation:
$$
d_1 = c_{33}\nu_1 - c_{13}k\epsilon_1
$$
$$
d_2 = c_{33}\nu_3 - c_{13}k\epsilon_3
$$
$$
d_3 = k + \nu_1\epsilon_1
$$
$$
d_4 = k + \nu_3\epsilon_3
$$
$$
d_5 = \nu_1\epsilon_1 - \nu_3\epsilon_3
$$
$$
d_6 = \nu_3\epsilon_1 - \nu_1\epsilon_3
$$

### 1.5 Layer Matrix Elements (5x5)

For positive layer thickness, $B$ propagates the bottom minor vector upward to
the top. The solid state is `[uz, szz, ur, szr]` with upward vertical
coordinate. For two independent solutions that decay in the lower half-space,
the minor formed from rows $i,j$ is

$$
Y_{ij}=y_i^{(1)}y_j^{(2)}-y_i^{(2)}y_j^{(1)}.
$$

In this component order $Y_{34}=-Y_{12}$. The columns of $B$ follow the
order of the reduced vector $\mathbf{Y}=(Y_{12},Y_{13},Y_{14},Y_{23},Y_{24})^\mathsf{T}$.
In C++, `I14` is the zero-based index 2, so `Y(I14)` returns $Y_{14}$.

Define propagator terms:
$$
C_1 = \cosh(\nu_1 H), \quad S_1 = \sinh(\nu_1 H)
$$
$$
C_3 = \cosh(\nu_3 H), \quad S_3 = \sinh(\nu_3 H)
$$

Define auxiliary scaling factors:
$$
k_{33}^{(d6)} = \frac{1}{c_{33} d_6}, \quad k_{44}^{(d5)} = \frac{1}{c_{44} d_5}
$$
$$
k_{all} = \frac{1}{c_{33} c_{44} d_5 d_6}
$$
$$
k_{33}^{(d5d6)} = \frac{1}{c_{33} d_5 d_6}
$$

**Matrix Elements $B_{ij}$:**

**Column 1 ($Y_{12}$):**
$$
b_{1234} = k_{all} \left[ 2 d_1 d_2 (C_1 C_3 - 1) - (d_1^2 + d_2^2) S_1 S_3 \right]
$$
$$
B_{11} = 1 - 2 b_{1234}
$$
$$
B_{21} = 2 k_{all} \left[ (\epsilon_1 d_2 + \epsilon_3 d_1)(1 - C_1 C_3) + (\epsilon_1 d_1 + \epsilon_3 d_2) S_1 S_3 \right]
$$
$$
B_{31} = 2 k_{33}^{(d6)} (d_1 C_1 S_3 - d_2 C_3 S_1)
$$
$$
B_{41} = 2 k_{44}^{(d5)} (-d_2 C_1 S_3 + d_1 C_3 S_1)
$$
$$
B_{51} = 2 k_{33}^{(d5d6)} \left[ d_1 d_2 (d_3 + d_4)(C_1 C_3 - 1) - (d_2^2 d_3 + d_1^2 d_4) S_1 S_3 \right]
$$

**Column 2 ($Y_{13}$):**
$$
B_{12} = \frac{1}{2} B_{51}
$$
$$
B_{22} = C_1 C_3 + b_{1234}
$$
$$
B_{32} = c_{44} k_{33}^{(d6)} (-d_1 d_4 C_1 S_3 + d_2 d_3 C_3 S_1)
$$
$$
B_{42} = \frac{1}{d_5} (d_2 d_3 C_1 S_3 - d_1 d_4 C_3 S_1)
$$
$$
B_{52} = c_{44} k_{33}^{(d5d6)} \left[ 2 d_1 d_2 d_3 d_4 (1 - C_1 C_3) + (d_1^2 d_4^2 + d_2^2 d_3^2) S_1 S_3 \right]
$$

**Column 3 ($Y_{14}$):**
$$
B_{13} = \frac{1}{2} B_{41}
$$
$$
B_{23} = k_{44}^{(d5)} (-\epsilon_3 C_1 S_3 + \epsilon_1 C_3 S_1)
$$
$$
B_{33} = C_1 C_3
$$
$$
B_{43} = c_{33} d_6 k_{44}^{(d5)} S_1 S_3
$$
$$
B_{53} = B_{42} \quad (\text{Symmetry})
$$

**Column 4 ($Y_{23}$):**
$$
B_{14} = \frac{1}{2} B_{31}
$$
$$
B_{24} = k_{33}^{(d6)} (\epsilon_1 C_1 S_3 - \epsilon_3 C_3 S_1)
$$
$$
B_{34} = c_{44} d_5 k_{33}^{(d6)} S_1 S_3
$$
$$
B_{44} = B_{33} \quad (\text{Symmetry})
$$
$$
B_{54} = B_{32} \quad (\text{Symmetry})
$$

**Column 5 ($Y_{24}$):**
$$
B_{15} = \frac{1}{2} B_{21}
$$
$$
B_{25} = k_{all} \left[ 2 \epsilon_1 \epsilon_3 (1 - C_1 C_3) + (\epsilon_1^2 + \epsilon_3^2) S_1 S_3 \right]
$$
$$
B_{35} = B_{24} \quad (\text{Symmetry})
$$
$$
B_{45} = B_{23} \quad (\text{Symmetry})
$$
$$
B_{55} = B_{22} \quad (\text{Symmetry})
$$

## 2. Fluid Layer Formulation

In a fluid layer ($c_{44}=0$), the $2\times2$ acoustic matrix propagates the state
$[w,T_{zz}]^\mathsf{T}$ (upward displacement and Lagrangian normal traction).
The matrix below propagates upward through thickness $H$ at zero gravity.
At the seabed, choosing a common normalization gives
$(w,T_{zz})=(Y_{14},Y_{24})$ in terms of the solid-side minors.
See the [fluid boundary conditions](fluid_layers.md#solid-fluid-transition).

### 2.1 Parameters
*   $V_p = \sqrt{c_{33} / \rho}$ (P-wave velocity)
*   $\nu^2 = k^2(1-c^2/V_p^2)=-k_z^2$, where $k_z^2=\omega^2/V_p^2-k^2$.

For $c<V_p$, $\nu$ is real and the two solutions decay or grow exponentially
with depth. For $c>V_p$, $\nu$ is purely imaginary and the solutions
oscillate with depth. At $c=V_p$, $\nu=0$.

### 2.2 Matrix Elements (2x2)

$$
B_{\mathrm{fluid}} = \begin{pmatrix}
\cosh(\nu H) & -\frac{\nu}{\rho \omega^2} \sinh(\nu H) \\
-\frac{\rho \omega^2}{\nu} \sinh(\nu H) & \cosh(\nu H)
\end{pmatrix}
$$

Note: In the limit $\nu \to 0$, ratio terms are handled as:
$$
\lim_{\nu \to 0} \frac{\sinh(\nu H)}{\nu} = H
$$
