"""High-precision diagnostic reference, independent of finite-layer modal roots.

Materials are [rho,A,F,C,L,N] in the core's units, converted from the exact
binary64 inputs. The finite-layer reference is exp(G*h), then its second
compound for Rayleigh. No finite-layer nu or sinh/nu division is formed.
Only the decaying halfspace basis uses the ordinary modal roots. Gravity is
not part of this diagnostic reference; acoustic top fluids are supported.
"""
from mpmath import mp


def derived(material, omega, velocity):
    rho, a, f, cc, l, nn = map(mp.mpf, material)
    k = omega/velocity
    x, y = rho*omega**2-l*k**2, rho*omega**2-a*k**2
    m1 = l*x+cc*y+(f+l)**2*k**2
    m2 = x*y
    disc = mp.sqrt(m1*m1-4*cc*l*m2)
    q1, q3 = (-m1+disc)/(2*cc*l), (-m1-disc)/(2*cc*l)
    n1, n3 = mp.sqrt(q1), mp.sqrt(q3)
    e1 = (f+l)*k*n1/(a*k*k-rho*omega**2-l*q1)
    e3 = (f+l)*k*n3/(a*k*k-rho*omega**2-l*q3)
    d1, d2 = cc*n1-f*k*e1, cc*n3-f*k*e3
    d3, d4 = k+n1*e1, k+n3*e3
    d5, d6 = n1*e1-n3*e3, n3*e1-n1*e3
    return q1, q3, (d2-d1, e3-e1, -l*d5, -cc*d6, l*(d1*d4-d2*d3))


def initial(material, omega, velocity, wave):
    rho, a, f, cc, l, nn = map(mp.mpf, material)
    k = omega/velocity
    if wave == "Love":
        return mp.matrix([1, -l*mp.sqrt((nn*k*k-rho*omega**2)/l)])
    _, _, raw = derived(material, omega, velocity)
    # Match the halfspace phase convention; it is constant in each tested
    # neighborhood, so differentiation does not cross a phase selection.
    component = mp.im if sum(abs(mp.im(x)) for x in raw) > sum(abs(mp.re(x)) for x in raw) else mp.re
    out = mp.matrix([component(x) for x in raw])
    return -out if out[0] < 0 else out


def layer_matrix(material, h, omega, velocity, wave):
    rho, a, f, cc, l, nn = map(mp.mpf, material)
    h = mp.mpf(h)
    k = omega/velocity
    if wave == "Love":
        # The core propagates SH with z=-h.
        g = mp.matrix([[0, 1/l], [nn*k*k-rho*omega**2, 0]])
        return mp.expm(-h*g)
    if not l:
        g = mp.matrix([[0, 1/cc-k*k/(rho*omega**2)], [-rho*omega**2, 0]])
        return mp.expm(h*g)
    # State [uz,szz,ur,szr], same physical first-order system as the core's
    # independently derived regular propagator, evaluated by mpmath expm.
    g = mp.matrix([[0, 1/cc, f*k/cc, 0],
                   [-rho*omega**2, 0, 0, k],
                   [-k, 0, 0, 1/l],
                   [0, -f*k/cc, (a-f*f/cc)*k*k-rho*omega**2, 0]])
    p = mp.expm(h*g)
    pairs = ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    compound = mp.matrix(6)
    for r, (i,j) in enumerate(pairs):
        for col, (k0,l0) in enumerate(pairs):
            compound[r,col] = p[i,k0]*p[j,l0]-p[i,l0]*p[j,k0]
    out = mp.matrix(5)
    for r in range(5):
        out[r,0] = compound[r,0]-compound[r,5]  # Y34=-Y12.
        for col in range(1,5):
            out[r,col] = compound[r,col]
    return out


def surface(case, omega, velocity):
    wave = case["wave"]
    y = initial(case["halfspace"], omega, velocity, wave)
    liquid = False
    cache = {}
    for p, h in reversed(list(zip(case["layers"], case["thicknesses"]))):
        if wave == "Rayleigh" and p[4] == 0 and not liquid:
            y = mp.matrix([y[2],y[4]])
            liquid = True
        key = tuple(p)+(h,)
        if key not in cache:
            cache[key] = layer_matrix(p,h,omega,velocity,wave)
        y = cache[key]*y
        if abs(y[0]) > mp.mpf(1e-20):
            y = y/y[0]
    return y, liquid


def secular(case, omega, velocity):
    y, liquid = surface(case,omega,velocity)
    return y[1] if liquid or case["wave"] == "Love" else y[4]


def triplets(function, w, c):
    value = function(w,c)
    dw = mp.diff(lambda x:function(x,c),w)
    dc = mp.diff(lambda x:function(w,x),c)
    if isinstance(value, mp.matrix):
        return [[float(part[i,j]) for part in (value,dw,dc)]
                for i in range(value.rows) for j in range(value.cols)]
    return [float(part) for part in (value,dw,dc)]


def reference(case, digits):
    with mp.workdps(digits):
        w,c = mp.mpf(case["omega"]),mp.mpf(case["c"])
        y,fluid = surface(case,w,c)
        # Rayleigh applies one final magnitude scale with its derivative
        # explicitly stopped. Freeze it at the evaluation point also here.
        scale = abs(y[2]) if not fluid and case["wave"] == "Rayleigh" else mp.mpf(1)
        if scale <= mp.mpf(1e-30): scale = mp.mpf(1)
        d = triplets(lambda ww,cc:secular(case,ww,cc)/scale,w,c)
        dw = mp.diff(lambda ww:secular(case,ww,c),w)
        dc = mp.diff(lambda cc:secular(case,w,cc),c)
        u = c/(1+w/c*dw/dc)
        out = {"D": d, "U": float(u), "U_decimal": mp.nstr(u,digits),
               "initial": triplets(lambda ww,cc:initial(case["halfspace"],ww,cc,case["wave"]),w,c)}
        p,h = case["layers"][case["selected"]],case["thicknesses"][case["selected"]]
        out["matrix"] = triplets(lambda ww,cc:layer_matrix(p,h,ww,cc,case["wave"]),w,c)
        if case["wave"] == "Rayleigh" and p[4] != 0:
            for index,name in enumerate(("nu1_sq","nu3_sq")):
                out[name] = [triplets(lambda ww,cc:mp.re(derived(p,ww,cc)[index]),w,c),
                             triplets(lambda ww,cc:mp.im(derived(p,ww,cc)[index]),w,c)]
        return out
