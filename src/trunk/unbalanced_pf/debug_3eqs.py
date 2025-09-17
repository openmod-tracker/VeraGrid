import numpy as np

C = np.array([
    [-0.06094868484727337+0.1108157906314061j,  0j,                                0.06094868484727337-0.1108157906314061j],
    [ 0.06094868484727337-0.1108157906314061j, -0.06094868484727337+0.1108157906314061j, 0j],
    [ 0j,                                        0.06094868484727337-0.1108157906314061j, -0.06094868484727337+0.1108157906314061j]
])

extra = 1e-12
D = np.array([
    [0.08237747920665388-0.13695905310300704j + extra, -0.04218873960332694+0.06897952655150352j, -0.04018873960332694+0.06797952655150352j],
    [-0.04218873960332694+0.06897952655150352j,  0.08337747920665388-0.13745905310300705j + extra, -0.04118873960332694+0.06847952655150352j],
    [-0.04018873960332694+0.06797952655150352j, -0.04118873960332694+0.06847952655150352j,  0.08137747920665388-0.13645905310300704j + extra]
])

# C = np.array([
#     [-0.10556621880998082+0.19193857965451055j,  0j,                                 0j],
#     [ 0j,                                       -0.10556621880998082+0.19193857965451055j, 0j],
#     [ 0j,                                        0j,                                -0.10556621880998082+0.19193857965451055j]
# ])

# D = np.array([
#     [0.12656621890998082-0.20693857965451057j,  0j,                                 0j],
#     [0j,                                        0.12356621890998082-0.20543857965451057j, 0j],
#     [0j,                                        0j,                                 0.12056621890998082-0.20393857965451057j]
# ])


# 0 = C * Vsl + D * Vload

Vsl = np.array([1.0 * np.exp(1j * 0.0), 1.0 * np.exp(1j * -120 * np.pi / 180), 1.0 * np.exp(1j * 120 * np.pi / 180)])


# Direct formulation with current
Vload = np.linalg.solve(D, -C @ Vsl)
Vload2 = np.linalg.inv(D) @ (-C @ Vsl)

print(abs(Vload))
print(abs(Vload2))

cond_number = np.linalg.cond(D)
print(cond_number )
print(np.linalg.det(D))

check = C @ Vsl + D @ Vload
print(check)

print('--------')

# Power formulation
# x = [vre1, vei1, vere2, vei2, vere3, vei3]
x = np.ones(6, dtype=float)
ang_init = 00

v1a = 0.91773 * np.exp(1j * -32.6 * np.pi / 180)
v1b = 0.90638 * np.exp(1j * -152.2 * np.pi / 180)
v1c = 0.9172 * np.exp(1j * 88.2 * np.pi / 180)

x[0] = np.real(v1a) 
x[1] = np.imag(v1a)
x[2] = np.real(v1b)
x[3] = np.imag(v1b)
x[4] = np.real(v1c)
x[5] = np.imag(v1c)


# x[0] = 1.0 * np.cos(ang_init * np.pi / 180)
# x[1] = 1.0 * np.sin(ang_init * np.pi / 180)
# x[2] = 1.0 * np.cos((ang_init - 120) * np.pi / 180)
# x[3] = 1.0 * np.sin((ang_init - 120) * np.pi / 180)
# x[4] = 1.0 * np.cos((ang_init + 120) * np.pi / 180)
# x[5] = 1.0 * np.sin((ang_init + 120) * np.pi / 180)


for i in range(1000):
    xcomplex = np.array([x[0] + 1j * x[1], x[2] + 1j * x[3], x[4] + 1j * x[5]])
    res_pq = xcomplex * np.conj(D @ xcomplex + C @ Vsl)
    res_p = np.real(res_pq)
    res_q = np.imag(res_pq)

    res_tot_old = np.array([res_p, res_q]).flatten()

    xc = np.copy(x)

    J = np.zeros((len(x), len(x)))

    for j in range(len(x)):
        xc[j] += 1e-6
        xcomplex = np.array([xc[0] + 1j * xc[1], xc[2] + 1j * xc[3], xc[4] + 1j * xc[5]])
        res_pq = xcomplex * np.conj(D @ xcomplex + C @ Vsl)
        res_p = np.real(res_pq)
        res_q = np.imag(res_pq)
        res_tot = np.array([res_p, res_q]).flatten()
        J[:, j] = (res_tot - res_tot_old) / 1e-6
        xc[j] -= 1e-6

    print(J)

    dx = np.linalg.solve(J, -res_tot_old)
    x += 0.01 * dx

    print(x)
    print(abs(res_tot_old))
    print(abs(res_tot))
    print(x)
    print('--------')

    if np.max(abs(dx)) < 1e-6:
        break

va = x[0] + 1j * x[1]
vb = x[2] + 1j * x[3]
vc = x[4] + 1j * x[5]

print(va)
print(vb)
print(vc)

print(abs(va))
print(abs(vb))
print(abs(vc))