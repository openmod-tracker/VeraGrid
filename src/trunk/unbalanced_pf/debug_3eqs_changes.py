import numpy as np

# Matrices del trafo+carga (Ytf, Ytt)
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

# Slack de 1∠0, -120, 120 grados
Vsl = np.array([
    1.0 * np.exp(1j * 0.0),
    1.0 * np.exp(1j * -120 * np.pi / 180),
    1.0 * np.exp(1j * 120 * np.pi / 180)
])

# Inicialización con formulación directa de corriente
Vload0 = np.linalg.solve(D, -C @ Vsl)

# Vector de variables reales: [Re(Va), Im(Va), Re(Vb), Im(Vb), Re(Vc), Im(Vc)]
x = np.zeros(6, dtype=float)
x[0], x[1] = np.real(Vload0[0]), np.imag(Vload0[0])
x[2], x[3] = np.real(Vload0[1]), np.imag(Vload0[1])
x[4], x[5] = np.real(Vload0[2]), np.imag(Vload0[2])

# Iteraciones Newton–Raphson
for it in range(50):
    # Calcula residuo actual
    xcomplex = np.array([x[0] + 1j * x[1], x[2] + 1j * x[3], x[4] + 1j * x[5]])
    res_pq = xcomplex * np.conj(D @ xcomplex + C @ Vsl)
    res_tot = np.array([np.real(res_pq), np.imag(res_pq)]).flatten()

    norm_res = np.linalg.norm(res_tot, np.inf)
    print(f"Iter {it}: Residuo = {norm_res:.3e}")
    if norm_res < 1e-8:
        break

    # Jacobiano numérico
    J = np.zeros((len(x), len(x)))
    for j in range(len(x)):
        xc = np.copy(x)
        xc[j] += 1e-6
        xcomplex = np.array([xc[0] + 1j * xc[1], xc[2] + 1j * xc[3], xc[4] + 1j * xc[5]])
        res_pq = xcomplex * np.conj(D @ xcomplex + C @ Vsl)
        res_new = np.array([np.real(res_pq), np.imag(res_pq)]).flatten()
        J[:, j] = (res_new - res_tot) / 1e-6

    dx = np.linalg.solve(J, -res_tot)

    # Damping adaptativo
    alpha = 1.0
    while alpha > 1e-4:
        xtest = x + alpha * dx
        xcomplex = np.array([xtest[0] + 1j * xtest[1], xtest[2] + 1j * xtest[3], xtest[4] + 1j * xtest[5]])
        res_pq = xcomplex * np.conj(D @ xcomplex + C @ Vsl)
        res_test = np.array([np.real(res_pq), np.imag(res_pq)]).flatten()
        if np.linalg.norm(res_test) < norm_res:
            x = xtest
            break
        alpha *= 0.5

# Resultado final
va = x[0] + 1j * x[1]
vb = x[2] + 1j * x[3]
vc = x[4] + 1j * x[5]

print("Tensiones finales en bus 2:")
print(f"Va = {abs(va):.5f} ∠ {np.angle(va, deg=True):.1f}°")
print(f"Vb = {abs(vb):.5f} ∠ {np.angle(vb, deg=True):.1f}°")
print(f"Vc = {abs(vc):.5f} ∠ {np.angle(vc, deg=True):.1f}°")
