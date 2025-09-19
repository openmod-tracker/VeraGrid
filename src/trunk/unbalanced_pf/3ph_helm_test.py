import numpy as np
from numpy.linalg import solve, LinAlgError

# ---------------------------
# Utilidades Padé y residuo
# ---------------------------
def pade_num_den(c, M):
    """
    Aproximante de Padé [M/M] de la serie c(s)=sum c_k s^k.
    Devuelve (num, den) con Q(0)=1. Requiere c_0..c_{2M}.
    """
    if len(c) < 2*M + 1:
        raise ValueError("No hay coeficientes suficientes para Padé [M/M].")
    # Sistema para den[1..M]: sum_{j=1..M} den[j] * c_{k-j} = -c_k, k=M+1..2M
    A = np.zeros((M, M), dtype=complex)
    b = np.zeros(M, dtype=complex)
    for i in range(M):
        k = M + 1 + i
        for j in range(M):
            A[i, j] = c[k - 1 - j]  # c_{k-j-1}
        b[i] = -c[k]
    den_tail = solve(A, b)  # den[1..M]
    den = np.empty(M + 1, dtype=complex)
    den[0] = 1.0
    den[1:] = den_tail

    # Numerador: p_k = sum_{m=0..k} c_m * den_{k-m}, k=0..M
    num = np.zeros(M + 1, dtype=complex)
    for k in range(M + 1):
        num[k] = sum(c[m] * den[k - m] for m in range(k + 1))
    return num, den

def eval_rational(num, den, s=1.0):
    # num y den en potencias ascendentes: num[0] + num[1] s + ...
    # np.polyval espera descendentes -> invertimos
    return np.polyval(num[::-1], s) / np.polyval(den[::-1], s)

def power_flow_residual(V2, y, S2):
    """
    Residuo de potencia en el bus PQ:
    resid = S_calc - S_spec, con Ybus serie puro para 2-bus.
    """
    V1 = 1.0 + 0.0j
    Y = np.array([[ y, -y],
                  [-y,  y]], dtype=complex)
    V = np.array([V1, V2], dtype=complex)
    I = Y @ V
    S_inj = V * np.conjugate(I)
    return S_inj[1] - S2  # complejo


# ---------------------------
# HELM canónico 2-bus
# ---------------------------
def helm_series_coeffs_2bus(R=0.01, X=0.10, P_load=0.9, Q_load=0.3, max_k=400):
    """
    Devuelve coeficientes v, w, u hasta k=max_k para:
      Y21 V1 + Y22 V2 = s S2* U2
      Y21* W1 + Y22* W2 = s S2  V2
      U2 * W2 = 1
    Serie pura (sin shunt). P,Q en p.u. (consumo positivo).
    """
    Z = R + 1j*X
    y = 1 / Z
    Y22 = y

    V1 = 1.0 + 0.0j
    W1 = np.conjugate(V1)

    # Inyección (carga positiva => inyección negativa)
    S2  = -(P_load + 1j*Q_load)
    S2c = np.conjugate(S2)

    v = np.zeros(max_k + 1, dtype=complex)
    w = np.zeros(max_k + 1, dtype=complex)
    u = np.zeros(max_k + 1, dtype=complex)

    # Orden 0 (s=0): v0 = w0 = 1 en 2-bus serie puro
    v[0] = 1.0 + 0j
    w[0] = 1.0 + 0j
    u[0] = 1.0 + 0j

    for k in range(1, max_k + 1):
        # De la red reflejada (lineal):
        w[k] = (S2 / np.conjugate(Y22)) * v[k-1]
        # Recíproco U*W = 1:
        u[k] = -sum(u[k-m] * w[m] for m in range(1, k+1))
        # Red original (lineal):
        v[k] = (S2c / Y22) * u[k-1]

    return v, w, u, y, S2


def helm_2bus_accel(R=0.01, X=0.10, P_load=0.9, Q_load=0.3,
                    max_k=400, M_min=6, M_max=80):
    """
    Calcula V2 con:
      - suma directa de la serie (baseline),
      - búsqueda de Padé [M/M] con M en [M_min..M_max] que minimiza |residuo|.
    Devuelve (V2_best, info) con detalles.
    """
    v, w, u, y, S2 = helm_series_coeffs_2bus(R, X, P_load, Q_load, max_k=max_k)

    # Suma directa (puede ser mala en s=1 para cargas fuertes)
    V2_series = np.sum(v)
    resid_series = power_flow_residual(V2_series, y, S2)

    best = {
        "method": "series",
        "V2": V2_series,
        "resid": resid_series,
        "M": None
    }

    # Búsqueda adaptativa de Padé por residuo
    for M in range(M_min, min(M_max, (len(v)-1)//2) + 1):
        try:
            num, den = pade_num_den(v, M)
            # Evitar denominadores patológicos en s=1
            denom_at_1 = np.polyval(den[::-1], 1.0)
            if abs(denom_at_1) < 1e-14:
                continue
            V2_pade = eval_rational(num, den, 1.0)
            resid = power_flow_residual(V2_pade, y, S2)
            if abs(resid) < abs(best["resid"]):
                best = {"method": "pade", "V2": V2_pade, "resid": resid, "M": M}
        except (LinAlgError, ValueError, FloatingPointError, OverflowError):
            continue

    return best["V2"], {
        "best_method": best["method"],
        "best_M": best["M"],
        "best_resid": best["resid"],
        "V2_series": V2_series,
        "resid_series": resid_series
    }


# ---------------------------
# Demo con tu caso
# ---------------------------
if __name__ == "__main__":
    V2, info = helm_2bus_accel(
        R=0.01, X=0.10, P_load=0.9, Q_load=0.3,
        max_k=500,   # genera más coeficientes -> más libertad para Padé
        M_min=6, M_max=100
    )
    print(f"V2 ≈ {V2}  |V2| ≈ {abs(V2):.6f}")
    print(f"Mejor método: {info['best_method']}  M={info['best_M']}")
    print(f"|Residuo| ≈ {abs(info['best_resid']):.3e}")
