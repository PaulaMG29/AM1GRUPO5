"""Cálculos del motor cohete de propulsante sólido: apartados 1 y 2."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapezoid


# Datos del enunciado (SI)
r_e = 0.762
r_i = 0.254
L = 2.921
epsilon = 60.0
p_c_max = 5_963_954.274
c_star = 1527.0
r_dot_ref = 5.791e-3
p_c_ref = 6_894_744.825
n = 0.3
rho_pr = 1801.952
gamma_t = 1.18
gamma_c = 1.14
p_a = 0.0


def gamma_factor(gamma: float) -> float:
    """Factor geométrico del flujo compresible en la tobera."""
    return np.sqrt(gamma) * (2.0 / (gamma + 1.0)) ** (
        (gamma + 1.0) / (2.0 * (gamma - 1.0))
    )


def area_ratio_from_pressure_ratio(pressure_ratio: float, gamma: float) -> float:
    """Relación de áreas de una tobera para una relación de presiones dada."""
    momentum_term = (
        2.0
        * gamma
        / (gamma - 1.0)
        * (1.0 - pressure_ratio ** ((gamma - 1.0) / gamma))
    )
    return gamma_factor(gamma) / (
        pressure_ratio ** (1.0 / gamma) * np.sqrt(momentum_term)
    )


def supersonic_exit_pressure_ratio(area_ratio: float, gamma: float) -> float:
    """Resuelve la rama supersónica de la relación área-presión."""
    critical_ratio = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
    low = 1e-12
    high = critical_ratio * (1.0 - 1e-12)

    # En la rama supersónica, la relación de áreas decrece al aumentar p_s/p_c.
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if area_ratio_from_pressure_ratio(midpoint, gamma) > area_ratio:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def thrust_coefficient(
    pressure_ratio: float, area_ratio: float, gamma: float
) -> float:
    """Coeficiente de empuje para expansión en vacío."""
    momentum_term = (
        2.0
        * gamma
        / (gamma - 1.0)
        * (1.0 - pressure_ratio ** ((gamma - 1.0) / gamma))
    )
    return gamma_factor(gamma) * np.sqrt(momentum_term) + pressure_ratio * area_ratio


# ---------------------------------------------------------------------------
# Apartado 1: geometría y prestaciones iniciales
# ---------------------------------------------------------------------------
h_0 = np.sqrt(3.0) / 2.0 * (r_e - r_i)
m_pr = 6.0 * rho_pr * L * r_i * h_0
A_b_0 = 6.0 * L * (r_i + 2.0 * h_0)
a = r_dot_ref / p_c_ref**n

# El área de garganta se dimensiona para que p_c(0) = p_c,max.
A_g = c_star * rho_pr * A_b_0 * a * p_c_max ** (n - 1.0)
A_s = epsilon * A_g
r_dot_0 = a * p_c_max**n
m_dot_0 = rho_pr * A_b_0 * r_dot_0

p_s_over_p_c = supersonic_exit_pressure_ratio(epsilon, gamma_t)
p_s_0 = p_s_over_p_c * p_c_max
gamma_t_factor = gamma_factor(gamma_t)
#R_T_c_0 = (c_star * gamma_t_factor) ** 2
R_T_c_0 = (c_star * gamma_factor(gamma_c)) ** 2
v_s_0 = np.sqrt(
    2.0
    * gamma_t
    / (gamma_t - 1.0)
    * R_T_c_0
    * (1.0 - p_s_over_p_c ** ((gamma_t - 1.0) / gamma_t))
)
E_0 = m_dot_0 * v_s_0 + (p_s_0 - p_a) * A_s
I_sp_0 = E_0 / m_dot_0

print("APARTADO 1 - Prestaciones iniciales")
print(f"Altura de la sección hexagonal, h_0 = {h_0:.6f} m")
print(f"Masa inicial de propelente, m_pr = {m_pr:.3f} kg")
print(f"Área de quemado inicial, A_b,0 = {A_b_0:.3f} m^2")
print(f"Área de garganta, A_g = {A_g:.6f} m^2")
print(f"Área de salida, A_s = {A_s:.3f} m^2")
print(f"Velocidad de regresión inicial = {r_dot_0:.6f} m/s")
print(f"Gasto másico inicial, m_dot_0 = {m_dot_0:.3f} kg/s")
print(f"Relación de presiones p_s/p_c = {p_s_over_p_c:.6f}")
print(f"Presión de salida inicial, p_s,0 = {p_s_0:.3f} Pa")
print(f"R_T_c,0 = {R_T_c_0:.3f} J/kg")
print(f"Velocidad de salida, v_s,0 = {v_s_0:.3f} m/s")
print(f"Empuje inicial, E_0 = {E_0:.3f} N")
print(f"Impulso específico inicial, I_sp,0 = {I_sp_0:.3f} m/s")


# ---------------------------------------------------------------------------
# Apartado 2: evolución cuasiestacionaria de p_c(t) y E(t)
# ---------------------------------------------------------------------------
# A_b(y) = A_b,0 - 24 L y. El modelo geométrico se detiene cuando
# desaparece la primera de las dimensiones de propelente de la sección.
y_end = min(r_i / 2.0, h_0)
C = c_star * rho_pr * a / A_g
N = C * A_b_0
M = 24.0 * C * L
exponent = n / (1.0 - n)
integration_power = 1.0 - exponent

x_0 = N
x_end = C * (A_b_0 - 24.0 * L * y_end)
burn_time = (
    x_0**integration_power - x_end**integration_power
) / (integration_power * M * a)

t = np.linspace(0.0, burn_time, 1000)
x = (
    N**integration_power
    - integration_power * M * a * t
) ** (1.0 / integration_power)
y = (N - x) / M
A_b = A_b_0 - 24.0 * L * y
p_c = (C * A_b) ** (1.0 / (1.0 - n))

C_F = thrust_coefficient(p_s_over_p_c, epsilon, gamma_t)
E = C_F * A_g * p_c


print("\nAPARTADO 2 - Evolución cuasiestacionaria")
print(f"Tiempo de quemado hasta que desaparece la primera dimensión de propelente = {burn_time:.3f} s")
print(f"Presión de cámara inicial = {p_c[0]:.3f} Pa")
print(f"Presión de cámara final = {p_c[-1]:.3f} Pa")
print(f"Empuje inicial = {E[0]:.3f} N")
print(f"Empuje final = {E[-1]:.3f} N")


fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(t, p_c / 1e6, color="tab:blue")
axes[0].set_ylabel(r"$p_c(t)$ [MPa]")
axes[0].set_title("Evolución cuasiestacionaria del motor")
axes[0].grid(True)

axes[1].plot(t, E / 1e3, color="tab:red")
axes[1].set_xlabel("Tiempo [s]")
axes[1].set_ylabel(r"$E(t)$ [kN]")
axes[1].grid(True)

fig.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Apartado 2: sistema transitorio reteniendo la acumulación de masa
# ---------------------------------------------------------------------------
def chamber_volume(y_value: float) -> float:
    """Volumen de gas: volumen interior menos el volumen de propelente restante."""
    return 6.0 * L * (
        np.sqrt(3.0) / 4.0 * r_e**2
        - (h_0 - y_value) * (r_i - 2.0 * y_value)
    )



def transient_system(_time: float, state: np.ndarray) -> tuple[float, float]:
    pressure, regression = state
    if pressure < 0.0:
        raise RuntimeError("La presión de cámara no puede ser negativa.")

    burn_area = A_b_0 - 24.0 * L * regression
    gas_density = pressure / R_T_c_0
    pressure_rate = R_T_c_0 / chamber_volume(regression) * (
        burn_area * a * pressure**n * (rho_pr - gas_density)
        - pressure * A_g / c_star
    )
    regression_rate = a * pressure**n
    return pressure_rate, regression_rate


def burnout_event(_time: float, state: np.ndarray) -> float:
    return state[1] - r_i / 2.0


burnout_event.terminal = True
burnout_event.direction = 1.0

transient_solution = solve_ivp(
    transient_system,
    t_span=(0.0, 200.0),
    #y0=(p_c_max, 0.0),
    y0=(0.000001, 0.0),
    events=burnout_event,
    max_step=0.05,
    rtol=1e-9,
    atol=1e-11,
)
if not transient_solution.success:
    raise RuntimeError(
        f"No se pudo integrar el sistema transitorio: {transient_solution.message}"
    )
if len(transient_solution.t_events[0]) == 0:
    raise RuntimeError("La integración terminó sin alcanza el fin geométrico.")

t_transient = transient_solution.t
p_c_transient = transient_solution.y[0]
y_transient = transient_solution.y[1]
E_transient = C_F * A_g * p_c_transient

max_thrust_transient_index = np.argmax(E_transient)
max_thrust_transient = E_transient[max_thrust_transient_index]
thrust_10_percent = 0.10 * max_thrust_transient
threshold_crossings = np.flatnonzero(E_transient >= thrust_10_percent)
if len(threshold_crossings) == 0:
    raise RuntimeError(
        "El empuje transitorio no alcanza el 10 % de su máximo durante la "
        "integración."
    )
threshold_index = threshold_crossings[0]
if threshold_index == 0:
    t_0 = t_transient[0]
else:
    t_0 = t_transient[threshold_index - 1] + (
        (thrust_10_percent - E_transient[threshold_index - 1])
        * (t_transient[threshold_index] - t_transient[threshold_index - 1])
        / (E_transient[threshold_index] - E_transient[threshold_index - 1])
    )

print("\nAPARTADO 2 - Reteniendo la acumulación de masa")
print(f"Volumen inicial de cámara, V_c,0 = {chamber_volume(0.0):.3f} m^3")
print(f"Tiempo hasta y = r_i/2 = {transient_solution.t_events[0][0]:.3f} s")
print(f"Presión de cámara final = {p_c_transient[-1]:.3f} Pa")
print(f"Empuje final = {E_transient[-1]:.3f} N")
print(f"Máximo empuje alcanzado = {max_thrust_transient:.3f} N")
print(f"Umbral del 10 % del empuje máximo = {thrust_10_percent:.3f} N")
print(f"t_0 (subida al 10 % del empuje máximo) = {t_0:.6f} s")

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(t_transient, p_c_transient / 1e6, color="tab:blue", label="Transitorio")
axes[0].plot(t, p_c / 1e6, color="tab:green", label="Cuasi-estacionario")
axes[0].axvline(
    t_0, color="black", linestyle=":", label=fr"$t_0={t_0:.3f}$ s"
)
axes[0].set_ylabel(r"$p_c(t)$ [MPa]")
#axes[0].set_title("Modelo transitorio con acumulación de masa vs modelo cuasi-estacionario.")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t_transient, E_transient / 1e3, color="tab:red", label="Transitorio")
axes[1].plot(t, E / 1e3, color="tab:orange", label="Cuasi-estacionario")
axes[1].axhline(
    thrust_10_percent / 1e3,
    color="gray",
    linestyle="--",
    label="10 % del empuje máximo",
)
axes[1].axvline(t_0, color="black", linestyle=":", label=fr"$t_0={t_0:.3f}$ s")
axes[1].set_xlabel("Tiempo [s]")
axes[1].set_ylabel(r"$E(t)$ [kN]")
axes[1].grid(True)
axes[1].legend()

fig.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Pregunta 3: fase de cola, desde el fin de la combustión
# ---------------------------------------------------------------------------
t_burn = transient_solution.t_events[0][0]
p_c_burn = p_c_transient[-1]
y_burn = y_transient[-1]
V_c_burn = chamber_volume(y_burn)
rho_c_burn = p_c_burn / R_T_c_0
c_star_burn = c_star

# Escala temporal definida en el enunciado. En la fase de cola V_c es constante.
t_hat = c_star_burn * V_c_burn * rho_c_burn / (p_c_burn * A_g)

# En vacío p_c=p_a se alcanza asintóticamente; se grafica hasta esta fracción.
pressure_plot_ratio = 1e-2
delta_tau_adiabatic = (
    2.0
    / (gamma_c - 1.0)
    * (
        pressure_plot_ratio ** (-(gamma_c - 1.0) / (2.0 * gamma_c))
        - 1.0
    )
)
delta_tau_isothermal = -np.log(pressure_plot_ratio)
delta_t_adiabatic = t_hat * delta_tau_adiabatic
delta_t_isothermal = t_hat * delta_tau_isothermal

tau_adiabatic = np.linspace(0.0, delta_tau_adiabatic, 1000)
tau_isothermal = np.linspace(0.0, delta_tau_isothermal, 1000)
pc_ratio_adiabatic = (
    1.0 + (gamma_c - 1.0) / 2.0 * tau_adiabatic
) ** (-2.0 * gamma_c / (gamma_c - 1.0))
pc_ratio_isothermal = np.exp(-tau_isothermal)

t_adiabatic = t_burn + t_hat * tau_adiabatic
t_isothermal = t_burn + t_hat * tau_isothermal
pc_adiabatic = p_c_burn * pc_ratio_adiabatic
pc_isothermal = p_c_burn * pc_ratio_isothermal
thrust_adiabatic = C_F * A_g * pc_adiabatic
thrust_isothermal = C_F * A_g * pc_isothermal
tail_thrust_ratio_10_percent = thrust_10_percent / (C_F * A_g * p_c_burn)
if not 0.0 < tail_thrust_ratio_10_percent < 1.0:
    raise RuntimeError(
        "El 10 % del empuje máximo no queda por debajo del empuje al inicio "
        "de la fase de cola; no existe un cruce posterior en la cola."
    )
t_a1_adiabatic = t_burn + t_hat * (
    2.0
    / (gamma_c - 1.0)
    * (
        tail_thrust_ratio_10_percent
        ** (-(gamma_c - 1.0) / (2.0 * gamma_c))
        - 1.0
    )
)
t_a1_isothermal = t_burn + t_hat * np.log(
    1.0 / tail_thrust_ratio_10_percent
)

print("\nPREGUNTA 3 - Fase de cola")
print(f"Tiempo de quemado, t_burn = {t_burn:.3f} s")
print(f"Presión al inicio de la cola, p_c(t_burn) = {p_c_burn:.3f} Pa")
print(f"Escala temporal, t_hat = {t_hat:.6f} s")
print(
    "Instante en que la cola vuelve al 10 % del empuje máximo: "
    f"t_a1_adiabatic = {t_a1_adiabatic:.6f} s (adiabático), "
    f"t_a1_isothermal = {t_a1_isothermal:.6f} s (isotérmico)"
)
print(
    f"Tiempo adicional hasta p_c/p_c(t_burn) = {pressure_plot_ratio:g}: "
    f"adiabático {delta_t_adiabatic:.3f} s, "
    f"isotérmico {delta_t_isothermal:.3f} s"
)

t_application_adiabatic = t_a1_adiabatic-t_0
t_application_isothermal = t_a1_isothermal-t_0
t_burn_real = t_burn-t_0

print(
        "Tiempo de aplicación y tiempo burn real: "
        f"adiabático {t_application_adiabatic:.3f} s, "
        f"isotérmico {t_application_isothermal:.3f} s"
        f"t_burn_real {t_burn_real:.3f} s"
    )

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
axes[0].plot(t_adiabatic, pc_adiabatic / 1e6, label="Adiabático")
axes[0].plot(t_isothermal, pc_isothermal / 1e6, "--", label="Isotérmico")
axes[0].axvline(t_burn, color="black", linestyle=":", label=r"$t_{b}$")
axes[0].axvline(
    t_a1_adiabatic,
    color="tab:blue",
    linestyle=":",
    label=fr"$t_{{a,ad}}={t_a1_adiabatic:.3f}$ s",
)
axes[0].axvline(
    t_a1_isothermal,
    color="tab:orange",
    linestyle=":",
    label=fr"$t_{{a,isot}}={t_a1_isothermal:.3f}$ s",
)
axes[0].set_ylabel(r"$p_c(t)$ [MPa]")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t_adiabatic, thrust_adiabatic / 1e3, label="Adiabático")
axes[1].plot(t_isothermal, thrust_isothermal / 1e3, "--", label="Isotérmico")
axes[1].axvline(t_burn, color="black", linestyle=":")
axes[1].axhline(
    thrust_10_percent / 1e3,
    color="gray",
    linestyle="--",
    label="10 % del empuje máximo",
)
axes[1].axvline(t_a1_adiabatic, color="tab:blue", linestyle=":")
axes[1].axvline(t_a1_isothermal, color="tab:orange", linestyle=":")
axes[1].set_xlabel("Tiempo desde el fin de combustion [s]")
axes[1].set_ylabel(r"$E(t)$ [kN]")
axes[1].grid(True)
axes[1].legend()

fig.tight_layout()
plt.show()


print("\nCONTRIBUCIÓN AL IMPULSO TOTAL - FASE DE COLA")

transient_interval_mask = (t_transient > t_0) & (t_transient < t_burn)
t_impulse_transient = np.concatenate(
    ([t_0], t_transient[transient_interval_mask], [t_burn])
)
thrust_impulse_transient = np.interp(
    t_impulse_transient, t_transient, E_transient
)
impulse_permanent_transient_t0 = trapezoid(
    thrust_impulse_transient, x=t_impulse_transient
)
print(
    "Impulso de la fase permanente transitoria (de t_0 a t_burn) = "
    f"{impulse_permanent_transient_t0:.3f} N·s"
)

adiabatic_interval_mask = (t_adiabatic > t_burn) & (
    t_adiabatic < t_a1_adiabatic
)
t_impulse_adiabatic = np.concatenate(
    ([t_burn], t_adiabatic[adiabatic_interval_mask], [t_a1_adiabatic])
)
thrust_impulse_adiabatic = np.interp(
    t_impulse_adiabatic, t_adiabatic, thrust_adiabatic
)
tail_impulse_adiabatic = trapezoid(
    thrust_impulse_adiabatic, x=t_impulse_adiabatic
)

isothermal_interval_mask = (t_isothermal > t_burn) & (
    t_isothermal < t_a1_isothermal
)
t_impulse_isothermal = np.concatenate(
    ([t_burn], t_isothermal[isothermal_interval_mask], [t_a1_isothermal])
)
thrust_impulse_isothermal = np.interp(
    t_impulse_isothermal, t_isothermal, thrust_isothermal
)
tail_impulse_isothermal = trapezoid(
    thrust_impulse_isothermal, x=t_impulse_isothermal
)

print(f"Impulso de cola adiabático = {tail_impulse_adiabatic:.3f} N·s")
print(f"Impulso de cola isotérmico = {tail_impulse_isothermal:.3f} N·s")
print(
    "Diferencia (isotérmico - adiabático) = "
    f"{tail_impulse_isothermal - tail_impulse_adiabatic:.3f} N·s"
)