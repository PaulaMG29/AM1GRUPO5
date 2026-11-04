---
title: PEI 1 · Simulador de actitud satelital
markmap:
  colorFreezeLevel: 2
  initialExpandLevel: -1
  maxWidth: 340
  spacingVertical: 6
  spacingHorizontal: 90
---

# PEI 1 · Simulador de actitud satelital

## numerical_engine
### `linear_solvers.py`
- `partition(a, b, N)` → malla temporal uniforme de $N$ pasos
### `non_linear_solvers.py`
- `our_jacobian(F, U, t)`
  - Diferencias centradas de 2º orden, error $\mathcal{O}(h^2)$
  - $h = 10^{-6} \approx \varepsilon_{\text{máq}}^{1/3}$
  - Admite $U$ complejo (regiones de estabilidad)
- `our_Newton(F, U0, t, tol, max_iter)`
  - Resuelve $J\,\Delta U = -F(U)$ en cada iteración
  - Para cuando $\lVert \Delta U \rVert < 10^{-6}$ (máx. 100 iteraciones)
### `numerical_schemes.py`
- Explícitos
  - `Euler` → orden 1
  - `RK4` → orden 4
- Implícitos (resueltos con `our_Newton`)
  - `Crank_Nicolson` → orden 2
  - `Inverse_Euler` → orden 1
### `Cauchy_problem.py`
- `Cauchy_Problem(Esquema, U0, F, t)`
  - Bucle $U^{n+1} = \text{Esquema}(U^n, t_n, \Delta t, F)$
  - Cualquier esquema con cualquier $F$
### `Richardson_error.py`
- `estimate_scheme_order`
  - 5 mallas: de $N/4$ a $4N$ pasos
  - Compara cada malla con la de paso mitad
  - Órdenes locales → mediana → ajuste de la recta $\ln E$ – $\ln N$
  - Descarta los puntos fuera de la región asintótica
- `Richardson_Error`
  - $E = (U_{\Delta t/2} - U_{\Delta t})\,/\,(1 - 2^{-q})$
### `numerical_stability_region.py`
- `create_mesh` → malla del plano complejo $z$
- `stability_region`
  - Ecuación test $U' = zU$ con $\Delta t = 1$
  - Un paso desde $U_0 = 1$ → factor $R(z)$
  - Estable si $|R(z)| \le 1$

## space_physics
### `Euler_equations.py`
- `Euler_Equations(omega, t, inertia=None)`
  - $I_1\dot\omega_1 = (I_2 - I_3)\,\omega_2\omega_3$ (y cíclicas)
  - Inercias globales $I_1, I_2, I_3$ que fija la GUI
  - `inertia` opcional para no tocar las globales
### `attitude_equations.py`
- `Attitude_Equations(U, t)`
  - Estado de 12 componentes: $\omega$ + matriz $R$
  - Dinámica de Euler + cinemática $\dot R = R\,[\omega]_\times$
- `nearest_rotation(R)` → rotación más cercana (SVD) para dibujar
### Invariantes
- `kinetic_energy.py` → `T` = energía cinética
- `angular_momentum.py` → `L` = $\lVert \mathbf{L} \rVert$
- `invariants_drift.py` → deriva relativa de $2T$ y $L^2$
### Estabilidad de los equilibrios
- `equilibrium_espectrum.py` → `equilibrium_spectrum`
  - Equilibrio $\omega^*_k = \Omega\,\mathbf{e}_k$
  - Autovalores del Jacobiano de `Euler_Equations`
  - El Jacobiano llega como argumento: la GUI pasa `our_jacobian`
  - Silla (inestable) o centro (estable)
- `exact_jacobian.py` → `exact_Jacobian` (referencia analítica)
- Efecto Dzhanibekov: el eje intermedio $I_2$ es inestable
### Geometría del movimiento
- `separatrix.py` → separatriz sobre el elipsoide $L^2$

## gui_reactive
### Capas
- `main.py` → punto de entrada
- `app.py` → carcasa
  - Estado compartido entre ventanas
  - Menú lateral y navegación
  - Cálculo en un hilo secundario
  - Selector de esquema (ventanas 3 y 4)
- `config.py` → `SCHEMES`, colores, títulos, tipografía LaTeX
- `widgets.py`
  - `MathLabel` → texto LaTeX en Tk
  - `make_canvas`, `make_card`, `slider_control`
- `helpers.py` → post-proceso y dibujo
  - `satellite_model` (CubeSat 3U), `shaded_colors`, `ellipsoid`
  - `stability_map` (en caché), `amplification_at`
  - `error_summary`, `latex_sci`, `set_inertias`
- `simulation.py` → puente con los motores
  - `run_scheme`, `richardson`
### Ventana 1 · Satélite y condiciones iniciales
- Sliders $I_1 < I_2 < I_3$, $\Omega$ y $\varepsilon$
- Eje del giro nominal
- CubeSat 3D con las proporciones de las inercias
- Elipsoides $2T$ y $L^2$ con la separatriz
- Tarjeta espectral: autovalores y tiempo del primer volteo
### Ventana 2 · Esquemas numéricos
- $\Delta t$, $t_f$ y esquema
- 4 regiones de estabilidad con los puntos $\lambda\,\Delta t$
- Resumen: factor $|R(z)|$ y número de pasos
- Botón Ejecutar
### Ventana 3 · Integración numérica
- Actitud animada + viñetas $t$ y ángulo $(\mathbf{e}_k, \mathbf{L})$
- Ángulo entre $\mathbf{e}_k$ y $\mathbf{L}$
- Espacio de fases: polhodia
- $\omega(t)$
### Ventana 4 · Error estimado
- Tarjeta de error máximo
- Error de Richardson $E(t)$
- Convergencia + orden estimado $q$ y teórico $p$
- Deriva de energía y de momento (4 esquemas)
