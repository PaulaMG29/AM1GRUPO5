"""Retratos de fase del sistema lineal asociado a la matriz 10.

    x' = y
    y' = a*x - 2*y

Las trayectorias se integran numéricamente con solve_ivp. Ejecuta este
archivo para imprimir la clasificación y mostrar los retratos de fase.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Se elige un valor representativo en cada régimen y el caso degenerado a=-1.
VALORES_A = [-2, -1.0, -0.5, 0.0, 1.0]
TIEMPO_MAXIMO = 8.0
LIMITE_TRAYECTORIAS = 6.0
LIMITE_GRAFICA = 3.5


def matriz(a):
    """Devuelve la matriz del sistema para el parámetro a."""
    return np.array([[0.0, 1.0], [a, -2.0]])


def campo(t, estado, a):
    """Campo vectorial (x', y') del sistema."""
    x, y = estado
    return np.array([y, a * x - 2.0 * y])


def autovalores(a):
    """Autovalores exactos expresados como números complejos."""
    return -1.0 + complex(1.0 + a) ** 0.5, -1.0 - complex(1.0 + a) ** 0.5


def clasificar(a):
    """Clasifica el origen según el valor del parámetro."""
    tolerancia = 1e-10
    if abs(a) < tolerancia:
        return "Equilibrios no aislados: toda la recta y=0"
    if a > 0:
        return "Punto silla (inestable)"
    if abs(a + 1.0) < tolerancia:
        return "Nodo impropio estable (autovalor doble)"
    if a < -1.0:
        return "Foco estable"
    return "Nodo estable"


def integrar_desde(estado_inicial, a, direccion):
    """Integra una semiórbita y la detiene al salir de la región dibujada."""

    def salir_de_ventana(t, estado, parametro):
        return LIMITE_TRAYECTORIAS - np.max(np.abs(estado))

    salir_de_ventana.terminal = True
    salir_de_ventana.direction = -1

    tiempos = np.linspace(0.0, direccion * TIEMPO_MAXIMO, 1200)
    solucion = solve_ivp(
        campo,
        (0.0, direccion * TIEMPO_MAXIMO),
        estado_inicial,
        args=(a,),
        t_eval=tiempos,
        events=salir_de_ventana,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.04,
    )
    return solucion.y.T


def integrar_orbita(estado_inicial, a):
    """Construye una órbita con su pasado y su futuro numéricos."""
    pasado = integrar_desde(estado_inicial, a, -1.0)[::-1]
    futuro = integrar_desde(estado_inicial, a, 1.0)
    return np.vstack((pasado, futuro[1:]))


def dibujar_flecha(eje, puntos, color):
    """Añade una flecha siguiendo el sentido creciente del tiempo."""
    if len(puntos) < 4:
        return
    indice = min(int(len(puntos) * 0.62), len(puntos) - 2)
    eje.annotate(
        "",
        xy=puntos[indice + 1],
        xytext=puntos[indice],
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4},
    )


def direcciones_propias_reales(a):
    """Devuelve vectores propios reales de autovalores no nulos."""
    valores, vectores = np.linalg.eig(matriz(a))
    direcciones = []
    for indice, valor in enumerate(valores):
        if abs(valor.imag) > 1e-9 or abs(valor.real) < 1e-9:
            continue
        vector = np.real(vectores[:, indice])
        vector /= np.linalg.norm(vector)
        if any(abs(np.dot(vector, anterior)) > 1.0 - 1e-8 for anterior in direcciones):
            continue
        direcciones.append(vector)
    return direcciones


def dibujar_retrato(eje, a):
    """Dibuja órbitas numéricas, flechas y puntos/rectas de equilibrio."""
    semillas = [
        (1.8, 0.0),
        (-1.8, 0.0),
        (0.0, 1.8),
        (0.0, -1.8),
        (1.5, 1.5),
        (-1.5, 1.5),
        (1.5, -1.5),
        (-1.5, -1.5),
    ]

    for semilla in semillas:
        puntos = integrar_orbita(np.array(semilla, dtype=float), a)
        linea, = eje.plot(puntos[:, 0], puntos[:, 1], color="C0", lw=1.15)
        dibujar_flecha(eje, puntos, linea.get_color())

    # Las direcciones propias reales generan órbitas rectas; se señalan aparte.
    for direccion in direcciones_propias_reales(a):
        puntos = integrar_orbita(0.35 * direccion, a)
        linea, = eje.plot(
            puntos[:, 0],
            puntos[:, 1],
            color="C3",
            lw=1.8,
            ls="--",
        )
        dibujar_flecha(eje, puntos, linea.get_color())

    if abs(a) < 1e-10:
        eje.plot(
            [-LIMITE_GRAFICA, LIMITE_GRAFICA],
            [0.0, 0.0],
            color="C2",
            lw=2.0,
            label="Recta de puntos de equilibrio",
        )

    eje.plot(0.0, 0.0, "ko", ms=5, label="Origen")
    eje.set_title(f"a = {a:g}: {clasificar(a)}")
    eje.set_xlabel("x")
    eje.set_ylabel("y = x'")
    eje.set_xlim(-LIMITE_GRAFICA, LIMITE_GRAFICA)
    eje.set_ylim(-LIMITE_GRAFICA, LIMITE_GRAFICA)
    eje.set_aspect("equal", adjustable="box")
    eje.grid(True, alpha=0.35)


def main():
    print("Sistema: x' = y,  y' = a*x - 2*y")
    print("Autovalores: lambda = -1 +/- sqrt(1+a)")
    print("\nClasificación para los valores representados:")
    for a in VALORES_A:
        lam1, lam2 = autovalores(a)
        print(
            f"  a = {a:5g}: lambda = {lam1:.5g}, {lam2:.5g}; "
            f"{clasificar(a)}"
        )

    print(
        "\nCambios en la clasificación espectral: a=-1 (nodo/foco) y a=0 "
        "(aparición de una recta de equilibrios y pérdida de estabilidad "
        "asintótica)."
    )
    print(
        "Como bifurcación topológica del retrato de fase, a=0 es el valor "
        "relevante: para a<-1 hay sumideros focales, y para -1<a<0 "
        "sumideros nodales, que son topológicamente equivalentes."
    )

    figura, ejes = plt.subplots(2, 3, figsize=(15, 9))
    for eje, a in zip(ejes.flat, VALORES_A):
        dibujar_retrato(eje, a)

    ejes.flat[-1].axis("off")
    figura.suptitle(
        "Retratos de fase numéricos para A = [[0, 1], [a, -2]]\n"
        "Las flechas indican el sentido de avance del tiempo; las líneas "
        "discontinuas rojas son órbitas rectas",
        fontsize=13,
    )
    figura.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
