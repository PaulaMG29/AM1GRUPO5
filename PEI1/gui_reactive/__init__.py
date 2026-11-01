"""
Grupo gui_reactive: interfaz gráfica (Tkinter + Matplotlib) dividida en capas.

    main                         punto de entrada: crea la ventana Tk y lanza la app
    app                          carcasa: estado, menú lateral, navegación y cálculo en segundo plano
    window_initial_conditions    ventana 1 «Satélite y Condiciones Iniciales»
    window_numerical_schemes     ventana 2 «Esquemas Numéricos» (+ botón Ejecutar)
    window_numerical_integration ventana 3 «Integración numérica»
    window_estimated_error       ventana 4 «Error Estimado»
    simulation                   puente con numerical_engine y space_physics (sin Tkinter)
    widgets                      widgets reutilizables (etiqueta LaTeX, lienzo, tarjeta, slider)
    helpers                      geometría del CubeSat y post-proceso para dibujar
    config                       constantes y esquemas disponibles
"""
