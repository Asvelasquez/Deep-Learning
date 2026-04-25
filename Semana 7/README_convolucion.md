# Convolucion de Matrices e Imagenes — Padding y Stride

## Objetivo

Implementar la operacion de convolucion manualmente con NumPy y demostrar con evidencia practica como padding y stride modifican las dimensiones del mapa de características y la informacion preservada.

---

## Qué se comparo

Misma imagen sintética 32×32 y mismo kernel Sobel horizontal (3×3) en todas las comparaciones.

| Experimento | Variable | Valores |
|-------------|----------|---------|
| Padding | Filas/columnas de ceros agregadas | 0, 1, 2 (stride=1 fijo) |
| Stride | Paso de desplazamiento del kernel | 1, 2, 3 (padding=0 fijo) |
| Kernels | Tipo de filtro | Sobel H, Sobel V, Sharpen, Blur (padding=1, stride=1) |

---

## Evidencia principal

Con `padding=0` y `stride=1` sobre una imagen 32×32, el mapa de salida es 30×30 (−12% de área). Con `padding=1` la salida preserva las dimensiones originales (32×32). Con `stride=2` el mapa se reduce a 15×15 (solo 22% del área original), perdiendo detalle espacial pero reduciendo el costo computacional.

---

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/drive/1_5Sm9oY-9iAIp1Go2NlAT2LWcmpXv0fn)
2. Ejecutar las celdas en orden

No requiere instalación adicional. Solo usa `numpy` y `matplotlib`, preinstalados en Colab.
