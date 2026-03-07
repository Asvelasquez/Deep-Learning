# Semana 2 - Red Neuronal Básica (Perceptrón)

## Objetivo

Implementar y validar una neurona/perceptrón desde cero en Google Colab, con el fin de consolidar los conceptos fundamentales del aprendizaje profundo: entradas, pesos, sesgo (bias), cálculo del puntaje z, función de activación y clasificación binaria (0/1).

---

## ¿Qué se implementó?

- Una neuronal con dos entradas (`x1`, `x2`), pesos (`w1`, `w2`) y un sesgo (`b`).
- La función de activación tipo **escalón (step)**: retorna `1` si `z >= 0`, de lo contrario `0`.
- El cálculo del puntaje z: `z = x1·w1 + x2·w2 + b`.
- Simulación del comportamiento de una **compuerta lógica AND**.

---

## Pruebas realizadas

Se evaluaron los 4 casos posibles de entrada binaria para una compuerta AND:

| x1 | x2 | z    | Salida |
|----|----|------|--------|
| 0  | 0  | -2.0 | 0      |
| 0  | 1  | -1.0 | 0      |
| 1  | 0  | -1.0 | 0      |
| 1  | 1  |  0.0 | 1      |

Parámetros usados: `w1 = 1`, `w2 = 1`, `b = -2.0`

---

## Resultados principales

Con `b = -2.0`, la neurona replica correctamente el comportamiento de una compuerta AND, activándose únicamente cuando ambas entradas son 1. Se observó que aumentar el bias genera más salidas en 1, mientras que bajarlo produce más salidas en 0.

---

## Conclusiones

- El **bias** controla el umbral de activación: valores más altos facilitan la activación, valores más bajos la restringen.
- Los **pesos** determinan la influencia de cada entrada en la decisión final.
- Una sola neurona es suficiente para modelar funciones linealmente separables como AND.

---

## Cómo ejecutar el notebook

1. Abre el archivo `Semana1.ipynb` en [Google Colab](https://colab.research.google.com/)
2. Ve a **Runtime → Run all** (`Ctrl + F9`)
3. Verifica las salidas impresas para cada caso de prueba

> No se requieren librerías externas adicionales. Solo se usa `numpy`.

---

## Estructura del repositorio

```
Deep-Learning/
└── Semana2/
    ├── Ejercicio Semana 2.ipynb   # Notebook principal
    └── README.md       # Este archivo
```
