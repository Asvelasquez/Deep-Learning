# Semana 4 — Tecnicas de Optimizacion en Redes Neuronales

## Objetivo

Aplicar y comparar tecnicas de optimizacion en el entrenamiento de una red neuronal, evidenciando el optimizador y la tasa de aprendizaje afectan la velocidad de convergencia, la estabilidad y el desempeño final del modelo.

---

## Tecnicas comparadas

| Experimento | Variable | Valores |
|-------------|----------|---------|
| Exp. 1 | Optimizador | SGD · SGD+Momentum · Adam |
| Exp. 2 | Learning rate | 0.001 · 0.01 · 0.1 (con SGD) |

---

## Configuracion base

Red neuronal 2 → 8 → 4 → 1, activación ReLU en capas ocultas, Sigmoid en salida, loss Binary Cross-Entropy, dataset sintetico de clasificación binaria con 200 muestras, 500 épocas de entrenamiento.

---

## Resultado principal

Adam convergio mas rapido en el experimento 1, alcanzando el loss mas bajo en las primeras epocas. 
En el Experimento 2, LR=0.01 ofrecio el mejor balance entre velocidad y estabilidad
LR=0.1 produjo oscilaciones y LR=0.001 fue demasiado lento para converger en 500 epocas.

---

## Cómo ejecutar el notebook

1. Abrir URL(https://colab.research.google.com/drive/1VhFPme_RUC-I7wq0WJXe4jWfC0cG4SVT)
2. Ejecutar las celdas en orden 

No requiere instalación adicional. Solo usa `numpy`, `matplotlib` y `pandas`, que vienen preinstalados en Colab.

---

## Estructura del repositorio

```
Semana 4/
├── Ejercicio Semana 4.ipynb   # Notebook principal
└── README.md                  # Este archivo
```
