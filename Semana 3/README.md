# Backpropagation y Funciones de Activación

## Objetivo

Implementar y validar el proceso de aprendizaje de redes neuronales mediante backpropagation, evidenciando cómo se ajustan los parámetros (pesos y sesgos) para reducir el error, y cómo las funciones de activación influyen en el comportamiento del modelo.

---

## Qué se implementó

Tres modelos de red neuronal construidos desde cero en NumPy (sin frameworks):

| Modelo | Arquitectura | Tarea |
|--------|-------------|-------|
| Perceptrón simple | 2 → 1 | Clasificación binaria AND |
| Red 1 capa oculta | 2 → 4 → 1 | Clasificación binaria XOR |
| Red multicapa | 2 → 8 → 4 → 1 | Clasificación binaria XOR |

Cada modelo incluye forward pass, cálculo de loss (MSE), backpropagation explícito y actualización de parámetros por gradient descent, tambien se añade evidencia de pesos antes y después del entrenamiento para verificar el ajuste.

---

## Activaciones comparadas

**Sigmoid** vs **ReLU**  misma semilla, mismos datos, misma arquitectura y mismo learning rate en cada modelo. Solo varía la función de activación de las capas ocultas. La capa de salida usa Sigmoid en todos los casos para producir una probabilidad entre 0 y 1.

---

## Resultados principales

Sigmoid resulto mas estable en los tres modelos, con curvas de descenso suaves en AND y XOR
ReLU alcanzó resultados similares en los Modelos 2 y 3, pero mostró mayor inestabilidad en el perceptrón simple al operar sin capa de amortiguación.

---

## Cómo ejecutar el notebook

1. Abrir URL (https://colab.research.google.com/drive/1q78epdxfBUaHLecdjH8IUDMPe5HhhrUa)
3. Ejecutar las celdas en orden con **Runtime → Run all**

Solo usa `numpy`, `matplotlib` y `pandas`, que vienen preinstalados en Colab.

## Estructura del repositorio

```
Deep-Learning/
└── Semana3/
    ├── Ejercicio Semana 3.ipynb   # Notebook principal
    └── README.md       # Este archivo
```