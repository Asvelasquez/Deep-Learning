# Semana 4 — Regularizacion en Redes Neuronales

## Objetivo

Aplicar métodos de regularización a una red neuronal para reducir el sobreajuste (overfitting) y mejorar la capacidad de generalización del modelo, comparando el comportamiento en entrenamiento y validación con y sin regularización.

---

## Métodos aplicados

| Técnica | Descripción |
|---------|-------------|
| L2 (weight decay) | Penaliza pesos grandes sumando `λ‖W‖²` al loss |
| Dropout | Desactiva aleatoriamente el 30% de neuronas por época durante entrenamiento |
| L2 + Dropout | Combinación de ambas técnicas |

---

## Comparación realizada

Cuatro modelos con la misma arquitectura (2 → 16 → 16 → 1), mismo dataset y mismo optimizador (SGD, lr=0.05), variando unicamente el metodo de regularización: baseline sin regularizacion, L2 (λ=0.01), Dropout (p=0.3), y L2+Dropout combinados.

---

## Resultado principal

El modelo baseline presento overfitting claro ( creciente entre loss de train y validacion). La combinación L2+Dropout redujo ese valor al mínimo y obtuvo el mejor accuracy de validacion, confirmando que la regularizacion mejora la generalizacion a costa de un leve aumento del loss de entrenamiento.

---

## Cómo ejecutar el notebook

1. Abrir url (https://colab.research.google.com/drive/1WhvIsi6VbD3QFqKiSuzdzKbGoGNf8ekx)
2. Ejecutar las celdas en orden

No requiere instalación adicional. Solo usa `numpy`, `matplotlib` y `pandas`, preinstalados en Colab.

---

## Estructura del repositorio

```
Semana/
├── semana 4 Regularizacion.ipynb # Este notebook — regularización    
├── week4_regularizacion.ipynb   # Actividad anterior — tecnicas de optimizacion
├── Readme Optimizacion # Actividad anterior — tecnicas de optimizacion
└── Readme Regualarizacion.md                    # Este archivo
```
