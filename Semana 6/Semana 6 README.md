# Actividad 6 — Métricas, Preprocesamiento y Regularización

## Objetivo

Evidenciar el fenómeno de overfitting en un modelo base y demostrar cómo las técnicas de regularización (L2, Dropout, Early Stopping) mejoran la capacidad de generalización de una red neuronal densa para clasificación multiclase.

---

## Métodos de regularización aplicados

| Técnica | Descripción |
|---------|-------------|
| L2 (weight decay) | Penaliza pesos grandes  en cada capa oculta |
| Dropout (p=0.4) | Desactiva aleatoriamente el 40% de neuronas por paso de entrenamiento |
| Early Stopping | Detiene el entrenamiento cuando val_loss no mejora en 20 épocas consecutivas |

---

## Comparación realizada

Cuatro modelos con la misma arquitectura (13→256→256→128→3), mismo optimizador (Adam lr=0.001) y mismo dataset (Wine UCI, 178 muestras, 3 clases), variando únicamente el método de regularización: baseline sin regularización, L2, Dropout, y L2+Dropout+EarlyStopping combinados.

---

## Resultado principal

El modelo base mostró overfitting claro con gap creciente entre loss de entrenamiento y validación. El modelo con L2+Dropout+EarlyStopping redujo ese gap al mínimo y obtuvo el mejor F1 macro en el test set, confirmando que la regularización combinada mejora la generalización en datasets pequeños con arquitecturas sobredimensionadas.

---

## Cómo ejecutar el notebook

1. Abrir [Google Colab](https://colab.research.google.com/drive/19QfW5bNg4PcOSuC5Kwk0D2kHbio28u0C)
2. Ejecutar las celdas en orden

Requiere: `tensorflow`, `scikit-learn`, `numpy`, `matplotlib`, `pandas` — todos preinstalados en Colab.
