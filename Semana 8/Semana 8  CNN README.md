# CNN para Clasificación de Imágenes + Transfer Learning

## Dataset

CIFAR-10: 60 000 imagenes a color (32×32 px) en 10 clases (avion, auto, pájaro, gato, ciervo, perro, rana, caballo, barco, camión). Disponible directamente en Keras.

---

## Arquitecturas comparadas

| Modelo | Descripción |
|--------|-------------|
| **CNN Base** | 3 bloques Conv→BatchNorm→MaxPool + Dense(256) + Dropout(0.4) + Softmax(10) |
| **Transfer Learning** | MobileNetV2 (ImageNet) congelado + cabeza propia + fine-tuning últimas 20 capas |

---

## Comparacion realizada

Mismas metricas (accuracy, loss, precision, recall, F1, matriz de confusion) sobre el mismo test set (10 000 imagenes), entrenando la CNN base desde cero y el modelo TL con estrategia de dos fases (base congelada → fine-tuning).

---

## Resultado principal

Transfer Learning con MobileNetV2 alcanza mayor accuracy en test con menos epocas totales que la CNN base entrenada desde cero, confirmando la utilidad del transfer learning cuando el dataset local es limitado en comparacion con ImageNet.

---

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/drive/15LLyvJkwofO6M0AL73ruJDLcJcsILE7K)
2. Ejecutar las celdas en orden

Requiere: `tensorflow`, `scikit-learn`, `numpy`, `matplotlib` — todos preinstalados en Colab. El dataset CIFAR-10 se descarga automáticamente.