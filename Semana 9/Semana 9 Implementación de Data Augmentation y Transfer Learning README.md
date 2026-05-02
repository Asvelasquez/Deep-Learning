# Data Augmentation y Transfer Learning para Clasificación de Imágenes

## Dataset

**CIFAR-10:** 60 000 imágenes a color (32×32 px) en 10 clases. Disponible directamente en Keras, no requiere descarga manual. División: 80% entrenamiento / 10% validación / 10% test.

---

## Modelos comparados

| Modelo | Descripción |
|--------|-------------|
| **A — CNN Base** | 3 bloques Conv→BatchNorm→MaxPool, sin augmentation |
| **B — CNN + Augmentation** | Misma arquitectura + RandomFlip, RandomRotation, RandomZoom, RandomTranslation |
| **C — Transfer Learning** | MobileNetV2 (ImageNet) + augmentation + fine-tuning últimas 20 capas |

---

## Comparación realizada

Los tres modelos se evalúan con las mismas métricas (accuracy, loss, precision, recall, F1, matriz de confusión) sobre el mismo test set. La arquitectura de la cabeza de clasificación es idéntica en los tres para aislar el efecto de cada técnica.

---

## Resultado principal

Transfer Learning (Modelo C) alcanza el mayor accuracy con menos épocas totales. Data Augmentation (Modelo B) reduce la brecha train/val respecto al Modelo A, confirmando su efecto regularizador. La combinación augmentation + transfer learning es la estrategia más robusta para datasets de tamaño limitado.

---

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/drive/1MNLH2oHtIGD62ktTDbfzwoxveUtaArdm)
2. Ejecutar las celdas en orden

Requiere: `tensorflow`, `scikit-learn` — preinstalados en Colab. CIFAR-10 se descarga automáticamente (~170 MB).
