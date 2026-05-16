# Autoencoder Denoising con MNIST

## Dataset

**MNIST:** 70 000 imágenes de dígitos manuscritos (0-9), 28×28 px, escala de grises. Disponible directamente en Keras — no requiere descarga manual.

---

## Modelos implementados

| Modelo | Arquitectura | Espacio latente |
|--------|-------------|-----------------|
| **DAE Denso** | 784→256→128→64→128→256→784 | 64 dimensiones |
| **DAE Convolucional** | Conv(32)→MaxPool→Conv(16)→MaxPool→ConvT(16)→Up→ConvT(32)→Up→Conv(1) | 7×7×16 |

---

## Proceso

1. Agregar ruido gaussiano N(0, 0.3) a las imágenes de entrenamiento
2. Entrenar con: entrada = imagen ruidosa, objetivo = imagen original
3. Evaluar reconstrucción con MSE, PSNR y SSIM en el test set
4. Comparar los dos modelos y analizar robustez ante distintos niveles de ruido

---

## Visualizaciones

- Original vs Ruidosa vs Reconstruida (DAE Denso y DAE Conv) — 10 dígitos
- Robustez ante sigma = 0.1, 0.3, 0.5, 0.7
- Espacio latente en 2D con t-SNE (2000 muestras)
- Curvas de entrenamiento (BCE loss)

---

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/drive/1Oz_7BWhe6ewX5zm-W7nzShHa9moV-N5C)
2. Ejecutar las celdas en orden

Requiere: `tensorflow`, `scikit-learn` — preinstalados en Colab. MNIST (~11 MB) se descarga automáticamente.
