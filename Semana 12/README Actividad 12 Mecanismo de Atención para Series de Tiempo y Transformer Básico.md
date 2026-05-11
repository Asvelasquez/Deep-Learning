# Mecanismo de Atención y Transformer para Predicción de Series de Tiempo

## Dataset

**Airline Passengers** (Box & Jenkins, 1976): mismo dataset de la Actividad 11. Pasajeros aéreos mensuales 1949-1960 (144 muestras). Permite comparación directa con los modelos recurrentes previos.

---

## Modelos comparados

| Modelo | Arquitectura | Novedad |
|--------|-------------|---------|
| **A — LSTM** | LSTM(64) + Dropout + Dense | Baseline (Act.11) |
| **B — LSTM + Atención** | LSTM(64, return_seq=True) + Atención Aditiva + Dense | Pesos de atención interpretables |
| **C — Transformer** | PosEncoding + 2×BloqueTransformer(d=32, heads=4) + GAP + Dense | Sin recurrencia, acceso directo a toda la secuencia |

---

## Componentes implementados

- **Positional Encoding** con funciones seno/coseno (Vaswani et al. 2017)
- **Atención Aditiva** (Bahdanau): scores = v·tanh(W·h_t), weights = softmax(scores)
- **Multi-Head Self-Attention** con `layers.MultiHeadAttention` de Keras
- **Bloque Transformer**: MHA + FFN + residuals + LayerNorm

---

## Visualizaciones incluidas

- Predicciones vs real para los tres modelos
- Heatmap de pesos de atención aditiva por muestra
- Matrices de Multi-Head Self-Attention por cabeza
- Pesos de atención vs autocorrelación (análisis de dependencias)

---

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/drive/1AmEEJWI07t-REEfpFHNUG-ZSatrC3UjR)
2. Ejecutar las celdas en orden

Requiere: `tensorflow`, `scikit-learn`, `pandas` — preinstalados en Colab.
