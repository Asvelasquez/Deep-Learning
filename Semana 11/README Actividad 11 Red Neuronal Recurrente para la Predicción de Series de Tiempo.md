# Red Neuronal Recurrente para Predicción de Series de Tiempo

## Dataset

**Airline Passengers** número mensual de pasajeros aéreos internacionales (miles), enero 1949 — diciembre 1960 (144 muestras). Descarga automática desde repositorio público. Características: tendencia creciente + estacionalidad anual clara.

---

## Modelos implementados

| Modelo | Arquitectura | Descripción |
|--------|-------------|-------------|
| **RNN Simple** | SimpleRNN(64) + Dropout(0.2) + Dense(32) + Dense(1) | Baseline recurrente |
| **LSTM** | LSTM(64) + Dropout(0.2) + Dense(32) + Dense(1) | Captura dependencias largas con gates |
| **GRU** | GRU(64) + Dropout(0.2) + Dense(32) + Dense(1) | Balance eficiencia/rendimiento |

---

## Preprocesamiento

- Normalización a [0, 1] con MinMaxScaler
- División temporal 80/20 (sin shuffle — el orden es crítico)
- Ventanas de 12 pasos (un año) como entrada, siguiente valor como objetivo
- El test incluye los últimos 12 puntos del train como contexto inicial

---

## Métricas reportadas

MSE, RMSE, MAE y MAPE en escala original (miles de pasajeros), más análisis de residuos y ratio train/val para detectar sobreajuste.

---

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/drive/1wMbTBXOvzcdjRBZV5TmZi8mijjXrWcFh)
2. Ejecutar las celdas en orden

Requiere: `tensorflow`, `scikit-learn`, `pandas` — preinstalados en Colab. El dataset se descarga automáticamente (~3 KB).
