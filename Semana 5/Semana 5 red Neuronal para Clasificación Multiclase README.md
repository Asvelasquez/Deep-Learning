# Clasificacion Multiclase con TensorFlow/Keras - Hiperparametros y Optimizadores

## Objetivo

Evidenciar cómo el comportamiento del entrenamiento de una red neuronal densa cambia al ajustar hiperparámetros clave (tasa de aprendizaje, tamaño de lote) y al modificar el optimizador, usando clasificación multiclase como tarea de referencia.

---

## Qué se comparó

| Experimento | Variable | Valores |
|-------------|----------|---------|
| Exp. 1 | Optimizador | SGD · RMSprop · Adam |
| Exp. 2 | Learning rate | 0.001 · 0.01 · 0.1 (Adam) |
| Exp. 3 | Batch size | 8 · 32 · 128 (Adam lr=0.01) |

Arquitectura fija en todos los experimentos: red densa 4 → 64 → 32 → 3, ReLU + Dropout(0.2), Softmax, dataset Iris (150 muestras, 3 clases).

---

## Resultado principal

Adam con lr=0.01 y batch=32 fue la configuración más estable y rápida. SGD requirió más épocas para converger y mostró mayor oscilación. lr=0.1 produjo inestabilidad; lr=0.001 convergió demasiado lento en 150 épocas. Lotes pequeños (batch=8) introdujeron ruido visible sin mejorar el resultado final.

---

## Cómo ejecutar el notebook

1. Abrir [Google Colab](https://colab.research.google.com/drive/16nEXMlCtAqCjfkloYorMnjg0L4hVVTPw)
2. Ejecutar las celdas en orden

Requiere: `tensorflow`, `scikit-learn`, `numpy`, `matplotlib`, `pandas` - todos preinstalados en Colab.
