# Actividad 15 — Data Journey, Monitoreo y Model Serving

## Descripción

Notebook cooperativo que implementa el ciclo de vida completo de un modelo de aprendizaje profundo:  
**Datos → Modelo → Monitoreo → Puesta en operación**

Se utiliza una CNN sobre MNIST como modelo de referencia, un **ExperimentLogger** local equivalente a Weights & Biases, y una simulación de API REST para demostrar el concepto de Model Serving.

---

## Estructura del notebook

| Sección | Descripción |
|---------|-------------|
| **1. Data Journey** | Diagrama y tabla del ciclo de vida de los datos: Origen → Acceso → Preparación → Uso |
| **2. Acceso y manipulación** | Carga de MNIST, EDA (distribución de clases, histograma de píxeles), preprocesamiento |
| **3. Modelo CNN** | Arquitectura reutilizada del CADI: 2× Conv2D + Dense(128) + Softmax(10) |
| **4. Logger local (W&B)** | `ExperimentLogger` + `WandbLocalCallback` para registro de hiperparámetros y métricas |
| **5. Entrenamiento** | Entrenamiento con EarlyStopping, ReduceLROnPlateau y logging por época |
| **5. Análisis de métricas** | Tabla de épocas, dashboard loss/accuracy, matriz de confusión, ejemplos de errores |
| **6. Model Serving** | Guardado del modelo, simulación de API `/predict`, tabla comparativa de estrategias |
| **7. Conclusiones** | Aprendizajes, dificultades y oportunidades de mejora |

---

## Data Journey

```
[Origen]          [Acceso]            [Preparación]        [Uso en Modelo]
NIST / LeCun  →  keras.datasets  →  Normalización     →  Entrenamiento
(1998)            .mnist.load_data    [0,255]→[0,1]         Validación
Escaneo de        (descarga auto)     Reshape (28,28,1)      Evaluación
formularios                           One-hot encoding       Predicción
```

---

## Arquitectura del modelo

```
Input (28×28×1)
  → Conv2D(32, 3×3, ReLU) + MaxPool(2×2)   → (13×13×32)
  → Conv2D(64, 3×3, ReLU) + MaxPool(2×2)   → (5×5×64)
  → Flatten → Dropout(0.3)
  → Dense(128, ReLU)
  → Dense(10, Softmax)
```

**Compilación:** Adam (lr=1e-3) · Categorical Crossentropy · Accuracy

---

## Monitoreo y Logging — Alternativa a W&B

Se implementa **ExperimentLogger** que simula las funcionalidades de Weights & Biases:

| Funcionalidad W&B | Equivalente local |
|-------------------|-------------------|
| `wandb.init(config=...)` | `ExperimentLogger(nombre, config)` |
| `wandb.log({loss, acc})` | `logger.log(epoch, loss=, val_loss=, ...)` |
| Runs Table | `logger.resumen()` → DataFrame |
| Curvas en dashboard | Matplotlib: loss + accuracy por época |
| Export JSON | `logger.exportar_json()` |

---

## Parámetros de entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Épocas máximas | 15 |
| Batch size | 128 |
| Learning rate | 1e-3 |
| Optimizador | Adam (β₁=0.9) |
| Loss | Categorical Crossentropy |
| Dropout | 0.3 |
| EarlyStopping | patience=3, monitor=val_accuracy |
| ReduceLROnPlateau | factor=0.5, patience=2 |

---

## Model Serving

### Estrategia recomendada: FastAPI + Docker

```
modelo.save()  →  Dockerfile  →  docker build  →  docker push  →  Cloud Run deploy
   (Colab)           (local)         (local)        (Docker Hub)     (GCP gratis)
```

### Comparación de alternativas

| Estrategia | Latencia | Escalabilidad | Costo | Recomendado para |
|-----------|----------|---------------|-------|-----------------|
| FastAPI REST | Baja | Media | Gratis | Prototipos y MVPs |
| TF Serving | Muy baja | Alta | Gratis/Propio | Producción TF a escala |
| Google Cloud AI | Baja | Muy alta | Pago | Entornos GCP |
| AWS SageMaker | Baja | Muy alta | Pago | Entornos AWS |
| Docker | Variable | Alta | Gratis/Propio | Portabilidad total |

---

## Resultados esperados

- **val_accuracy:** ~99% en MNIST con esta arquitectura
- **test_accuracy:** ~99%
- Curvas de loss y accuracy convergentes sin overfitting significativo
- Matriz de confusión con errores concentrados en dígitos similares (4↔9, 3↔5)

---

## Requisitos

```
tensorflow >= 2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Ejecutar en **Google Colab** (Runtime → Run all).

---

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/drive/1ZNzsAkHuWOZlAI6qdYNKsgLyQAumqDSN)
2. Ejecutar las celdas en orden


## Flujo completo

```
DATOS          →    MODELO      →    MONITOREO        →    SERVING
────────            ────────         ──────────             ────────
Origen NIST         CNN 3 bloques    ExperimentLogger       SavedModel
keras.load()        Adam+EarlyStop   Curvas loss/acc        API /predict
Norm+Reshape        val_acc~99%      JSON trazabilidad      Docker/Cloud
```
