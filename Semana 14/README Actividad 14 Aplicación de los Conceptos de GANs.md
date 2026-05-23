# Actividad 14 — Aplicación de los Conceptos de GANs

## Descripción

Notebook desarrollado en Google Colab que implementa una **Red Generativa Adversaria (GAN)** básica sobre el dataset **MNIST**. El objetivo es comprender el flujo técnico del entrenamiento adversarial mediante la construcción e interacción de los dos componentes principales: el **Generador** y el **Discriminador**.

---

## Contenido del notebook

| Sección | Descripción |
|---------|-------------|
| **1. ¿Qué es una GAN?** | Explicación de la dinámica adversarial, función objetivo minimax y roles de G y D |
| **2. Dataset MNIST** | Carga, normalización `[0,255] → [-1,1]` y visualización de muestras reales |
| **3. Generador** | Arquitectura DCGAN: Dense → Conv2DTranspose ×3 → tanh |
| **4. Discriminador** | Arquitectura: Conv2D ×3 → Dense(1) → sigmoid |
| **5. Entrenamiento adversarial** | Definición de pérdidas BCE, optimizadores Adam y función `@tf.function` |
| **6. Ejecución** | 30 épocas, batch 256, registro de pérdidas por época |
| **7. Visualizaciones** | Curvas de pérdida, grilla de imágenes, progreso, interpolación y comparación |
| **8. Análisis** | Dificultades, posibles mejoras y aprendizajes técnicos |

---

## Arquitectura

### Generador
```
Entrada: z ∈ ℝ^100  (ruido aleatorio)
  Dense(7×7×256) + BatchNorm + LeakyReLU
  Reshape → (7, 7, 256)
  Conv2DTranspose(128, 5×5, stride=1) + BatchNorm + LeakyReLU  → (7, 7, 128)
  Conv2DTranspose( 64, 5×5, stride=2) + BatchNorm + LeakyReLU  → (14, 14, 64)
  Conv2DTranspose(  1, 5×5, stride=2) + tanh                   → (28, 28, 1)
Salida: imagen ∈ [-1, 1]
```

### Discriminador
```
Entrada: imagen (28, 28, 1)
  Conv2D( 64, 5×5, stride=2) + LeakyReLU + Dropout(0.3)  → (14, 14, 64)
  Conv2D(128, 5×5, stride=2) + LeakyReLU + Dropout(0.3)  → (7, 7, 128)
  Conv2D(256, 5×5, stride=2) + LeakyReLU + Dropout(0.3)  → (4, 4, 256)
  Flatten → Dense(1) + sigmoid
Salida: probabilidad ∈ [0, 1]  (1 = real, 0 = falsa)
```

---

## Parámetros de entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Dimensión espacio latente | 100 |
| Épocas | 30 |
| Batch size | 256 |
| Optimizador G y D | Adam (lr=2e-4, β₁=0.5) |
| Función de pérdida | Binary Cross-Entropy |
| Dataset | MNIST (60 000 imágenes de entrenamiento) |

---

## Visualizaciones incluidas

1. **Curvas de pérdida** — Loss del Generador y Discriminador por época, con línea de equilibrio teórico `ln(2) ≈ 0.693`
2. **Grilla de imágenes generadas** — 64 imágenes al final del entrenamiento
3. **Progreso por época** — Misma semilla de ruido fijo en épocas 1, 5, 10, 15, 20, 25 y 30
4. **Interpolación en espacio latente** — Transición suave entre dos vectores z1 y z2
5. **Comparación real vs. generada** — 8 imágenes reales (fila 1) y 8 generadas (fila 2)

---

## Resultados

- Con **30 épocas** el Generador produce dígitos reconocibles con estructura global correcta.
- Las pérdidas oscilan en torno al equilibrio teórico `ln(2)`, comportamiento esperado en entrenamiento adversarial.
- La interpolación lineal en el espacio latente demuestra que el Generador aprende una representación **continua y estructurada**.

---

## Dificultades y soluciones

| Dificultad | Solución aplicada |
|-----------|-------------------|
| Inestabilidad de pérdidas | `beta_1=0.5` en Adam · `BatchNorm` en G · `Dropout` en D |
| Mode collapse parcial | Arquitectura moderada · LR bajo (2e-4) |
| Velocidad de entrenamiento | `@tf.function` · `prefetch(AUTOTUNE)` en el dataset |


## Requisitos

```
tensorflow >= 2.10
numpy
matplotlib
```

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/drive/1bx4pEyXHBpKqdyXcqh6hYzcY41ipeA3v)
2. Ejecutar las celdas en orden
