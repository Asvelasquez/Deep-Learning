# Red Neuronal Siamesa para Reconocimiento Facial

## Dataset

**Olivetti Faces** 40 personas, 10 fotos por persona (400 imágenes, 64×64 px, escala de grises). Disponible en scikit-learn — no requiere descarga manual. Split a nivel de **persona**: 32 personas en train, 8 en test (evita data leakage).

---

## Arquitectura


 **Subred de embedding**  3 bloques Conv→BatchNorm→MaxPool + GlobalAvgPool + Dense(128) + UnitNormalization L2 
 **Entradas**  3 imágenes ancla, positivo, negativo (pesos compartidos) 
 **Distancia** Distancia euclidiana entre pares de embeddings 
 **Loss** Triplet Loss `max(0, d_pos - d_neg + margin)` con margin=0.5 

---

## Dataset de triplets

Cada triplet contiene:
- **Ancla:** imagen de referencia
- **Positivo:** otra imagen de la misma persona
- **Negativo:** imagen de una persona distinta

1000 triplets de entrenamiento / 300 de test, balanceados aleatoriamente.

---

## Resultado principal

El modelo aprende un espacio de embedding donde `d(ancla, positivo) < d(ancla, negativo)`. La métrica principal es **Triplet Accuracy** (fracción de triplets correctamente ordenados). Se reportan también accuracy binaria, precision, recall, F1 y AUC-ROC sobre pares construidos desde los triplets de test.

---

## Cómo ejecutar

1. Abrir [Google Colab](https://colab.research.google.com/1LltIZZ_yI9O_4x1IrQRGsj318VUNueQz)
2. Ejecutar las celdas en orden

Requiere: `tensorflow`, `scikit-learn` — preinstalados en Colab.
