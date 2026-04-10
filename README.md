# Clasificación fisiológica de estrés agudo y actividad física

Proyecto de aprendizaje supervisado para clasificar tres estados fisiológicos a partir de señales obtenidas con dispositivos wearables:

- STRESS
- AEROBIC
- ANAEROBIC

## Descripción

Este proyecto analiza señales fisiológicas registradas en condiciones controladas de estrés agudo, ejercicio aeróbico y ejercicio anaeróbico. Se implementa un pipeline reproducible que incluye:

1. Carga y limpieza de archivos CSV por sujeto
2. Preprocesamiento mínimo y manejo de datos faltantes
3. Extracción de 31 características fisiológicamente interpretables
4. Construcción del dataset final
5. Entrenamiento de un clasificador Random Forest
6. Evaluación mediante holdout, matriz de confusión y validación cruzada estratificada

## Señales utilizadas

- ACC
- BVP
- EDA
- HR
- IBI
- TEMP

## Características

Se extraen 31 características a partir de:

- acelerometría por eje
- magnitud del acelerómetro
- jerk
- frecuencia cardiaca
- intervalos entre latidos
- pulso de volumen sanguíneo
- actividad electrodermal

## Modelo

Se utiliza un clasificador `RandomForestClassifier` con:

- `n_estimators = 300`
- `max_features = log2`
- `min_samples_split = 6`
- `class_weight = balanced`

## Resultados

- Exactitud en prueba: `0.95`
- Validación cruzada estratificada de 5 folds: `0.9184 ± 0.0248`

La principal confusión observada ocurre entre `STRESS` y `ANAEROBIC`.

## Estructura del proyecto

```text
data/
docs/
notebooks/
src/
results/
