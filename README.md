# Clasificación fisiológica de estrés agudo y actividad física

Proyecto de aprendizaje supervisado para clasificar tres estados fisiológicos a partir de señales obtenidas con dispositivos wearables:

STRESS
AEROBIC
ANAEROBIC

## Descripción

Este proyecto analiza señales fisiológicas registradas en condiciones controladas de estrés agudo, ejercicio aeróbico y ejercicio anaeróbico. Se implementa un pipeline reproducible que incluye:

- Carga y limpieza de archivos CSV por sujeto
- Preprocesamiento mínimo y manejo de datos faltantes
- Extracción de 31 características fisiológicamente interpretables
- Construcción del dataset final
- Entrenamiento de un clasificador Random Forest
- Evaluación mediante holdout, matriz de confusión y validación cruzada estratificada

## Dataset

El proyecto utiliza el conjunto de datos Wearable Device Dataset from Induced Stress and Structured Exercise Sessions.

En este repositorio, la base de datos se organiza dentro de la siguiente ruta:

data/raw/Wearable_Dataset/
├── STRESS/
├── AEROBIC/
└── ANAEROBIC/

### Señales utilizadas

ACC
BVP
EDA
HR
IBI
TEMP

## Características extraídas

Se extraen 31 características a partir de:

- acelerometría por eje
- magnitud del acelerómetro
- jerk
- frecuencia cardiaca
- intervalos entre latidos
- pulso de volumen sanguíneo
- actividad electrodermal

## Modelo utilizado

Se emplea un clasificador RandomForestClassifier con la siguiente configuración principal:

n_estimators = 300
max_depth = None
max_features = 'log2'
min_samples_split = 6
class_weight = 'balanced'

## Resultados principales

Exactitud en prueba: 0.95
Validación cruzada estratificada de 5 folds: 0.9184 ± 0.0248

La principal confusión observada ocurre entre las clases STRESS y ANAEROBIC.

## Estructura del proyecto

clasificacion-fisiologica-estres-ejercicio/

├── README.md

├── .gitignore

├── requirements.txt

├── src/pipeline_clasificacion.py

├── docs/articulo.pdf

├── data/raw/Wearable_Dataset

## Requisitos

Instala las dependencias con:

pip install -r requirements.txt

## Ejecución

Asegúrate de que la base de datos esté en la ruta:

data/raw/Wearable_Dataset

Después ejecuta:

python src/pipeline_clasificacion.py

## Autor

Braulio Sayd Gutiérrez Hernández
CICESE - Unidad Académica Tepic
