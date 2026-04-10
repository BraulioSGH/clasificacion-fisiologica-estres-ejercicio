import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# rutas y configuracion general
# revisar al volver a cambiar a la lap o a la pc
ruta_base = r"data/raw/Wearable_Dataset"

estados = ["STRESS", "AEROBIC", "ANAEROBIC"]
SEMILLA = 42


# funciones de preprocesamiento y calculo de caracteristicas

#sale nan

def limpiar_señal(x):
    # limpieza basica de la señal
    x = np.asarray(x, dtype=float).flatten()
    x = x[np.isfinite(x)]
    return x


def correlacion_robusta(a, b):
    a = limpiar_señal(a)
    b = limpiar_señal(b)

    n = min(len(a), len(b))
    if n < 3:
        return 0.0

    a, b = a[:n], b[:n]

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


def raiz_cuadratica_media(x):
    x = limpiar_señal(x)
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(x**2)))


def media_diferencias_absolutas(x):
    x = limpiar_señal(x)
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(x))))


def rango_intercuartil(x):
    x = limpiar_señal(x)
    if len(x) == 0:
        return 0.0
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def tasa_cruces_cero(x):
    x = limpiar_señal(x)
    if len(x) < 2:
        return 0.0

    centrada = x - np.mean(x)
    return float(np.mean(np.abs(np.diff(np.sign(centrada))) > 0))


def pendiente_lineal(x):
    x = limpiar_señal(x)
    if len(x) < 2:
        return 0.0

    tiempo = np.arange(len(x))
    coefs = np.polyfit(tiempo, x, 1)
    return float(coefs[0])


def contar_picos(x):
    # pendiente revisar si conviene luego usar scipy
    x = limpiar_señal(x)
    if len(x) < 5:
        return 0.0

    umbral = np.mean(x) + 0.5 * np.std(x)
    cuenta = 0

    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > umbral:
            cuenta += 1

    return float(cuenta)


def densidad_picos(x):
    x = limpiar_señal(x)
    if len(x) < 5:
        return 0.0
    return float(contar_picos(x) / len(x))


def potencia_banda_fft(señal, frec_muestreo, frec_baja, frec_alta):
    # posible fuente de error si la frecuencia viene mal leida
    x = limpiar_señal(señal)

    if len(x) < 16 or frec_muestreo <= 0:
        return 0.0

    x = x - np.mean(x)
    transformada = np.fft.rfft(x)
    frecuencias = np.fft.rfftfreq(len(x), d=1.0/frec_muestreo)
    potencia = np.abs(transformada)**2

    total = np.sum(potencia) + 1e-12
    banda = (frecuencias >= frec_baja) & (frecuencias < frec_alta)

    return float(np.sum(potencia[banda]) / total)


# carga de archivos de un sujeto

def cargar_sujeto(ruta):
    archivos = ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'TEMP.csv']
    datos = {}

    for archivo in archivos:
        ruta_archivo = os.path.join(ruta, archivo)

        if not os.path.exists(ruta_archivo):
            # omitir por falta de archivos
            continue

        try:
            df = pd.read_csv(ruta_archivo, header=None)
            frec_muestreo = float(df.iloc[1, 0])

            if archivo == 'ACC.csv':
                señal = df.iloc[2:, :3].values.astype(float)

            elif archivo == 'IBI.csv':
                # aqui IBI dio problemas en algunas pruebas
                # revisar despues si siempre conviene tomar la segunda columna
                if df.shape[1] >= 2:
                    señal = df.iloc[2:, 1].values.astype(float)
                else:
                    señal = df.iloc[2:, 0].values.astype(float)

            else:
                señal = df.iloc[2:, 0].values.astype(float)

            datos[archivo.replace(".csv", "")] = {
                "frec_muestreo": frec_muestreo,
                "señal": señal
            }

        except Exception:
            ## archivo faltante o corrupto, omitir señal
            continue

    return datos


# extraccion de las 31 caracteristicas por sujeto

def extraer_caracteristicas(datos):
    caracteristicas = {}

    if "ACC" in datos:
        acc = np.asarray(datos["ACC"]["señal"], dtype=float)

        if acc.ndim == 2 and acc.shape[1] >= 3:
            eje_x = limpiar_señal(acc[:, 0])
            eje_y = limpiar_señal(acc[:, 1])
            eje_z = limpiar_señal(acc[:, 2])

            n = min(len(eje_x), len(eje_y), len(eje_z))
            eje_x, eje_y, eje_z = eje_x[:n], eje_y[:n], eje_z[:n]

            magnitud = np.sqrt(eje_x**2 + eje_y**2 + eje_z**2)
            jerk = np.diff(magnitud) if len(magnitud) >= 2 else np.array([])

            # eje Y
            caracteristicas["accy_rms"]     = raiz_cuadratica_media(eje_y)
            caracteristicas["accy_energy"]  = float(np.mean(eje_y**2)) if len(eje_y) > 0 else 0.0

            caracteristicas["accy_madiff"]  = media_diferencias_absolutas(eje_y)
            caracteristicas["accy_q90"]     = float(np.percentile(eje_y, 90)) if len(eje_y) > 0 else 0.0

            # ejes X y Z
            caracteristicas["accx_madiff"]  = media_diferencias_absolutas(eje_x)
            caracteristicas["accz_madiff"]  = media_diferencias_absolutas(eje_z)

            caracteristicas["accx_zcr"]     = tasa_cruces_cero(eje_x)
            
            # magnitud
            caracteristicas["accmag_madiff"]       = media_diferencias_absolutas(magnitud)
            caracteristicas["accmag_iqr"]          = rango_intercuartil(magnitud)

            caracteristicas["accmag_q90"]          = float(np.percentile(magnitud, 90)) if len(magnitud) > 0 else 0.0
            caracteristicas["accmag_slope"]        = pendiente_lineal(magnitud)

            caracteristicas["accmag_n_peaks"]      = contar_picos(magnitud)
            caracteristicas["accmag_peak_density"] = densidad_picos(magnitud)

            # jerk
            caracteristicas["jerk_q10"]    = float(np.percentile(jerk, 10)) if len(jerk) > 0 else 0.0
            caracteristicas["jerk_q25"]    = float(np.percentile(jerk, 25)) if len(jerk) > 0 else 0.0

            caracteristicas["jerk_q75"]    = float(np.percentile(jerk, 75)) if len(jerk) > 0 else 0.0
            caracteristicas["jerk_q90"]    = float(np.percentile(jerk, 90)) if len(jerk) > 0 else 0.0

            caracteristicas["jerk_iqr"]    = rango_intercuartil(jerk)
            caracteristicas["jerk_madiff"] = media_diferencias_absolutas(jerk)

    if "HR" in datos:
        frec_cardiaca = limpiar_señal(datos["HR"]["señal"])
        caracteristicas["hr_madiff"] = media_diferencias_absolutas(frec_cardiaca)
        caracteristicas["hr_mean"]   = float(np.mean(frec_cardiaca)) if len(frec_cardiaca) > 0 else 0.0

    if "IBI" in datos:
        intervalos = limpiar_señal(datos["IBI"]["señal"])
        caracteristicas["ibi_mean"]  = float(np.mean(intervalos)) if len(intervalos) > 0 else 0.0
        caracteristicas["ibi_min"]   = float(np.min(intervalos)) if len(intervalos) > 0 else 0.0

        caracteristicas["ibi_q10"]   = float(np.percentile(intervalos, 10)) if len(intervalos) > 0 else 0.0
        caracteristicas["ibi_t_min"] = caracteristicas["ibi_min"]

        caracteristicas["ibi_t_q10"] = caracteristicas["ibi_q10"]
        caracteristicas["ibi_t_q25"] = float(np.percentile(intervalos, 25)) if len(intervalos) > 0 else 0.0

    if "BVP" in datos:
        pulso = limpiar_señal(datos["BVP"]["señal"])
        caracteristicas["bvp_zcr"]     = tasa_cruces_cero(pulso)
        caracteristicas["bvp_n_peaks"] = contar_picos(pulso)
        caracteristicas["bvp_median"]  = float(np.median(pulso)) if len(pulso) > 0 else 0.0

    if "EDA" in datos:
        conductancia = limpiar_señal(datos["EDA"]["señal"])
        frec_eda = datos["EDA"]["frec_muestreo"]

        # esta parte se ajusto despues de varias pruebas
        caracteristicas["eda_band_010_050"] = potencia_banda_fft(conductancia, frec_eda, 0.10, 0.50)

    return caracteristicas


# construccion del dataset completo

def construir_dataset():
    X = []
    y = []
    sujetos = []

    nombres_caracteristicas = [
        "jerk_iqr", "jerk_q25", "jerk_q75", "jerk_q90", "jerk_q10", "jerk_madiff",
        "accy_rms", "accy_energy", "accy_madiff", "accy_q90",
        "accx_madiff", "accz_madiff", "accx_zcr",
        "accmag_madiff", "accmag_iqr", "accmag_q90", "accmag_slope",
        "accmag_n_peaks", "accmag_peak_density",
        "hr_madiff", "hr_mean",
        "ibi_mean", "ibi_min", "ibi_q10", "ibi_t_min", "ibi_t_q10", "ibi_t_q25",
        "bvp_zcr", "bvp_n_peaks", "bvp_median",
        "eda_band_010_050"
    ]

    for estado in estados:
        ruta_estado = os.path.join(ruta_base, estado)

        for sujeto in os.listdir(ruta_estado):
            ruta_sujeto = os.path.join(ruta_estado, sujeto)

            if not os.path.isdir(ruta_sujeto):
                continue

            try:
                datos = cargar_sujeto(ruta_sujeto)

                # se pidio al menos ACC, IBI y HR
                if "ACC" not in datos or "IBI" not in datos or "HR" not in datos:
                    continue

                caract = extraer_caracteristicas(datos)

                # pendiente revisar si rellenar con 0.0 mete sesgo
                vector = [caract.get(c, 0.0) for c in nombres_caracteristicas]

                X.append(vector)
                y.append(estado)
                sujetos.append(sujeto)

            except Exception:
                # investigar fallo en esta seccion si no cuadran los sujetos finales
                continue

    return np.array(X, dtype=float), np.array(y), np.array(sujetos), nombres_caracteristicas


# grafica de las señales crudas de un sujeto de ejemplo

def graficar_sujeto(estado, num_sujeto=0):
    ruta_estado = os.path.join(ruta_base, estado)
    sujetos_disponibles = [s for s in os.listdir(ruta_estado)
                           if os.path.isdir(os.path.join(ruta_estado, s))]

    if num_sujeto >= len(sujetos_disponibles):
        print("no hay tantos sujetos en ese estado")
        return

    sujeto = sujetos_disponibles[num_sujeto]
    ruta_sujeto = os.path.join(ruta_estado, sujeto)
    datos = cargar_sujeto(ruta_sujeto)

    print(f"mostrando señales del sujeto: {sujeto}, estado: {estado}")

    señales_a_graficar = []

    if "ACC" in datos:
        acc = np.asarray(datos["ACC"]["señal"], dtype=float)

        if acc.ndim == 2 and acc.shape[1] >= 3:
            eje_x = limpiar_señal(acc[:, 0])
            eje_y = limpiar_señal(acc[:, 1])
            eje_z = limpiar_señal(acc[:, 2])

            n = min(len(eje_x), len(eje_y), len(eje_z))
            magnitud = np.sqrt(eje_x[:n]**2 + eje_y[:n]**2 + eje_z[:n]**2)

            señales_a_graficar.append((eje_x[:n], "ACC eje X", "aceleracion"))
            señales_a_graficar.append((eje_y[:n], "ACC eje Y", "aceleracion"))
            señales_a_graficar.append((eje_z[:n], "ACC eje Z", "aceleracion"))
            señales_a_graficar.append((magnitud, "ACC magnitud", "aceleracion"))

            jerk = np.diff(magnitud)
            señales_a_graficar.append((jerk, "Jerk (cambio de magnitud)", "jerk"))

    if "HR" in datos:
        frec_cardiaca = limpiar_señal(datos["HR"]["señal"])
        señales_a_graficar.append((frec_cardiaca, "Frecuencia cardiaca (HR)", "bpm"))

    if "IBI" in datos:
        intervalos = limpiar_señal(datos["IBI"]["señal"])
        señales_a_graficar.append((intervalos, "Intervalos entre latidos (IBI)", "segundos"))

    if "BVP" in datos:
        pulso = limpiar_señal(datos["BVP"]["señal"])
        señales_a_graficar.append((pulso, "Pulso de volumen sanguineo (BVP)", "amplitud"))

    if "EDA" in datos:
        conductancia = limpiar_señal(datos["EDA"]["señal"])
        señales_a_graficar.append((conductancia, "Conductancia de la piel (EDA)", "microsiemens"))

    num_graficas = len(señales_a_graficar)
    fig, ejes = plt.subplots(num_graficas, 1, figsize=(12, 2.5 * num_graficas))
    fig.suptitle(f"Señales fisiologicas, sujeto {sujeto}, estado {estado}", fontsize=13)

    for i, (señal, titulo, unidad) in enumerate(señales_a_graficar):
        eje_grafica = ejes[i] if num_graficas > 1 else ejes
        eje_grafica.plot(señal, linewidth=0.7, color="steelblue")
        eje_grafica.set_title(titulo, fontsize=10)
        eje_grafica.set_ylabel(unidad, fontsize=8)
        eje_grafica.set_xlabel("muestras", fontsize=8)
        eje_grafica.tick_params(labelsize=7)

    plt.tight_layout()
    plt.show()
    plt.close()

    # error de tikner al cerrar rápido, o comentar figuras para no tener error


# aqui arranca todo

X, y, ids_sujetos, nombres_caracteristicas = construir_dataset()

print("dataset listo")
print("dimensiones de X:", X.shape)
print("numero de caracteristicas:", len(nombres_caracteristicas))
print("clases:", np.unique(y))
print(pd.Series(y).value_counts())
print()

# esta grafica se dejo para inspeccion manual
for estado in estados:
    graficar_sujeto(estado, num_sujeto=0)

# particion train/test
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEMILLA,
    stratify=y
)

print("entrenamiento:", X_entreno.shape, " prueba:", X_prueba.shape)
print()

# entrenamiento del modelo
# esta configuracion se fue ajustando poco a poco
modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features='log2',
    min_samples_split=6,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=SEMILLA,
    n_jobs=-1
)

modelo.fit(X_entreno, y_entreno)
predicciones = modelo.predict(X_prueba)

# resultados en el conjunto de prueba
print("resultados en conjunto de prueba")
print(classification_report(y_prueba, predicciones))

exactitud = accuracy_score(y_prueba, predicciones)
print("exactitud:", round(exactitud, 4))
print()

matriz_confusion = confusion_matrix(y_prueba, predicciones, labels=estados)

plt.figure(figsize=(6, 5))
sns.heatmap(
    matriz_confusion,
    annot=True,
    fmt='d',
    xticklabels=estados,
    yticklabels=estados,
    cmap="Blues"
)
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.title("Matriz de confusion")
plt.tight_layout()
plt.show()
plt.close()

# investigacion de kfold pq en particiones aleatorias varia, sacar acuracy entre los mismos 

# validacion cruzada para ver si el modelo se mantiene

validacion_cruzada = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
puntuaciones = cross_val_score(modelo, X, y, cv=validacion_cruzada, scoring="accuracy", n_jobs=-1)

print("validacion cruzada 5 folds")
print(f"exactitud promedio: {puntuaciones.mean():.4f} +/- {puntuaciones.std():.4f}")
print("puntuacion por fold:", np.round(puntuaciones, 4))
print()

# importancia de caracteristicas
importancias = modelo.feature_importances_
orden = np.argsort(importancias)[::-1]

tabla_importancias = pd.DataFrame({
    "caracteristica": np.array(nombres_caracteristicas)[orden],
    "importancia": importancias[orden]
})

print("importancia de caracteristicas")
print(tabla_importancias.to_string(index=False))
print()

plt.figure(figsize=(10, 7))
plt.barh(
    tabla_importancias["caracteristica"][::-1],
    tabla_importancias["importancia"][::-1]
)
plt.xlabel("importancia")
plt.title("importancia de caracteristicas")
plt.tight_layout()
plt.show()
plt.close()

# resumen final
print("resumen final")
print("numero de caracteristicas usadas:", len(nombres_caracteristicas))
print("exactitud en prueba:", round(exactitud, 4))
print(f"exactitud en validacion cruzada: {puntuaciones.mean():.4f} +/- {puntuaciones.std():.4f}")
print("top 10 caracteristicas:")

for c in tabla_importancias["caracteristica"].head(10):
    print("-", c)
