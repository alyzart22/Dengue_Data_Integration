#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera la base climática estatal semanal de México para 2025
a partir de los JSON semanales de ERA5-Land.

Para cada entidad federativa y semana calcula:

    - Temperatura mínima semanal (°C)
    - Temperatura media semanal (°C)
    - Temperatura máxima semanal (°C)
    - Precipitación acumulada semanal (mm)

La agregación espacial se realiza utilizando las celdas
de ERA5-Land cuyo centro se encuentra dentro del polígono
de cada entidad federativa.

python scripts/02_generar_clima_estatal_semanal_2025.py

Salida
------
data/clima/clima_estados_2025_semanal.csv
"""

from pathlib import Path
import unicodedata
import json

import pandas as pd
import geopandas as gpd
import numpy as np


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

# GeoJSON de las entidades federativas
GEOJSON = (
    BASE_DIR
    / "data"
    / "geografia"
    / "mexico_estados.geojson"
)

# Datos climáticos SEMANALES
CLIMA_DIR = (
    BASE_DIR
    / "data"
    / "clima"
    / "semanal"
)

# Archivo de salida
OUTPUT_CSV = (
    BASE_DIR
    / "data"
    / "clima"
    / "clima_estados_2025_semanal.csv"
)


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def normalizar(texto):
    """
    Normaliza nombres de entidades para facilitar
    el cruce entre diferentes fuentes.
    """

    if pd.isna(texto):
        return ""

    texto = str(texto).strip().lower()

    texto = "".join(
        c
        for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

    reemplazos = {
        "veracruz de ignacio de la llave": "veracruz",
        "coahuila de zaragoza": "coahuila",
        "michoacan de ocampo": "michoacan",
        "estado de mexico": "mexico",
        "mexico city": "ciudad de mexico",
        "distrito federal": "ciudad de mexico",
    }

    return reemplazos.get(texto, texto)


def detectar_columna(df, opciones):
    """
    Busca una columna utilizando varios nombres posibles.
    """

    columnas = {
        normalizar(c): c
        for c in df.columns
    }

    for opcion in opciones:

        opcion_norm = normalizar(opcion)

        if opcion_norm in columnas:
            return columnas[opcion_norm]

    return None


# ============================================================
# 3. PROCESAMIENTO CLIMÁTICO SEMANAL
# ============================================================

def cargar_clima():

    # --------------------------------------------------------
    # Carpetas de las cuatro variables
    # --------------------------------------------------------

    carpetas = {
        "TEMPERATURA_MIN_C": (
            CLIMA_DIR / "tmin_c",
            "tmin_c_2025-*.json",
        ),
        "TEMPERATURA_MEDIA_C": (
            CLIMA_DIR / "tmean_c",
            "tmean_c_2025-*.json",
        ),
        "TEMPERATURA_MAX_C": (
            CLIMA_DIR / "tmax_c",
            "tmax_c_2025-*.json",
        ),
        "PRECIPITACION_MM": (
            CLIMA_DIR / "precip_mm",
            "precip_mm_2025-*.json",
        ),
    }

    # --------------------------------------------------------
    # Localizar archivos
    # --------------------------------------------------------

    archivos_variables = {}

    print("\n==============================================")
    print(" DATOS CLIMÁTICOS SEMANALES ERA5-LAND")
    print("==============================================")

    for variable, (carpeta, patron) in carpetas.items():

        if not carpeta.exists():

            raise FileNotFoundError(
                f"\nNo existe la carpeta:\n{carpeta}"
            )

        archivos = sorted(
            carpeta.glob(patron)
        )

        if len(archivos) == 0:

            raise FileNotFoundError(
                f"\nNo encontré archivos con patrón:\n"
                f"{carpeta / patron}"
            )

        archivos_variables[variable] = archivos

        print(
            f"{variable:<22}: "
            f"{len(archivos)} archivos"
        )

    if not GEOJSON.exists():

        raise FileNotFoundError(
            f"\nNo existe el GeoJSON:\n{GEOJSON}"
        )

    # --------------------------------------------------------
    # Comprobar que las cuatro variables tienen
    # el mismo número de semanas
    # --------------------------------------------------------

    cantidades = {
        variable: len(archivos)
        for variable, archivos
        in archivos_variables.items()
    }

    if len(set(cantidades.values())) != 1:

        raise ValueError(
            "\nLas variables climáticas no tienen "
            "el mismo número de archivos semanales:\n"
            f"{cantidades}"
        )

    numero_archivos = next(
        iter(cantidades.values())
    )

    print(
        f"\nPeriodos semanales encontrados: "
        f"{numero_archivos}"
    )

    # --------------------------------------------------------
    # Cargar geometría estatal
    # --------------------------------------------------------

    geo = gpd.read_file(
        GEOJSON
    )

    nombre_col = detectar_columna(
        geo,
        [
            "nomgeo",
            "NOMGEO",
            "nombre",
            "name",
            "estado",
            "entidad",
            "NOM_ENT",
        ]
    )

    if nombre_col is None:

        raise ValueError(
            "\nNo pude identificar la columna "
            "con el nombre de las entidades.\n"
            f"Columnas disponibles:\n"
            f"{geo.columns.tolist()}"
        )

    print(
        f"\nColumna de entidad utilizada: "
        f"{nombre_col}"
    )

    geo["ESTADO_NORM"] = (
        geo[nombre_col]
        .astype(str)
        .apply(normalizar)
    )

    # --------------------------------------------------------
    # Trabajar en EPSG:4326
    # --------------------------------------------------------

    if geo.crs is None:

        geo = geo.set_crs(
            "EPSG:4326"
        )

    else:

        geo = geo.to_crs(
            "EPSG:4326"
        )

    # --------------------------------------------------------
    # Seleccionar entidades
    # --------------------------------------------------------

    geo_sel = geo[
        [
            nombre_col,
            "ESTADO_NORM",
            "geometry",
        ]
    ].copy()

    geo_sel = geo_sel.dropna(
        subset=["geometry"]
    ).copy()

    geo_sel = geo_sel.drop_duplicates(
        subset=["ESTADO_NORM"]
    ).copy()

    estados_norm = sorted(
        geo_sel[
            "ESTADO_NORM"
        ].unique()
    )

    print(
        f"\nEntidades encontradas: "
        f"{len(estados_norm)}"
    )

    for estado in estados_norm:

        print(
            f"  ✓ {estado}"
        )

    if len(estados_norm) != 32:

        raise ValueError(
            f"\nSe esperaban 32 entidades federativas "
            f"y se encontraron {len(estados_norm)}."
        )

    # --------------------------------------------------------
    # Leer JSON completo para construir la malla
    # --------------------------------------------------------

    def leer_json_era5(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            datos_json = json.load(f)

        meta = datos_json["meta"]

        lat = np.asarray(
            meta["latitude"],
            dtype=float
        )

        lon = np.asarray(
            meta["longitude"],
            dtype=float
        )

        values = np.asarray(
            datos_json["values"],
            dtype=float
        )

        fecha = pd.to_datetime(
            meta["date"]
        )

        esperado = (
            len(lat)
            * len(lon)
        )

        if values.size != esperado:

            raise ValueError(
                f"\nDimensiones incompatibles en:\n"
                f"{path}\n\n"
                f"Latitudes:  {len(lat)}\n"
                f"Longitudes: {len(lon)}\n"
                f"Esperado:   {esperado}\n"
                f"Values:     {values.size}"
            )

        lon_grid, lat_grid = (
            np.meshgrid(
                lon,
                lat
            )
        )

        lon_flat = (
            lon_grid.ravel()
        )

        lat_flat = (
            lat_grid.ravel()
        )

        val_flat = (
            values.ravel()
        )

        validos = np.isfinite(
            val_flat
        )

        df = pd.DataFrame(
            {
                "GRID_INDEX": (
                    np.arange(
                        val_flat.size
                    )[validos]
                ),
                "longitude": (
                    lon_flat[validos]
                ),
                "latitude": (
                    lat_flat[validos]
                ),
                "valor": (
                    val_flat[validos]
                ),
            }
        )

        puntos = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(
                df["longitude"],
                df["latitude"]
            ),
            crs="EPSG:4326",
        )

        return (
            puntos,
            fecha,
            meta,
        )

    # --------------------------------------------------------
    # Construir máscara espacial una sola vez
    # --------------------------------------------------------

    print(
        "\nConstruyendo máscara espacial "
        "ERA5-Land → entidades..."
    )

    archivo_base = (
        archivos_variables[
            "TEMPERATURA_MEDIA_C"
        ][0]
    )

    puntos_base, _, _ = (
        leer_json_era5(
            archivo_base
        )
    )

    puntos_estados = (
        gpd.sjoin(
            puntos_base,
            geo_sel[
                [
                    "ESTADO_NORM",
                    "geometry",
                ]
            ],
            how="inner",
            predicate="within",
        )
    )

    # --------------------------------------------------------
    # Índices de las celdas de cada entidad
    # --------------------------------------------------------

    indices_estado = {}

    print(
        "\nCeldas ERA5-Land por entidad:"
    )

    for estado in estados_norm:

        indices = (
            puntos_estados.loc[
                puntos_estados[
                    "ESTADO_NORM"
                ] == estado,
                "GRID_INDEX",
            ]
            .astype(int)
            .to_numpy()
        )

        if len(indices) == 0:

            raise ValueError(
                f"\nNo se encontraron celdas "
                f"ERA5-Land para:\n{estado}"
            )

        indices_estado[
            estado
        ] = indices

        print(
            f"  {estado:<25} "
            f"{len(indices):>5} celdas"
        )

    # --------------------------------------------------------
    # Función para leer valores de cada JSON
    # --------------------------------------------------------

    def valores_json(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            datos_json = json.load(f)

        meta = datos_json["meta"]

        valores = np.asarray(
            datos_json["values"],
            dtype=float
        )

        fecha = pd.to_datetime(
            meta["date"]
        )

        # Semana ISO correspondiente a la fecha
        iso = fecha.isocalendar()

        semana = int(
            iso.week
        )

        return (
            valores,
            fecha,
            semana,
            meta,
        )

    # --------------------------------------------------------
    # Procesar cada variable
    # --------------------------------------------------------

    dataframes = []

    for variable, archivos in archivos_variables.items():

        print(
            f"\nProcesando {variable}..."
        )

        registros = []

        for archivo in archivos:

            valores, fecha, semana, _ = (
                valores_json(
                    archivo
                )
            )

            for estado in estados_norm:

                idx = (
                    indices_estado[
                        estado
                    ]
                )

                datos = (
                    valores[idx]
                )

                datos = datos[
                    np.isfinite(
                        datos
                    )
                ]

                valor_estado = (
                    float(
                        np.mean(
                            datos
                        )
                    )
                    if len(datos)
                    else np.nan
                )

                registros.append(
                    {
                        "ESTADO_NORM": estado,
                        "SEMANA": semana,
                        "FECHA": (
                            fecha.strftime(
                                "%Y-%m-%d"
                            )
                        ),
                        variable: valor_estado,
                    }
                )

        df_variable = pd.DataFrame(
            registros
        )

        dataframes.append(
            df_variable
        )

    # --------------------------------------------------------
    # Combinar las cuatro variables
    # --------------------------------------------------------

    clima = dataframes[0]

    for df_variable in dataframes[1:]:

        clima = clima.merge(
            df_variable,
            on=[
                "ESTADO_NORM",
                "SEMANA",
                "FECHA",
            ],
            how="outer",
        )

    # --------------------------------------------------------
    # Recuperar nombre original del estado
    # --------------------------------------------------------

    nombre_estado = (
        geo_sel[
            [
                nombre_col,
                "ESTADO_NORM",
            ]
        ]
        .drop_duplicates(
            subset=[
                "ESTADO_NORM"
            ]
        )
        .set_index(
            "ESTADO_NORM"
        )[nombre_col]
        .to_dict()
    )

    clima["ESTADO"] = (
        clima[
            "ESTADO_NORM"
        ]
        .map(
            nombre_estado
        )
    )

    # --------------------------------------------------------
    # Orden final
    # --------------------------------------------------------

    clima = clima[
        [
            "ESTADO",
            "ESTADO_NORM",
            "SEMANA",
            "FECHA",
            "TEMPERATURA_MIN_C",
            "TEMPERATURA_MEDIA_C",
            "TEMPERATURA_MAX_C",
            "PRECIPITACION_MM",
        ]
    ]

    clima = (
        clima
        .sort_values(
            [
                "ESTADO_NORM",
                "SEMANA",
                "FECHA",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Validaciones
    # --------------------------------------------------------

    semanas = sorted(
        clima[
            "SEMANA"
        ].unique()
    )

    fechas = sorted(
        clima[
            "FECHA"
        ].unique()
    )

    print(
        "\n=============================================="
    )
    print(
        " VALIDACIÓN TEMPORAL"
    )
    print(
        "=============================================="
    )

    print(
        f"Semanas únicas: "
        f"{len(semanas)}"
    )

    print(
        f"Primera semana: "
        f"{min(semanas)}"
    )

    print(
        f"Última semana: "
        f"{max(semanas)}"
    )

    print(
        f"Fechas semanales: "
        f"{len(fechas)}"
    )

    # Debe existir exactamente un registro por
    # estado y periodo semanal.
    filas_esperadas = (
        len(estados_norm)
        * len(fechas)
    )

    if len(clima) != filas_esperadas:

        raise ValueError(
            f"\nSe esperaban "
            f"{filas_esperadas} registros "
            f"y se obtuvieron "
            f"{len(clima)}."
        )

    # --------------------------------------------------------
    # Verificar duplicados
    # --------------------------------------------------------

    duplicados = clima.duplicated(
        subset=[
            "ESTADO_NORM",
            "SEMANA",
            "FECHA",
        ]
    )

    if duplicados.any():

        raise ValueError(
            "\nSe encontraron registros "
            "climáticos duplicados."
        )

    # --------------------------------------------------------
    # Verificar valores faltantes
    # --------------------------------------------------------

    columnas_clima = [
        "TEMPERATURA_MIN_C",
        "TEMPERATURA_MEDIA_C",
        "TEMPERATURA_MAX_C",
        "PRECIPITACION_MM",
    ]

    if (
        clima[
            columnas_clima
        ]
        .isna()
        .any()
        .any()
    ):

        print(
            "\n⚠ Advertencia: existen "
            "valores climáticos faltantes."
        )

        print(
            clima[
                columnas_clima
            ]
            .isna()
            .sum()
        )

    # --------------------------------------------------------
    # Redondear
    # --------------------------------------------------------

    for columna in columnas_clima:

        clima[columna] = (
            clima[columna]
            .round(3)
        )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    print(
        "\n=============================================="
    )
    print(
        " RESUMEN CLIMÁTICO ESTATAL SEMANAL"
    )
    print(
        "=============================================="
    )

    print(
        f"Entidades: "
        f"{clima['ESTADO_NORM'].nunique()}"
    )

    print(
        f"Periodos semanales: "
        f"{clima['FECHA'].nunique()}"
    )

    print(
        f"Semanas ISO diferentes: "
        f"{clima['SEMANA'].nunique()}"
    )

    print(
        f"Registros: "
        f"{len(clima)}"
    )

    print(
        "\nPrimeros registros:"
    )

    print(
        clima
        .head(15)
        .to_string(
            index=False
        )
    )

    return clima


# ============================================================
# 4. MAIN
# ============================================================

def main():

    print(
        "\n=============================================="
    )
    print(
        " CLIMA ESTATAL SEMANAL ERA5-LAND · MÉXICO 2025"
    )
    print(
        "=============================================="
    )

    clima = cargar_clima()

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    clima.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    print(
        "\n=============================================="
    )
    print(
        " ARCHIVO GENERADO"
    )
    print(
        "=============================================="
    )

    print(
        f"\n{OUTPUT_CSV}"
    )

    print(
        f"\nEntidades: "
        f"{clima['ESTADO_NORM'].nunique()}"
    )

    print(
        f"Periodos semanales: "
        f"{clima['FECHA'].nunique()}"
    )

    print(
        f"Registros: "
        f"{len(clima)}"
    )


if __name__ == "__main__":
    main()