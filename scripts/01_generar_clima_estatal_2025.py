#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera la base climática estatal de México para 2025
a partir de datos mensuales de ERA5-Land.

Para cada entidad federativa y mes calcula:

    - Temperatura media mensual (°C)
    - Precipitación acumulada mensual (mm)

La agregación espacial se realiza utilizando las celdas
de ERA5-Land cuyo centro se encuentra dentro del polígono
de cada entidad federativa.

Salida
------
data/clima/clima_estados_2025_mensual.csv
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

# Datos climáticos mensuales ERA5-Land
CLIMA_DIR = (
    BASE_DIR
    / "data"
    / "clima"
    / "mensual"
)

# Archivo generado para alimentar el dashboard
OUTPUT_CSV = (
    BASE_DIR
    / "data"
    / "clima"
    / "clima_estados_2025_mensual.csv"
)


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def normalizar(texto):
    """
    Normaliza nombres para facilitar cruces entre
    diferentes fuentes de datos.
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
# 3. LEER Y AGREGAR DATOS CLIMÁTICOS
# ============================================================

def cargar_clima():
    """
    Lee los JSON mensuales de ERA5-Land y calcula valores
    climáticos mensuales para las 32 entidades federativas.

    Variables
    ---------
    tmean_c:
        Temperatura media mensual (°C).

    precip_mm:
        Precipitación acumulada mensual (mm).

    Metodología
    -----------
    Los JSON contienen una malla regular:

        latitude x longitude -> values

    Se construyen puntos correspondientes al centro de cada
    celda de ERA5-Land.

    Posteriormente se identifica qué celdas se encuentran
    dentro del polígono de cada entidad federativa.

    Para cada estado y mes se calcula:

        TEMPERATURA_MEDIA_C
            Promedio espacial de tmean_c.

        PRECIPITACION_MM
            Promedio espacial de la precipitación mensual
            acumulada en las celdas pertenecientes al estado.

    Returns
    -------
    pandas.DataFrame
    """

    # --------------------------------------------------------
    # Carpetas climáticas
    # --------------------------------------------------------

    temp_dir = (
        CLIMA_DIR
        / "tmean_c"
    )

    precip_dir = (
        CLIMA_DIR
        / "precip_mm"
    )

    if not temp_dir.exists():
        raise FileNotFoundError(
            f"\nNo existe:\n{temp_dir}"
        )

    if not precip_dir.exists():
        raise FileNotFoundError(
            f"\nNo existe:\n{precip_dir}"
        )

    if not GEOJSON.exists():
        raise FileNotFoundError(
            f"\nNo existe:\n{GEOJSON}"
        )

    # --------------------------------------------------------
    # Localizar archivos mensuales
    # --------------------------------------------------------

    archivos_temp = sorted(
        temp_dir.glob(
            "tmean_c_2025-*.json"
        )
    )

    archivos_precip = sorted(
        precip_dir.glob(
            "precip_mm_2025-*.json"
        )
    )

    print("\n==============================================")
    print(" DATOS CLIMÁTICOS ERA5-LAND")
    print("==============================================")

    print(
        f"Temperatura:   "
        f"{len(archivos_temp)} meses"
    )

    print(
        f"Precipitación: "
        f"{len(archivos_precip)} meses"
    )

    if len(archivos_temp) != 12:
        raise ValueError(
            f"Esperaba 12 archivos de temperatura "
            f"y encontré {len(archivos_temp)}."
        )

    if len(archivos_precip) != 12:
        raise ValueError(
            f"Esperaba 12 archivos de precipitación "
            f"y encontré {len(archivos_precip)}."
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
    # ERA5-Land utiliza coordenadas geográficas lat/lon.
    # Trabajaremos en EPSG:4326.
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
    # Utilizar todas las entidades del GeoJSON
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
        f"\nEntidades encontradas en GeoJSON: "
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
    # Leer un JSON ERA5-Land y construir la malla geográfica
    # --------------------------------------------------------

    def leer_json_era5(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            datos_json = json.load(f)

        meta = (
            datos_json["meta"]
        )

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

        # ----------------------------------------------------
        # Validar dimensiones
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Construir coordenadas de la malla
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Conservar celdas con valores válidos
        # ----------------------------------------------------

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
            fecha.month,
            meta,
        )

    # --------------------------------------------------------
    # Construir máscara espacial
    #
    # La geometría de la malla ERA5-Land no cambia entre
    # meses, por lo que esta operación se realiza una sola vez.
    # --------------------------------------------------------

    print(
        "\nConstruyendo máscara espacial "
        "ERA5-Land → entidades..."
    )

    puntos_base, _, _ = (
        leer_json_era5(
            archivos_temp[0]
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
    # Guardar los índices originales de las celdas
    # correspondientes a cada entidad.
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
    # Leer únicamente los valores de un JSON
    # --------------------------------------------------------

    def valores_json(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            datos_json = (
                json.load(f)
            )

        meta = (
            datos_json["meta"]
        )

        valores = np.asarray(
            datos_json["values"],
            dtype=float
        )

        mes = pd.to_datetime(
            meta["date"]
        ).month

        return (
            valores,
            mes,
            meta,
        )

    # --------------------------------------------------------
    # Procesar temperatura media mensual
    # --------------------------------------------------------

    registros_temp = []

    print(
        "\nProcesando temperatura "
        "media mensual..."
    )

    for archivo in archivos_temp:

        valores, mes, _ = (
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

            media = (
                float(
                    np.mean(
                        datos
                    )
                )
                if len(datos)
                else np.nan
            )

            registros_temp.append(
                {
                    "ESTADO_NORM": estado,
                    "MES": mes,
                    "TEMPERATURA": media,
                }
            )

    temp_df = pd.DataFrame(
        registros_temp
    )

    # --------------------------------------------------------
    # Procesar precipitación mensual
    # --------------------------------------------------------

    registros_precip = []

    print(
        "Procesando precipitación "
        "mensual..."
    )

    for archivo in archivos_precip:

        valores, mes, _ = (
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

            media = (
                float(
                    np.mean(
                        datos
                    )
                )
                if len(datos)
                else np.nan
            )

            registros_precip.append(
                {
                    "ESTADO_NORM": estado,
                    "MES": mes,
                    "PRECIPITACION": media,
                }
            )

    precip_df = pd.DataFrame(
        registros_precip
    )

    # --------------------------------------------------------
    # Combinar temperatura y precipitación
    # --------------------------------------------------------

    clima = temp_df.merge(
        precip_df,
        on=[
            "ESTADO_NORM",
            "MES",
        ],
        how="outer",
    )

    # --------------------------------------------------------
    # Recuperar el nombre original de cada entidad
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
    # Renombrar variables
    # --------------------------------------------------------

    clima = clima.rename(
        columns={
            "TEMPERATURA": (
                "TEMPERATURA_MEDIA_C"
            ),
            "PRECIPITACION": (
                "PRECIPITACION_MM"
            ),
        }
    )

    # --------------------------------------------------------
    # Orden final de columnas
    # --------------------------------------------------------

    clima = clima[
        [
            "ESTADO",
            "ESTADO_NORM",
            "MES",
            "TEMPERATURA_MEDIA_C",
            "PRECIPITACION_MM",
        ]
    ]

    clima = (
        clima
        .sort_values(
            [
                "ESTADO_NORM",
                "MES",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Validación
    # --------------------------------------------------------

    filas_esperadas = (
        len(estados_norm)
        * 12
    )

    if len(clima) != filas_esperadas:

        raise ValueError(
            f"\nSe esperaban "
            f"{filas_esperadas} registros "
            f"y se obtuvieron "
            f"{len(clima)}."
        )

    columnas_clima = [
        "TEMPERATURA_MEDIA_C",
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

    # --------------------------------------------------------
    # Redondear valores
    # --------------------------------------------------------

    clima[
        "TEMPERATURA_MEDIA_C"
    ] = (
        clima[
            "TEMPERATURA_MEDIA_C"
        ]
        .round(3)
    )

    clima[
        "PRECIPITACION_MM"
    ] = (
        clima[
            "PRECIPITACION_MM"
        ]
        .round(3)
    )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    print(
        "\n=============================================="
    )
    print(
        " RESUMEN CLIMÁTICO ESTATAL"
    )
    print(
        "=============================================="
    )

    print(
        f"Entidades: "
        f"{clima['ESTADO_NORM'].nunique()}"
    )

    print(
        f"Meses:     "
        f"{clima['MES'].nunique()}"
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
        .head(12)
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
        " CLIMA ESTATAL ERA5-LAND · MÉXICO 2025"
    )
    print(
        "=============================================="
    )

    clima = cargar_clima()

    # Crear carpeta de salida si no existe
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Guardar CSV
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
        f"Meses: "
        f"{clima['MES'].nunique()}"
    )

    print(
        f"Registros: "
        f"{len(clima)}"
    )


if __name__ == "__main__":
    main()