#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figura para cartel científico:
Datos epidemiológicos y ambientales de entrada al modelo predictivo de dengue.

python Code/graficas/04_figura_entrada_modelo_2025.py

Panel A
-------
Distribución de casos confirmados acumulados de dengue
por entidad federativa, México, 2025.

Panel B
-------
Dinámica temporal para cuatro estados seleccionados:
    - Sonora
    - Veracruz
    - Jalisco
    - Sinaloa

Variables:
    - Casos confirmados semanales
    - Temperatura mensual
    - Precipitación mensual

Salida:
    PNG y PDF a 300 dpi.
"""

from pathlib import Path
import unicodedata
import warnings

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(".")

EPIDEMIOLOGIA = (
    BASE_DIR
    / "Data"
    / "epidemiologia"
    / "resultados"
    / "dengue_mexico_2025_estados_semanal.csv"
)

CLIMA_DIR = (
    BASE_DIR
    / "Data"
    / "Clima"
    / "ERA5_Land_Mexico_Dashboard"
    / "2025"
    / "monthly"
)

GEOJSON = (
    BASE_DIR
    / "Data"
    / "epidemiologia"
    / "geografia"
    / "mexico_estados.geojson"
)

OUTPUT_DIR = BASE_DIR /  "Data"  / "epidemiologia" / "resultados" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PNG = OUTPUT_DIR / "Figura_Entrada_Modelo_Dengue_2025.png"
OUTPUT_PDF = OUTPUT_DIR / "Figura_Entrada_Modelo_Dengue_2025.pdf"

DPI = 300

ESTADOS = [
    "Sonora",
    "Veracruz",
    "Jalisco"
    #"Sinaloa",
]

# Colores por tipo de variable
COLOR_CASOS = "#A61B1B"
COLOR_TEMP = "#D95F02"
COLOR_PRECIP = "#2B6CB0"

COLOR_GRID = "#D9D9D9"
COLOR_TEXT = "#252525"
COLOR_SECONDARY = "#666666"


# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================

def normalizar(texto):
    """
    Normaliza nombres para facilitar cruces entre
    epidemiología, clima y GeoJSON.
    """
    if pd.isna(texto):
        return ""

    texto = str(texto).strip().lower()

    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
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
    Busca una columna usando varios nombres posibles.
    """
    columnas = {
        normalizar(c): c
        for c in df.columns
    }

    for opcion in opciones:
        op = normalizar(opcion)

        if op in columnas:
            return columnas[op]

    return None


def configurar_estilo():

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ============================================================
# 3. LEER EPIDEMIOLOGÍA
# ============================================================

def cargar_epidemiologia():

    if not EPIDEMIOLOGIA.exists():
        raise FileNotFoundError(
            f"\nNo encontré:\n{EPIDEMIOLOGIA}\n"
        )

    df = pd.read_csv(EPIDEMIOLOGIA)

    print("\nColumnas epidemiología:")
    print(df.columns.tolist())

    requeridas = [
        "YEAR",
        "SEMANA",
        "ESTADO",
        "TOTAL_CONFIRMADOS",
    ]

    faltantes = [
        c for c in requeridas
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas epidemiológicas: {faltantes}"
        )

    df = df[df["YEAR"] == 2025].copy()

    df["ESTADO_NORM"] = (
        df["ESTADO"]
        .astype(str)
        .apply(normalizar)
    )

    # --------------------------------------------------------
    # Casos semanales
    # --------------------------------------------------------

    if "TOTAL_CONFIRMADOS_SEMANA" in df.columns:

        df["CASOS_SEMANA"] = pd.to_numeric(
            df["TOTAL_CONFIRMADOS_SEMANA"],
            errors="coerce"
        )

    else:

        # Si no existe la columna semanal,
        # se obtiene mediante diferencias del acumulado.

        df = df.sort_values(
            ["ESTADO_NORM", "SEMANA"]
        )

        df["CASOS_SEMANA"] = (
            df
            .groupby("ESTADO_NORM")["TOTAL_CONFIRMADOS"]
            .diff()
        )

        # Primera semana
        primera = (
            df.groupby("ESTADO_NORM")
            .cumcount()
            .eq(0)
        )

        df.loc[
            primera,
            "CASOS_SEMANA"
        ] = df.loc[
            primera,
            "TOTAL_CONFIRMADOS"
        ]

    # Evitar negativos producidos por correcciones
    df["CASOS_SEMANA"] = (
        df["CASOS_SEMANA"]
        .clip(lower=0)
    )

    return df


# ============================================================
# 4. DATOS FINALES 2025 PARA EL MAPA
# ============================================================

def obtener_cierre_2025(df):

    ultima_semana = int(
        df["SEMANA"].max()
    )

    print(
        f"\nÚltima semana epidemiológica encontrada: "
        f"SE{ultima_semana}"
    )

    cierre = (
        df[df["SEMANA"] == ultima_semana]
        .copy()
    )

    cierre = cierre[
        [
            "ESTADO",
            "ESTADO_NORM",
            "TOTAL_CONFIRMADOS"
        ]
    ]

    print("\nEstados con más casos al cierre:")

    print(
        cierre
        .sort_values(
            "TOTAL_CONFIRMADOS",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    return cierre, ultima_semana


# ============================================================
# 5. LEER GEOJSON
# ============================================================

def cargar_geografia():

    if not GEOJSON.exists():
        raise FileNotFoundError(
            f"\nNo encontré:\n{GEOJSON}\n"
        )

    geo = gpd.read_file(GEOJSON)

    print("\nColumnas GeoJSON:")
    print(geo.columns.tolist())

    nombre_col = detectar_columna(
        geo,
        [
            "name",
            "nombre",
            "estado",
            "entidad",
            "NOMGEO",
            "NOM_ENT",
        ]
    )

    if nombre_col is None:

        raise ValueError(
            "\nNo pude detectar la columna "
            "con el nombre del estado.\n"
            f"Columnas disponibles:\n{geo.columns.tolist()}"
        )

    print(
        f"Columna geográfica utilizada: {nombre_col}"
    )

    geo["ESTADO_NORM"] = (
        geo[nombre_col]
        .astype(str)
        .apply(normalizar)
    )

    return geo


# ============================================================
# 6. LEER DATOS CLIMÁTICOS
# ============================================================

def cargar_clima():
    """
    Lee los JSON mensuales generados para el dashboard de ERA5-Land
    y calcula valores climáticos mensuales para los estados seleccionados.

    Variables utilizadas
    --------------------
    tmean_c:
        Temperatura media mensual (°C).

    precip_mm:
        Precipitación acumulada mensual (mm).

    Metodología
    -----------
    Cada JSON contiene una malla regular:
        latitude x longitude -> values

    Se construyen puntos geográficos para las celdas válidas de ERA5-Land
    y se seleccionan aquellas cuyo centro cae dentro de cada entidad
    federativa.

    Para cada estado y mes:
        TEMPERATURA = promedio espacial de tmean_c
        PRECIPITACION = promedio espacial de precip_mm

    Returns
    -------
    pandas.DataFrame
        ESTADO_NORM | MES | TEMPERATURA | PRECIPITACION
    """

    import json

    # --------------------------------------------------------
    # Carpetas
    # --------------------------------------------------------

    temp_dir = CLIMA_DIR / "tmean_c"
    precip_dir = CLIMA_DIR / "precip_mm"

    if not temp_dir.exists():
        raise FileNotFoundError(
            f"\nNo existe:\n{temp_dir}"
        )

    if not precip_dir.exists():
        raise FileNotFoundError(
            f"\nNo existe:\n{precip_dir}"
        )

    archivos_temp = sorted(
        temp_dir.glob("tmean_c_2025-*.json")
    )

    archivos_precip = sorted(
        precip_dir.glob("precip_mm_2025-*.json")
    )

    print("\n==============================================")
    print(" DATOS CLIMÁTICOS ERA5-LAND")
    print("==============================================")

    print(
        f"Temperatura:   {len(archivos_temp)} meses"
    )

    print(
        f"Precipitación: {len(archivos_precip)} meses"
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

    geo = gpd.read_file(GEOJSON)

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
            "No pude identificar la columna "
            "con el nombre de los estados."
        )

    geo["ESTADO_NORM"] = (
        geo[nombre_col]
        .astype(str)
        .apply(normalizar)
    )

    # ERA5-Land usa lat/lon -> EPSG:4326
    if geo.crs is None:
        geo = geo.set_crs(
            "EPSG:4326"
        )
    else:
        geo = geo.to_crs(
            "EPSG:4326"
        )

    # Solo necesitamos los cuatro estados
    estados_norm = [
        normalizar(e)
        for e in ESTADOS
    ]

    geo_sel = geo[
        geo["ESTADO_NORM"].isin(
            estados_norm
        )
    ].copy()

    print("\nEstados encontrados en GeoJSON:")

    for estado in ESTADOS:

        encontrado = (
            normalizar(estado)
            in geo_sel["ESTADO_NORM"].values
        )

        print(
            f"  {estado:<10} "
            f"{'✓' if encontrado else '✗'}"
        )

    if len(geo_sel) != len(ESTADOS):
        raise ValueError(
            "\nNo se encontraron correctamente "
            "los cuatro estados en el GeoJSON."
        )

    # --------------------------------------------------------
    # Función para leer un JSON ERA5-Land
    # --------------------------------------------------------

    def leer_json_era5(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            d = json.load(f)

        meta = d["meta"]

        lat = np.asarray(
            meta["latitude"],
            dtype=float
        )

        lon = np.asarray(
            meta["longitude"],
            dtype=float
        )

        values = np.asarray(
            d["values"],
            dtype=float
        )

        fecha = pd.to_datetime(
            meta["date"]
        )

        # ----------------------------------------------------
        # Validar dimensiones
        # ----------------------------------------------------

        esperado = len(lat) * len(lon)

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
        # Crear coordenadas correspondientes a values
        #
        # La matriz está almacenada como:
        # latitude x longitude
        # ----------------------------------------------------

        lon_grid, lat_grid = np.meshgrid(
            lon,
            lat
        )

        lon_flat = lon_grid.ravel()
        lat_flat = lat_grid.ravel()
        val_flat = values.ravel()

        # ----------------------------------------------------
        # Quitar NaN
        # ----------------------------------------------------

        validos = np.isfinite(
            val_flat
        )

        df = pd.DataFrame({
            "GRID_INDEX": np.arange(
                val_flat.size
            )[validos],
            "longitude": lon_flat[validos],
            "latitude": lat_flat[validos],
            "valor": val_flat[validos],
        })

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
            meta
        )

    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # La geometría de la malla NO cambia entre meses.
    # Por eso calculamos qué píxeles pertenecen a cada estado
    # UNA SOLA VEZ.
    # --------------------------------------------------------

    print(
        "\nConstruyendo máscara espacial ERA5-Land → estados..."
    )

    puntos_base, _, meta_base = (
        leer_json_era5(
            archivos_temp[0]
        )
    )

    # --------------------------------------------------------
    # Spatial join
    # --------------------------------------------------------

    puntos_estados = gpd.sjoin(
        puntos_base,
        geo_sel[
            [
                "ESTADO_NORM",
                "geometry"
            ]
        ],
        how="inner",
        predicate="within",
    )

    # Guardamos índices originales de la malla
    indices_estado = {}

    for estado in estados_norm:

        indices = (
            puntos_estados.loc[
                puntos_estados["ESTADO_NORM"] == estado,
                "GRID_INDEX"
            ]
            .astype(int)
            .to_numpy()
        )

        indices_estado[estado] = indices

        print(
            f"  {estado:<10}: "
            f"{len(indices):>4} celdas ERA5-Land"
        )

    # --------------------------------------------------------
    # Función para extraer valores directamente de JSON
    # usando la misma malla
    # --------------------------------------------------------

    def valores_json(path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            d = json.load(f)

        meta = d["meta"]

        valores = np.asarray(
            d["values"],
            dtype=float
        )

        mes = pd.to_datetime(
            meta["date"]
        ).month

        return (
            valores,
            mes,
            meta
        )

    # --------------------------------------------------------
    # Procesar temperatura
    # --------------------------------------------------------

    registros_temp = []

    print(
        "\nProcesando temperatura media mensual..."
    )

    for archivo in archivos_temp:

        valores, mes, meta = (
            valores_json(
                archivo
            )
        )

        for estado in estados_norm:

            idx = indices_estado[
                estado
            ]

            datos = valores[idx]

            datos = datos[
                np.isfinite(datos)
            ]

            media = (
                float(np.mean(datos))
                if len(datos)
                else np.nan
            )

            registros_temp.append({
                "ESTADO_NORM": estado,
                "MES": mes,
                "TEMPERATURA": media,
            })

    temp_df = pd.DataFrame(
        registros_temp
    )

    # --------------------------------------------------------
    # Procesar precipitación
    # --------------------------------------------------------

    registros_precip = []

    print(
        "Procesando precipitación mensual..."
    )

    for archivo in archivos_precip:

        valores, mes, meta = (
            valores_json(
                archivo
            )
        )

        for estado in estados_norm:

            idx = indices_estado[
                estado
            ]

            datos = valores[idx]

            datos = datos[
                np.isfinite(datos)
            ]

            media = (
                float(np.mean(datos))
                if len(datos)
                else np.nan
            )

            registros_precip.append({
                "ESTADO_NORM": estado,
                "MES": mes,
                "PRECIPITACION": media,
            })

    precip_df = pd.DataFrame(
        registros_precip
    )

    # --------------------------------------------------------
    # Combinar
    # --------------------------------------------------------

    clima = temp_df.merge(
        precip_df,
        on=[
            "ESTADO_NORM",
            "MES"
        ],
        how="outer",
    )

    clima = clima.sort_values(
        [
            "ESTADO_NORM",
            "MES"
        ]
    )

    # --------------------------------------------------------
    # Validación
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

    for estado in ESTADOS:

        estado_norm = normalizar(
            estado
        )

        sub = clima[
            clima["ESTADO_NORM"]
            == estado_norm
        ]

        print(
            f"\n{estado.upper()}"
        )

        print(
            sub[
                [
                    "MES",
                    "TEMPERATURA",
                    "PRECIPITACION"
                ]
            ]
            .round(2)
            .to_string(
                index=False
            )
        )

    return clima

# ============================================================
# 7. PANEL A — MAPA NACIONAL
# ============================================================

def dibujar_mapa(
    ax,
    geo,
    cierre,
    ultima_semana
):

    mapa = geo.merge(
        cierre,
        on="ESTADO_NORM",
        how="left"
    )

    # Cuantiles altos pueden comprimir mucho el mapa.
    # Usamos escala continua con límite basado en máximo real.

    mapa.plot(
        column="TOTAL_CONFIRMADOS",
        cmap="YlOrRd",
        linewidth=0.55,
        edgecolor="white",
        ax=ax,
        missing_kwds={
            "color": "#EFEFEF",
            "edgecolor": "white",
        },
    )

    # Contorno nacional
    mapa.boundary.plot(
        ax=ax,
        linewidth=0.65,
        color="#555555"
    )

    # --------------------------------------------------------
    # Resaltar cuatro estados
    # --------------------------------------------------------

    seleccion = mapa[
        mapa["ESTADO_NORM"].isin(
            [normalizar(e) for e in ESTADOS]
        )
    ]

    seleccion.boundary.plot(
        ax=ax,
        linewidth=1.8,
        color="#222222"
    )

    # Etiquetas
    for _, row in seleccion.iterrows():

        punto = row.geometry.representative_point()

        casos = int(
            row["TOTAL_CONFIRMADOS"]
        )

        nombre = row["ESTADO"]

        ax.annotate(
            f"{nombre}\n{casos:,}",
            xy=(punto.x, punto.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=COLOR_TEXT,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="#777777",
                lw=0.5,
                alpha=0.92,
            ),
        )

    ax.set_title(
        "A. Distribución de casos confirmados de dengue en México · 2025",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    ax.text(
        0,
        1.005,
        f"Casos acumulados por entidad federativa al cierre de SE{ultima_semana}",
        transform=ax.transAxes,
        fontsize=9,
        color=COLOR_SECONDARY,
        va="bottom",
    )

    ax.axis("off")

    # --------------------------------------------------------
    # Barra de color
    # --------------------------------------------------------

    vmax = mapa["TOTAL_CONFIRMADOS"].max()

    norm = mpl.colors.Normalize(
        vmin=0,
        vmax=vmax
    )

    sm = mpl.cm.ScalarMappable(
        cmap="YlOrRd",
        norm=norm
    )

    sm.set_array([])

    cbar = ax.figure.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.025,
        pad=0.015,
    )

    cbar.set_label(
        "Casos confirmados",
        fontsize=13,
        fontweight="bold",
        labelpad=10
    )

    cbar.ax.tick_params(
        labelsize=11
    )




# ============================================================
# 8. SERIES EPIDEMIOLÓGICAS
# ============================================================

def dibujar_casos(
    ax,
    df,
    estado
):

    sub = (
        df[
            df["ESTADO_NORM"]
            == normalizar(estado)
        ]
        .sort_values("SEMANA")
    )

    ax.plot(
        sub["SEMANA"],
        sub["CASOS_SEMANA"],
        color=COLOR_CASOS,
        linewidth=1.8,
    )

    ax.fill_between(
        sub["SEMANA"],
        sub["CASOS_SEMANA"],
        color=COLOR_CASOS,
        alpha=0.12,
    )

    ax.set_xlim(
        1,
        max(53, sub["SEMANA"].max())
    )

    ax.set_ylabel(
        "Casos\nsemanales",
        fontsize=8
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.5
    )

    ax.tick_params(
        axis="x",
        labelbottom=False
    )


# ============================================================
# 9. TEMPERATURA
# ============================================================

def dibujar_temperatura(
    ax,
    clima,
    estado
):

    sub = (
        clima[
            clima["ESTADO_NORM"]
            == normalizar(estado)
        ]
        .sort_values("MES")
    )

    ax.plot(
        sub["MES"],
        sub["TEMPERATURA"],
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        color=COLOR_TEMP,
    )

    ax.set_xlim(
        1,
        12
    )

    ax.set_ylabel(
        "Temp.\n(°C)",
        fontsize=8
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.5
    )

    ax.tick_params(
        axis="x",
        labelbottom=False
    )


# ============================================================
# 10. PRECIPITACIÓN
# ============================================================

def dibujar_precipitacion(
    ax,
    clima,
    estado
):

    sub = (
        clima[
            clima["ESTADO_NORM"]
            == normalizar(estado)
        ]
        .sort_values("MES")
    )

    ax.bar(
        sub["MES"],
        sub["PRECIPITACION"],
        width=0.72,
        color=COLOR_PRECIP,
        alpha=0.8,
    )

    ax.set_xlim(
        0.5,
        12.5
    )

    ax.set_ylabel(
        "Precip.\n(mm)",
        fontsize=8
    )

    ax.set_xticks(
        range(1, 13)
    )

    ax.set_xticklabels(
        [
            "E", "F", "M", "A",
            "M", "J", "J", "A",
            "S", "O", "N", "D"
        ],
        fontsize=7
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        linewidth=0.5
    )


# ============================================================
# 11. FRANJA DE ENTRADA AL MODELO
# ============================================================

def dibujar_modelo(ax):

    ax.axis("off")

    ax.text(
        0.5,
        0.88,
        "VARIABLES DE ENTRADA AL MODELO PREDICTIVO",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLOR_TEXT,
    )

    elementos = [
        (
            0.13,
            "EPIDEMIOLOGÍA",
            "Casos confirmados\npor semana"
        ),
        (
            0.38,
            "TEMPERATURA",
            "ERA5-Land\nmensual"
        ),
        (
            0.63,
            "PRECIPITACIÓN",
            "ERA5-Land\nmensual"
        ),
    ]

    for x, titulo, subtitulo in elementos:

        caja = FancyBboxPatch(
            (x - 0.09, 0.20),
            0.18,
            0.43,
            boxstyle="round,pad=0.015",
            linewidth=0.8,
            edgecolor="#888888",
            facecolor="#F7F7F7",
            transform=ax.transAxes,
        )

        ax.add_patch(caja)

        ax.text(
            x,
            0.47,
            titulo,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            transform=ax.transAxes,
        )

        ax.text(
            x,
            0.30,
            subtitulo,
            ha="center",
            va="center",
            fontsize=7.5,
            color=COLOR_SECONDARY,
            transform=ax.transAxes,
        )

    # Flechas
    ax.annotate(
        "",
        xy=(0.84, 0.42),
        xytext=(0.73, 0.42),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            lw=1.5,
            color="#444444"
        ),
    )

    modelo = FancyBboxPatch(
        (0.84, 0.17),
        0.14,
        0.50,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="#444444",
        facecolor="#EEEEEE",
        transform=ax.transAxes,
    )

    ax.add_patch(modelo)

    ax.text(
        0.91,
        0.47,
        "MODELO",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        0.91,
        0.32,
        "PREDICTIVO",
        ha="center",
        va="center",
        fontsize=8,
        transform=ax.transAxes,
    )


# ============================================================
# 12. FIGURA COMPLETA
# ============================================================

def crear_figura(
    epidemiologia,
    cierre,
    ultima_semana,
    geo,
    clima
):

    configurar_estilo()

    fig = plt.figure(
        figsize=(14, 8),
        constrained_layout=False
    )

    # Solo 3 filas:
    # casos, temperatura y precipitación
    gs = GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        height_ratios=[
            1.7,   # casos
            1.35,  # temperatura
            1.35,  # precipitación
        ],
        hspace=0.48,
        wspace=0.30,
    )

    # ========================================================
    # TRES ESTADOS
    # ========================================================

    for col, estado in enumerate(ESTADOS):

        # ----------------------------------------------------
        # CASOS DE DENGUE
        # ----------------------------------------------------

        ax_casos = fig.add_subplot(
            gs[0, col]
        )

        dibujar_casos(
            ax_casos,
            epidemiologia,
            estado
        )

        fila = cierre[
            cierre["ESTADO_NORM"]
            == normalizar(estado)
        ]

        if len(fila):

            total = int(
                fila.iloc[0]["TOTAL_CONFIRMADOS"]
            )

            titulo = (
                f"{estado}\n"
                f"{total:,} casos confirmados"
            )

        else:
            titulo = estado

        ax_casos.set_title(
            titulo,
            fontsize=16,
            fontweight="bold",
            pad=12
        )

        # ----------------------------------------------------
        # TEMPERATURA
        # ----------------------------------------------------

        ax_temp = fig.add_subplot(
            gs[1, col]
        )

        dibujar_temperatura(
            ax_temp,
            clima,
            estado
        )

        # ----------------------------------------------------
        # PRECIPITACIÓN
        # ----------------------------------------------------

        ax_prec = fig.add_subplot(
            gs[2, col]
        )

        dibujar_precipitacion(
            ax_prec,
            clima,
            estado
        )

    # ========================================================
    # AUMENTAR TEXTO DE TODOS LOS EJES
    # ========================================================

    for ax in fig.axes:

        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)

        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")

        ax.tick_params(
            axis="both",
            labelsize=11
        )

    # ========================================================
    # AJUSTAR ESPACIO
    # ========================================================

    plt.subplots_adjust(
        top=0.94,
        bottom=0.08,
        left=0.07,
        right=0.98
    )

    # ========================================================
    # GUARDAR
    # ========================================================

    output_png = (
        OUTPUT_DIR
        / "Figura_Dengue_Clima_Estados_2025.png"
    )

    output_pdf = (
        OUTPUT_DIR
        / "Figura_Dengue_Clima_Estados_2025.pdf"
    )

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        facecolor="white"
    )

    print("\nFigura temporal guardada:")
    print(output_png)
    print(output_pdf)

    plt.show()
def crear_mapa_nacional(
    cierre,
    ultima_semana,
    geo
):

    configurar_estilo()

    fig, ax = plt.subplots(
        figsize=(10, 6.5)
    )

    dibujar_mapa(
        ax,
        geo,
        cierre,
        ultima_semana
    )

    fig.suptitle(
        "Distribución de casos confirmados de dengue",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    fig.text(
        0.5,
        0.92,
        "México · 2025",
        ha="center",
        fontsize=11,
        color=COLOR_SECONDARY,
    )

    output_png = (
        OUTPUT_DIR
        / "Mapa_Dengue_Mexico_2025.png"
    )

    output_pdf = (
        OUTPUT_DIR
        / "Mapa_Dengue_Mexico_2025.pdf"
    )

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        facecolor="white"
    )

    print("\nMapa nacional guardado:")
    print(output_png)
    print(output_pdf)

    plt.show()
# ============================================================
# 13. MAIN
# ============================================================

def main():

    print(
        "\n=============================================="
    )
    print(
        " FIGURA DENGUE + CLIMA · MÉXICO 2025"
    )
    print(
        "=============================================="
    )

    epidemiologia = cargar_epidemiologia()

    cierre, ultima_semana = (
        obtener_cierre_2025(
            epidemiologia
        )
    )

    geo = cargar_geografia()

    clima = cargar_clima()

    crear_mapa_nacional(
    cierre,
    ultima_semana,
    geo
)

    crear_figura(
        epidemiologia,
        cierre,
        ultima_semana,
        geo,
        clima
    )


if __name__ == "__main__":
    main()