#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_integrar_dengue_clima_2025.py

Integra los datos epidemiológicos semanales de dengue con los
datos climáticos semanales agregados por estado para México, 2025.

Para la visualización del dashboard se consideran únicamente
las semanas 1 a 52.
python scripts/03_integrar_dengue_clima_2025.py

Entradas:
    data/epidemiologia/dengue_mexico_2025_estados_semanal.csv
    data/clima/clima_estados_2025_semanal.csv

Salida:
    data/integrada/dengue_clima_2025_estatal_semanal.csv
"""

from pathlib import Path
import unicodedata

import pandas as pd


# ============================================================
# RUTAS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

EPI_PATH = (
    PROJECT_DIR
    / "data"
    / "epidemiologia"
    / "dengue_mexico_2025_estados_semanal.csv"
)

CLIMA_PATH = (
    PROJECT_DIR
    / "data"
    / "clima"
    / "clima_estados_2025_semanal.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "integrada"

OUTPUT_PATH = (
    OUTPUT_DIR
    / "dengue_clima_2025_estatal_semanal.csv"
)


# ============================================================
# FUNCIONES
# ============================================================

def normalizar_estado(texto):
    """
    Normaliza el nombre de un estado para facilitar la unión
    entre las fuentes epidemiológica y climática.
    """

    if pd.isna(texto):
        return None

    texto = str(texto).strip().upper()

    # Eliminar acentos
    texto = "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )

    # Eliminar espacios duplicados
    texto = " ".join(texto.split())

    return texto


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("INTEGRACIÓN DENGUE + CLIMA | MÉXICO 2025")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Verificar archivos
    # --------------------------------------------------------

    if not EPI_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo epidemiológico:\n{EPI_PATH}"
        )

    if not CLIMA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo climático:\n{CLIMA_PATH}"
        )

    print("\nArchivos encontrados:")
    print(f"  Epidemiología: {EPI_PATH}")
    print(f"  Clima:         {CLIMA_PATH}")

    # --------------------------------------------------------
    # 2. Leer datos
    # --------------------------------------------------------

    epi = pd.read_csv(EPI_PATH)
    clima = pd.read_csv(CLIMA_PATH)

    print("\nDatos originales:")
    print(f"  Epidemiología: {len(epi):,} filas")
    print(f"  Clima:         {len(clima):,} filas")

    # --------------------------------------------------------
    # 3. Conservar semanas 1–52
    # --------------------------------------------------------

    epi["SEMANA"] = pd.to_numeric(
        epi["SEMANA"],
        errors="coerce"
    )

    clima["SEMANA"] = pd.to_numeric(
        clima["SEMANA"],
        errors="coerce"
    )

    epi = epi[
        epi["SEMANA"].between(1, 52)
    ].copy()

    clima = clima[
        clima["SEMANA"].between(1, 52)
    ].copy()

    epi["SEMANA"] = epi["SEMANA"].astype(int)
    clima["SEMANA"] = clima["SEMANA"].astype(int)

    print("\nDespués de conservar semanas 1–52:")
    print(f"  Epidemiología: {len(epi):,} filas")
    print(f"  Clima:         {len(clima):,} filas")

    print(
        "  Semanas epidemiología:",
        epi["SEMANA"].nunique()
    )

    print(
        "  Semanas clima:",
        clima["SEMANA"].nunique()
    )

    # --------------------------------------------------------
    # 4. Normalizar estados
    # --------------------------------------------------------

    epi["ESTADO_NORM"] = epi["ESTADO"].apply(
        normalizar_estado
    )

    # El archivo climático ya debería tener ESTADO_NORM,
    # pero lo recalculamos para asegurar consistencia.
    clima["ESTADO_NORM"] = clima["ESTADO"].apply(
        normalizar_estado
    )

    # --------------------------------------------------------
    # 5. Revisar estados
    # --------------------------------------------------------

    estados_epi = set(epi["ESTADO_NORM"].dropna())
    estados_clima = set(clima["ESTADO_NORM"].dropna())

    solo_epi = sorted(
        estados_epi - estados_clima
    )

    solo_clima = sorted(
        estados_clima - estados_epi
    )

    print("\nComparación de estados:")

    print(
        f"  Estados epidemiología: {len(estados_epi)}"
    )

    print(
        f"  Estados clima:         {len(estados_clima)}"
    )

    if solo_epi:
        print("\n⚠ Estados solamente en epidemiología:")
        for estado in solo_epi:
            print(f"   - {estado}")

    if solo_clima:
        print("\n⚠ Estados solamente en clima:")
        for estado in solo_clima:
            print(f"   - {estado}")

    if not solo_epi and not solo_clima:
        print("  ✓ Los estados coinciden.")

    # --------------------------------------------------------
    # 6. Verificar duplicados antes del merge
    # --------------------------------------------------------

    dup_epi = epi.duplicated(
        subset=["ESTADO_NORM", "SEMANA"]
    ).sum()

    dup_clima = clima.duplicated(
        subset=["ESTADO_NORM", "SEMANA"]
    ).sum()

    print("\nDuplicados ESTADO + SEMANA:")
    print(f"  Epidemiología: {dup_epi}")
    print(f"  Clima:         {dup_clima}")

    if dup_epi > 0:
        raise ValueError(
            "Hay combinaciones ESTADO + SEMANA duplicadas "
            "en epidemiología."
        )

    if dup_clima > 0:
        raise ValueError(
            "Hay combinaciones ESTADO + SEMANA duplicadas "
            "en clima."
        )

    # --------------------------------------------------------
    # 7. Integrar
    # --------------------------------------------------------

    # Quitamos ESTADO de clima para no generar
    # ESTADO_x / ESTADO_y.
    columnas_clima = [
        "ESTADO_NORM",
        "SEMANA",
        "FECHA",
        "TEMPERATURA_MIN_C",
        "TEMPERATURA_MEDIA_C",
        "TEMPERATURA_MAX_C",
        "PRECIPITACION_MM",
    ]

    integrado = epi.merge(
        clima[columnas_clima],
        on=["ESTADO_NORM", "SEMANA"],
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # 8. Revisar filas sin clima
    # --------------------------------------------------------

    sin_clima = integrado[
        integrado["TEMPERATURA_MEDIA_C"].isna()
    ]

    print("\nResultado del merge:")
    print(f"  Filas integradas: {len(integrado):,}")
    print(f"  Filas sin clima:  {len(sin_clima):,}")

    if len(sin_clima) > 0:

        print(
            "\n⚠ Combinaciones epidemiológicas "
            "sin información climática:"
        )

        print(
            sin_clima[
                ["ESTADO", "SEMANA"]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 9. Ordenar
    # --------------------------------------------------------

    integrado = integrado.sort_values(
        ["ESTADO_NORM", "SEMANA"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 10. Guardar
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    integrado.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("ARCHIVO GENERADO")
    print("=" * 70)

    print(f"\n{OUTPUT_PATH}")

    print("\nDimensiones:")
    print(
        f"  {len(integrado):,} filas × "
        f"{len(integrado.columns)} columnas"
    )

    print("\nSemanas presentes:")
    print(
        sorted(
            integrado["SEMANA"]
            .dropna()
            .unique()
        )
    )

    print("\n✓ Integración terminada.")


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()