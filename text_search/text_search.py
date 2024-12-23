import psycopg2
from openai import OpenAI
import pandas as pd
import os

from rich import print

relevant_columns = {
    "_SkuId (Not changeable)": ("ID SKU", "id_sku"),
    "_SkuName": ("Nombre SKU", "nombre_sku"),
    "_ProductShortDescription": ("Descripción Corta", "descripcion_corta"),
    "_ProductDescription": ("Descripción Larga", "descripcion_larga"),
    "_Keywords": ("Palabras Clave", "palabras_clave"),
    "_MetaTagDescription": ("Descripción Meta", "descripcion_meta"),
    "_DepartamentName": ("Departamento", "departamento"),
    "_CategoryName": ("Categoría", "categoria"),
    "_Brand": ("Marca", "marca"),
}


# Decorador para conectar a la base de datos
def connect(func):
    def wrapper(*args, **kwargs):
        try:
            conn = psycopg2.connect(
                dbname="vectorsearch",
                user="postgres",
                password="",
                host="localhost",
            )
            cur = conn.cursor()
            func(cur, *args, **kwargs)
            conn.commit()
            cur.close()
            conn.close()
        except psycopg2.Error as e:
            print(f"Error de PostgreSQL: {e}")

    return wrapper


def col_format(col):
    new_col = col.replace("_", "")
    new_col = new_col.split("(")[0].strip()
    return new_col.lower()


# @connect
def crear_tabla_desde_excel(excel_file=None):
    if not excel_file:
        raise ValueError("El archivo Excel no puede estar vacío")

    if not os.path.exists(excel_file):
        raise FileNotFoundError("El archivo Excel no existe")

    if not excel_file.endswith(".xlsx"):
        raise ValueError("El archivo debe ser un Excel (.xlsx)")

    # Obtener columnas de la hoja de cálculo
    df = pd.read_excel(excel_file, usecols=relevant_columns.keys())
    df.fillna("No info", inplace=True)
