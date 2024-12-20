import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuración
ruta_excel = "./wong_catalogo_prueba.xlsx"
ruta_embeddings = "./wong_catalogo_prueba.npy"
ruta_datos = "./wong_catalogo_prueba.csv"

# 1. Cargar el archivo Excel
df = pd.read_excel(ruta_excel)

# Verificar que las columnas requeridas estén presentes
columnas_requeridas = [
    "_SkuName",
    "_DepartamentName",
    "_CategoryName",
    "_ProductDescription",
]
if not all(col in df.columns for col in columnas_requeridas):
    raise ValueError(
        f"Faltan columnas necesarias en el archivo Excel: {columnas_requeridas}"
    )

# 2. Combinar columnas relevantes para crear un texto representativo
df["texto_combinado"] = (
    df["_SkuName"].fillna("")
    + " "
    + df["_DepartamentName"].fillna("")
    + " "
    + df["_CategoryName"].fillna("")
    + " "
    + df["_ProductDescription"].fillna("")
)

# 3. Cargar el modelo de embeddings textuales
modelo = SentenceTransformer("PlanTL-GOB-ES/roberta-base-bne")

# 4. Generar embeddings
print("Generando embeddings, por favor espera...")
embeddings = modelo.encode(
    df["texto_combinado"].tolist(), batch_size=32, show_progress_bar=True
)

# 5. Guardar los embeddings y los datos del producto
np.save(ruta_embeddings, embeddings)  # Guardar los embeddings como archivo .npy
df.to_csv(
    ruta_datos, index=False, encoding="utf-8-sig"
)  # Guardar datos como referencia

# Mensajes informativos
print(f"Embeddings generados y guardados en '{ruta_embeddings}'")
print(f"Datos del producto guardados en '{ruta_datos}'")
print(f"Procesados {len(df)} productos.")
