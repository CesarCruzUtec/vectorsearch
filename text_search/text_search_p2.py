import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración
ruta_embeddings = "./wong_catalogo_prueba.npy"
ruta_datos = "./wong_catalogo_prueba.csv"

# 1. Cargar los embeddings y los datos del producto
print("Cargando datos y embeddings...")
embeddings = np.load(ruta_embeddings)
df = pd.read_csv(ruta_datos)

# Verificar que los tamaños coincidan
if len(embeddings) != len(df):
    raise ValueError(
        "La cantidad de embeddings no coincide con la cantidad de registros en el archivo de datos."
    )

# 2. Cargar el modelo de embeddings textuales
modelo = SentenceTransformer("PlanTL-GOB-ES/roberta-base-bne")


# 3. Función para buscar productos
def buscar_productos(consulta, top_n=5, umbral=0.5):
    # Generar embedding para la consulta
    embedding_consulta = modelo.encode([consulta.lower()])

    # Calcular similitud de coseno entre la consulta y los embeddings
    similitudes = cosine_similarity(embedding_consulta, embeddings)

    # Obtener los índices de los productos más similares
    indices_similares = np.argsort(similitudes[0])[::-1][:top_n]

    # Filtrar resultados por umbral
    indices_filtrados = [i for i in indices_similares if similitudes[0][i] >= umbral]

    # Retornar los productos más similares
    return df.iloc[indices_filtrados]


# 4. Interfaz de búsqueda
print("Sistema de búsqueda por texto iniciado.")
while True:
    consulta = input("\nEscribe tu búsqueda (o 'salir' para terminar): ")
    if consulta.lower() == "salir":
        print("Cerrando el sistema de búsqueda. ¡Hasta luego!")
        break

    # Realizar búsqueda
    resultados = buscar_productos(consulta)
    if resultados.empty:
        print("No se encontraron productos relevantes.")
    else:
        print("\nProductos más similares:")
        print(
            resultados[
                ["_SkuName", "_DepartamentName", "_CategoryName", "_ProductDescription"]
            ].to_string(index=False)
        )
