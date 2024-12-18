import os
import shutil
import clip
import torch
import psycopg2
from PIL import Image

# Configuración de CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Conexión a la base de datos
conn = psycopg2.connect(
    dbname="vectorsearch",
    user="yubi",
    password="",
    host="localhost",
)
cur = conn.cursor()

# Directorio para guardar las imágenes similares
output_directory = "./closest_matches"
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)  # Limpiar la carpeta antes de copiar
os.makedirs(output_directory, exist_ok=True)

# Imagen de consulta
ruta_imagen_consulta = "./test/peine.png"
imagen = Image.open(ruta_imagen_consulta)

# Convertir a RGBA si la imagen contiene transparencia
if imagen.mode in ("RGBA", "LA") or (imagen.mode == "P" and "transparency" in imagen.info):
    imagen = imagen.convert("RGBA")

imagen_consulta = preprocess(imagen).unsqueeze(0).to(device)
with torch.no_grad():
    embedding_consulta = (
        model.encode_image(imagen_consulta).cpu().numpy().flatten().tolist()
    )

# Buscar imágenes similares
cur.execute(
    """
    SELECT nombre, embedding <-> %s::vector AS distancia
    FROM imagenes
    ORDER BY distancia
    LIMIT 5
    """,
    (embedding_consulta,),
)
resultados = cur.fetchall()

print("Imágenes similares:")
for i, (nombre_imagen, distancia) in enumerate(resultados):
    print(f"{i + 1}. {nombre_imagen} (Similitud: {distancia:.4f})")

    # Copiar imagen similar al directorio de salida con índice en el nombre
    origen = os.path.join("./images", nombre_imagen)
    destino = os.path.join(output_directory, f"{i + 1}_{nombre_imagen}")
    shutil.copy(origen, destino)

print(f"Imágenes similares copiadas a {output_directory}")

# Cerrar conexión
cur.close()
conn.close()
