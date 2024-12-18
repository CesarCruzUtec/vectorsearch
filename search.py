import clip
import torch
import faiss
import numpy as np
from PIL import Image
import shutil
import os

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Cargar el modelo CLIP y la función de preprocesamiento
model, preprocess = clip.load("ViT-B/32", device=device)

# Cargar el índice de Faiss y los nombres de las imágenes
index = faiss.read_index("indice_productos.faiss")
nombres_imagenes = np.load("nombres_imagenes.npy")

# Ruta a la imagen de consulta
ruta_imagen_consulta = "./test/incakola.jpg"

# Preprocesar la imagen de consulta y obtener su embedding
imagen_consulta = preprocess(Image.open(ruta_imagen_consulta)).unsqueeze(0).to(device)
with torch.no_grad():
    embedding_consulta = model.encode_image(imagen_consulta).cpu().numpy()

# Realizar la búsqueda en el índice
k = 5  # Número de resultados a recuperar
distancias, indices = index.search(embedding_consulta, k)

# Mostrar los resultados
print("Resultados de la búsqueda:")
for i, indice in enumerate(indices[0]):
    print(
        "{}. {} (distancia: {:.4f})".format(
            i + 1, nombres_imagenes[indice], distancias[0][i]
        )
    )

# Eliminar previamente cualquier imagen en la carpeta de resultados
resultados_dir = "./closest_matches"
if os.path.exists(resultados_dir):
    for archivo in os.listdir(resultados_dir):
        archivo_path = os.path.join(resultados_dir, archivo)
        try:
            if os.path.isfile(archivo_path) or os.path.islink(archivo_path):
                os.unlink(archivo_path)
            elif os.path.isdir(archivo_path):
                shutil.rmtree(archivo_path)
        except Exception as e:
            print(f"No se pudo eliminar {archivo_path}. Razón: {e}")

# Crear la carpeta si no existe
os.makedirs(resultados_dir, exist_ok=True)

for i, indice in enumerate(indices[0]):
    # Nombre de la images {indice}{distancia}{nombre}.{extensión}
    shutil.copy(
        nombres_imagenes[indice],
        f"{resultados_dir}/{i+1}_{distancias[0][i]:.4f}_{os.path.basename(nombres_imagenes[indice])}",
    )
