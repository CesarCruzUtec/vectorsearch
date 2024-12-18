import os
import clip
import torch
import faiss
import numpy as np
from PIL import Image

from rich import print

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Cargar el modelo CLIP y la función de preprocesamiento
model, preprocess = clip.load("ViT-B/32", device=device)

# Ruta al directorio que contiene las imágenes del catálogo
directorio_imagenes = "./images"

# Lista para almacenar los embeddings y los nombres de los archivos
embeddings = []
nombres_imagenes = []

# Procesar cada imagen en el directorio
for root, _, files in os.walk(directorio_imagenes):
    for nombre_archivo in files:
        if nombre_archivo.endswith((".png", ".jpg", ".jpeg", ".JPG")):
            # print(f"Procesando imagen: {nombre_archivo}")
            ruta_imagen = os.path.join(root, nombre_archivo)
            imagen = preprocess(Image.open(ruta_imagen)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(imagen).cpu().numpy()
            embeddings.append(embedding)
            nombres_imagenes.append(ruta_imagen)

# DEBUG: Mostrar informacion sobre un embedding
print(embeddings[0].shape)


# Convertir la lista de embeddings a un array de NumPy
embeddings_np = np.vstack(embeddings)

# Crear un índice de Faiss y agregar los embeddings
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_np)

# Guardar el índice y los nombres de las imágenes
faiss.write_index(index, "indice_productos.faiss")
np.save("nombres_imagenes.npy", np.array(nombres_imagenes))

print(
    "Indexación completada. Se han procesado {} imágenes.".format(len(nombres_imagenes))
)
