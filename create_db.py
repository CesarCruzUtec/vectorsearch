import os
import clip
import torch
import faiss
import shutil
import psycopg2
import numpy as np
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

# Parámetros
EMBEDDING_DIM = 512
INDEX_FILE = "./faiss_index.ivf"  # Archivo para persistir el índice FAISS
IDS_FILE = "./ids_faiss.npy"  # Archivo para persistir los IDs de las imágenes
OUTPUT_DIR = "./closest_matches"  # Directorio para guardar las imágenes similares


def agregar_embedding(ruta_imagen, nombre_archivo=None):
    try:
        if nombre_archivo is None:
            nombre_archivo = os.path.basename(ruta_imagen)

        cur.execute("SELECT 1 FROM imagenes WHERE nombre = %s", (nombre_archivo,))
        if cur.fetchone():
            print(f"El embedding para '{nombre_archivo}' ya existe. No se agregó.")
            return

        imagen = preprocess(Image.open(ruta_imagen)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(imagen).cpu().numpy().flatten().tolist()
        cur.execute(
            "INSERT INTO imagenes (nombre, embedding) VALUES (%s, %s)",
            (nombre_archivo, embedding),
        )
        print(f"Embedding añadido para {nombre_archivo}")
    except Exception as e:
        print(f"Error al añadir embedding para {nombre_archivo}: {e}")


def eliminar_embedding(nombre_archivo):
    cur.execute("DELETE FROM imagenes WHERE nombre = %s", (nombre_archivo,))
    if cur.rowcount > 0:
        print(f"Embedding eliminado para {nombre_archivo}")
    else:
        print(f"No se encontró {nombre_archivo} en la base de datos")


def modificar_embedding(nueva_ruta_imagen, nombre_archivo=None):
    try:
        if nombre_archivo is None:
            nombre_archivo = os.path.basename(nueva_ruta_imagen)

        imagen = preprocess(Image.open(nueva_ruta_imagen)).unsqueeze(0).to(device)
        with torch.no_grad():
            nuevo_embedding = (
                model.encode_image(imagen).cpu().numpy().flatten().tolist()
            )
        cur.execute(
            "UPDATE imagenes SET embedding = %s WHERE nombre = %s",
            (nuevo_embedding, nombre_archivo),
        )
        if cur.rowcount > 0:
            print(f"Embedding modificado para {nombre_archivo}")
        else:
            print(f"No se encontró {nombre_archivo} en la base de datos")
    except Exception as e:
        print(f"Error al modificar embedding para {nombre_archivo}: {e}")


def agregar_embeddings_masa(ruta_carpeta):
    for root, _, files in os.walk(ruta_carpeta):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg", ".JPG")):
                ruta_imagen = os.path.join(root, file)
                try:
                    agregar_embedding(ruta_imagen)
                except Exception as e:
                    print(f"Error al procesar {file}: {e}")


# Función para eliminar todos los datos de la tabla
def eliminar_todos_los_embeddings():
    cur.execute("DELETE FROM imagenes")
    cur.execute("ALTER SEQUENCE imagenes_id_seq RESTART WITH 1")
    print("Todos los embeddings han sido eliminados.")


# Función para cargar embeddings desde PostgreSQL a FAISS
def cargar_embeddings_a_faiss():
    cur.execute("SELECT id, embedding FROM imagenes")
    datos = cur.fetchall()

    ids_faiss = []
    embeddings = []

    for row in datos:
        ids_faiss.append(row[0])
        embeddings.append(np.array(row[1], dtype="float32"))

    # Crear un índice IVFFlat
    index = faiss.IndexIVFFlat(faiss.IndexFlatL2(EMBEDDING_DIM), EMBEDDING_DIM, 100)
    if embeddings:
        embeddings_array = np.vstack(embeddings)
        index.train(embeddings_array)  # Entrenar el índice
        index.add(embeddings_array)  # Añadir embeddings al índice
    print(f"Índice FAISS creado con {len(ids_faiss)} embeddings.")

    # Guardar índice en disco
    faiss.write_index(index, INDEX_FILE)
    np.save(IDS_FILE, np.array(ids_faiss))
    print(f"Índice FAISS y IDs guardados en disco con {len(ids_faiss)} embeddings.")
    return index, ids_faiss


# Función para cargar un índice FAISS desde disco
def cargar_faiss_desde_disco():
    if os.path.exists(INDEX_FILE) and os.path.exists(IDS_FILE):
        index = faiss.read_index(INDEX_FILE)
        ids_faiss = np.load(IDS_FILE).tolist()
        print("Índice FAISS cargado desde disco.")
        return index, ids_faiss
    else:
        print("No se encontró un índice en disco. Creando uno nuevo.")
        return cargar_embeddings_a_faiss()


# Función para sincronizar FAISS incrementalmente después de agregar datos
def sincronizar_faiss_incremental():
    # Cargar índice y IDs desde disco
    index, ids_faiss = cargar_faiss_desde_disco()

    # Obtener los embeddings actualizados desde PostgreSQL
    cur.execute("SELECT id, embedding FROM imagenes")
    datos = cur.fetchall()

    nuevos_embeddings = []
    nuevos_ids = []

    for row in datos:
        if row[0] not in ids_faiss:  # Solo añadir los nuevos embeddings
            nuevos_ids.append(row[0])
            nuevos_embeddings.append(np.array(row[1], dtype="float32"))

    if nuevos_embeddings:
        index.add(np.vstack(nuevos_embeddings))
        ids_faiss.extend(nuevos_ids)
        print(
            f"Índice FAISS sincronizado con {len(nuevos_embeddings)} nuevos embeddings."
        )
        faiss.write_index(index, INDEX_FILE)
        np.save(IDS_FILE, np.array(ids_faiss))
    else:
        print("No hay nuevos embeddings para sincronizar.")


# Función para realizar búsquedas en FAISS
def buscar_con_faiss(ruta_imagen, k=5):
    # Cargar índice y IDs desde disco
    index, ids_faiss = cargar_faiss_desde_disco()

    # Directorio para guardar las imágenes similares
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)  # Limpiar la carpeta antes de copiar
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Imagen de consulta
    imagen = Image.open(ruta_imagen)

    # Convertir a RGBA si la imagen contiene transparencia
    if imagen.mode in ("RGBA", "LA") or (
        imagen.mode == "P" and "transparency" in imagen.info
    ):
        imagen = imagen.convert("RGBA")

    imagen_consulta = preprocess(imagen).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding_consulta = (
            model.encode_image(imagen_consulta).cpu().numpy().flatten().tolist()
        )

    D, ID = index.search(np.array([embedding_consulta], dtype="float32"), k)
    resultados = [(ids_faiss[i], D[0][j]) for j, i in enumerate(ID[0]) if i != -1]

    # Obtener nombres de las imágenes desde PostgreSQL
    if resultados:
        ids_encontrados = [r[0] for r in resultados]
        cur.execute(
            "SELECT id, nombre FROM imagenes WHERE id = ANY(%s)", (ids_encontrados,)
        )
        metadatos = {row[0]: row[1] for row in cur.fetchall()}
        return [(metadatos[r[0]], r[1]) for r in resultados]
    return []


# Confirmar y cerrar conexión
conn.commit()
cur.close()
conn.close()
