import os
import clip
import torch
import faiss
import shutil
import psycopg2
import numpy as np
from PIL import Image

if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Parámetros
OUTPUT_DIR = "./image_search/closest_matches"

model = None
preprocess = None
model_name: str = None
model_dim: int = None
model_id: int = None
index_file: str = None
ids_file: str = None

MODELOS = {
    "RN50": {"name": "RN50", "dim": 1024, "id": 1},
    "RN101": {"name": "RN101", "dim": 512, "id": 2},
    "RN50x4": {"name": "RN50x4", "dim": 640, "id": 3},
    "RN50x16": {"name": "RN50x16", "dim": 768, "id": 4},
    "RN50x64": {"name": "RN50x64", "dim": 1024, "id": 5},
    "ViT-B/32": {"name": "ViTB32", "dim": 512, "id": 6},
    "ViT-B/16": {"name": "ViTB16", "dim": 512, "id": 7},
    "ViT-L/14": {"name": "ViTL14", "dim": 768, "id": 8},
    "ViT-L/14@336px": {"name": "ViTL14_336px", "dim": 768, "id": 9},
}

# Configuración de CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"


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


def cargar_modelo_clip(modelo="ViT-B/32"):
    global model, preprocess, model_dim, model_name, index_file, ids_file, model_id

    if modelo not in MODELOS:
        print(f"El modelo {modelo} no está disponible.")
        return

    model_dim = MODELOS[modelo]["dim"]
    model_name = MODELOS[modelo]["name"]
    model_id = MODELOS[modelo]["id"] + 1  # Columna en la tabla 'imagenes'
    model, preprocess = clip.load(modelo, device=device)

    model_folder = os.path.join("./image_search/models", model_name)
    os.makedirs(model_folder, exist_ok=True)

    index_file = os.path.join(model_folder, "index.faiss")
    ids_file = os.path.join(model_folder, "ids.npy")

    print(f"Modelo {modelo} cargado.")


@connect
def crear_tabla_imagenes(cur):
    if model_name is None or model_dim is None:
        print(
            "No se ha cargado un modelo CLIP. Ejecute 'cargar_modelo_clip()' primero."
        )
        return

    try:
        # Crear la tabla si no existe
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS imagenes (
                id SERIAL PRIMARY KEY,
                nombre TEXT NOT NULL
            )
            """
        )

        # Añadir una columna para cada modelo disponible
        for modelo in MODELOS.values():
            cur.execute(
                f"""
                ALTER TABLE imagenes 
                ADD COLUMN IF NOT EXISTS {model_name} FLOAT[{model_dim}]
                """
            )

        print("Tabla 'imagenes' creada y columnas añadidas.")
    except Exception as e:
        print(f"Error al crear la tabla 'imagenes': {e}")


@connect
def agregar_embedding(cur, ruta_imagen, nombre_archivo=None):
    if model is None or preprocess is None:
        print(
            "No se ha cargado un modelo CLIP. Ejecute 'cargar_modelo_clip()' primero."
        )
        return

    if nombre_archivo is None:
        nombre_archivo = os.path.basename(ruta_imagen)

    try:
        cur.execute(
            f"SELECT {model_name} FROM imagenes WHERE nombre = %s", (nombre_archivo,)
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            print(f"El embedding para {nombre_archivo} con {model_name} ya existe.")
            return

        imagen = Image.open(ruta_imagen)

        # Convertir a RGBA si la imagen contiene transparencia
        if imagen.mode in ("RGBA", "LA") or (
            imagen.mode == "P" and "transparency" in imagen.info
        ):
            imagen = imagen.convert("RGBA")

        imagen = preprocess(imagen).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(imagen).cpu().numpy().flatten().tolist()

        if not row:
            cur.execute(
                f"INSERT INTO imagenes (nombre, {model_name}) VALUES (%s, %s)",
                (nombre_archivo, embedding),
            )
            print(f"Embedding añadido para {nombre_archivo} con {model_name}")
        else:
            cur.execute(
                f"UPDATE imagenes SET {model_name} = %s WHERE nombre = %s",
                (embedding, nombre_archivo),
            )
            print(f"Embedding actualizado para {nombre_archivo} con {model_name}")
    except Exception as e:
        print(f"Error al añadir embedding para {nombre_archivo}: {e}")


@connect
def eliminar_embedding(cur, nombre_archivo):
    cur.execute("DELETE FROM imagenes WHERE nombre = %s", (nombre_archivo,))
    if cur.rowcount > 0:
        print(f"Embedding eliminado para {nombre_archivo}")
    else:
        print(f"No se encontró {nombre_archivo} en la base de datos")


@connect
def modificar_embedding(cur, nueva_ruta_imagen, nombre_archivo=None):
    if model is None or preprocess is None:
        print(
            "No se ha cargado un modelo CLIP. Ejecute 'cargar_modelo_clip()' primero."
        )
        return

    try:
        if nombre_archivo is None:
            nombre_archivo = os.path.basename(nueva_ruta_imagen)

        imagen = preprocess(Image.open(nueva_ruta_imagen)).unsqueeze(0).to(device)
        with torch.no_grad():
            nuevo_embedding = (
                model.encode_image(imagen).cpu().numpy().flatten().tolist()
            )
        cur.execute(
            f"UPDATE imagenes SET {model_name} = %s WHERE nombre = %s",
            (nuevo_embedding, nombre_archivo),
        )
        if cur.rowcount > 0:
            print(f"Embedding modificado para {nombre_archivo}")
        else:
            print(f"No se encontró {nombre_archivo} en la base de datos")
    except Exception as e:
        print(f"Error al modificar embedding para {nombre_archivo}: {e}")


def agregar_embeddings_masa(ruta_carpeta):
    if model is None or preprocess is None:
        print(
            "No se ha cargado un modelo CLIP. Ejecute 'cargar_modelo_clip()' primero."
        )
        return

    for root, _, files in os.walk(ruta_carpeta):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg", ".JPG")):
                ruta_imagen = os.path.join(root, file)
                try:
                    agregar_embedding(ruta_imagen)
                except Exception as e:
                    print(f"Error al procesar {file}: {e}")


@connect
def eliminar_todos_los_embeddings(cur):
    cur.execute("DELETE FROM imagenes")
    cur.execute("ALTER SEQUENCE imagenes_id_seq RESTART WITH 1")
    print("Todos los embeddings han sido eliminados.")


@connect
def cargar_embeddings_a_faiss(cur):
    if model is None or preprocess is None:
        print(
            "No se ha cargado un modelo CLIP. Ejecute 'cargar_modelo_clip()' primero."
        )
        return

    cur.execute(f"SELECT id, {model_name} FROM imagenes")
    datos = cur.fetchall()

    ids_faiss = []
    embeddings = []

    for row in datos:
        ids_faiss.append(row[0])
        embeddings.append(np.array(row[1], dtype="float32"))

    # Crear un índice IVFFlat
    # index = faiss.IndexIVFFlat(faiss.IndexFlatL2(EMBEDDING_DIM), EMBEDDING_DIM, 72)
    index = faiss.IndexFlatL2(model_dim)
    if embeddings:
        embeddings_array = np.vstack(embeddings)
        # index.train(embeddings_array)  # Entrenar el índice
        index.add(embeddings_array)  # Añadir embeddings al índice

    # Guardar índice en disco
    faiss.write_index(index, index_file)
    np.save(ids_file, np.array(ids_faiss))
    print(
        f"Índice FAISS y IDs guardados en disco con {len(ids_faiss)} embeddings para {model_name}."
    )
    return index, ids_faiss


def cargar_faiss_desde_disco():
    if os.path.exists(index_file) and os.path.exists(ids_file):
        index = faiss.read_index(index_file)
        ids_faiss = np.load(ids_file).tolist()
        print("Índice FAISS cargado desde disco.")
        return index, ids_faiss
    else:
        print("No se encontró un índice en disco. Creando uno nuevo.")
        return cargar_embeddings_a_faiss()


@connect
def sincronizar_faiss_incremental(cur):
    # Cargar índice y IDs desde disco
    index, ids_faiss = cargar_faiss_desde_disco()

    # Obtener los embeddings actualizados desde PostgreSQL
    cur.execute("SELECT id, %s FROM imagenes", (model_name,))
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
        faiss.write_index(index, index_file)
        np.save(ids_file, np.array(ids_faiss))
    else:
        print("No hay nuevos embeddings para sincronizar.")


@connect
def buscar_con_faiss(cur, ruta_imagen, k=5):
    if model is None or preprocess is None:
        print(
            "No se ha cargado un modelo CLIP. Ejecute 'cargar_modelo_clip()' primero."
        )
        return

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

        # Copy and rename images to the output folder
        for id_result, distance in resultados:
            if id_result in metadatos:
                nombre_archivo = metadatos[id_result]
                nuevo_nombre = f"{distance:.4f}-{nombre_archivo}"
                ruta_origen = os.path.join("images", nombre_archivo)
                ruta_destino = os.path.join(OUTPUT_DIR, nuevo_nombre)

                if os.path.exists(ruta_origen):
                    shutil.copy(ruta_origen, ruta_destino)
                else:
                    print(f"Advertencia: La imagen {ruta_origen} no existe.")

        # Return results with similarity and filenames
        return [(metadatos[r[0]], r[1]) for r in resultados]
    return []
