import torch
import clip
from PIL import Image
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_models = clip.available_models()

TEST_IMAGE = "./image_search/test/peine.png"

for model_name in clip_models:
    model, preprocess = clip.load(model_name, device=device)

    time_start = time.time()
    imagen = Image.open(TEST_IMAGE)

    if imagen.mode in ("RGBA", "LA") or (
        imagen.mode == "P" and "transparency" in imagen.info
    ):
        imagen = imagen.convert("RGBA")

    imagen = preprocess(imagen).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(imagen).cpu().numpy().flatten().tolist()
    time_end = time.time()

    # Mostrar información del embedding
    print(f"Modelo: {model_name}")
    print(f"Embedding: {embedding[:5]}...")
    print(f"Dimensiones: {len(embedding)}")
    print(f"Tiempo de ejecución: {time_end - time_start:.6f} segundos")
    print()
