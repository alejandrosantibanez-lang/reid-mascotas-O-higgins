# 06_extraer_embeddings.py
# Extrae embeddings de todas las imágenes de train y los guarda en un .pt
# Estos vectores serán la "galería" para búsquedas futuras.

import os
import torch
from torch.utils.data import DataLoader

from config import (
    DATA_ROOT, BASE_DIR, BATCH_SIZE, NUM_WORKERS,
    MODEL_SAVE_PATH, EMBEDDINGS_PATH,
)
from utils.datos_utils import (
    DatasetReID, cargar_split, transform_evaluacion,
)
from importlib import import_module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
modelo_modulo = import_module("03_construir_modelo")
construir_modelo = modelo_modulo.construir_modelo


def extraer_y_guardar():
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}")

    # 1) Cargar split (usaremos train para construir la galería base)
    ruta_split = os.path.join(BASE_DIR, "data", "split.json")
    train_ids, test_ids = cargar_split(ruta_split)

    # 2) Dataset de train (sin aumentación, transform de evaluación)
    dataset_train = DatasetReID(
        data_root=DATA_ROOT,
        identidades=train_ids,
        transform=transform_evaluacion,
    )
    print(f"Imágenes de train para galería: {len(dataset_train)}")

    # 3) DataLoader (sin shuffle, para mantener orden)
    dataloader = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # 4) Cargar modelo entrenado
    num_clases = len(train_ids)
    modelo = construir_modelo(num_clases=num_clases)
    ruta_modelo = os.path.join(MODEL_SAVE_PATH, "modelo_reid.pth")
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
    modelo = modelo.to(dispositivo)
    modelo.eval()
    print(f"Modelo cargado desde: {ruta_modelo}")

    # 5) Extraer embeddings de todo el train
    print("\nExtrayendo embeddings...")
    todos_embeddings, todas_etiquetas, todas_rutas = [], [], []

    with torch.no_grad():
        for imagenes, etiquetas in dataloader:
            imagenes = imagenes.to(dispositivo)
            embs = modelo(imagenes)  # Sin etiquetas → devuelve embeddings
            todos_embeddings.append(embs.cpu())
            todas_etiquetas.extend(etiquetas.tolist())

    # Recuperar las rutas en el mismo orden que el DataLoader
    for ruta, _ in dataset_train.muestras:
        todas_rutas.append(ruta)

    embeddings = torch.cat(todos_embeddings)
    etiquetas = torch.tensor(todas_etiquetas)

    # 6) Guardar todo en un solo archivo .pt
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
    ruta_salida = os.path.join(EMBEDDINGS_PATH, "galeria_embeddings.pt")

    # Crear mapa índice → nombre de identidad (mismo orden que DatasetReID)
    indice_a_identidad = {v: k for k, v in dataset_train.etiqueta_a_indice.items()}

    identidad_por_imagen = [indice_a_identidad[e] for e in etiquetas.tolist()]

    torch.save({
        "embeddings": embeddings,       # (num_imagenes, 512)
        "etiquetas": etiquetas,          # (num_imagenes,)
        "rutas": todas_rutas,            # lista de rutas
        "identidad_por_imagen": identidad_por_imagen,        # lista de identidades
    }, ruta_salida)

    print(f"\nEmbeddings guardados en: {ruta_salida}")
    print(f"Forma de embeddings: {embeddings.shape}")
    print(f"Total imágenes en galería: {len(todas_rutas)}")


if __name__ == "__main__":
    extraer_y_guardar()