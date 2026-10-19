# Simula la búsqueda real: dada una imagen de una mascota "perdida" (query),
# la compara contra la galería de embeddings y devuelve las top-K coincidencias.
#
# Uso:
#   python 07_buscar_mascota.py --query ../data_ohiggins/dogs/perro_001/foto1.jpg

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    DATA_ROOT, BASE_DIR, MODEL_SAVE_PATH, EMBEDDINGS_PATH,
)
from utils.datos_utils import transform_evaluacion
from importlib import import_module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
modelo_modulo = import_module("03_construir_modelo")
construir_modelo = modelo_modulo.construir_modelo


def extraer_embedding_query(ruta_imagen, modelo, dispositivo):
    """Convierte una imagen en un embedding (vector de 512)."""
    imagen = Image.open(ruta_imagen).convert("RGB")
    imagen = transform_evaluacion(imagen)
    imagen = imagen.unsqueeze(0).to(dispositivo)  # (1, 3, 224, 224)

    modelo.eval()
    with torch.no_grad():
        embedding = modelo(imagen)  # (1, 512)
    return embedding.cpu()


def buscar_top_k(embedding_query, galeria, top_k=5):
    """
    Compara el embedding de la query contra toda la galería
    y devuelve las top_k coincidencias (ruta, identidad, similitud).
    """
    # Normalizar para similitud coseno
    embedding_query = F.normalize(embedding_query, p=2, dim=1)
    embeddings_galeria = F.normalize(galeria["embeddings"], p=2, dim=1)

    # Similitud coseno: (1, num_galeria)
    similitudes = (embedding_query @ embeddings_galeria.t()).squeeze(0)

    # Ordenar de mayor a menor y tomar top_k
    top_indices = torch.argsort(similitudes, descending=True)[:top_k]

    resultados = []
    for idx in top_indices:
        idx = idx.item()
        resultados.append({
            "ruta": galeria["rutas"][idx],
            "identidad": galeria["identidad_por_imagen"][idx],
            "similitud": similitudes[idx].item(),
        })
    return resultados


def main():
    parser = argparse.ArgumentParser(description="Buscar mascota por imagen")
    parser.add_argument("--query", type=str, required=True,
                        help="Ruta de la imagen de la mascota perdida")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Cuántas coincidencias mostrar")
    args = parser.parse_args()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}")
    print(f"Imagen query: {args.query}")

    # 1) Cargar galería de embeddings
    ruta_galeria = os.path.join(EMBEDDINGS_PATH, "galeria_embeddings.pt")
    galeria = torch.load(ruta_galeria, map_location=dispositivo, weights_only=False)
    print(f"Galería cargada: {galeria['embeddings'].shape[0]} imágenes")

    # 2) Cargar modelo entrenado
    num_clases = len(set(galeria["identidad_por_imagen"]))  # Número de identidades únicas
    modelo = construir_modelo(num_clases=num_clases)
    ruta_modelo = os.path.join(MODEL_SAVE_PATH, "modelo_reid.pth")
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
    modelo = modelo.to(dispositivo)

    # 3) Extraer embedding de la query
    embedding_query = extraer_embedding_query(args.query, modelo, dispositivo)

    # 4) Buscar top-K
    resultados = buscar_top_k(embedding_query, galeria, top_k=args.top_k)

    # 5) Mostrar resultados
    print(f"\nTop {args.top_k} coincidencias:")
    print("-" * 60)
    for i, res in enumerate(resultados, 1):
        print(f"{i}. Similitud: {res['similitud']:.4f}")
        print(f"   Identidad: {res['identidad']}")
        print(f"   Ruta: {res['ruta']}")
        print("-" * 60)


if __name__ == "__main__":
    main()