# 05_evaluar_modelo.py
# Evalúa el modelo entrenado sobre el conjunto de test.
# Para cada identidad de test:
#   - 1 imagen → query (la "mascota perdida")
#   - resto    → gallery (candidatas)
# Calcula Rank-1, Rank-5 y mAP.

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DATA_ROOT, BASE_DIR, BATCH_SIZE, NUM_WORKERS,
    MODEL_SAVE_PATH, EMBEDDING_DIM,
)
from utils.datos_utils import (
    DatasetReID, cargar_split, transform_evaluacion,
)
from importlib import import_module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
modelo_modulo = import_module("03_construir_modelo")
construir_modelo = modelo_modulo.construir_modelo


# ─── Preparar query y gallery desde el test ────────────────────────────────
def separar_query_gallery(dataset_test):
    """
    Para cada identidad de test:
      - la primera imagen → query
      - las demás        → gallery
    Devuelve dos listas de (ruta, etiqueta).
    """
    por_identidad = {}
    for ruta, etiqueta in dataset_test.muestras:
        por_identidad.setdefault(etiqueta, []).append(ruta)

    queries, gallery = [], []
    for etiqueta, rutas in sorted(por_identidad.items()):
        queries.append((rutas[0], etiqueta))       # 1ª imagen = query
        for ruta in rutas[1:]:                     # resto = gallery
            gallery.append((ruta, etiqueta))

    return queries, gallery


# ─── Extraer embeddings para una lista de (ruta, etiqueta) ─────────────────
def extraer_embeddings(modelo, muestras, dispositivo):
    """
    Pasa cada imagen por el modelo y devuelve (embeddings, etiquetas).
    """
    modelo.eval()  # Modo evaluación (desactiva dropout, etc.)
    embeddings_todos, etiquetas_todas = [], []

    with torch.no_grad():  # No necesitamos gradientes aquí
        for i in range(0, len(muestras), BATCH_SIZE):
            lote = muestras[i:i + BATCH_SIZE]
            imagenes = []
            for ruta, _ in lote:
                from PIL import Image
                from utils.datos_utils import transform_evaluacion
                img = Image.open(ruta).convert("RGB")
                img = transform_evaluacion(img)
                imagenes.append(img)

            imagenes = torch.stack(imagenes).to(dispositivo)
            embs = modelo(imagenes)  # Sin etiquetas → devuelve embeddings
            embeddings_todos.append(embs.cpu())
            etiquetas_todas.extend([etq for _, etq in lote])

    return torch.cat(embeddings_todos), torch.tensor(etiquetas_todas)


# ─── Calcular métricas Rank-K y mAP ────────────────────────────────────────
def calcular_metricas(emb_query, etq_query, emb_gallery, etq_gallery):
    """
    Para cada query, ordena la galería por similitud coseno y calcula:
      - Rank-1: ¿la 1ª coincidencia es de la misma identidad?
      - Rank-5: ¿está dentro de las 5 primeras?
      - mAP: promedio de precisión en las posiciones correctas
    """
    # Normalizar para similitud coseno
    emb_query = F.normalize(emb_query, p=2, dim=1)
    emb_gallery = F.normalize(emb_gallery, p=2, dim=1)

    # Similitud: (num_queries, num_gallery)
    similitudes = emb_query @ emb_gallery.t()

    rank1_aciertos, rank5_aciertos, ap_total = 0, 0, 0.0

    for i in range(similitudes.size(0)):
        # Ordenar galería por similitud descendente
        orden = torch.argsort(similitudes[i], descending=True)
        etiquetas_ordenadas = etq_gallery[orden]

        # Posiciones donde la identidad coincide con la query
        aciertos = (etiquetas_ordenadas == etq_query[i]).nonzero(as_tuple=True)[0]

        if len(aciertos) > 0:
            # Rank-1: la primera coincidencia está en la posición 0
            if aciertos[0].item() == 0:
                rank1_aciertos += 1
            # Rank-5: alguna coincidencia dentro de las 5 primeras
            if aciertos[0].item() < 5:
                rank5_aciertos += 1

            # mAP: precisión promedio en las posiciones correctas
            precision_acumulada, relevantes = 0.0, 0
            for pos in aciertos:
                relevantes += 1
                precision_acumulada += relevantes / (pos.item() + 1)
            ap_total += precision_acumulada / len(aciertos)

    num_queries = similitudes.size(0)
    rank1 = rank1_aciertos / num_queries
    rank5 = rank5_aciertos / num_queries
    map_score = ap_total / num_queries

    return rank1, rank5, map_score


# ─── Función principal ──────────────────────────────────────────────────────
def evaluar():
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {dispositivo}")

    # 1) Cargar split
    ruta_split = os.path.join(BASE_DIR, "data", "split.json")
    train_ids, test_ids = cargar_split(ruta_split)

    # 2) Crear dataset de test (sin aumentación, transform de evaluación)
    dataset_test = DatasetReID(
        data_root=DATA_ROOT,
        identidades=test_ids,
        transform=transform_evaluacion,
    )
    print(f"Imágenes de test: {len(dataset_test)}")

    # 3) Separar en query y gallery
    queries, gallery = separar_query_gallery(dataset_test)
    print(f"Queries:  {len(queries)}")
    print(f"Gallery:  {len(gallery)}")

    # 4) Cargar modelo entrenado
    num_clases = len(train_ids)
    modelo = construir_modelo(num_clases=num_clases)
    ruta_modelo = os.path.join(MODEL_SAVE_PATH, "modelo_reid.pth")
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=dispositivo))
    modelo = modelo.to(dispositivo)
    print(f"Modelo cargado desde: {ruta_modelo}")

    # 5) Extraer embeddings
    print("\nExtrayendo embeddings de queries...")
    emb_q, etq_q = extraer_embeddings(modelo, queries, dispositivo)
    print("Extrayendo embeddings de gallery...")
    emb_g, etq_g = extraer_embeddings(modelo, gallery, dispositivo)

    # 6) Calcular métricas
    print("\nCalculando métricas...")
    rank1, rank5, map_score = calcular_metricas(emb_q, etq_q, emb_g, etq_g)

    # 7) Mostrar resultados
    print("\n" + "=" * 40)
    print("RESULTADOS DE EVALUACIÓN")
    print("=" * 40)
    print(f"Rank-1: {rank1:.2%}")
    print(f"Rank-5: {rank5:.2%}")
    print(f"mAP:    {map_score:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    evaluar()