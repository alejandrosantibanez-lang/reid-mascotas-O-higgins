# Divide las identidades (carpetas de perros/gatos) en train y test.
# Garantiza que todas las fotos de un mismo animal queden en el mismo conjunto.

import os
import random
import json

from config import DATA_ROOT, TRAIN_RATIO, BASE_DIR


def listar_identidades(data_root):
    """
    Recorre las subcarpetas de especies (dogs, cats) y devuelve la lista
    de identidades reales con su ruta relativa (ej: "dogs/perro_001").
    """
    identidades = []
    especies = ["dogs", "cats"]
    
    for especie in especies:
        ruta_especie = os.path.join(data_root, especie)
        
        # Verificamos si la carpeta de la especie existe antes de listar
        if os.path.exists(ruta_especie):
            for nombre in os.listdir(ruta_especie):
                # Validamos que sea una carpeta de animal (identidad)
                if os.path.isdir(os.path.join(ruta_especie, nombre)):
                    # Guardamos el formato "especie/nombre_animal"
                    identidades.append(f"{especie}/{nombre}")
                    
    return sorted(identidades)


def dividir_por_identidad(identidades, train_ratio=0.8, semilla=42):
    """
    Divide las identidades en train y test, estratificando por especie
    (dogs/cats) para que ambos conjuntos mantengan la proporción original.
    """
    random.seed(semilla)

    train, test = [], []
    especies = sorted({identidad.split("/")[0] for identidad in identidades})

    for especie in especies:
        del_especie = [i for i in identidades if i.startswith(especie + "/")]
        random.shuffle(del_especie)

        corte = int(train_ratio * len(del_especie))
        train.extend(del_especie[:corte])
        test.extend(del_especie[corte:])

    return train, test


def guardar_split(train, test, ruta_salida):
    """Guarda la división en un JSON para que otros scripts la reutilicen."""
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump({"train": train, "test": test}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    identidades = listar_identidades(DATA_ROOT)
    train, test = dividir_por_identidad(identidades, TRAIN_RATIO)

    print(f"Total identidades encontradas: {len(identidades)}")
    print(f"Train: {len(train)} animales")
    print(f"Test:  {len(test)} animales")

    ruta_split = os.path.join(BASE_DIR, "data", "split.json")
    guardar_split(train, test, ruta_split)
    print(f"División guardada con éxito en: {ruta_split}")
