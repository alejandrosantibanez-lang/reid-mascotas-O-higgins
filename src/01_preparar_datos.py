# Revisa todas las imágenes del dataset (dogs/ y cats/), detecta corruptas y genera un reporte. Estructura esperada:
#   data_ohiggins/dogs/identidad/foto.jpg
#   data_ohiggins/cats/identidad/foto.jpg

import os
import json

from PIL import Image, UnidentifiedImageError

from config import DATA_ROOT, BASE_DIR

EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".webp"}
ESPECIES = ["dogs", "cats"]


def es_imagen_valida(ruta_imagen):
    """Revisa si una imagen se puede abrir y no está corrupta."""
    try:
        with Image.open(ruta_imagen) as img:
            img.verify()
        return True, None
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        return False, str(e)


def revisar_dataset(data_root):
    """
    Recorre dogs/ y cats/, y dentro de cada una las identidades.
    Devuelve un resumen con válidas, corruptas y extensiones raras.
    """
    resumen = {
        "total": 0, "validas": 0,
        "corruptas": [], "extension_rara": [],
        "por_especie": {"dogs": 0, "cats": 0},
    }

    for especie in ESPECIES:
        ruta_especie = os.path.join(data_root, especie)
        if not os.path.isdir(ruta_especie):
            print(f"⚠️  No existe la carpeta: {ruta_especie}")
            continue

        for identidad in sorted(os.listdir(ruta_especie)):
            carpeta_identidad = os.path.join(ruta_especie, identidad)
            if not os.path.isdir(carpeta_identidad):
                continue

            for archivo in sorted(os.listdir(carpeta_identidad)):
                ruta = os.path.join(carpeta_identidad, archivo)
                if not os.path.isfile(ruta):
                    continue

                resumen["total"] += 1
                resumen["por_especie"][especie] += 1
                extension = os.path.splitext(archivo)[1].lower()

                if extension not in EXTENSIONES_VALIDAS:
                    resumen["extension_rara"].append(ruta)
                    continue

                valida, error = es_imagen_valida(ruta)
                if valida:
                    resumen["validas"] += 1
                else:
                    resumen["corruptas"].append({"ruta": ruta, "error": error})

    return resumen


def guardar_reporte(resumen, ruta_salida):
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print(f"Revisando imágenes en: {DATA_ROOT}")
    resumen = revisar_dataset(DATA_ROOT)

    print(f"\nTotal imágenes:   {resumen['total']}")
    print(f"  Perros (dogs):  {resumen['por_especie']['dogs']}")
    print(f"  Gatos (cats):   {resumen['por_especie']['cats']}")
    print(f"Válidas:          {resumen['validas']}")
    print(f"Corruptas:        {len(resumen['corruptas'])}")
    print(f"Extensión rara:   {len(resumen['extension_rara'])}")

    if resumen["corruptas"]:
        print("\n⚠️  Imágenes corruptas:")
        for item in resumen["corruptas"][:10]:
            print(f"  - {item['ruta']} ({item['error']})")

    ruta_reporte = os.path.join(BASE_DIR, "data", "reporte_preparacion.json")
    guardar_reporte(resumen, ruta_reporte)
    print(f"\nReporte guardado en: {ruta_reporte}")