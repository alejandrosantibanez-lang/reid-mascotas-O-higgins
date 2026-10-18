import random
import csv
from pathlib import Path


# CONFIGURACIÓN

DATA_DIR = Path(__file__).resolve().parent.parent / 'data_ohiggins'
SPLITS_DIR = Path(__file__).resolve().parent.parent / 'splits'
SPLITS_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)

# FUNCIÓN ADAPTATIVA DE SPLIT


def assign_split(num_photos):
    """
    Reparte fotos según cuántas tenga la identidad:
      - 3 fotos: 1 train, 1 gallery, 1 probe
      - 4 fotos: 2 train, 1 gallery, 1 probe
      - 5 fotos: 3 train, 1 gallery, 1 probe
      - 6+ fotos: 4 train (o más), 1 gallery, 1 probe
    Devuelve (n_train, n_gallery, n_probe) o None si no alcanza.
    """
    if num_photos < 3:
        return None  # no alcanza ni para 1/1/1
    elif num_photos == 3:
        return (1, 1, 1)
    elif num_photos == 4:
        return (2, 1, 1)
    elif num_photos == 5:
        return (3, 1, 1)
    else:  # 6 o más
        # Train se lleva todo menos 2 (1 gallery + 1 probe)
        n_train = num_photos - 2
        return (n_train, 1, 1)


# RECORRER IDENTIDADES Y DIVIDIR

train_rows = []
gallery_rows = []
probe_rows = []
skipped = []

for species in ['dogs', 'cats']:
    species_dir = DATA_DIR / species
    if not species_dir.exists():
        continue

    for identity_folder in sorted(species_dir.iterdir()):
        if not identity_folder.is_dir():
            continue

        identity_name = identity_folder.name

        images = sorted(
            p for p in identity_folder.iterdir()
            if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        )

        split_counts = assign_split(len(images))

        if split_counts is None:
            skipped.append((identity_name, len(images)))
            continue

        n_train, n_gallery, n_probe = split_counts

        # Mezclar con semilla fija
        random.shuffle(images)

        train_imgs = images[:n_train]
        gallery_imgs = images[n_train:n_train + n_gallery]
        probe_imgs = images[n_train + n_gallery:
                            n_train + n_gallery + n_probe]

        for img in train_imgs:
            train_rows.append([str(img), identity_name, species, 'train'])
        for img in gallery_imgs:
            gallery_rows.append([str(img), identity_name, species, 'gallery'])
        for img in probe_imgs:
            probe_rows.append([str(img), identity_name, species, 'probe'])


# GUARDAR CSVs

def save_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'identity', 'species', 'split'])
        writer.writerows(rows)

save_csv(SPLITS_DIR / 'train.csv', train_rows)
save_csv(SPLITS_DIR / 'gallery.csv', gallery_rows)
save_csv(SPLITS_DIR / 'probe.csv', probe_rows)

# RESUMEN

print('=' * 60)
print('SPLIT ADAPTATIVO GENERADO')
print('=' * 60)
print(f'Train:   {len(train_rows)} imágenes')
print(f'Gallery: {len(gallery_rows)} imágenes')
print(f'Probe:   {len(probe_rows)} imágenes')
print(f'Identidades incluidas: {len(set(r[1] for r in train_rows))}')

if skipped:
    print(f'\n⚠️  Identidades omitidas (menos de 3 fotos):')
    for name, n in skipped:
        print(f'   - {name}: {n} fotos')

print(f'\nArchivos guardados en: {SPLITS_DIR}')
print('  - train.csv')
print('  - gallery.csv')
print('  - probe.csv')
print('=' * 60)