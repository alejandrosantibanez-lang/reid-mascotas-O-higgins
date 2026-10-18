"""
05_train_petface.py
Entrenamiento de Re-ID de mascotas: ViT + ArcFace + Triplet Loss.
Compatible con datasets pequeños y grandes (PetFace).

Uso:
  python 05_train_petface.py --data-root ../data_ohiggins --epochs 10 --num-workers 0
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import timm
from PIL import Image
from torchvision import transforms

# ARGUMENTOS

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='../data_ohiggins')
    p.add_argument('--output-dir', type=str, default='../models')
    p.add_argument('--model-name', type=str, default='vit_base_patch16_224')
    p.add_argument('--embed-dim', type=int, default=512)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--p-identities', type=int, default=16)
    p.add_argument('--k-images', type=int, default=4)
    p.add_argument('--lr-backbone', type=float, default=1e-5)
    p.add_argument('--lr-head', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--arcface-s', type=float, default=30.0)
    p.add_argument('--arcface-m', type=float, default=0.3)
    p.add_argument('--triplet-margin', type=float, default=0.3)
    p.add_argument('--triplet-weight', type=float, default=0.5)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-photos', type=int, default=4)
    p.add_argument('--max-identities', type=int, default=0)
    return p.parse_args()

args = parse_args()

# Reproducibilidad
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ROOT = Path(args.data_root)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = args.p_identities * args.k_images

print(f'Dispositivo: {DEVICE}')
print(f'Modelo: {args.model_name}')
print(f'Batch: {args.p_identities} identidades x {args.k_images} fotos = {BATCH_SIZE}')


# ESCANEAR DATASET

def scan_dataset(data_root, min_photos, max_identities):
    samples = []
    identities = {}

    for species_dir in sorted(data_root.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name

        for identity_folder in sorted(species_dir.iterdir()):
            if not identity_folder.is_dir():
                continue

            images = sorted(
                p for p in identity_folder.iterdir()
                if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
            )

            if len(images) < min_photos:
                continue

            identity_name = identity_folder.name
            identities[identity_name] = images

            for img in images:
                samples.append((img, identity_name, species))

    if max_identities > 0 and len(identities) > max_identities:
        keep = set(list(identities.keys())[:max_identities])
        samples = [s for s in samples if s[1] in keep]
        identities = {k: v for k, v in identities.items() if k in keep}

    return samples, identities

print('Escaneando dataset...')
samples, identities = scan_dataset(DATA_ROOT, args.min_photos, args.max_identities)

if len(samples) == 0:
    raise RuntimeError(f'No se encontraron imágenes válidas en {DATA_ROOT}')

NUM_CLASSES = len(identities)
print(f'Imágenes válidas: {len(samples)}')
print(f'Identidades: {NUM_CLASSES}')
print(f'Usando ArcFace con s={args.arcface_s}, m={args.arcface_m}')


# PK SAMPLER

class PKSampler(Sampler):
    def __init__(self, samples, p_identities, k_images):
        self.p = p_identities
        self.k = k_images

        self.id_to_indices = {}
        for idx, (_, identity, _) in enumerate(samples):
            self.id_to_indices.setdefault(identity, []).append(idx)

        self.valid_ids = [
            i for i, idxs in self.id_to_indices.items()
            if len(idxs) >= self.k
        ]

        if len(self.valid_ids) < self.p:
            raise RuntimeError(
                f'Solo hay {len(self.valid_ids)} identidades con >= {self.k} fotos. '
                f'Se necesitan al menos {self.p}. Reduce --p-identities o --k-images.'
            )

        self.num_batches = len(self.valid_ids) // self.p

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_ids = random.sample(self.valid_ids, self.p)
            batch = []
            for identity in chosen_ids:
                idxs = self.id_to_indices[identity]
                batch.extend(random.sample(idxs, self.k))
            yield batch

    def __len__(self):
        return self.num_batches

# DATASET

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class PetFaceDataset(Dataset):
    def __init__(self, samples, identities, transform=None):
        self.samples = samples
        self.transform = transform
        self.label_to_index = {
            identity: i for i, identity in enumerate(sorted(identities.keys()))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, identity, species = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label_to_index[identity]
        return img, label, identity

dataset = PetFaceDataset(samples, identities, train_transform)

sampler = PKSampler(samples, args.p_identities, args.k_images)
loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    num_workers=args.num_workers,
    pin_memory=(DEVICE == 'cuda'),
)

# MODELO

class PetReIDModel(nn.Module):
    def __init__(self, model_name, embed_dim, num_classes, arcface_s, arcface_m):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features

        self.projection = nn.Linear(backbone_dim, embed_dim)
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.s = arcface_s
        self.m = arcface_m

    def forward(self, x, labels=None):
        features = self.backbone(x)
        embeddings = F.normalize(self.projection(features), dim=1)

        if labels is None:
            return embeddings

        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_cosine = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        output = cosine * (1 - one_hot) + target_cosine * one_hot
        output = output * self.s
        return output, embeddings

model = PetReIDModel(
    args.model_name, args.embed_dim, NUM_CLASSES,
    args.arcface_s, args.arcface_m
).to(DEVICE)


# OPTIMIZADOR Y PÉRDIDAS

optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': args.lr_backbone},
    {'params': model.projection.parameters(), 'lr': args.lr_head},
    {'params': [model.weight], 'lr': args.lr_head},
], weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs
)

criterion_arcface = nn.CrossEntropyLoss()
criterion_triplet = nn.TripletMarginLoss(margin=args.triplet_margin, p=2)

# TRIPLETS DESDE ESTRUCTURA PK

def build_triplets_from_pk(embeddings, labels, k_images):
    anchors, positives, negatives = [], [], []

    num_groups = len(labels) // k_images

    for g in range(num_groups):
        group_indices = list(range(g * k_images, (g + 1) * k_images))

        other_groups = [i for i in range(num_groups) if i != g]
        if not other_groups:
            continue
        neg_group = random.choice(other_groups)
        neg_idx = random.choice(
            range(neg_group * k_images, (neg_group + 1) * k_images)
        )

        for i in group_indices:
            pos_candidates = [j for j in group_indices if j != i]
            if not pos_candidates:
                continue
            pos_idx = random.choice(pos_candidates)

            anchors.append(embeddings[i])
            positives.append(embeddings[pos_idx])
            negatives.append(embeddings[neg_idx])

    if not anchors:
        return None
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

# ENTRENAMIENTO

model.train()

for epoch in range(1, args.epochs + 1):
    total_loss = 0.0
    total_arc = 0.0
    total_trip = 0.0
    correct = 0
    total = 0

    for imgs, labels, identities_batch in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, embeddings = model(imgs, labels=labels)

        loss_arc = criterion_arcface(logits, labels)

        triplets = build_triplets_from_pk(
            embeddings, labels, args.k_images
        )
        if triplets is not None:
            a, p_, n = triplets
            loss_trip = criterion_triplet(a, p_, n)
        else:
            loss_trip = torch.tensor(0.0, device=DEVICE)

        loss = loss_arc + args.triplet_weight * loss_trip

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_arc += loss_arc.item()
        total_trip += loss_trip.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    scheduler.step()

    print(f'Epoch {epoch:02d}/{args.epochs} | '
          f'Loss: {total_loss/len(loader):.4f} | '
          f'ArcFace: {total_arc/len(loader):.4f} | '
          f'Triplet: {total_trip/len(loader):.4f} | '
          f'Acc: {100.0*correct/total:.2f}% | '
          f'LR: {scheduler.get_last_lr()[0]:.2e}')

# GUARDAR MODELO

checkpoint = {
    'model_state_dict': model.state_dict(),
    'label_to_index': dataset.label_to_index,
    'embed_dim': args.embed_dim,
    'num_classes': NUM_CLASSES,
    'arcface_s': args.arcface_s,
    'arcface_m': args.arcface_m,
    'model_name': args.model_name,
    'args': vars(args),
}

out_path = OUTPUT_DIR / 'vit_arcface_petface.pth'
torch.save(checkpoint, out_path)

print('\n' + '=' * 60)
print('ENTRENAMIENTO COMPLETADO')
print('=' * 60)
print(f'Modelo guardado en: {out_path}')
print(f'Identidades entrenadas: {NUM_CLASSES}')
print(f'Imágenes: {len(samples)}')
print('=' * 60)