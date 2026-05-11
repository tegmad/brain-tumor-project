# training/dataset.py
# Датасет в формате COCO.
# Маски не отдельные файлы — они описаны как полигоны в JSON.
# Мы читаем JSON, находим полигоны для каждого снимка
# и рисуем маску программно.

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Читаем COCO аннотации
        json_path = os.path.join(data_dir, "_annotations.coco.json")
        with open(json_path, "r") as f:
            self.coco = json.load(f)
        # coco — словарь с ключами:
        # images      — список всех снимков (id, filename, width, height)
        # annotations — список всех аннотаций (полигоны опухолей)
        # categories  — классы (у нас один: Tumor)

        # Строим словарь: image_id → список аннотаций
        self.annotations = {}
        for ann in self.coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        # Теперь self.annotations[0] — все полигоны для снимка с id=0

        self.images = self.coco["images"]
        print(f"Датасет загружен: {len(self.images)} снимков")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        width = img_info["width"]
        height = img_info["height"]

        # Загружаем снимок
        img_path = os.path.join(self.data_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Создаём пустую маску — чёрное изображение
        mask = Image.new("L", (width, height), 0)
        # "L" — grayscale
        # 0   — чёрный фон (нет опухоли)

        # Рисуем полигоны опухолей на маске
        draw = ImageDraw.Draw(mask)
        if img_id in self.annotations:
            for ann in self.annotations[img_id]:
                if "segmentation" in ann and ann["segmentation"]:
                    for seg in ann["segmentation"]:
                        # seg — список координат [x1,y1,x2,y2,...]
                        # Преобразуем в список пар [(x1,y1),(x2,y2),...]
                        points = list(zip(seg[0::2], seg[1::2]))
                        if len(points) >= 3:
                            draw.polygon(points, fill=255)
                            # fill=255 — белый = опухоль

        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
            # Маску ресайзим отдельно без нормализации
            mask = T.Resize((256, 256))(mask)
            mask = T.ToTensor()(mask)
            # ToTensor переводит [0,255] → [0.0,1.0]

        return image, mask


def get_dataloaders(data_dir="/data/raw", batch_size=8):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = BrainTumorDataset(
        data_dir=os.path.join(data_dir, "train"),
        transform=transform
    )
    val_dataset = BrainTumorDataset(
        data_dir=os.path.join(data_dir, "valid"),
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
        # num_workers=2 — два параллельных процесса загрузки данных
        # ускоряет обучение на CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader
