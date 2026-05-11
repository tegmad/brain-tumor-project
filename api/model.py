# training/model.py
# Здесь описываем архитектуру нейросети.
# Мы используем U-Net — классическую архитектуру
# для задачи сегментации медицинских изображений.
# Называется U-Net потому что на схеме похожа на букву U.

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    # Базовый блок U-Net — два свёрточных слоя подряд.
    # Используется много раз и в encoder и в decoder.

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            # Sequential — выполняет слои последовательно

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Conv2d — свёрточный слой.
            # kernel_size=3 — смотрим на окно 3x3 пикселей
            # padding=1 — добавляем рамку чтобы размер не уменьшался

            nn.BatchNorm2d(out_channels),
            # BatchNorm — нормализует выходы слоя.
            # Стабилизирует обучение, позволяет использовать больший lr

            nn.ReLU(inplace=True),
            # ReLU — функция активации: f(x) = max(0, x)
            # inplace=True — экономим память, меняем tensor на месте

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        # in_channels=3  — RGB изображение (3 канала)
        # out_channels=1 — маска сегментации (1 канал, бинарная)
        super().__init__()

        # ── ENCODER (левая часть U) ──────────────────
        # Encoder сжимает изображение и извлекает признаки.
        # С каждым шагом: размер уменьшается, каналов больше.

        self.enc1 = DoubleConv(3, 64)
        # 3 канала → 64 карты признаков
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)
        # MaxPool2d(2) — уменьшает размер в 2 раза
        # берёт максимальный пиксель из окна 2x2
        # 256x256 → 128x128 → 64x64 → 32x32

        # ── BOTTLENECK (дно U) ───────────────────────
        # Самое узкое место — здесь самое абстрактное
        # представление изображения
        self.bottleneck = DoubleConv(512, 1024)

        # ── DECODER (правая часть U) ─────────────────
        # Decoder восстанавливает размер изображения.
        # На каждом шаге: размер увеличивается вдвое.
        # Skip connections — добавляем признаки из encoder.

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # ConvTranspose2d — обратная свёртка, увеличивает размер в 2 раза
        self.dec4 = DoubleConv(1024, 512)
        # 1024 = 512 (from up4) + 512 (skip connection from enc4)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # ── ВЫХОДНОЙ СЛОЙ ────────────────────────────
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        # kernel_size=1 — просто меняем количество каналов
        # 64 карты признаков → 1 маска сегментации

    def forward(self, x):
        # forward — описывает как данные проходят через сеть.
        # PyTorch вызывает его при model(image).

        # Encoder — сжимаем с сохранением промежуточных результатов
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder — расширяем и добавляем skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        # torch.cat — склеиваем тензоры по оси каналов (dim=1)
        # self.up4(b) — увеличили размер
        # e4 — признаки из encoder на том же уровне
        # это и есть skip connection — "U" в U-Net

        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))
        # sigmoid — сжимает выход в диапазон [0, 1]
        # 0 = точно фон, 1 = точно опухоль