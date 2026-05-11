import torch
import torch.nn as nn
from torch.optim import Adam
import mlflow
import mlflow.pytorch
import sys
import os

# Добавляем путь к папке с кастомными модулями
sys.path.append('./training')
from dataset import get_dataloaders
from model import UNet

# --- Глобальный конфиг ---
CONFIG = {
    "epochs": 100,            # Увеличиваем, так как есть Early Stopping
    "batch_size": 8,          # Для GPU 8-16 обычно оптимально
    "learning_rate": 1e-4,
    "data_dir": "./data/raw",
    "model_path": "./models/best_brain_model.pth",
    "early_stop_patience": 10 # Остановка, если 10 эпох нет прогресса
}

# --- Лосс-функции ---

class SoftDiceLoss(nn.Module):
    """
    Дифференцируемый Dice Loss для обучения. 
    Помогает модели фокусироваться на пересечении масок.
    """
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = preds.contiguous()
        targets = targets.contiguous()
        intersection = (preds * targets).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        return 1 - dice.mean()

def dice_score(pred, target, threshold=0.5):
    """Метрика для оценки качества на валидации."""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection) / (pred.sum() + target.sum() + 1e-8)

# --- Основной цикл обучения ---

def train():
    # 1. Проверка GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Запуск обучения. Используемое устройство: {device}")
    if device.type == 'cuda':
        print(f"Видеокарта: {torch.cuda.get_device_name(0)}")

    # 2. Подготовка данных
    train_loader, val_loader = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"]
    )

    # 3. Инициализация модели и оптимизатора
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    bce_criterion = nn.BCELoss()
    dice_criterion = SoftDiceLoss()
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    # Планировщик: уменьшает LR в 2 раза, если Dice не растет 4 эпохи
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4
    )

    # 4. MLflow Setup
    mlflow.set_tracking_uri("./mlflow/store")
    mlflow.set_experiment("brain-segmentation-v2")

    best_val_dice = 0.0
    patience_counter = 0

    with mlflow.start_run():
        mlflow.log_params(CONFIG)

        for epoch in range(CONFIG["epochs"]):
            # --- PHASE: TRAIN ---
            model.train()
            epoch_train_loss = 0.0
            
            for images, masks in train_loader:
                # Перенос батча на GPU
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                
                # Комбинированный лосс: BCE + Dice
                loss = 0.5 * bce_criterion(outputs, masks) + 0.5 * dice_criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)

            # --- PHASE: VALIDATION ---
            model.eval()
            epoch_val_dice = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    preds = model(images)
                    epoch_val_dice += dice_score(preds, masks).item()

            avg_val_dice = epoch_val_dice / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']

            # Шаг планировщика по метрике Dice
            scheduler.step(avg_val_dice)

            # Логирование в MLflow
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_dice": avg_val_dice,
                "lr": current_lr
            }, step=epoch)

            print(f"[{epoch+1}/{CONFIG['epochs']}] Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {current_lr}")

            # --- CHECKPOINT & EARLY STOPPING ---
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                patience_counter = 0
                os.makedirs("./models", exist_ok=True)
                torch.save(model.state_dict(), CONFIG["model_path"])
                print(f"⭐ New Best Dice: {best_val_dice:.4f}. Model saved!")
            else:
                patience_counter += 1

            if patience_counter >= CONFIG["early_stop_patience"]:
                print(f"🛑 Early stopping. No improvement for {CONFIG['early_stop_patience']} epochs.")
                break

        # Сохранение финальных артефактов
        mlflow.pytorch.log_model(model, "final_model")
        print(f"Обучение завершено. Лучший результат Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    train()