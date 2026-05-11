# 🧠 Brain Tumor Segmentation (Full-Cycle MLOps)

Проект по автоматизированной сегментации опухолей головного мозга. Это не просто скрипт обучения, а полноценная инфраструктура с оркестрацией, трекингом экспериментов и сервисом предсказаний.

## 🏗 Архитектура и Поток данных (Data Flow)

Ниже описана логика взаимодействия компонентов:

1.  **Airflow (Orchestrator):** 
    *   **Откуда:** Забирает сырые снимки МРТ (из внешнего S3 или Kaggle).
    *   **Куда:** Складывает в локальное хранилище `./data/raw/` внутри контейнера.
    *   **Зачем:** Автоматизирует регулярное обновление данных и запуск обучения.
2.  **Training (PyTorch + MLflow):**
    *   **Input:** Берет данные из `./data/raw/`.
    *   **Processing:** Обучает нейросеть `U-Net` с оптимизатором `Adam`.
    *   **Tracking:** Все метрики (Loss, Dice) и параметры (`LR=2.5e-05`) летят в **MLflow Server**.
    *   **GPU:** NVIDIA GeForce RTX 3060 (12 ГБ), на  .
    *   **Output:** Лучшие веса сохраняются в `./models/model.pth`.
3.  **Inference (FastAPI):**
    *   **Input:** Принимает JPG/PNG снимок от пользователя.
    *   **Logic:** Подгружает веса из `./models/model.pth`, делает сегментацию маски.
    *   **Database:** Метаданные запроса и путь к маске сохраняются в **PostgreSQL**.

---

## 📊 Технические метрики и Стек

### Результаты обучения:
*   **Best Val Dice Score:** `0.7963` (Высокая точность наложения маски).
*   **Final Train Loss:** `0.0356`.
*   **PR-AUC Score:** `0.8042`
*   **Early Stopping:** Остановка на 35-й эпохе (спасло от переобучения).

### Стек технологий:
*   **Core:** `Python 3.10`, `PyTorch` (Deep Learning).
*   **Optimizer:** `Adam` (адаптивный шаг градиента).
*   **Loss Function:** `Dice Loss` + `BCEWithLogits` (стабильность на несбалансированных данных).
*   **MLOps:** `MLflow`, `Apache Airflow`.
*   **DevOps:** `Docker`, `Docker Compose`.
*   **Storage:** `PostgreSQL`.

---

## 🛠 Инструкция по развертыванию

Проект полностью контейнеризирован. Вам не нужно устанавливать библиотеки локально.

1.  **Клонировать репозиторий:**
    ```bash
    git clone [https://github.com/tegmad/brain-tumor-project.git](https://github.com/tegmad/brain-tumor-project.git)
    cd brain-tumor-project

2. **Запустить всю инфраструктуру::**
    docker-compose up --build


*После этого поднимутся: API (порт 8000), MLflow (порт 5000), Airflow (порт 8080) и база данных.*

---

## 🖼 Схема архитектуры

```text
       [ Внешний Источник ]
               |
        (Airflow DAG)
               v
        [ ./data/raw/ ] <----------- [ PostgreSQL ]
               |                          ^
        (train.py / PyTorch)              | (Log Metadata)
               |                          |
        +------+------+            [ FastAPI Service ]
        |             |                   ^
    [ MLflow ]   [ ./models/model.pth ] --+ (Load Weights)
