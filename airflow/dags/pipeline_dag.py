# airflow/dags/pipeline_dag.py
# DAG (Directed Acyclic Graph) — это граф задач.
# Airflow читает этот файл и знает:
# - какие задачи выполнять
# - в каком порядке
# - по какому расписанию
# Думай об этом как о сценарии для оркестра —
# Airflow дирижёр, а задачи — музыканты.

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────
# Настройки DAG по умолчанию
# ─────────────────────────────────────────
default_args = {
    "owner": "brain-tumor-project",
    # owner — кто владелец этого DAG, просто метка

    "retries": 2,
    # retries — сколько раз повторить задачу если она упала.
    # Например сеть моргнула при скачивании датасета —
    # Airflow сам попробует ещё раз.

    "retry_delay": timedelta(minutes=5),
    # retry_delay — подождать 5 минут перед повторной попыткой.

    "email_on_failure": False,
    # email_on_failure — не слать email при ошибке.
    # Можно включить и настроить SMTP в продакшне.
}

# ─────────────────────────────────────────
# Задачи — функции которые будет вызывать Airflow
# ─────────────────────────────────────────

def download_dataset():
    # Задача 1: скачиваем датасет с Kaggle.
    # Запускается только если данных ещё нет.
    import kaggle

    data_path = "/data/raw"

    if os.path.exists(f"{data_path}/images") and \
       len(os.listdir(f"{data_path}/images")) > 0:
        # Проверяем есть ли уже данные.
        # Если есть — пропускаем скачивание.
        # Это называется idempotency — задача безопасна
        # для повторного запуска, не сломает уже скачанное.
        print("Данные уже есть, пропускаем скачивание")
        return

    kaggle.api.authenticate()
    # authenticate() читает ~/.kaggle/kaggle.json
    # и авторизуется через API токен

    kaggle.api.dataset_download_files(
        "pkdarabi/brain-tumor-image-dataset-semantic-segmentation",
        path=data_path,
        unzip=True
        # unzip=True — сразу распаковывает архив
    )
    print(f"Датасет скачан в {data_path}")


def preprocess_data():
    # Задача 2: предобработка данных.
    # Проверяем что все изображения одного размера,
    # убираем битые файлы, создаём train/val разбивку.
    from PIL import Image
    import shutil

    raw_images = "/data/raw/images"
    raw_masks = "/data/raw/masks"
    processed = "/data/processed"

    os.makedirs(f"{processed}/train/images", exist_ok=True)
    os.makedirs(f"{processed}/train/masks", exist_ok=True)
    os.makedirs(f"{processed}/val/images", exist_ok=True)
    os.makedirs(f"{processed}/val/masks", exist_ok=True)
    # exist_ok=True — не ругаться если папки уже есть

    filenames = sorted(os.listdir(raw_images))
    # берём все файлы и сортируем для стабильности

    valid_files = []
    for fname in filenames:
        try:
            img = Image.open(f"{raw_images}/{fname}")
            img.verify()
            # verify() — проверяем что файл не битый.
            # Если битый — бросает исключение, мы пропускаем.
            valid_files.append(fname)
        except Exception as e:
            print(f"Битый файл пропущен: {fname} — {e}")

    # Делим на train (80%) и val (20%)
    split = int(0.8 * len(valid_files))
    train_files = valid_files[:split]
    val_files = valid_files[split:]

    for fname in train_files:
        shutil.copy(f"{raw_images}/{fname}", f"{processed}/train/images/{fname}")
        shutil.copy(f"{raw_masks}/{fname}", f"{processed}/train/masks/{fname}")

    for fname in val_files:
        shutil.copy(f"{raw_images}/{fname}", f"{processed}/val/images/{fname}")
        shutil.copy(f"{raw_masks}/{fname}", f"{processed}/val/masks/{fname}")

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")


def train_model():
    # Задача 3: запускаем обучение модели.
    # Просто вызываем нашу функцию train() из training/train.py.
    # Airflow запустит это в отдельном процессе.
    import sys
    sys.path.append("/training")
    # добавляем папку training в Python path
    # чтобы можно было импортировать наши модули

    from train import train
    train()
    # вызываем функцию которую написали раньше


def validate_model():
    # Задача 4: проверяем что модель достаточно хороша.
    # Если Dice Score ниже порога — не деплоим модель.
    # Это называется model validation gate.
    import mlflow

    mlflow.set_tracking_uri("http://mlflow:5000")

    client = mlflow.tracking.MlflowClient()
    # MlflowClient — API для работы с MLflow программно

    experiment = client.get_experiment_by_name("brain-tumor-segmentation")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    # берём последний запуск обучения

    if not runs:
        raise ValueError("Нет запусков в MLflow!")

    last_run = runs[0]
    dice = last_run.data.metrics.get("val_dice", 0)
    # достаём метрику val_dice из последнего запуска

    DICE_THRESHOLD = 0.7
    # минимальный приемлемый Dice Score.
    # Если ниже — модель недостаточно хорошая для продакшна.

    if dice < DICE_THRESHOLD:
        raise ValueError(
            f"Dice Score {dice:.4f} ниже порога {DICE_THRESHOLD}. "
            f"Модель не прошла валидацию!"
        )
        # raise ValueError — задача упадёт с ошибкой.
        # Airflow остановит пайплайн и не задеплоит плохую модель.

    print(f"Модель прошла валидацию! Dice Score: {dice:.4f}")


def notify_success():
    # Задача 5: уведомляем что пайплайн прошёл успешно.
    # В продакшне тут был бы Telegram/Slack/email.
    print("=" * 50)
    print("Пайплайн завершён успешно!")
    print("Новая модель готова к использованию.")
    print("=" * 50)


# ─────────────────────────────────────────
# Создаём DAG
# ─────────────────────────────────────────
with DAG(
    dag_id="brain_tumor_pipeline",
    # dag_id — уникальное имя DAG в Airflow UI

    default_args=default_args,

    description="Полный ML пайплайн: скачать данные → обучить → задеплоить",

    schedule_interval="@weekly",
    # schedule_interval — как часто запускать.
    # @weekly  — раз в неделю
    # @daily   — каждый день
    # "0 2 * * *" — каждый день в 2 ночи (cron формат)
    # None     — только вручную

    start_date=datetime(2024, 1, 1),
    # start_date — с какой даты считать расписание.

    catchup=False,
    # catchup=False — не запускать пропущенные запуски.
    # Если DAG был выключен неделю — не будет
    # наверстывать все пропущенные недели.

    tags=["ml", "brain-tumor", "segmentation"],
    # tags — метки для удобного поиска в Airflow UI

) as dag:

    # ─────────────────────────────────────────
    # Объявляем задачи
    # ─────────────────────────────────────────

    task_download = PythonOperator(
        task_id="download_dataset",
        # task_id — уникальное имя задачи внутри DAG
        python_callable=download_dataset,
        # python_callable — какую функцию вызвать
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task_validate = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
    )

    task_notify = PythonOperator(
        task_id="notify_success",
        python_callable=notify_success,
    )

    # ─────────────────────────────────────────
    # Описываем порядок выполнения задач
    # ─────────────────────────────────────────
    task_download >> task_preprocess >> task_train >> task_validate >> task_notify
    # >> означает "после этого выполни следующее"
    # download → preprocess → train → validate → notify
    #
    # Если task_validate упадёт (Dice < 0.7) —
    # task_notify не запустится.
    # Airflow пометит запуск как failed.