# api/models.py
# Здесь описываем таблицы базы данных через Python классы.
# SQLAlchemy читает эти классы и создаёт реальные таблицы в PostgreSQL.
# Это называется ORM — Object Relational Mapping.
# Смысл: вместо SQL пишем обычный Python.

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from database import Base

# ─────────────────────────────────────────
# Таблица 1: логи предсказаний
# ─────────────────────────────────────────
class PredictionLog(Base):
    # Base — говорит SQLAlchemy что этот класс = таблица в БД
    __tablename__ = "prediction_logs"
    # __tablename__ — реальное имя таблицы в PostgreSQL

    id = Column(Integer, primary_key=True, index=True)
    # Integer      — тип данных: целое число
    # primary_key  — уникальный идентификатор каждой строки
    # index=True   — создаём индекс для быстрого поиска

    filename = Column(String(255))
    # String(255)  — строка максимум 255 символов
    # имя загруженного MRI файла

    confidence = Column(Float)
    # Float        — число с плавающей точкой
    # уверенность модели: 0.0 → 1.0

    tumor_detected = Column(Boolean, default=False)
    # Boolean      — True или False
    # default=False — если не передали, ставим False

    model_version = Column(String(50), default="v1.0")
    # версия модели которая делала предсказание

    error_message = Column(Text, nullable=True)
    # Text         — длинная строка, без ограничения
    # nullable=True — поле может быть пустым (None)
    # если что-то пошло не так — запишем ошибку сюда

    created_at = Column(DateTime, server_default=func.now())
    # DateTime         — дата и время
    # server_default   — PostgreSQL сам проставит текущее время
    # func.now()       — SQL функция NOW()

    def __repr__(self):
        # __repr__ — как объект выглядит при печати в терминале
        # удобно для дебага
        return f"<PredictionLog id={self.id} file={self.filename} tumor={self.tumor_detected}>"


# ─────────────────────────────────────────
# Таблица 2: версии моделей
# ─────────────────────────────────────────
class ModelVersion(Base):
    __tablename__ = "model_versions"
    # Здесь храним информацию о каждой обученной модели.
    # Когда обучили новую модель — пишем запись сюда.

    id = Column(Integer, primary_key=True, index=True)

    version = Column(String(50), unique=True)
    # unique=True — две модели не могут иметь одинаковую версию

    dice_score = Column(Float)
    # метрика качества модели на валидационной выборке

    train_loss = Column(Float)
    val_loss = Column(Float)

    epochs = Column(Integer)
    # сколько эпох обучали

    dataset_size = Column(Integer)
    # на скольких снимках обучали

    is_active = Column(Boolean, default=False)
    # is_active — это текущая активная модель?
    # True только у одной модели — той что сейчас в продакшне

    notes = Column(Text, nullable=True)
    # любые заметки об этой версии модели

    created_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<ModelVersion {self.version} dice={self.dice_score} active={self.is_active}>"