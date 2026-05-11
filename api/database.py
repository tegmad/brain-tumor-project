# api/database.py
# Подключение к PostgreSQL.
# Этот файл импортируют main.py и models.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("DATABASE_URL")
# os.getenv — читает переменную окружения.
# Docker подставит её из docker-compose.yml.
# Например: postgresql://admin:secret123@db:5432/brain_tumor

if not DATABASE_URL:
    raise ValueError("DATABASE_URL не задан! Проверь docker-compose.yml")
# Если забыли передать переменную — сразу падаем с понятной ошибкой.
# Лучше упасть здесь чем получить непонятную ошибку глубже в коде.

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    # pool_pre_ping — перед каждым запросом проверяет
    # что соединение с БД живое.
    # Если нет — переподключается автоматически.
    pool_size=5,
    # pool_size — сколько соединений держим открытыми.
    # Не создаём новое соединение на каждый запрос —
    # берём готовое из пула. Быстрее и экономнее.
    max_overflow=10,
    # max_overflow — сколько дополнительных соединений
    # можно создать если пул заполнен.
    # Итого максимум: pool_size + max_overflow = 15
)

SessionLocal = sessionmaker(
    autocommit=False,
    # autocommit=False — сами решаем когда делать commit.
    # Так мы контролируем транзакции.
    autoflush=False,
    # autoflush=False — не отправляем SQL
    # до явного flush() или commit().
    bind=engine
)

Base = declarative_base()
# Base — родительский класс для всех моделей.
# Все классы в models.py наследуются от него.
# Через Base.metadata SQLAlchemy знает все таблицы.