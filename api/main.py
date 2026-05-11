from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import models
from database import SessionLocal, engine
import torch
from PIL import Image
import io

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Brain Tumor Detection API",
    description="Загрузи MRI снимок — получи сегментацию опухоли",
    version="1.0.0"
)

ml_model = None

@app.on_event("startup")
async def load_model():
    global ml_model
    try:
        # Импортируем архитектуру UNet
        # model.py лежит рядом в папке api/
        from model import UNet

        # Создаём пустую модель с той же архитектурой
        ml_model = UNet(in_channels=3, out_channels=1)

        # Загружаем сохранённые веса в модель
        # state_dict — это словарь весов, не вся модель
        state_dict = torch.load("/models/model.pth", map_location="cpu")
        ml_model.load_state_dict(state_dict)

        ml_model.eval()
        print("Модель загружена!")
    except FileNotFoundError:
        print("model.pth не найден — сначала запусти обучение")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Brain Tumor API работает!",
        "model_loaded": ml_model is not None
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Сначала запусти обучение."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Ожидается изображение, получен: {file.content_type}"
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = ml_model(tensor)
            confidence = float(output.max())
            tumor_detected = confidence > 0.5

        log = models.PredictionLog(
            filename=file.filename,
            confidence=confidence,
            tumor_detected=tumor_detected,
            model_version="v1.0"
        )
        db.add(log)
        db.commit()
        db.refresh(log)

        return {
            "prediction_id": log.id,
            "filename": file.filename,
            "tumor_detected": tumor_detected,
            "confidence": round(confidence, 4),
            "created_at": str(log.created_at)
        }

    except Exception as e:
        log = models.PredictionLog(
            filename=file.filename,
            confidence=0.0,
            tumor_detected=False,
            error_message=str(e)
        )
        db.add(log)
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def history(
    limit: int = 20,
    tumor_only: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.PredictionLog)

    if tumor_only is not None:
        query = query.filter(
            models.PredictionLog.tumor_detected == tumor_only
        )

    logs = query.order_by(
        models.PredictionLog.created_at.desc()
    ).limit(limit).all()

    return {
        "total": len(logs),
        "predictions": [
            {
                "id": log.id,
                "filename": log.filename,
                "tumor_detected": log.tumor_detected,
                "confidence": log.confidence,
                "model_version": log.model_version,
                "error": log.error_message,
                "created_at": str(log.created_at)
            }
            for log in logs
        ]
    }

@app.get("/stats")
async def stats(db: Session = Depends(get_db)):
    total = db.query(models.PredictionLog).count()
    tumor_count = db.query(models.PredictionLog).filter(
        models.PredictionLog.tumor_detected == True
    ).count()

    return {
        "total_predictions": total,
        "tumor_detected": tumor_count,
        "healthy": total - tumor_count,
        "tumor_rate": round(tumor_count / total, 4) if total > 0 else 0
    }
