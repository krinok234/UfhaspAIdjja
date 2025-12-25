"""
JmAC AI Server - FastAPI сервер для inference
"""
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import flatbuffers
from pathlib import Path

from model import CheatDetector, SimpleCheatDetector, extract_features

# FlatBuffers schema (сгенерированный код)
# Пока используем простой парсинг, потом можно добавить сгенерированные классы

app = FastAPI(title="JmAC AI Server", version="1.0.0")

# Global model
model = None
model_type = None
sequence_length = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TickData(BaseModel):
    delta_yaw: float
    delta_pitch: float
    accel_yaw: float
    accel_pitch: float
    jerk_yaw: float
    jerk_pitch: float
    gcd_error_yaw: float
    gcd_error_pitch: float


class InferenceRequest(BaseModel):
    ticks: List[TickData]


class InferenceResponse(BaseModel):
    probability: float
    is_cheating: bool


@app.on_event("startup")
async def load_model():
    """Загружает модель при старте сервера"""
    global model, model_type, sequence_length
    
    model_path = Path("model/cheat_detector.pt")
    
    if not model_path.exists():
        print("WARNING: No model found! Server will return 0.0 probability.")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get('model_type', 'lstm')
    sequence_length = checkpoint.get('sequence_length', 40)
    
    if model_type == 'simple':
        model = SimpleCheatDetector(input_size=32).to(device)
    else:
        model = CheatDetector(input_size=8, hidden_size=64, num_layers=2).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {model_type}, sequence_length={sequence_length}")
    print(f"Best F1 during training: {checkpoint.get('best_f1', 'N/A')}")


@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Анализирует последовательность тиков и возвращает вероятность читерства.
    """
    if model is None:
        return InferenceResponse(probability=0.0, is_cheating=False)
    
    if len(request.ticks) != sequence_length:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_SEQUENCE",
                "details": {"sequence": sequence_length}
            }
        )
    
    # Конвертируем в тензор
    features = []
    for tick in request.ticks:
        features.append([
            tick.delta_yaw, tick.delta_pitch,
            tick.accel_yaw, tick.accel_pitch,
            tick.jerk_yaw, tick.jerk_pitch,
            tick.gcd_error_yaw, tick.gcd_error_pitch
        ])
    
    sequence = torch.tensor([features], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        if model_type == 'simple':
            feat = extract_features(sequence)
            probability = model(feat).item()
        else:
            probability = model(sequence).item()
    
    return InferenceResponse(
        probability=probability,
        is_cheating=probability > 0.5
    )


@app.post("/v1/inference/binary")
async def inference_binary(request: Request):
    """
    Принимает FlatBuffers бинарные данные (как отправляет плагин).
    """
    if model is None:
        return JSONResponse({"probability": 0.0, "is_cheating": False})
    
    body = await request.body()
    
    try:
        # Парсим FlatBuffers
        # Структура: TickDataSequence -> ticks[] -> TickData
        buf = bytearray(body)
        
        # Простой парсинг (без сгенерированного кода)
        # TODO: использовать сгенерированные FlatBuffers классы
        
        # Пока возвращаем заглушку
        return JSONResponse({
            "probability": 0.0,
            "is_cheating": False,
            "error": "Binary parsing not implemented yet. Use JSON endpoint."
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": model_type,
        "sequence_length": sequence_length,
        "device": str(device)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
