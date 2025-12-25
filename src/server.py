"""
JmAC AI Server - FastAPI сервер для inference
"""
import torch
import numpy as np
import struct
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from model import CheatDetector, SimpleCheatDetector, extract_features

# Global model
model = None
model_type = None
sequence_length = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель при старте сервера"""
    global model, model_type, sequence_length
    
    model_path = Path("model/cheat_detector.pt")
    
    if not model_path.exists():
        print("WARNING: No model found! Server will return 0.5 probability.")
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
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
    
    yield


app = FastAPI(title="JmAC AI Server", version="1.0.0", lifespan=lifespan)


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


def parse_flatbuffer(data: bytes) -> List[List[float]]:
    """
    Парсит FlatBuffer данные от плагина.
    
    Структура FlatBuffer:
    - TickDataSequence содержит вектор TickData
    - Каждый TickData содержит 8 float полей
    """
    try:
        # FlatBuffer начинается с offset к root table
        if len(data) < 4:
            return []
        
        # Читаем offset к root table
        root_offset = struct.unpack('<I', data[0:4])[0]
        root_pos = root_offset
        
        # Читаем vtable offset (отрицательный offset от начала table)
        vtable_offset = struct.unpack('<i', data[root_pos:root_pos+4])[0]
        vtable_pos = root_pos - vtable_offset
        
        # Читаем размер vtable
        vtable_size = struct.unpack('<H', data[vtable_pos:vtable_pos+2])[0]
        
        # Читаем offset к вектору ticks (первое поле после vtable header)
        if vtable_size < 6:
            return []
        
        ticks_field_offset = struct.unpack('<H', data[vtable_pos+4:vtable_pos+6])[0]
        if ticks_field_offset == 0:
            return []
        
        # Позиция вектора
        vector_offset_pos = root_pos + ticks_field_offset
        vector_offset = struct.unpack('<I', data[vector_offset_pos:vector_offset_pos+4])[0]
        vector_pos = vector_offset_pos + vector_offset
        
        # Читаем длину вектора
        vector_len = struct.unpack('<I', data[vector_pos:vector_pos+4])[0]
        vector_data_pos = vector_pos + 4
        
        ticks = []
        for i in range(vector_len):
            # Каждый элемент - offset к TickData struct
            tick_offset = struct.unpack('<I', data[vector_data_pos + i*4:vector_data_pos + i*4 + 4])[0]
            tick_pos = vector_data_pos + i*4 + tick_offset
            
            # Читаем vtable для TickData
            tick_vtable_offset = struct.unpack('<i', data[tick_pos:tick_pos+4])[0]
            tick_vtable_pos = tick_pos - tick_vtable_offset
            
            # Читаем 8 float полей
            tick_data = []
            for field_idx in range(8):
                field_offset_pos = tick_vtable_pos + 4 + field_idx * 2
                if field_offset_pos + 2 > len(data):
                    tick_data.append(0.0)
                    continue
                    
                field_offset = struct.unpack('<H', data[field_offset_pos:field_offset_pos+2])[0]
                if field_offset == 0:
                    tick_data.append(0.0)
                else:
                    field_pos = tick_pos + field_offset
                    if field_pos + 4 <= len(data):
                        value = struct.unpack('<f', data[field_pos:field_pos+4])[0]
                        tick_data.append(value)
                    else:
                        tick_data.append(0.0)
            
            ticks.append(tick_data)
        
        return ticks
        
    except Exception as e:
        print(f"FlatBuffer parse error: {e}")
        return []


def simple_parse_floats(data: bytes, expected_ticks: int = 40) -> List[List[float]]:
    """
    Простой парсинг - ищем float значения в данных.
    FlatBuffers хранит данные в little-endian формате.
    """
    ticks = []
    
    # Пробуем найти паттерн из 8 float подряд
    # Каждый TickData = 8 floats = 32 bytes
    tick_size = 8 * 4  # 8 floats * 4 bytes
    
    # Ищем начало данных (пропускаем FlatBuffer header)
    # Обычно данные начинаются после нескольких offset'ов
    
    for start in range(0, min(100, len(data)), 4):
        test_ticks = []
        pos = start
        
        while pos + tick_size <= len(data) and len(test_ticks) < expected_ticks:
            tick = []
            valid = True
            
            for j in range(8):
                if pos + 4 > len(data):
                    valid = False
                    break
                value = struct.unpack('<f', data[pos:pos+4])[0]
                # Проверяем что значение разумное (не NaN, не слишком большое)
                if np.isnan(value) or np.isinf(value) or abs(value) > 1000:
                    valid = False
                    break
                tick.append(value)
                pos += 4
            
            if valid and len(tick) == 8:
                test_ticks.append(tick)
            else:
                break
        
        if len(test_ticks) >= expected_ticks // 2:
            return test_ticks
    
    return ticks


@app.post("/v1/inference")
async def inference_binary(request: Request):
    """
    Принимает данные от плагина (FlatBuffers или JSON).
    """
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        # JSON запрос
        try:
            json_data = await request.json()
            ticks_data = []
            for tick in json_data.get("ticks", []):
                ticks_data.append([
                    tick.get("delta_yaw", 0),
                    tick.get("delta_pitch", 0),
                    tick.get("accel_yaw", 0),
                    tick.get("accel_pitch", 0),
                    tick.get("jerk_yaw", 0),
                    tick.get("jerk_pitch", 0),
                    tick.get("gcd_error_yaw", 0),
                    tick.get("gcd_error_pitch", 0)
                ])
        except Exception as e:
            return JSONResponse({"probability": 0.0, "is_cheating": False, "error": str(e)})
    else:
        # Binary (FlatBuffers) запрос
        body = await request.body()
        print(f"Received binary data: {len(body)} bytes")
        
        # Пробуем распарсить FlatBuffer
        ticks_data = parse_flatbuffer(body)
        
        if not ticks_data:
            # Fallback - простой парсинг
            ticks_data = simple_parse_floats(body, sequence_length)
        
        if not ticks_data:
            print(f"Failed to parse data, first 100 bytes: {body[:100].hex()}")
            return JSONResponse({"probability": 0.0, "is_cheating": False, "error": "Failed to parse data"})
        
        print(f"Parsed {len(ticks_data)} ticks")
    
    if model is None:
        return JSONResponse({"probability": 0.5, "is_cheating": False})
    
    if len(ticks_data) != sequence_length:
        return JSONResponse(
            status_code=422,
            content={
                "error": "INVALID_SEQUENCE",
                "details": {"sequence": sequence_length, "received": len(ticks_data)}
            }
        )
    
    # Конвертируем в тензор
    sequence = torch.tensor([ticks_data], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        if model_type == 'simple':
            feat = extract_features(sequence)
            probability = model(feat).item()
        else:
            probability = model(sequence).item()
    
    print(f"Inference result: probability={probability:.4f}")
    
    return JSONResponse({
        "probability": probability,
        "is_cheating": probability > 0.5
    })


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


@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "JmAC AI Server is running", "health": "/health", "inference": "/v1/inference"}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
