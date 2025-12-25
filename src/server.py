"""
JmAC AI Server - FastAPI сервер для inference
"""
import os
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


def parse_flatbuffer(data: bytes) -> list:
    """Парсит FlatBuffer данные от Java клиента"""
    try:
        if len(data) < 8:
            return []
        
        # FlatBuffer root offset
        root_offset = struct.unpack('<I', data[0:4])[0]
        root_pos = root_offset
        
        # Table offset
        table_offset = struct.unpack('<i', data[root_pos:root_pos+4])[0]
        table_pos = root_pos - table_offset
        
        # VTable
        vtable_offset = struct.unpack('<i', data[table_pos:table_pos+4])[0]
        vtable_pos = table_pos - vtable_offset
        
        vtable_size = struct.unpack('<H', data[vtable_pos:vtable_pos+2])[0]
        
        # Ticks vector offset (field 0)
        if vtable_size < 6:
            return []
        
        ticks_field_offset = struct.unpack('<H', data[vtable_pos+4:vtable_pos+6])[0]
        if ticks_field_offset == 0:
            return []
        
        ticks_vector_offset_pos = table_pos + ticks_field_offset
        ticks_vector_offset = struct.unpack('<I', data[ticks_vector_offset_pos:ticks_vector_offset_pos+4])[0]
        ticks_vector_pos = ticks_vector_offset_pos + ticks_vector_offset
        
        # Vector length
        num_ticks = struct.unpack('<I', data[ticks_vector_pos:ticks_vector_pos+4])[0]
        
        ticks = []
        for i in range(num_ticks):
            tick_offset_pos = ticks_vector_pos + 4 + i * 4
            tick_offset = struct.unpack('<I', data[tick_offset_pos:tick_offset_pos+4])[0]
            tick_pos = tick_offset_pos + tick_offset
            
            # Parse tick table
            tick_table_offset = struct.unpack('<i', data[tick_pos:tick_pos+4])[0]
            tick_table_pos = tick_pos - tick_table_offset
            
            tick_vtable_offset = struct.unpack('<i', data[tick_table_pos:tick_table_pos+4])[0]
            tick_vtable_pos = tick_table_pos - tick_vtable_offset
            
            tick_vtable_size = struct.unpack('<H', data[tick_vtable_pos:tick_vtable_pos+2])[0]
            
            # Read 8 float fields
            tick_data = []
            for field_idx in range(8):
                field_offset_pos = tick_vtable_pos + 4 + field_idx * 2
                if field_offset_pos + 2 > tick_vtable_pos + tick_vtable_size:
                    tick_data.append(0.0)
                    continue
                    
                field_offset = struct.unpack('<H', data[field_offset_pos:field_offset_pos+2])[0]
                if field_offset == 0:
                    tick_data.append(0.0)
                else:
                    field_pos = tick_table_pos + field_offset
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


def analyze_movement(ticks: list) -> float:
    """
    Анализирует движения и возвращает вероятность читерства.
    Использует статистический анализ вместо нейросети.
    """
    if not ticks or len(ticks) < 10:
        return 0.0
    
    arr = np.array(ticks)
    
    # Индексы: 0=delta_yaw, 1=delta_pitch, 2=accel_yaw, 3=accel_pitch,
    #          4=jerk_yaw, 5=jerk_pitch, 6=gcd_error_yaw, 7=gcd_error_pitch
    
    delta_yaw = arr[:, 0]
    delta_pitch = arr[:, 1]
    accel_yaw = arr[:, 2]
    accel_pitch = arr[:, 3]
    jerk_yaw = arr[:, 4]
    jerk_pitch = arr[:, 5]
    gcd_error_yaw = arr[:, 6]
    gcd_error_pitch = arr[:, 7]
    
    score = 0.0
    
    # 1. GCD Error - высокая ошибка = программное движение
    avg_gcd_error = (np.mean(np.abs(gcd_error_yaw)) + np.mean(np.abs(gcd_error_pitch))) / 2
    if avg_gcd_error > 0.01:
        score += min(0.3, avg_gcd_error * 10)
    
    # 2. Jerk (рывки) - аимботы делают резкие движения
    jerk_std = (np.std(jerk_yaw) + np.std(jerk_pitch)) / 2
    if jerk_std > 5:
        score += min(0.2, jerk_std / 50)
    
    # 3. Слишком стабильное движение - KillAura
    delta_std = np.std(delta_yaw)
    if 0.1 < delta_std < 2.0:  # Подозрительно стабильно
        accel_std = np.std(accel_yaw)
        if accel_std < 1.0:
            score += 0.15
    
    # 4. Snap detection - резкие изменения направления
    snaps = 0
    for i in range(1, len(delta_yaw)):
        if abs(delta_yaw[i]) > 10 and abs(delta_yaw[i-1]) < 2:
            snaps += 1
    if snaps > 3:
        score += min(0.25, snaps * 0.05)
    
    # 5. Паттерн повторения - боты часто повторяют движения
    if len(delta_yaw) >= 20:
        first_half = delta_yaw[:len(delta_yaw)//2]
        second_half = delta_yaw[len(delta_yaw)//2:]
        if len(first_half) == len(second_half):
            correlation = np.corrcoef(first_half, second_half)[0, 1]
            if not np.isnan(correlation) and correlation > 0.8:
                score += 0.2
    
    # Нормализуем
    probability = min(1.0, max(0.0, score))
    
    return probability


@app.get("/v1/inference")
async def inference_get():
    """GET эндпоинт для базовых проверок - возвращает статус сервера"""
    return JSONResponse({
        "status": "ok",
        "message": "AI Inference endpoint. Use POST to send data.",
        "model_loaded": model is not None,
        "model_type": model_type,
        "sequence_length": sequence_length
    })


@app.post("/v1/inference")
async def inference(request: Request):
    """Принимает данные от плагина и анализирует."""
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        try:
            json_data = await request.json()
            ticks_data = []
            for tick in json_data.get("ticks", []):
                ticks_data.append([
                    tick.get("delta_yaw", 0), tick.get("delta_pitch", 0),
                    tick.get("accel_yaw", 0), tick.get("accel_pitch", 0),
                    tick.get("jerk_yaw", 0), tick.get("jerk_pitch", 0),
                    tick.get("gcd_error_yaw", 0), tick.get("gcd_error_pitch", 0)
                ])
        except Exception as e:
            return JSONResponse({"probability": 0.0, "is_cheating": False, "error": str(e)})
    else:
        body = await request.body()
        print(f"Received {len(body)} bytes")
        ticks_data = parse_flatbuffer(body)
        
        if not ticks_data:
            print(f"Parse failed, hex: {body[:50].hex() if body else 'empty'}")
            return JSONResponse({"probability": 0.0, "is_cheating": False})
        
        print(f"Parsed {len(ticks_data)} ticks")
    
    probability = analyze_movement(ticks_data)
    print(f"Analysis result: {probability:.4f}")
    
    return JSONResponse({
        "probability": probability,
        "is_cheating": probability > 0.5
    })


@app.get("/health")
async def health():
    return {"status": "ok", "mode": "lightweight", "sequence_length": sequence_length}


@app.get("/")
async def root():
    return {"status": "JmAC AI Server running", "mode": "lightweight"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
