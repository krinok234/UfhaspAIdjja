"""
Генератор синтетических данных для обучения модели.

Это временное решение пока нет реальных данных.
Реальные данные собираются через команду /jmac dc в игре.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_legit_movement(n_ticks: int = 1000) -> np.ndarray:
    """
    Генерирует движения легитного игрока.
    
    Характеристики:
    - Плавные движения с небольшим шумом
    - GCD ошибка близка к 0 (настоящая мышь)
    - Естественные паузы и ускорения
    """
    data = []
    
    # Базовые параметры
    sensitivity = np.random.uniform(0.3, 1.5)  # Чувствительность мыши
    
    prev_delta_yaw = 0
    prev_delta_pitch = 0
    prev_accel_yaw = 0
    prev_accel_pitch = 0
    
    for i in range(n_ticks):
        # Иногда игрок не двигает мышью
        if np.random.random() < 0.3:
            delta_yaw = 0
            delta_pitch = 0
        else:
            # Плавное движение с шумом
            target_yaw = np.random.normal(0, 5 * sensitivity)
            target_pitch = np.random.normal(0, 2 * sensitivity)
            
            # Сглаживание
            delta_yaw = prev_delta_yaw * 0.3 + target_yaw * 0.7
            delta_pitch = prev_delta_pitch * 0.3 + target_pitch * 0.7
            
            # Добавляем человеческий шум
            delta_yaw += np.random.normal(0, 0.5)
            delta_pitch += np.random.normal(0, 0.3)
        
        # Ускорение
        accel_yaw = delta_yaw - prev_delta_yaw
        accel_pitch = delta_pitch - prev_delta_pitch
        
        # Jerk
        jerk_yaw = accel_yaw - prev_accel_yaw
        jerk_pitch = accel_pitch - prev_accel_pitch
        
        # GCD ошибка - у легитов близка к 0
        gcd_error_yaw = np.random.exponential(0.001)
        gcd_error_pitch = np.random.exponential(0.001)
        
        data.append([
            delta_yaw, delta_pitch,
            accel_yaw, accel_pitch,
            jerk_yaw, jerk_pitch,
            gcd_error_yaw, gcd_error_pitch
        ])
        
        prev_delta_yaw = delta_yaw
        prev_delta_pitch = delta_pitch
        prev_accel_yaw = accel_yaw
        prev_accel_pitch = accel_pitch
    
    return np.array(data)


def generate_aimbot_movement(n_ticks: int = 1000) -> np.ndarray:
    """
    Генерирует движения аимбота.
    
    Характеристики:
    - Резкие snap-движения к цели
    - Неестественно точные углы
    - Высокая GCD ошибка (программное движение)
    - Паттерны повторяются
    """
    data = []
    
    prev_delta_yaw = 0
    prev_delta_pitch = 0
    prev_accel_yaw = 0
    prev_accel_pitch = 0
    
    in_snap = False
    snap_duration = 0
    
    for i in range(n_ticks):
        # Случайно начинаем snap к цели
        if not in_snap and np.random.random() < 0.1:
            in_snap = True
            snap_duration = np.random.randint(2, 5)
            snap_target_yaw = np.random.uniform(-30, 30)
            snap_target_pitch = np.random.uniform(-10, 10)
        
        if in_snap:
            # Резкое движение к цели
            delta_yaw = snap_target_yaw / snap_duration
            delta_pitch = snap_target_pitch / snap_duration
            snap_duration -= 1
            if snap_duration <= 0:
                in_snap = False
        else:
            # Небольшие корректировки (слишком точные)
            delta_yaw = np.random.normal(0, 0.5)
            delta_pitch = np.random.normal(0, 0.3)
        
        # Ускорение
        accel_yaw = delta_yaw - prev_delta_yaw
        accel_pitch = delta_pitch - prev_delta_pitch
        
        # Jerk - у аимботов часто резкий
        jerk_yaw = accel_yaw - prev_accel_yaw
        jerk_pitch = accel_pitch - prev_accel_pitch
        
        # GCD ошибка - у аимботов высокая (не кратно чувствительности мыши)
        gcd_error_yaw = np.random.uniform(0.01, 0.1)
        gcd_error_pitch = np.random.uniform(0.01, 0.1)
        
        data.append([
            delta_yaw, delta_pitch,
            accel_yaw, accel_pitch,
            jerk_yaw, jerk_pitch,
            gcd_error_yaw, gcd_error_pitch
        ])
        
        prev_delta_yaw = delta_yaw
        prev_delta_pitch = delta_pitch
        prev_accel_yaw = accel_yaw
        prev_accel_pitch = accel_pitch
    
    return np.array(data)


def generate_killaura_movement(n_ticks: int = 1000) -> np.ndarray:
    """
    Генерирует движения KillAura.
    
    Характеристики:
    - Постоянные мелкие корректировки
    - Слишком стабильное отслеживание
    - Неестественная плавность
    """
    data = []
    
    prev_delta_yaw = 0
    prev_delta_pitch = 0
    prev_accel_yaw = 0
    prev_accel_pitch = 0
    
    # Симулируем отслеживание движущейся цели
    target_angle = 0
    target_pitch = 0
    
    for i in range(n_ticks):
        # Цель двигается
        target_angle += np.random.normal(0, 2)
        target_pitch += np.random.normal(0, 0.5)
        target_pitch = np.clip(target_pitch, -30, 30)
        
        # KillAura слишком точно следует за целью
        delta_yaw = target_angle * 0.8 + np.random.normal(0, 0.1)
        delta_pitch = target_pitch * 0.3 + np.random.normal(0, 0.05)
        
        target_angle *= 0.5  # Затухание
        target_pitch *= 0.5
        
        # Ускорение
        accel_yaw = delta_yaw - prev_delta_yaw
        accel_pitch = delta_pitch - prev_delta_pitch
        
        # Jerk
        jerk_yaw = accel_yaw - prev_accel_yaw
        jerk_pitch = accel_pitch - prev_accel_pitch
        
        # GCD ошибка
        gcd_error_yaw = np.random.uniform(0.005, 0.05)
        gcd_error_pitch = np.random.uniform(0.005, 0.05)
        
        data.append([
            delta_yaw, delta_pitch,
            accel_yaw, accel_pitch,
            jerk_yaw, jerk_pitch,
            gcd_error_yaw, gcd_error_pitch
        ])
        
        prev_delta_yaw = delta_yaw
        prev_delta_pitch = delta_pitch
        prev_accel_yaw = accel_yaw
        prev_accel_pitch = accel_pitch
    
    return np.array(data)


def generate_dataset(
    output_path: str = "data/synthetic_data.csv",
    n_legit_sessions: int = 100,
    n_cheat_sessions: int = 100,
    ticks_per_session: int = 500
):
    """
    Генерирует полный датасет.
    """
    all_data = []
    
    print(f"Generating {n_legit_sessions} legit sessions...")
    for i in range(n_legit_sessions):
        session_data = generate_legit_movement(ticks_per_session)
        for tick in session_data:
            all_data.append({
                'session_id': f'legit_{i}',
                'is_cheating': 0,
                'delta_yaw': tick[0],
                'delta_pitch': tick[1],
                'accel_yaw': tick[2],
                'accel_pitch': tick[3],
                'jerk_yaw': tick[4],
                'jerk_pitch': tick[5],
                'gcd_error_yaw': tick[6],
                'gcd_error_pitch': tick[7]
            })
    
    print(f"Generating {n_cheat_sessions} cheat sessions...")
    for i in range(n_cheat_sessions):
        # Половина аимботы, половина киллауры
        if i % 2 == 0:
            session_data = generate_aimbot_movement(ticks_per_session)
        else:
            session_data = generate_killaura_movement(ticks_per_session)
        
        for tick in session_data:
            all_data.append({
                'session_id': f'cheat_{i}',
                'is_cheating': 1,
                'delta_yaw': tick[0],
                'delta_pitch': tick[1],
                'accel_yaw': tick[2],
                'accel_pitch': tick[3],
                'jerk_yaw': tick[4],
                'jerk_pitch': tick[5],
                'gcd_error_yaw': tick[6],
                'gcd_error_pitch': tick[7]
            })
    
    df = pd.DataFrame(all_data)
    
    # Перемешиваем сессии
    sessions = df['session_id'].unique().tolist()
    np.random.shuffle(sessions)
    session_order = {s: i for i, s in enumerate(sessions)}
    df['session_order'] = df['session_id'].map(session_order)
    df = df.sort_values(['session_order']).drop('session_order', axis=1).reset_index(drop=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Legit: {len(df[df['is_cheating'] == 0])}")
    print(f"Cheat: {len(df[df['is_cheating'] == 1])}")


if __name__ == "__main__":
    generate_dataset()
