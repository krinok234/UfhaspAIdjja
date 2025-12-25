# JmAC AI Server

AI сервер для анализа движений мыши и детекта читов (аимботы, киллаура).

## Быстрый старт

### 1. Установка зависимостей

```bash
cd ai-server
pip install -r requirements.txt
```

### 2. Генерация синтетических данных (для теста)

```bash
python src/generate_synthetic_data.py
```

Это создаст `data/synthetic_data.csv` с примерами легитных игроков и читеров.

### 3. Обучение модели

```bash
python src/train.py --data data/synthetic_data.csv --epochs 50
```

Модель сохранится в `model/cheat_detector.pt`

### 4. Запуск сервера

```bash
python src/server.py
```

Сервер запустится на `http://localhost:8000`

### 5. Настройка плагина

В `config.yml` плагина:

```yaml
ai:
  enabled: true
  server: "http://localhost:8000/v1/inference"
  api-key: ""  # не нужен для локального сервера
```

## Сбор реальных данных

Синтетические данные — это только для теста. Для хорошей модели нужны реальные данные.

### Сбор данных в игре

1. Включи сбор данных для игрока:
   ```
   /jmac dc start <player> LEGIT
   /jmac dc start <player> CHEAT <описание чита>
   ```

2. Пусть игрок поиграет 5-10 минут

3. Останови сбор:
   ```
   /jmac dc stop <player>
   ```

4. Данные сохранятся в папку плагина

### Глобальный сбор

Для сбора данных со всех игроков:

```
/jmac dc global start UNLABELED
```

## Структура проекта

```
ai-server/
├── data/               # Датасеты
│   └── synthetic_data.csv
├── model/              # Обученные модели
│   └── cheat_detector.pt
├── src/
│   ├── model.py        # Архитектура нейросети
│   ├── train.py        # Скрипт обучения
│   ├── server.py       # FastAPI сервер
│   └── generate_synthetic_data.py
└── requirements.txt
```

## API

### POST /v1/inference

Анализирует последовательность движений.

Request:
```json
{
  "ticks": [
    {
      "delta_yaw": 0.5,
      "delta_pitch": 0.1,
      "accel_yaw": 0.2,
      "accel_pitch": 0.05,
      "jerk_yaw": 0.1,
      "jerk_pitch": 0.02,
      "gcd_error_yaw": 0.001,
      "gcd_error_pitch": 0.001
    }
    // ... 40 тиков
  ]
}
```

Response:
```json
{
  "probability": 0.85,
  "is_cheating": true
}
```

### GET /health

Проверка состояния сервера.

## Фичи модели

Модель анализирует 8 параметров на каждый тик:

| Фича | Описание |
|------|----------|
| delta_yaw | Изменение горизонтального угла |
| delta_pitch | Изменение вертикального угла |
| accel_yaw | Ускорение по yaw |
| accel_pitch | Ускорение по pitch |
| jerk_yaw | Рывок (производная ускорения) по yaw |
| jerk_pitch | Рывок по pitch |
| gcd_error_yaw | Ошибка GCD — детектит программное движение |
| gcd_error_pitch | Ошибка GCD по pitch |

## Советы по обучению

1. **Баланс классов** — нужно примерно равное количество легитов и читеров
2. **Разнообразие** — собирай данные с разных читов (aimbot, killaura, разные настройки)
3. **Реальные данные** — синтетика хороша для теста, но реальные данные критичны
4. **Валидация** — следи за F1 score, не только accuracy
