"""
JmAC AI Model - Нейросеть для детекта читов по движению мыши
"""
import torch
import torch.nn as nn


class CheatDetector(nn.Module):
    """
    LSTM модель для анализа последовательности движений мыши.
    
    Входные фичи (8 на каждый тик):
    - delta_yaw, delta_pitch: изменение углов
    - accel_yaw, accel_pitch: ускорение
    - jerk_yaw, jerk_pitch: рывок (производная ускорения)
    - gcd_error_yaw, gcd_error_pitch: ошибка GCD (детект Cinematic Camera)
    """
    
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden*2)
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classification
        output = self.classifier(context)
        return output.squeeze(-1)


class SimpleCheatDetector(nn.Module):
    """
    Простая MLP модель для быстрого inference.
    Принимает усреднённые статистики по последовательности.
    """
    
    def __init__(self, input_size=32, hidden_size=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_features(sequence):
    """
    Извлекает статистические фичи из последовательности тиков.
    
    Args:
        sequence: tensor shape (batch, seq_len, 8)
    
    Returns:
        features: tensor shape (batch, 32)
    """
    # Mean, std, min, max для каждой из 8 фич
    mean = sequence.mean(dim=1)
    std = sequence.std(dim=1)
    min_val = sequence.min(dim=1).values
    max_val = sequence.max(dim=1).values
    
    return torch.cat([mean, std, min_val, max_val], dim=1)
