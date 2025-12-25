"""
JmAC AI Training Script - Обучение модели на датасете
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import CheatDetector, SimpleCheatDetector, extract_features


class CheatDataset(Dataset):
    """Датасет для обучения модели"""
    
    def __init__(self, csv_path: str, sequence_length: int = 40):
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        
        # Загружаем CSV
        df = pd.read_csv(csv_path)
        
        # Колонки фич
        feature_cols = [
            'delta_yaw', 'delta_pitch',
            'accel_yaw', 'accel_pitch', 
            'jerk_yaw', 'jerk_pitch',
            'gcd_error_yaw', 'gcd_error_pitch'
        ]
        
        # Группируем по сессиям если есть session_id
        if 'session_id' in df.columns:
            for session_id, group in df.groupby('session_id'):
                self._process_group(group, feature_cols)
        else:
            self._process_group(df, feature_cols)
    
    def _process_group(self, df, feature_cols):
        """Разбивает группу на последовательности"""
        features = df[feature_cols].values
        labels = df['is_cheating'].values
        
        # Скользящее окно
        for i in range(0, len(features) - self.sequence_length + 1, self.sequence_length // 2):
            seq = features[i:i + self.sequence_length]
            if len(seq) == self.sequence_length:
                # Лейбл = большинство в окне
                label = 1 if labels[i:i + self.sequence_length].mean() > 0.5 else 0
                self.sequences.append(seq)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


def train_model(
    data_path: str,
    model_save_path: str = "model/cheat_detector.pt",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    sequence_length: int = 40,
    use_simple_model: bool = False
):
    """
    Обучает модель на датасете.
    
    Args:
        data_path: путь к CSV файлу с данными
        model_save_path: куда сохранить модель
        epochs: количество эпох
        batch_size: размер батча
        learning_rate: learning rate
        sequence_length: длина последовательности
        use_simple_model: использовать простую MLP вместо LSTM
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загружаем данные
    print(f"Loading data from {data_path}...")
    dataset = CheatDataset(data_path, sequence_length)
    
    if len(dataset) == 0:
        print("ERROR: No data loaded! Check your CSV file.")
        return
    
    print(f"Loaded {len(dataset)} sequences")
    
    # Считаем баланс классов
    labels = [dataset.labels[i] for i in range(len(dataset))]
    cheat_count = sum(labels)
    legit_count = len(labels) - cheat_count
    print(f"Class balance: {legit_count} legit, {cheat_count} cheat")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Модель
    if use_simple_model:
        model = SimpleCheatDetector(input_size=32).to(device)
    else:
        model = CheatDetector(input_size=8, hidden_size=64, num_layers=2).to(device)
    
    # Loss с весами для несбалансированных классов
    pos_weight = torch.tensor([legit_count / max(cheat_count, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if not use_simple_model else nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_f1 = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if use_simple_model:
                features = extract_features(sequences)
                outputs = model(features)
            else:
                outputs = model(sequences)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                if use_simple_model:
                    features = extract_features(sequences)
                    outputs = model(features)
                else:
                    outputs = model(sequences)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}")
        print(f"  Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': 'simple' if use_simple_model else 'lstm',
                'sequence_length': sequence_length,
                'best_f1': best_f1
            }, model_save_path)
            print(f"  Saved best model with F1: {best_f1:.4f}")
    
    print(f"\nTraining complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {model_save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train JmAC cheat detection model")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")
    parser.add_argument("--output", type=str, default="model/cheat_detector.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=40, help="Sequence length")
    parser.add_argument("--simple", action="store_true", help="Use simple MLP model")
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_save_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sequence_length=args.seq_len,
        use_simple_model=args.simple
    )
