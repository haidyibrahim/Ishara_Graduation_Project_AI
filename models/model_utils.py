import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import random
import json

# Automatic Path Discovery (CSV, JSON, BASE_DIR) 

BASE_DIR = "/kaggle/input/competitions/asl-signs"
CSV_PATH = "/kaggle/input/competitions/asl-signs/train.csv"
JSON_PATH ="/kaggle/input/competitions/asl-signs/sign_to_prediction_index_map.json"

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'train.csv':
            CSV_PATH = os.path.join(dirname, filename)
            BASE_DIR = os.path.dirname(CSV_PATH)
        if filename.endswith('.json'):
            JSON_PATH = os.path.join(dirname, filename)

if not CSV_PATH or not JSON_PATH:
    raise FileNotFoundError("Ensure train.csv and the JSON label map are added to the dataset.")

# Setup cache directory for potential training speedup

CACHE_DIR = "/kaggle/working/npy_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# Dataset Class with JSON Mapping and Augmentation
class GoogleSignDataset(Dataset):
    def __init__(self, df, base_dir, json_path, n_frames=30, augment=False):
        self.df = df
        self.base_dir = base_dir
        self.n_frames = n_frames
        self.augment = augment

        # Load label map from JSON to ensure index consistency
        with open(json_path, 'r') as f:
            self.label_map = json.load(f)

    def __len__(self): return len(self.df)

    def temporal_augmentation(self, x):
        #Randomly crops a sequence of frames from the video
        if len(x) > self.n_frames:
            start = random.randint(0, len(x) - self.n_frames)
            return x[start : start + self.n_frames]
        return x

    def spatial_augmentation(self, x):
        #Applies Gaussian noise, scaling, and shifting to the landmarks

        # Noise
        noise = np.random.normal(0, 0.002, x.shape)
        x = x + noise
        # Scale (90% - 110%)
        scale = random.uniform(0.9, 1.1)
        x = x * scale
        # Shift
        shift = random.uniform(-0.05, 0.05)
        x = x + shift
        return x

    def resample(self, x, size):
        #Resizes the frame sequence to a fixed length using linear interpolation
        if len(x) >= size: 
            indices = np.linspace(0, len(x) - 1, size).astype(int)
        else: 
            indices = np.pad(np.arange(len(x)), (0, max(0, size - len(x))), 'edge')
        return x[indices]

    def load_video(self, path):
        #Loads parquet file and converts XYZ coordinates to a numpy array
        full_path = os.path.join(self.base_dir, path)
        try:
            data = pd.read_parquet(full_path)
            # Reshape to (Frames, Landmarks, Coordinates)
            xyz = data[['x', 'y', 'z']].values.reshape(-1, 543, 3)
            # Replace NaNs with zeros immediately
            return np.nan_to_num(xyz, nan=0.0).astype(np.float32)
        except Exception as e:
            # Return zero-filled array in case of corrupted files
            return np.zeros((self.n_frames, 543, 3), dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        landmarks = self.load_video(row['path'])
        
        if self.augment:
            landmarks = self.temporal_augmentation(landmarks)
            landmarks = self.spatial_augmentation(landmarks)

# Split landmarks into Face (0-467) and Body/Hands (468-542)
        face = landmarks[:, 0:468, :].reshape(len(landmarks), -1)
        body = landmarks[:, 468:, :].reshape(len(landmarks), -1)

        # Ensure fixed frame count
        face = self.resample(face, self.n_frames)
        body = self.resample(body, self.n_frames)
        
        # Standardization (Zero Mean, Unit Variance)
        face = (face - face.mean()) / (face.std() + 1e-6)
        body = (body - body.mean()) / (body.std() + 1e-6)
        
        # Get integer label from JSON map
        label_idx = self.label_map[row['sign']]
        
        return torch.tensor(face, dtype=torch.float32), \
               torch.tensor(body, dtype=torch.float32), \
               torch.tensor(label_idx, dtype=torch.long)

#Data Preparation 
full_df = pd.read_csv(CSV_PATH)
# Split into 90% Training and 10% Validation (Stratified by sign)
train_df, val_df = train_test_split(full_df, test_size=0.10, stratify=full_df['sign'], random_state=42)

train_loader = DataLoader(GoogleSignDataset(train_df, BASE_DIR, JSON_PATH, augment=True), 
                          batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(GoogleSignDataset(val_df, BASE_DIR, JSON_PATH, augment=False), 
                        batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

n_classes = len(pd.read_csv(CSV_PATH)['sign'].unique())

#Model Architecture (Dual-Stream LSTM)
class DualStreamLSTM(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Face stream
        self.face_fc = nn.Linear(1404, 256)
        self.face_lstm = nn.LSTM(256, 256, batch_first=True, num_layers=2, dropout=0.3)

        # Body/Hand stream
        self.body_fc = nn.Linear(225, 256)
        self.body_lstm = nn.LSTM(256, 256, batch_first=True, num_layers=2, dropout=0.3)

        # Shared classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def forward(self, face, body):
        # Process face stream
        f = torch.relu(self.face_fc(face))
        _, (h_f, _) = self.face_lstm(f)

        # Process body stream
        b = torch.relu(self.body_fc(body))
        _, (h_b, _) = self.body_lstm(b)

        # Average pooling of the last hidden states
        combined = (h_f[-1] + h_b[-1]) / 2
        return self.classifier(combined)

#Training Setup and Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualStreamLSTM(n_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_acc = 0.0

for epoch in range(1, 101):
    model.train()
    t_loss, t_acc = 0, 0
    for f, b, l in train_loader:
        f, b, l = f.to(device), b.to(device), l.to(device)
        optimizer.zero_grad()
        out = model(f, b)
        loss = criterion(out, l)
        loss.backward()
        # Gradient Clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        t_loss += loss.item()
        t_acc += (out.argmax(1) == l).sum().item()

    # Validation Phase
    model.eval()
    v_loss, v_acc = 0, 0
    with torch.no_grad():
        for f, b, l in val_loader:
            f, b, l = f.to(device), b.to(device), l.to(device)
            out = model(f, b)
            v_loss += criterion(out, l).item()
            v_acc += (out.argmax(1) == l).sum().item()
    # Calculate metrics
    avg_v_acc = v_acc / len(val_df)
    avg_train_loss = t_loss / len(train_loader)
    avg_val_loss = v_loss / len(val_loader)
    avg_train_acc = t_acc / len(train_df)
    avg_val_acc = v_acc / len(val_df)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    # Save the best model
    if avg_v_acc > best_val_acc:
        best_val_acc = avg_v_acc
        
        torch.save(model.state_dict(), "best_asl_model_avg_66_modified.pt")
        print(f"New best model saved with Acc: {best_val_acc:.4f}")
    
    # Epoch summary logs
    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}")