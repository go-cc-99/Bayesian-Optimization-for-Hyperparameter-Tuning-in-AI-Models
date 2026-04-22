#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_3dcnn.py

Training function for 3D CNN GRB localization (w/o MEGAlib/ROOT).
- Caches the binned 3D histograms to avoid rebuilding per BO trial.
- Objective: minimize validation Mean Angular Deviation (MeanAD).
"""

import os
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import hashlib

def file_md5(path, chunk_size=1024*1024):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()[:8]  

@dataclass
class TrainResult:
    meanAD: float  # Validation MeanAD (minimize)
    val_loss: float
    train_loss: float
    best_step: int
    elapsed_sec: float

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _sph_to_cart(theta, phi):
    st = math.sin(theta)
    return (st * math.cos(phi), st * math.sin(phi), math.cos(theta))

def angular_deviation_deg(t_true, p_true, t_pred, p_pred):
    x1, y1, z1 = _sph_to_cart(float(t_true), float(p_true))
    x2, y2, z2 = _sph_to_cart(float(t_pred), float(p_pred))
    dot = x1*x2 + y1*y2 + z1*z2
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

class ComptonHistogramDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class GRBNet(nn.Module):
    def __init__(self, OutputDim=2):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=5, stride=2)   
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1)  
        self.pool  = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=2, stride=2) 
        self.conv4 = nn.Conv3d(128, 128, kernel_size=2, stride=2)
        self.relu  = nn.ReLU(inplace=True)
        self.fc1   = nn.LazyLinear(128)
        self.fc2   = nn.Linear(128, OutputDim)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def _load_or_build_data(cache_dir, inputs_path, labels_path, resolution=5.0):
    """Builds and caches the 3D histograms to save time across BO trials."""
    os.makedirs(cache_dir, exist_ok=True)
    input_hash = file_md5(inputs_path)
    label_hash = file_md5(labels_path)

    cache_file = os.path.join(
        cache_dir,
        f"binned_data_{resolution}deg_{input_hash}_{label_hash}.npz"
    )
    print(f"[CACHE] Using: {cache_file}")
    
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data['X'], data['Y']
        
    print(f"Cache not found. Building 3D histograms (Resolution {resolution}°)...")
    s_list = np.load(inputs_path, allow_pickle=True)
    Y_full = np.load(labels_path, allow_pickle=True).astype(np.float32)
    Y = Y_full[:, 1:3] if Y_full.shape[1] >= 3 else Y_full
    
    PsiMin, PsiMax = -np.pi, +np.pi
    ChiMin, ChiMax =  0.0,   +np.pi
    PhiMin, PhiMax =  0.0,   +np.pi

    PsiBins = int(round(360.0 / resolution))
    ChiBins = int(round(180.0 / resolution))
    PhiBins = int(round(180.0 / resolution))
    
    X = np.zeros((len(s_list), 1, PsiBins, ChiBins, PhiBins), dtype=np.float32)

    for i, events in enumerate(s_list):
        if len(events) == 0: continue
        events = np.asarray(events, dtype=np.float32)
        Chi, Psi, Phi = events[:,0], events[:,1], events[:,2]

        PsiBin = np.floor((Psi - PsiMin)/(PsiMax - PsiMin)*PsiBins).astype(int)
        ChiBin = np.floor((Chi - ChiMin)/(ChiMax - ChiMin)*ChiBins).astype(int)
        PhiBin = np.floor((Phi - PhiMin)/(PhiMax - PhiMin)*PhiBins).astype(int)
        
        PsiBin = np.clip(PsiBin, 0, PsiBins-1)
        ChiBin = np.clip(ChiBin, 0, ChiBins-1)
        PhiBin = np.clip(PhiBin, 0, PhiBins-1)

        lin = ((PsiBin * ChiBins) + ChiBin) * PhiBins + PhiBin
        idx, cnt = np.unique(lin, return_counts=True)

        p_idx = idx // (ChiBins * PhiBins)
        c_idx = (idx // PhiBins) % ChiBins
        f_idx = idx % PhiBins
        X[i, 0, p_idx, c_idx, f_idx] = cnt.astype(np.float32)

    N = min(X.shape[0], Y.shape[0])
    X, Y = X[:N], Y[:N]
    
    np.savez_compressed(cache_file, X=X, Y=Y)
    return X, Y

def eval_on_loader(model, loader, device, criterion):
    model.eval()
    total_loss, total_count = 0.0, 0
    deviations = []
    with torch.no_grad():
        for Xb, Yb in loader:
            Xb, Yb = Xb.to(device, non_blocking=True), Yb.to(device, non_blocking=True)
            preds = model(Xb)
            loss = criterion(preds, Yb)
            bs = Xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            
            preds_np = preds.detach().cpu().numpy()
            y_np = Yb.detach().cpu().numpy()
            for i in range(bs):
                deviations.append(angular_deviation_deg(y_np[i,0], y_np[i,1], preds_np[i,0], preds_np[i,1]))
                
    avg_loss = total_loss / max(1, total_count)
    meanAD = float(np.mean(deviations)) if deviations else float("nan")
    return avg_loss, meanAD

def train_fn(
    params: Dict[str, Any],
    *,
    seed: int = 42,
    inputs_path: str = "./Inputs_ChiPsiPhi_Local_WW.npy",
    labels_path: str = "./Inputs_ThetaPhi_WW.npy",
    output_dir: str = "./cnn_bo_trials",
    cache_dir: str = "./data_cache",
) -> TrainResult:
    
    t0 = time.time()
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparameters from BO
    lr = float(params.get("learning_rate", 1e-3))
    wd = float(params.get("weight_decay", 1e-4))
    batch_size = int(params.get("batch_size", 256))
    max_epochs = int(params.get("max_iterations", 200)) 
    patience = int(params.get("patience", 80))
    
    beta1 = 0.9
    beta2 = 0.999

    X, Y = _load_or_build_data(cache_dir, inputs_path, labels_path)

    # 80/10/10 Split
    X_trainval, _, Y_trainval, _ = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, random_state=42)

    train_loader = DataLoader(ComptonHistogramDataset(X_train, Y_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(ComptonHistogramDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = GRBNet(OutputDim=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
    criterion = nn.MSELoss(reduction="mean")

    best_meanAD = float('inf')
    best_step = 0
    no_improve = 0
    last_train_loss = float("nan")

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, total_count = 0.0, 0
        
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device, non_blocking=True), Yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            preds = model(Xb)
            loss = criterion(preds, Yb)
            loss.backward()
            optimizer.step()
            
            bs = Xb.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            
        last_train_loss = total_loss / max(1, total_count)
        val_loss, meanAD = eval_on_loader(model, val_loader, device, criterion)

        if meanAD < best_meanAD:
            best_meanAD = meanAD
            best_step = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # Clean up GPU memory before the next BO trial
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()

    return TrainResult(
        meanAD=float(best_meanAD),
        val_loss=float(val_loss),
        train_loss=float(last_train_loss),
        best_step=int(best_step),
        elapsed_sec=float(time.time() - t0),
    )