import copy
import os
import sys
from collections import defaultdict

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from config import (
        CLIENT_MAP, DATA_DIR, D_MODEL, DROPOUT, LR, MIN_DELTA,
        N_HEADS, N_LAYERS, PATIENCE, WEIGHT_DECAY,
    )
except ImportError:
    from pytorchexample.config import (
        CLIENT_MAP, DATA_DIR, D_MODEL, DROPOUT, LR, MIN_DELTA,
        N_HEADS, N_LAYERS, PATIENCE, WEIGHT_DECAY,
    )

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Steady-DHSV", "Transient-DHSV"]

# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n    = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x    = x + a
        x    = x + self.ff(self.norm2(x))
        return x

# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, class_weights=None):
        n      = logits.size(1)
        smooth = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        smooth = smooth * (1 - self.label_smoothing) + self.label_smoothing / n
        log_p  = F.log_softmax(logits, dim=1)
        ce     = -(smooth * log_p).sum(dim=1)
        pt     = torch.exp(-ce)
        loss   = (1 - pt) ** self.gamma * ce
        if class_weights is not None:
            loss = loss * class_weights[targets]
        return loss.mean()

# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, score):
        if self.best is None or score >= self.best + self.min_delta:
            self.best    = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

# ---------------------------------------------------------------------------
# GlobalModel
# ---------------------------------------------------------------------------

class GlobalModel(nn.Module):
    def __init__(self, all_sensors, window_size=16, num_classes=3,
                 d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.all_sensors = all_sensors
        self.n_sensors   = len(all_sensors)
        self.d_model     = d_model

        self.sensor_embed = nn.Linear(window_size, d_model)

        self.sensor_pos = nn.ParameterDict({
            name.replace('-', '_'): nn.Parameter(torch.randn(d_model) * 0.02)
            for name in all_sensors
        })

        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.out_norm   = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x, sensor_mask):
        x = x.permute(0, 2, 1)
        x = self.sensor_embed(x)

        pos = torch.stack([
            self.sensor_pos[name.replace('-', '_')] for name in self.all_sensors
        ], dim=0)
        x = x + pos.unsqueeze(0)

        if sensor_mask is not None:
            x = x * sensor_mask.unsqueeze(-1)

        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)

        if sensor_mask is not None:
            mask_sum = sensor_mask.sum(dim=1, keepdim=True) + 1e-8
            x = (x * sensor_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        else:
            x = x.mean(dim=1)

        x = self.dropout(self.out_norm(x))
        return self.classifier(x)

    def get_weights(self):
        return {name: param.data.clone() for name, param in self.named_parameters()}

    def load_weights(self, weights):
        state = self.state_dict()
        for k, v in weights.items():
            if k in state:
                state[k] = v.clone()
        self.load_state_dict(state)

# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_client_data(client_name):
    data = np.load(os.path.join(DATA_DIR, f"{client_name}.npz"))
    sensors_csv = os.path.join(DATA_DIR, f"{client_name}_sensors.csv")
    sensor_names = pd.read_csv(sensors_csv)["sensor_name"].tolist()
    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "sensor_names": sensor_names,
        "window_size": data["X_train"].shape[1],
    }

# ---------------------------------------------------------------------------
# Load all clients data (for server-side sensor map + two-phase training)
# ---------------------------------------------------------------------------

def load_all_clients_data():
    clients_data = {}
    for cid in sorted(CLIENT_MAP.values()):
        clients_data[cid] = load_client_data(cid)
    return clients_data

# ---------------------------------------------------------------------------
# Build sensor map (server-side, once at startup)
# ---------------------------------------------------------------------------

def build_sensor_map(clients_data):
    sensor_to_clients = defaultdict(list)
    all_sensors = set()

    for cid, data in clients_data.items():
        for sensor in data["sensor_names"]:
            sensor_to_clients[sensor].append(cid)
            all_sensors.add(sensor)

    all_sensors = sorted(list(all_sensors))

    return {
        "sensor_to_clients": sensor_to_clients,
        "all_sensors": all_sensors,
    }

# ---------------------------------------------------------------------------
# make_loaders
# ---------------------------------------------------------------------------

def make_loaders(X_tr, y_tr, X_te, y_te, batch_size):
    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    Xte = torch.tensor(X_te, dtype=torch.float32)
    yte = torch.tensor(y_te, dtype=torch.long)
    uniq, cnts = torch.unique(ytr, return_counts=True)
    w_cls  = 1.0 / cnts.float()
    w_samp = torch.zeros(len(ytr))
    for cls, w in zip(uniq, w_cls):
        w_samp[ytr == cls] = w
    sampler   = WeightedRandomSampler(w_samp, len(ytr), replacement=True)
    class_w   = (w_cls / w_cls.sum()).to(DEVICE)
    tr_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                           sampler=sampler, drop_last=True)
    tr_eval   = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size,
                           shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size,
                           shuffle=False)
    return tr_loader, tr_eval, te_loader, class_w

# ---------------------------------------------------------------------------
# prepare_client_data — zero-pad to full sensor universe + sensor_mask
# ---------------------------------------------------------------------------

def prepare_client_data(data, sensor_map, batch_size_override=None):
    n_train = len(data["X_train"])
    batch_size = batch_size_override or (
        64 if n_train > 20000 else (32 if n_train > 5000 else 16)
    )

    all_sensors = sensor_map["all_sensors"]
    sensor_mask = torch.zeros(1, len(all_sensors))
    for i, sensor in enumerate(all_sensors):
        if sensor in data["sensor_names"]:
            sensor_mask[0, i] = 1.0
    sensor_mask = sensor_mask.to(DEVICE)

    window = data["X_train"].shape[1]
    X_train_padded = torch.zeros(data["X_train"].shape[0], window, len(all_sensors), dtype=torch.float32)
    X_test_padded  = torch.zeros(data["X_test"].shape[0],  window, len(all_sensors), dtype=torch.float32)

    for j, sensor in enumerate(all_sensors):
        if sensor in data["sensor_names"]:
            idx = data["sensor_names"].index(sensor)
            X_train_padded[:, :, j] = torch.tensor(data["X_train"][:, :, idx])
            X_test_padded[:, :, j]  = torch.tensor(data["X_test"][:, :, idx])

    tr_loader, tr_eval, te_loader, class_w = make_loaders(
        X_train_padded.numpy(), data["y_train"],
        X_test_padded.numpy(),  data["y_test"],
        batch_size,
    )
    return tr_loader, tr_eval, te_loader, class_w, sensor_mask

# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def train_model(model, tr_loader, tr_eval, te_loader, class_w, sensor_mask,
                n_epochs, lr, use_early_stop=True, client_id=""):
    model.to(DEVICE)
    sensor_mask = sensor_mask.to(DEVICE) if sensor_mask is not None else None
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=8, min_lr=1e-6)
    early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

    best_f1, best_state, best_res = 0.0, None, None
    history = []

    n_train = len(tr_loader.dataset)
    print(f"\n{'='*65}")
    print(f"  {client_id} | sensors={model.n_sensors} | train={n_train:,} | batch={tr_loader.batch_size}")
    print(f"  {'Epoch':>6} {'Loss':>8} {'Train F1':>10} {'Test F1':>10} {'LR':>10}")
    print(f"  {'─'*50}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X, sensor_mask), y, class_weights=class_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps      += 1

        avg_loss  = total_loss / steps
        train_res = evaluate(model, tr_eval, sensor_mask)
        test_res  = evaluate(model, te_loader, sensor_mask)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(test_res["f1"])

        history.append({"epoch": epoch, "loss": avg_loss,
                        "train_f1": train_res["f1"], "test_f1": test_res["f1"]})

        if test_res["f1"] > best_f1:
            best_f1    = test_res["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_res   = {"train": train_res, "test": test_res, "epoch": epoch}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:>6} {avg_loss:>8.4f} "
                  f"{train_res['f1']:>10.4f} {test_res['f1']:>10.4f} "
                  f"{current_lr:>10.2e}")

        if use_early_stop and early_stop(test_res["f1"]):
            print(f"\n  early stopping @ epoch {epoch} "
                  f"(best={best_f1:.4f} @ epoch {best_res['epoch']})")
            break

    model.load_state_dict(best_state)

    print(f"\n  best epoch : {best_res['epoch']}")
    print(f"  train F1   : {best_res['train']['f1']:.4f}")
    print(f"  test  F1   : {best_res['test']['f1']:.4f}")
    print(f"  accuracy   : {best_res['test']['accuracy']:.4f}")
    print(classification_report(
        best_res["test"]["labels"], best_res["test"]["preds"],
        target_names=CLASS_NAMES, zero_division=0,
    ))
    return best_res, history

# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, sensor_mask):
    model.eval()
    logits_all, labels_all = [], []
    for X, y in loader:
        logits_all.append(model(X.to(DEVICE), sensor_mask).cpu())
        labels_all.append(y)
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    preds  = np.argmax(logits, axis=1)
    return {
        "f1"      : f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "labels"  : labels,
        "preds"   : preds,
    }

# ---------------------------------------------------------------------------
# fedavg_global_equal_stable
# ---------------------------------------------------------------------------

def fedavg_global_equal_stable(payloads, clients_data, global_weights, alpha=0.7):
    n_clients = len(payloads)
    equal_weight = 1.0 / n_clients
    new_weights = None

    for cid, weights in payloads.items():
        if new_weights is None:
            new_weights = {k: v.clone() * equal_weight for k, v in weights.items()}
        else:
            for k in new_weights:
                if k in weights:
                    new_weights[k] += weights[k] * equal_weight

    if global_weights is not None:
        for k in new_weights:
            if k in global_weights:
                new_weights[k] = alpha * new_weights[k] + (1 - alpha) * global_weights[k]

    return new_weights
