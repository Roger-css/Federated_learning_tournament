"""DHSV Fault Detection — Per-Sensor iTransformer Model.

Port of the per-sensor FL approach from Untitled74 (1).ipynb.
Each sensor name gets its own position embedding (via ParameterDict),
shared sensors are aggregated by name on the server, private sensors
stay local.  The classifier head is per-client (local).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Global hyper-parameters
# ---------------------------------------------------------------------------

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42
EPOCHS       = 100
PATIENCE     = 20
MIN_DELTA    = 0.003
D_MODEL      = 64
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 0.01
CLASS_NAMES  = ["Normal", "Steady-DHSV", "Transient-DHSV"]


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
# iTransformerPerSensor  — per-sensor position embeddings + local classifier
# ---------------------------------------------------------------------------

class iTransformerPerSensor(nn.Module):
    """iTransformer with per-sensor positional embeddings (ParameterDict).

    Each sensor *name* gets its own position embedding vector, enabling
    the server to aggregate embeddings for shared sensors by name while
    keeping private-sensor embeddings local.  The classifier head is also
    kept local (per-client).
    """

    def __init__(self, sensor_names, window_size, num_classes=3,
                 d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.sensor_names = sensor_names
        self.n_sensors    = len(sensor_names)
        self.d_model      = d_model

        # Shared backbone
        self.sensor_embed = nn.Linear(window_size, d_model)
        self.input_norm   = nn.LayerNorm(d_model)

        # Per-sensor position embeddings (keyed by sensor name, '-' → '_')
        self.sensor_pos = nn.ParameterDict({
            name.replace("-", "_"): nn.Parameter(torch.randn(d_model) * 0.02)
            for name in sensor_names
        })

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.out_norm  = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

        # Local classifier head (kept per client)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        # x: (B, T, S) → (B, S, T)
        x = x.permute(0, 2, 1)
        x = self.sensor_embed(x)                     # (B, S, d_model)

        # Stack per-sensor position embeddings in sensor order
        pos = torch.stack([
            self.sensor_pos[n.replace("-", "_")]
            for n in self.sensor_names
        ], dim=0)                                    # (S, d_model)
        x = x + pos.unsqueeze(0)                     # (B, S, d_model)

        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.dropout(self.out_norm(x).mean(dim=1))   # (B, d_model)
        return self.classifier(x)                         # (B, num_classes)

    # ------------------------------------------------------------------
    # Payload for Flower — shared weights + per-sensor embeddings
    # ------------------------------------------------------------------

    def get_payload(self):
        """Return dict with shared_weights + sensor_embeddings + sensor_names.

        Shared weights exclude sensor_pos (per-sensor) and classifier (local).
        """
        shared = {}
        for name, param in self.named_parameters():
            if "sensor_pos" not in name and "classifier" not in name:
                shared[name] = param.data.clone()

        sensor_embeddings = {
            sensor: self.sensor_pos[sensor.replace("-", "_")].data.clone()
            for sensor in self.sensor_names
        }

        return {
            "shared_weights"   : shared,
            "sensor_embeddings": sensor_embeddings,
            "sensor_names"     : self.sensor_names,
        }

    def load_global(self, payload):
        """Load aggregated global params into the model.

        Uses .data.copy_() on parameters directly (avoids state_dict
        device-mismatch issues).
        """
        global_shared   = payload["shared_weights"]
        global_sensor   = payload["sensor_embeddings"]

        param_dict = dict(self.named_parameters())
        for k, v in global_shared.items():
            if k in param_dict and "sensor_pos" not in k and "classifier" not in k:
                param_dict[k].data.copy_(v.to(device=param_dict[k].device))

        for sensor in self.sensor_names:
            key = sensor.replace("-", "_")
            if sensor in global_sensor:
                self.sensor_pos[key].data.copy_(
                    global_sensor[sensor].to(device=self.sensor_pos[key].device)
                )

    # ------------------------------------------------------------------
    # Local params (sensor_pos + classifier) persistence
    # ------------------------------------------------------------------

    def save_local_params(self, filepath):
        """Save per-client params: sensor_pos + classifier."""
        sensor_pos_state = {
            k: v.clone().cpu() for k, v in self.sensor_pos.items()
        }
        classifier_state = {
            k: v.clone().cpu() for k, v in self.classifier.state_dict().items()
        }
        torch.save({
            "sensor_pos" : sensor_pos_state,
            "classifier" : classifier_state,
        }, filepath)

    def load_local_params(self, filepath):
        """Restore per-client params: sensor_pos + classifier."""
        import os
        if os.path.isfile(filepath):
            local_state = torch.load(filepath, map_location="cpu", weights_only=True)
            with torch.no_grad():
                if "sensor_pos" in local_state:
                    for k, v in local_state["sensor_pos"].items():
                        self.sensor_pos[k].copy_(v.to(device=self.sensor_pos[k].device))
                if "classifier" in local_state:
                    self.classifier.load_state_dict(local_state["classifier"])
            return True
        return False

    # ------------------------------------------------------------------
    # Data loaders & evaluation helpers
    # ------------------------------------------------------------------

    def _make_loaders(self, X_tr, y_tr, X_te, y_te, batch_size):
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

    @torch.no_grad()
    def _evaluate(self, loader):
        self.eval()
        logits_all, labels_all = [], []
        for X, y in loader:
            logits_all.append(self(X.to(DEVICE)).cpu())
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

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_test, y_test,
            client_id="client", n_epochs=None, use_early_stop=True):
        epochs_to_run = n_epochs if n_epochs is not None else EPOCHS
        n_train       = len(X_train)
        batch_size    = 64 if n_train > 20000 else (32 if n_train > 5000 else 16)

        tr_loader, tr_eval, te_loader, class_w = self._make_loaders(
            X_train, y_train, X_test, y_test, batch_size
        )
        criterion  = FocalLoss(gamma=2.0, label_smoothing=0.05)
        optimizer  = AdamW(self.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler  = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                       patience=8, min_lr=1e-6)
        early_stop = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA)

        best_f1, best_state, best_res = 0.0, None, None
        history = []

        print(f"\n{'='*65}")
        print(f"  {client_id} | sensors={self.n_sensors} | train={n_train:,} | batch={batch_size}")
        print(f"  {'Epoch':>6} {'Loss':>8} {'Train F1':>10} {'Test F1':>10} {'LR':>10}")
        print(f"  {'─'*50}")

        for epoch in range(1, epochs_to_run + 1):
            self.train()
            total_loss, steps = 0.0, 0
            for X, y in tr_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self(X), y, class_weights=class_w)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                steps      += 1

            avg_loss  = total_loss / steps
            train_res = self._evaluate(tr_eval)
            test_res  = self._evaluate(te_loader)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(test_res["f1"])

            history.append({"epoch": epoch, "loss": avg_loss,
                            "train_f1": train_res["f1"], "test_f1": test_res["f1"]})

            if test_res["f1"] > best_f1:
                best_f1    = test_res["f1"]
                best_state = copy.deepcopy(self.state_dict())
                best_res   = {"train": train_res, "test": test_res, "epoch": epoch}

            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>6} {avg_loss:>8.4f} "
                      f"{train_res['f1']:>10.4f} {test_res['f1']:>10.4f} "
                      f"{current_lr:>10.2e}")

            if use_early_stop and early_stop(test_res["f1"]):
                print(f"\n  early stopping @ epoch {epoch} "
                      f"(best={best_f1:.4f} @ epoch {best_res['epoch']})")
                break

        self.load_state_dict(best_state)

        print(f"\n  best epoch : {best_res['epoch']}")
        print(f"  train F1   : {best_res['train']['f1']:.4f}")
        print(f"  test  F1   : {best_res['test']['f1']:.4f}")
        print(f"  accuracy   : {best_res['test']['accuracy']:.4f}")
        print(classification_report(
            best_res["test"]["labels"], best_res["test"]["preds"],
            target_names=CLASS_NAMES, zero_division=0,
        ))
        return best_res, history
