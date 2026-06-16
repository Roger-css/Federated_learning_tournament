"""DHSV Fault Detection — Flower Client (NumPyClient).

Weight exchange protocol
------------------------
Flower's NumPyClientWrapper passes whatever get_parameters() returns through
ndarrays_to_parameters(), which calls np.save() on each element.  Raw bytes
(pickle output) cannot be saved that way.  The fix: wrap the pickled payload
as a numpy uint8 array — np.save/np.load handles uint8 arrays natively, and
tobytes() recovers the original byte sequence for pickle.loads().

    get_parameters  → [np.frombuffer(pickle.dumps(payload), dtype=np.uint8)]
    set_parameters  → pickle.loads(bytes(parameters[0].tobytes()))
"""

import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import os
import pandas as pd
import flwr as fl
from flwr.client import ClientApp, NumPyClient

from config import CLIENT_MAP, DATA_DIR
from task import DEVICE, iTransformerGlobal


class FlowerClient(NumPyClient):
    """NumPyClient for a single oil-well client."""

    def __init__(
        self,
        model: iTransformerGlobal,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        client_id: str,
    ):
        self.model     = model.to(DEVICE)
        self.X_train   = X_train
        self.y_train   = y_train
        self.X_test    = X_test
        self.y_test    = y_test
        self.client_id = client_id

    # ------------------------------------------------------------------
    # Weight exchange helpers
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        """Serialize shared params as a single numpy uint8 array."""
        payload    = self.model.get_shared_params()
        serialized = pickle.dumps(payload)
        arr        = np.frombuffer(serialized, dtype=np.uint8).copy()
        return [arr]

    def set_parameters(self, parameters):
        """Deserialize and load shared params into the model."""
        arr     = parameters[0]                    # uint8 ndarray from server
        payload = pickle.loads(bytes(arr.tobytes()))
        self.model.load_shared_params(payload)

    # ------------------------------------------------------------------
    # Flower interface — fit
    # ------------------------------------------------------------------

    def fit(self, parameters, config):
        """Train for one FL round and return updated weights + metrics."""
        self.set_parameters(parameters)
        local_epochs = int(config.get("local_epochs", 5))

        best_res, _ = self.model.fit(
            self.X_train, self.y_train,
            self.X_test,  self.y_test,
            client_id    = self.client_id,
            n_epochs     = local_epochs,
            use_early_stop = False,
        )

        metrics = {
            "train_f1":  float(best_res["train"]["f1"]), # type: ignore
            "test_f1":   float(best_res["test"]["f1"]), # type: ignore
            "client_id": self.client_id,
        }
        return self.get_parameters(config={}), len(self.X_train), metrics

    # ------------------------------------------------------------------
    # Flower interface — evaluate
    # ------------------------------------------------------------------

    def evaluate(self, parameters, config):
        """Evaluate on the local test set and return metrics + predictions."""
        self.set_parameters(parameters)

        n          = len(self.X_test)
        batch_size = 64 if n > 20_000 else (32 if n > 5_000 else 16)
        Xte        = torch.tensor(self.X_test, dtype=torch.float32)
        yte        = torch.tensor(self.y_test,  dtype=torch.long)
        te_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size, shuffle=False)

        # Standard metrics (f1, accuracy)
        eval_res = self.model._evaluate(te_loader)

        # Separate forward pass to capture raw logits → softmax → confidences
        # (model._evaluate does not return logits, so we replicate the loop here
        #  without modifying the model class)
        self.model.eval()
        logits_list: list = []
        with torch.no_grad():
            for X, _ in te_loader:
                logits_list.append(self.model(X.to(DEVICE)).cpu())

        logits      = torch.cat(logits_list, dim=0)          # [N, num_classes]
        probs       = F.softmax(logits, dim=1)               # [N, num_classes]
        predictions = probs.argmax(dim=1).tolist()           # list[int]
        confidences = probs.max(dim=1).values.tolist()       # list[float]

        # Flower Scalar values must be bool | bytes | float | int | str
        # → serialize lists as JSON strings
        metrics = {
            "test_f1":     float(eval_res["f1"]),
            "accuracy":    float(eval_res["accuracy"]),
            "client_id":   self.client_id,
            "predictions": json.dumps(predictions),
            "confidences": json.dumps(confidences),
        }
        return 0.0, n, metrics


# ---------------------------------------------------------------------------
# ClientApp entrypoint & client factory
# ---------------------------------------------------------------------------


def client_fn(cid_or_context):
    """Load data + sensors for cid/context and return a FlowerClient."""
    if hasattr(cid_or_context, "node_config"):
        # Context object (from ClientApp)
        node_config = cid_or_context.node_config
        partition_id = node_config.get("partition-id", node_config.get("partition_id", 0))
        cid = str(partition_id)
    else:
        # Simple client ID string (from simulation / start_simulation)
        cid = str(cid_or_context)

    client_name = CLIENT_MAP[cid]

    # Load numpy arrays
    data    = np.load(os.path.join(DATA_DIR, f"{client_name}.npz"))
    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    # Load ordered sensor names
    sensors_csv  = os.path.join(DATA_DIR, f"{client_name}_sensors.csv")
    sensor_names = pd.read_csv(sensors_csv)["sensor_name"].tolist()

    # window_size is always X_train.shape[1] (16)
    window_size = X_train.shape[1]

    model = iTransformerGlobal(sensor_names=sensor_names, window_size=window_size)

    return FlowerClient(
        model     = model,
        X_train   = X_train,
        y_train   = y_train,
        X_test    = X_test,
        y_test    = y_test,
        client_id = client_name,
    ).to_client()


# App entrypoint for flwr run
app = ClientApp(client_fn=client_fn)
